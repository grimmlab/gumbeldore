import copy
import random
import pickle
import torch.optim
import numpy as np
import time

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from core.gumbeldore_dataset import GumbeldoreDataset
from core.train import main_train_cycle

from jssp.config import JSSPConfig
from jssp.dataset import RandomJSSPDataset
from jssp.instance_generator import JSSPInstanceGenerator
from jssp.trajectory import Trajectory as JSSPTrajectory
from jssp.network import JSSPPolicyNetwork

from tqdm import tqdm

from typing import Tuple, List, Optional

"""
Job Shop Scheduling Problem
===========================
Train only with Gumbeldore (no supervised training with expert trajectories).
"""


def get_network(config: JSSPConfig, device: torch.device) -> JSSPPolicyNetwork:
    net = JSSPPolicyNetwork(config, device)
    return net


def generate_instances(config: JSSPConfig):
    """
    Generate random instances for which we sample solutions to use as supervised signal.
    """
    if config.gumbeldore_config["active_search"] is None:
        # sample from the possible sizes to generate
        size_idx = random.randint(0, len(config.problem_sizes_to_generate) - 1)
        num_jobs, num_machines = config.problem_sizes_to_generate[size_idx]
        num_instances = config.gumbeldore_config["num_instances_to_generate"][size_idx]
        problem_instances = [JSSPInstanceGenerator.random_instance(num_jobs, num_machines) for _ in
                             range(num_instances)]
        batch_size_gpu = config.gumbeldore_config["batch_size_per_worker"][size_idx]
        batch_size_cpu = config.gumbeldore_config["batch_size_per_cpu_worker"][size_idx]
        print(f"Problem size: {num_jobs} x {num_machines}")
    else:
        print(f"Active search with instances from {config.gumbeldore_config['active_search']}")
        with open(config.gumbeldore_config["active_search"], "rb") as f:
            problem_instances = pickle.load(f)
        batch_size_gpu = config.gumbeldore_config["batch_size"][0]
        batch_size_cpu = config.gumbeldore_config["batch_size_per_cpu_worker"][0]

    return problem_instances, batch_size_gpu, batch_size_cpu


def beam_leaves_to_result(trajectories: List[JSSPTrajectory]):
    best_trajectory = sorted(trajectories, key=lambda y: y.objective)[0]
    return best_trajectory.job_sequence.copy(), best_trajectory.objective, best_trajectory.num_jobs, best_trajectory.num_machines


def save_search_results_to_dataset(destination_path: str, problem_instances, results, append_to_dataset):
    """
    Assumes all problem instances to be of the same (num_jobs, num_machines)-size.
    Returns the mean generated objective.
    """
    # Each result in `results` is a tuple (job sequence, objective, num_jobs, num_machines) (see above)
    dataset = []
    for i, instance in enumerate(problem_instances):
        instance["job_seq"] = results[i][0]
        dataset.append(instance)

    key = (results[0][2], results[0][3])  # (num_jobs, num_machines)
    if not append_to_dataset:
        with open(destination_path, "wb") as f:
            pickle.dump({key: dataset}, f)
    else:
        with open(destination_path, "rb") as f:
            instances = pickle.load(f)
        if not key in instances:
            instances[key] = []
        instances[key].extend(dataset)
        with open(destination_path, "wb") as f:
            pickle.dump(instances, f)

    return np.array([result[1] for result in results]).mean()


# EVALUATION
def evaluate(eval_type: str, config: JSSPConfig, network: JSSPPolicyNetwork, to_evaluate_path: str, num_instances: Optional[int] = None):
    def load_instances(conf):
        with open(to_evaluate_path, "rb") as f:
            instances = pickle.load(f)
            if num_instances is not None:
                instances = instances[:num_instances]
        return instances, conf.gumbeldore_config["batch_size_per_worker"], conf.gumbeldore_config["batch_size_per_cpu_worker"]

    def process_search_results(destination_path: str, problem_instances, results, append_to_dataset):
        # `destination_path` and `append_to_dataset` are not needed
        # Return mean optimality gap (if optimal objectives are known), else mean objective
        result_objectives = np.array([result[1] for result in results])
        optimal_objectives = None
        if "obj" in problem_instances[0]:
            # if we have an optimal solution, we can use it for optimality gap
            optimal_objectives = np.array([x["obj"] for x in problem_instances])
        mean_opt_gap = None
        if optimal_objectives is not None:
            gaps = (result_objectives - optimal_objectives) / optimal_objectives
            mean_opt_gap = gaps.mean()
        mean_obj = result_objectives.mean()
        return {"mean_obj": mean_obj, "mean_opt_gap": mean_opt_gap}

    if not config.gumbeldore_eval:
        loggable_results = dict()
        metric = None
        for beam_width, batch_size in config.beams_with_batch_sizes.items():
            print(f"Evaluating with beam search (k={beam_width})")
            _config = copy.deepcopy(config)
            _config.gumbeldore_config["search_type"] = "beam_search"
            _config.gumbeldore_config["beam_width"] = beam_width
            _config.gumbeldore_config["devices_for_workers"] = _config.devices_for_eval_workers
            _config.gumbeldore_config["batch_size_per_worker"] = batch_size
            _config.gumbeldore_config["batch_size_per_cpu_worker"] = batch_size
            beam_width_results = GumbeldoreDataset(
                config=_config, trajectory_cls=JSSPTrajectory, generate_instances_fn=load_instances,
                get_network_fn=get_network, beam_leaves_to_result_fn=beam_leaves_to_result, process_search_results_fn=process_search_results
            ).generate_dataset(copy.deepcopy(network.get_weights()), False)

            loggable_results[f"{eval_type} beam width {beam_width}. Obj."] = float(beam_width_results["mean_obj"])
            loggable_results[f"{eval_type} beam width {beam_width}. Opt. gap"] = float(beam_width_results["mean_opt_gap"]) if beam_width_results["mean_opt_gap"] is not None else 0
            if beam_width == config.validation_relevant_beam_width:
                # get metric used to decide whether network improved or not
                metric = beam_width_results["mean_opt_gap"] if beam_width_results["mean_opt_gap"] is not None else beam_width_results["mean_obj"]
        return metric, loggable_results
    else:
        results = GumbeldoreDataset(
            config=config, trajectory_cls=JSSPTrajectory, generate_instances_fn=load_instances,
            get_network_fn=get_network, beam_leaves_to_result_fn=beam_leaves_to_result,
            process_search_results_fn=process_search_results
        ).generate_dataset(copy.deepcopy(network.get_weights()), append_to_existing=False,
                           memory_aggressive=True)
        metric = results["mean_opt_gap"]
        if metric is None:
            results["mean_opt_gap"] = 0.
            metric = results["mean_obj"]
        return metric, {
            f"{eval_type} Gumbelore. Obj.": results["mean_obj"],
            f"{eval_type} Gumbelore. Opt. gap": results["mean_opt_gap"]
        }


def validate(config: JSSPConfig, network: JSSPPolicyNetwork):
    return evaluate("Validation", config, network, config.validation_set_path, config.validation_custom_num_instances)


def test(config: JSSPConfig, network: JSSPPolicyNetwork):
    _, loggable_test_metrics = evaluate("Test", config, network, config.test_set_path, None)
    return loggable_test_metrics


# TRAINING

def get_gumbeldore_dataloader(config: JSSPConfig, network_weights: dict, append_to_dataset: bool):
    gumbeldore_dataset = GumbeldoreDataset(
        config=config,
        trajectory_cls=JSSPTrajectory,
        generate_instances_fn=generate_instances,
        get_network_fn=get_network,
        beam_leaves_to_result_fn=beam_leaves_to_result,
        process_search_results_fn=save_search_results_to_dataset
    )
    mean_generated_obj = gumbeldore_dataset.generate_dataset(network_weights, append_to_dataset, True)
    print(f"Mean obj of generated data: {mean_generated_obj}")
    print("Training with generated data.")

    torch.cuda.empty_cache()

    time.sleep(10)
    # Load dataset.
    dataset = RandomJSSPDataset(expert_pickle_file=config.gumbeldore_config["destination_path"],
                                batch_size=config.batch_size_training,
                                custom_num_instances=None,
                                custom_num_batches=config.custom_num_batches)

    return (DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=config.num_dataloader_workers, pin_memory=True,
                            persistent_workers=True),
            float(mean_generated_obj))


def get_supervised_dataloader(config: JSSPConfig) -> DataLoader:
    print(f"Loading training dataset from {config.training_set_path}")
    dataset = RandomJSSPDataset(expert_pickle_file=config.training_set_path,
                                batch_size=config.batch_size_training,
                                custom_num_instances=config.custom_num_instances,
                                custom_num_batches=config.custom_num_batches)
    return DataLoader(dataset, batch_size=1, shuffle=True,
                      num_workers=config.num_dataloader_workers, pin_memory=True,
                      persistent_workers=True)


def train_with_dataloader(config: JSSPConfig, dataloader: DataLoader, network: JSSPPolicyNetwork, optimizer: torch.optim.Optimizer):
    """
    Iterates over dataloader and trains given network with optimizer.
    """
    network.train()

    accumulated_loss = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(data_iter)
        # Send everything to device.
        for key in ["operations", "job_ops_mask", "ops_machines_mask", "jobs_next_op_idx", "action_mask",
                    "next_action_idx"]:
            data[key] = data[key][0].to(network.device)

        logits = network(data)
        criterion = CrossEntropyLoss(reduction='mean')
        loss = criterion(logits, data["next_action_idx"])

        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()

        batch_loss = loss.item()
        accumulated_loss += batch_loss

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    avg_loss = accumulated_loss / num_batches
    return avg_loss


def train_for_one_epoch_gumbeldore(config: JSSPConfig, network: JSSPPolicyNetwork, network_weights: dict,
                                   optimizer: torch.optim.Optimizer, append_to_dataset: bool) -> Tuple[float, dict]:
    dataloader, mean_generated_obj = get_gumbeldore_dataloader(config, network_weights, append_to_dataset)
    avg_loss = train_with_dataloader(config, dataloader, network, optimizer)
    return avg_loss, {"Avg generated obj": float(mean_generated_obj)}


def train_for_one_epoch_supervised(config: JSSPConfig, network: JSSPPolicyNetwork, optimizer: torch.optim.Optimizer, dataloader: DataLoader):
    return train_with_dataloader(config, dataloader, network, optimizer)


if __name__ == '__main__':
    print(">> JSSP Gumbeldore <<")
    config = JSSPConfig()

    main_train_cycle(
        learning_type=config.learning_type,
        config=config,
        get_network_fn=get_network,
        validation_fn=validate,
        test_fn=test,
        get_supervised_dataloader=get_supervised_dataloader,
        train_for_one_epoch_supervised_fn=train_for_one_epoch_supervised,
        train_for_one_epoch_gumbeldore_fn=train_for_one_epoch_gumbeldore
    )