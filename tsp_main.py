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

from tsp.config import TSPConfig
from tsp.dataset import RandomTSPDataset
from tsp.trajectory import Trajectory as TSPTrajectory
from tsp.bq_network import BQPolicyNetwork
from tsp.lehd_network import LEHDPolicyNetwork

from tqdm import tqdm

from typing import Tuple, List, Optional, Union

"""
Traveling Salesman Problem
===========================
Can be trained in supervised or Gumbeldore-way. Specify in config-file which learning_type to choose.
"""

TSPNetwork = Union[BQPolicyNetwork, LEHDPolicyNetwork]

def get_network(config: TSPConfig, device: torch.device) -> TSPNetwork:
    if config.architecture == "BQ":
        network = BQPolicyNetwork(config, device)
    elif config.architecture == "LEHD":
        network = LEHDPolicyNetwork(config, device)
    else:
        raise ValueError(f"Unknown architecture {config.architecture}")
    return network


def generate_instances(config: TSPConfig):
    """
    Generate random instances for which we sample solutions to use as supervised signal.
    """
    if config.gumbeldore_config["active_search"] is None:
        num_instances = config.gumbeldore_config["num_instances_to_generate"]
        nodes = np.random.random((num_instances, config.num_nodes, 2))
        problem_instances = [{"inst": nodes[i]} for i in range(num_instances)]
    else:
        print(f"Active search with instances from {config.gumbeldore_config['active_search']}")
        with open(config.gumbeldore_config["active_search"], "rb") as f:
            problem_instances = pickle.load(f)

    return problem_instances, config.gumbeldore_config["batch_size_per_worker"], config.gumbeldore_config["batch_size_per_cpu_worker"]


def beam_leaves_to_result(trajectories: List[TSPTrajectory]):
    best_trajectory = sorted(trajectories, key=lambda y: y.objective)[0]
    return best_trajectory.partial_tour.copy(), best_trajectory.objective


def save_search_results_to_dataset(destination_path: str, problem_instances, results, append_to_dataset):
    """
    Assumes all problem instances to be of the same (num_jobs, num_machines)-size.
    Returns the mean generated objective.
    """
    # Each result in `results` is a tuple (tour, objective) (see above)
    dataset = [
        {"inst": instance["inst"], "tour": results[i][0]} for i, instance in enumerate(problem_instances)
    ]

    if not append_to_dataset:
        with open(destination_path, "wb") as f:
            pickle.dump(dataset, f)
    else:
        with open(destination_path, "rb") as f:
            instances = pickle.load(f)

        instances.extend(dataset)
        with open(destination_path, "wb") as f:
            pickle.dump(instances, f)

    return np.array([x[1] for x in results]).mean()


# EVALUATION
def evaluate(eval_type: str, config: TSPConfig, network: TSPNetwork, to_evaluate_path: str, num_instances: Optional[int] = None):
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
        optimal_objectives = np.array([x["sol"] for x in problem_instances])
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
                config=_config, trajectory_cls=TSPTrajectory, generate_instances_fn=load_instances,
                get_network_fn=get_network, beam_leaves_to_result_fn=beam_leaves_to_result, process_search_results_fn=process_search_results
            ).generate_dataset(copy.deepcopy(network.get_weights()), False)

            loggable_results[f"{eval_type} beam width {beam_width}. Obj."] = float(beam_width_results["mean_obj"])
            loggable_results[f"{eval_type} beam width {beam_width}. Opt. gap"] = float(beam_width_results["mean_opt_gap"])
            if beam_width == config.validation_relevant_beam_width:
                # get metric used to decide whether network improved or not
                metric = beam_width_results["mean_opt_gap"]
        return metric, loggable_results
    else:
        results = GumbeldoreDataset(
            config=config, trajectory_cls=TSPTrajectory, generate_instances_fn=load_instances,
            get_network_fn=get_network, beam_leaves_to_result_fn=beam_leaves_to_result,
            process_search_results_fn=process_search_results
        ).generate_dataset(copy.deepcopy(network.get_weights()), False)
        return results["mean_opt_gap"], {
            f"{eval_type} Gumbelore. Obj.": results["mean_obj"],
            f"{eval_type} Gumbelore. Opt. gap": results["mean_opt_gap"]
        }


def validate(config: TSPConfig, network: TSPNetwork):
    return evaluate("Validation", config, network, config.validation_set_path, config.validation_custom_num_instances)


def test(config: TSPConfig, network: TSPNetwork):
    _, loggable_test_metrics = evaluate("Test", config, network, config.test_set_path, None)
    return loggable_test_metrics


# TRAINING

def get_gumbeldore_dataloader(config: TSPConfig, network_weights: dict, append_to_dataset: bool):
    gumbeldore_dataset = GumbeldoreDataset(
        config=config,
        trajectory_cls=TSPTrajectory,
        generate_instances_fn=generate_instances,
        get_network_fn=get_network,
        beam_leaves_to_result_fn=beam_leaves_to_result,
        process_search_results_fn=save_search_results_to_dataset
    )
    mean_generated_obj = gumbeldore_dataset.generate_dataset(network_weights, append_to_dataset)
    print(f"Mean obj of generated data: {mean_generated_obj}")
    print("Training with generated data.")

    torch.cuda.empty_cache()

    time.sleep(10)
    # Load dataset.
    dataset = RandomTSPDataset(model_type=config.architecture,
                               expert_pickle_file=config.gumbeldore_config["destination_path"],
                               batch_size=config.batch_size_training,
                               data_augmentation=config.data_augmentation,
                               data_augmentation_linear_scale=config.data_augmentation_linear_scale,
                               augment_direction=config.augment_direction,
                               custom_num_instances=None,
                               custom_num_batches=config.custom_num_batches)

    return (DataLoader(dataset, batch_size=1, shuffle=True,
                       num_workers=config.num_dataloader_workers, pin_memory=True,
                       persistent_workers=True),
           float(mean_generated_obj))


def get_supervised_dataloader(config: TSPConfig) -> DataLoader:
    print(f"Loading training dataset from {config.training_set_path}")
    dataset = RandomTSPDataset(model_type=config.architecture, expert_pickle_file=config.training_set_path,
                               batch_size=config.batch_size_training,
                               data_augmentation=config.data_augmentation,
                               data_augmentation_linear_scale=config.data_augmentation_linear_scale,
                               augment_direction=config.augment_direction,
                               custom_num_instances=config.custom_num_instances,
                               custom_num_batches=config.custom_num_batches)
    return DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=config.num_dataloader_workers, pin_memory=True,
                            persistent_workers=True)


def train_with_dataloader(config: TSPConfig, dataloader: DataLoader, network: TSPNetwork, optimizer: torch.optim.Optimizer):
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
        data["nodes"] = data["nodes"][0].to(network.device)
        data["next_node_idx"] = data["next_node_idx"][0].to(network.device)

        logits = network(data)
        criterion = CrossEntropyLoss(reduction='mean')
        loss = criterion(logits, data["next_node_idx"])

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


def train_for_one_epoch_gumbeldore(config: TSPConfig, network: TSPNetwork, network_weights: dict,
                                   optimizer: torch.optim.Optimizer, append_to_dataset: bool) -> Tuple[float, dict]:

    dataloader, mean_generated_obj = get_gumbeldore_dataloader(config, network_weights, append_to_dataset)
    avg_loss = train_with_dataloader(config, dataloader, network, optimizer)
    return avg_loss, {"Avg generated obj": float(mean_generated_obj)}


def train_for_one_epoch_supervised(config: TSPConfig, network: TSPNetwork, optimizer: torch.optim.Optimizer, dataloader: DataLoader):
    return train_with_dataloader(config, dataloader, network, optimizer)


if __name__ == '__main__':
    print(f">> TSP <<")
    config = TSPConfig()

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