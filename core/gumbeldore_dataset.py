import copy
import os
os.environ["RAY_DEDUP_LOGS"] = "0"
import sys
import ray
import torch
import time
import numpy as np
from ray.thirdparty_files import psutil
from tqdm import tqdm

from core.abstracts import Config, Instance, BaseTrajectory
import core.stochastic_beam_search as sbs
from core.beam_search import beam_search as faster_beam_search

from typing import List, Callable, Tuple, Any, Type, Optional

from core.incremental_sbs import IncrementalSBS
from gomoku.config import GomokuConfig
from jssp.config import JSSPConfig


@ray.remote
class JobPool:
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results


class GumbeldoreDataset:
    def __init__(self, config: Config, trajectory_cls: Type[BaseTrajectory],
                 generate_instances_fn: Callable[[Config], Tuple[List[Instance], int, int]],
                 get_network_fn: Callable[[Config, torch.device], torch.nn.Module],
                 beam_leaves_to_result_fn: Callable[[List[sbs.State]], Any],
                 process_search_results_fn: Callable[[str, List[Instance], List[Any], bool], Any]
                ):
        """
        Parameters:
            config: [Config] Problem specific config object.
            trajectory_cls: [Type[BaseTrajectory]] Problem-specific subclass of BaseTrajectory.
            generate_instances_fn: Callable[[Config], Tuple[List[Instance], int, int]]
                Function taking config object and returning:
                    - a list of (random) problem instances to solve.
                    - depending on the generated dataset a batch size for GPU workers
                    - depending on the generated dataset a batch size for CPU workers
            get_network_fn: Callable[[Config, torch.device], torch.nn.Module]
                Method which takes a config object and a device and returns a fresh instance of a policy neural network
                on CPU, but with a device attribute.
            beam_leaves_to_result_fn: Callable[[List[sbs.State]], Any]
                Method which takes a list of finished trajectories obtained from search and returns
                the best trajectory (or other problem specific metrics) to use as solution for the generated
                problem instances.
            process_search_results_fn: Callable[[str, List[Instance], List[Any], bool], Any]
                Takes as arguments:
                    - path where to save dataset
                    - List of original problem instances
                    - List of results corresponding to problem instances
                    - bool flag indicating whether the newly generated data should be appended to the existing or not
                Returns any problem specific metrics needed within the `train_for_one_epoch_gumbeldore`-fn.
        Returns:
            Return values from `process_search_results_fn`.
        """
        self.config = config
        self.gumbeldore_config = config.gumbeldore_config
        self.devices_for_workers: List[str] = self.gumbeldore_config["devices_for_workers"]
        self.generate_instances_fn = generate_instances_fn
        self.beam_leaves_to_result_fn = beam_leaves_to_result_fn
        self.process_search_results_fn = process_search_results_fn
        self.get_network_fn = get_network_fn
        self.trajectory_cls = trajectory_cls

    def generate_dataset(self, network_weights: dict, append_to_existing: bool, memory_aggressive: bool = False):
        """
        Parameters:
            network_weights: [dict] Network weights to use for generating data.
            append_to_existing: [bool] If True, the generated data should be appended to existing dataset.
            memory_aggressive: [bool] If True, IncrementalSBS is performed "memory aggressive" meaning that
                intermediate states in the search tree are not stored after transitioning from them, only their
                policies.

        """
        problem_instances, batch_size_gpu, batch_size_cpu = self.generate_instances_fn(self.config)

        job_pool = JobPool.remote(problem_instances)
        results = [None] * len(problem_instances)

        # Check if we should pin the workers to core
        cpu_cores = [None] * len(self.devices_for_workers)
        if self.gumbeldore_config["pin_workers_to_core"] and sys.platform == "linux":
            # Get available core IDs
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        # Kick off workers
        future_tasks = [
            async_sbs_worker.remote(
                self.config, job_pool, network_weights, device,
                batch_size_gpu if device != "cpu" else batch_size_cpu,
                self.beam_leaves_to_result_fn,
                self.get_network_fn, self.trajectory_cls,
                cpu_cores[i], memory_aggressive
            )
            for i, device in enumerate(self.devices_for_workers)
        ]

        with tqdm(total=len(problem_instances)) as progress_bar:
            while None in results:
                time.sleep(0.001)
                fetched_results = ray.get(job_pool.fetch_results.remote())
                for (i, result) in fetched_results:
                    results[i] = result
                if len(fetched_results):
                    progress_bar.update(len(fetched_results))

        ray.get(future_tasks)
        del job_pool
        del network_weights
        torch.cuda.empty_cache()

        return self.process_search_results_fn(
            self.gumbeldore_config["destination_path"],
            problem_instances, results, append_to_existing
        )


@ray.remote(max_calls=1)
def async_sbs_worker(config: Config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     beam_leaves_to_result_fn: Callable[[List[sbs.State]], Any],
                     get_network_fn: Callable[[Config, torch.device], torch.nn.Module],
                     trajectory_cls: Type[BaseTrajectory],
                     cpu_core: Optional[int] = None,
                     memory_aggressive: bool = False,
                     ):
    def child_log_probability_fn(trajectories: List[BaseTrajectory]) -> [np.array]:
        return trajectory_cls.log_probability_fn(trajectories=trajectories, network=network, to_numpy=True)

    def child_transition_fn(trajectory_action_pairs):
        return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]

    # Pin worker to core if wanted
    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})
        psutil.Process().cpu_affinity([cpu_core])

    with torch.no_grad():

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        device = torch.device(device)
        network = get_network_fn(config, device)
        network.load_state_dict(network_weights)
        network.to(network.device)
        network.eval()

        while True:
            batch = ray.get(job_pool.get_jobs.remote(batch_size))
            if batch is None:
                break

            idx_list = [i for i, _ in batch]
            root_nodes = trajectory_cls.init_batch_from_instance_list(
                instances=[copy.deepcopy(instance) for _, instance in batch],
                network=network,
                device=torch.device(device)
            )

            if config.gumbeldore_config["search_type"] == "beam_search":
                # Deterministic beam search. For CVRP, TSP and JSSP we can use the optimized version as
                # all trajectories in a beam end at the same time.
                if not isinstance(config, GomokuConfig) and not isinstance(config, JSSPConfig):
                    # returns best trajectory per batch item
                    beam_trajectories_batch: List[BaseTrajectory] = faster_beam_search(
                        trajectories=root_nodes,
                        network=network,
                        beam_width=config.gumbeldore_config["beam_width"]
                    )
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = [[sbs.BeamLeaf(state=traj, log_probability=0., gumbel=0.)] for traj in beam_trajectories_batch]
                else:
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                        child_log_probability_fn=child_log_probability_fn,
                        child_transition_fn=child_transition_fn,
                        root_states=root_nodes,
                        beam_width=config.gumbeldore_config["beam_width"],
                        deterministic=True
                    )
            else:
                inc_sbs = IncrementalSBS(root_nodes, child_log_probability_fn, child_transition_fn,
                                         trajectory_cls.to_max_evaluation_fn,
                                         memory_aggressive=memory_aggressive)

                beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_incremental_sbs(
                    beam_width=config.gumbeldore_config["beam_width"],
                    num_rounds=config.gumbeldore_config["num_rounds"],
                    log_prob_update_type=config.gumbeldore_config["search_type"],
                    advantage_constant=config.gumbeldore_config["advantage_constant"],
                    min_max_normalize_advantage=config.gumbeldore_config["min_max_normalize_advantage"],
                    expected_value_use_simple_mean=config.gumbeldore_config["expected_value_use_simple_mean"],
                    use_pure_outcomes=config.gumbeldore_config["use_pure_outcomes"],
                    normalize_advantage_by_visit_count=config.gumbeldore_config["normalize_advantage_by_visit_count"],
                    perform_first_round_deterministic=config.gumbeldore_config["perform_first_round_deterministic"],
                    min_nucleus_top_p=config.gumbeldore_config["min_nucleus_top_p"]
                )

            results_to_push = []
            for j, result_idx in enumerate(idx_list):
                result = beam_leaves_to_result_fn([x.state for x in beam_leaves_batch[j]])
                results_to_push.append((result_idx, result))

            job_pool.push_results.remote(results_to_push)

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()

