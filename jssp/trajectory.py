import torch
import numpy as np
from typing import Optional, Union, List, Tuple
import copy
from jssp.network import JSSPPolicyNetwork, ALiBiPositionalEncoding
from core.abstracts import BaseTrajectory, Instance


class Trajectory(BaseTrajectory):
    """
    Represents a partial JSSP solution used for beam search/rolling out policy/incremental SBS.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.objective: Optional[float] = None  # Makespan
        self.operations_proc_times: Optional[torch.FloatTensor] = None  # operations processing times tensor (J, O, 1), unnormalized (!)
        self.job_earliest_start_time: Optional[torch.FloatTensor] = None  # (J), unnormalized (!)
        self.job_ops_mask: Optional[torch.FloatTensor] = None  # masking already scheduled operations, job-wise, also carries positional info
        self.ops_machines_mask: Optional[torch.BoolTensor] = None  # mask for operations-on-same-machine attention
        self.jobs_next_op_idx: Optional[torch.LongTensor] = None  # idx of next operation to schedule per job
        self.jobs_next_op_machine: Optional[torch.LongTensor] = None  # idx of the machine the next operation of a job has to run on
        self.action_mask: Optional[torch.BoolTensor] = None  # Masking already finished jobs
        self.proc_times: Optional[np.array] = None  # Processing times of the operations
        self.ops_machines: Optional[np.array] = None  # Indicates which on which machine the operations run

        self.num_operations_to_schedule: int = -1  # Number of operations left which need to be scheduled
        self.num_jobs: int = -1
        self.num_operations: int = -1
        self.num_machines: int = -1
        self.job_availability: Optional[torch.FloatTensor] = None
        self.machine_availability: Optional[torch.FloatTensor] = None

        self.job_sequence: List[int] = []  # Stores the ordered sequence of jobs to schedule, i.e. "the solution".

    def copy(self):
        new_traj = Trajectory(self.debug)
        new_traj.objective = self.objective
        new_traj.operations_proc_times = self.operations_proc_times
        new_traj.job_earliest_start_time = self.job_earliest_start_time.clone()
        new_traj.job_ops_mask = self.job_ops_mask.clone()
        new_traj.ops_machines_mask = self.ops_machines_mask.clone()
        new_traj.jobs_next_op_idx = self.jobs_next_op_idx.clone()
        new_traj.jobs_next_op_machine = self.jobs_next_op_machine.clone()
        new_traj.action_mask = self.action_mask.clone()

        new_traj.num_operations_to_schedule = self.num_operations_to_schedule
        new_traj.num_jobs = self.num_jobs
        new_traj.num_operations = self.num_operations
        new_traj.num_machines = self.num_machines
        new_traj.job_availability = self.job_availability.clone()
        new_traj.machine_availability = self.machine_availability.clone()

        # no need to copy
        new_traj.ops_machines = self.ops_machines
        new_traj.proc_times = self.proc_times

        new_traj.job_sequence = self.job_sequence.copy()

        return new_traj

    @staticmethod
    def init_from_instance(instance: dict, device: Optional[Union[torch.device, str]] = None):
        """
        Initializes a Trajectory from an instance as given in JSSPDataset.
        Already sends all tensors to the given device, if specified.
        """
        device = "cpu" if device is None else device
        traj = Trajectory()
        traj.proc_times = instance["proc_times"]
        traj.ops_machines = instance["ops_machines"]
        # Setup basic attributes of problem
        traj.num_jobs = traj.proc_times.shape[0]
        traj.num_operations = traj.proc_times.shape[1]
        traj.num_machines = traj.num_operations
        traj.num_operations_to_schedule = traj.num_jobs * traj.num_operations

        # Setup tensors. To save memory, we only keep the operations_proc_times on the GPU
        operations = np.zeros((traj.num_jobs, traj.num_operations, 1))
        operations[:, :, 0] = traj.proc_times
        traj.operations_proc_times = torch.from_numpy(operations).float().to(device)
        traj.job_availability = torch.zeros(traj.num_jobs, dtype=torch.float)
        traj.machine_availability = torch.zeros(traj.num_machines, dtype=torch.float)
        traj.job_earliest_start_time = torch.zeros(traj.num_jobs, dtype=torch.float)
        traj.jobs_next_op_idx = torch.zeros(traj.num_jobs, dtype=torch.long)
        traj.jobs_next_op_machine = torch.from_numpy(traj.ops_machines[:, 0]).long()
        traj.action_mask = torch.zeros(traj.num_jobs, dtype=torch.bool)

        traj.ops_machines_mask = torch.from_numpy(instance["op_mask_by_machines"]).bool()
        traj.job_ops_mask = ALiBiPositionalEncoding.get_position_matrix(traj.num_operations)[None, :, :]\
            .repeat((traj.num_jobs, 1, 1))  # (J, O, O)

        return traj

    @staticmethod
    def init_batch_from_instance_list(instances: List[Instance], network: torch.nn.Module, device: torch.device):
        return [Trajectory.init_from_instance(instance, device) for instance in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['Trajectory'], network: JSSPPolicyNetwork, to_numpy: bool) -> Union[
        torch.Tensor, List[np.array]]:
        with torch.no_grad():
            batch = Trajectory.trajectories_to_batch(trajectories)
            policy_logits = network(batch)
            batch_log_probs = torch.log_softmax(policy_logits, dim=1)

        if not to_numpy:
            return batch_log_probs

        batch_log_probs = batch_log_probs.cpu().numpy()
        return [batch_log_probs[i] for i in range(len(trajectories))]

    def transition_fn(self, action: int) -> Tuple['Trajectory', bool]:
        new_traj = self.make_copy_and_add_job_idx_to_schedule(action)
        is_finished = new_traj.num_operations_to_schedule == 0
        return new_traj, is_finished

    def to_max_evaluation_fn(self) -> float:
        return -1. * self.objective

    def num_actions(self) -> int:
        return self.num_operations_to_schedule

    def _add_job_idx_to_schedule(self, job_idx: int):
        if self.debug:
            assert self.action_mask[job_idx] == False, f"Job {job_idx} is already finished {self.action_mask[job_idx]}"

        self.job_sequence.append(job_idx)
        self.num_operations_to_schedule -= 1
        op_idx = self.jobs_next_op_idx[job_idx]  # operation idx which will be added to the schedule
        m_idx = self.ops_machines[job_idx, op_idx]
        end_time = self.job_earliest_start_time[job_idx] + self.proc_times[job_idx, op_idx]
        self.job_availability[job_idx] = end_time
        self.machine_availability[m_idx] = end_time

        # Set this operation as scheduled in the masks.

        self.job_ops_mask[job_idx, :, op_idx] = float("-inf")  # this operation should not attend to any other
        self.job_ops_mask[job_idx, op_idx, op_idx] = 0.  # hack to avoid NaNs
        _i = job_idx * self.num_operations + op_idx
        #self.ops_machines_mask[_i, :] = True  # no operation should attend to this
        self.ops_machines_mask[:, _i] = True  # this operation should not attend to any other
        self.ops_machines_mask[_i, _i] = False  # hack to avoid NaNs

        # Increase the next operation index to schedule, or mask the job if it is finished
        if self.jobs_next_op_idx[job_idx] < self.num_operations - 1:
            self.jobs_next_op_idx[job_idx] = self.jobs_next_op_idx[job_idx] + 1
            self.jobs_next_op_machine[job_idx] = self.ops_machines[job_idx, self.jobs_next_op_idx[job_idx]]
        else:
            # job is finished
            self.action_mask[job_idx] = True

        # Finally, update the next earliest time for _each_ unfinished job
        self.job_earliest_start_time = torch.maximum(self.job_availability, self.machine_availability[self.jobs_next_op_machine])

    def make_copy_and_add_job_idx_to_schedule(self, job_idx):
        """
        Returns new trajectory with added operation of given job. This is the method that should be called from outside.
        """
        traj = self.copy()
        traj._add_job_idx_to_schedule(job_idx)

        # If there are only two operations left to schedule, it's trivial and we can just schedule the remaining jobs
        if traj.num_operations_to_schedule == 2:
            for _ in range(2):
                idx = (traj.action_mask == False).nonzero()[0].item()
                traj._add_job_idx_to_schedule(idx)
            traj.objective = traj.machine_availability.max().item()
            traj.clear_memory()

        return traj

    def clear_memory(self):
        # When the trajectory is finished, we will not call any actions on it, so we can delete some properties
        # which are not shared (basically everything that was copied)
        self.job_earliest_start_time = None
        self.job_ops_mask = None
        self.ops_machines_mask = None
        self.jobs_next_op_idx = None
        self.jobs_next_op_machine = None
        self.action_mask = None
        self.job_availability = None
        self.machine_availability = None

    @staticmethod
    def trajectories_to_batch(trajectories: List):
        """
        Given a list of trajectories, returns a dict which can be passed through
        the policy neural network.
        """
        device = trajectories[0].operations_proc_times.device
        num_ops = trajectories[0].num_operations
        jobs_earliest_start = torch.stack([traj.job_earliest_start_time for traj in trajectories], dim=0).to(device)  # (B, J)
        # normalize by subtracting minimum in each batch item
        jobs_earliest_start = jobs_earliest_start - jobs_earliest_start.min(dim=1, keepdim=True)[0]
        jobs_earliest_start = jobs_earliest_start[:, :, None, None].repeat((1, 1, num_ops, 1))
        operations_proc_times = torch.stack([traj.operations_proc_times for traj in trajectories], dim=0)  # (B, J, O, 1)
        operations = torch.cat((operations_proc_times, jobs_earliest_start), dim=3)  # (B, J, O, 2)

        return dict(
            operations=operations / 100.,
            job_ops_mask=torch.stack([traj.job_ops_mask for traj in trajectories], dim=0).to(device),  # (B, J, O, O)
            ops_machines_mask=torch.stack([traj.ops_machines_mask for traj in trajectories], dim=0).to(device),
            jobs_next_op_idx=torch.stack([traj.jobs_next_op_idx for traj in trajectories])[:, :, None].to(device),
            action_mask=torch.stack([traj.action_mask for traj in trajectories]).to(device)
        )
