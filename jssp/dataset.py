import copy
import time
from typing import Optional, List, Tuple

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset

from jssp.network import ALiBiPositionalEncoding


class RandomJSSPDataset(Dataset):
    """
    Dataset for supervised learning of the standard Job Shop Scheduling problem.
    We use for an instance the abbreviation J = number of jobs, O = number of operations, M = number of machines.
    Indexing always starts from 0.
    The pickle file is expected to be a dictionary where each key represents the (J, M)-size as an (int, int)-Tuple,
    and the corresponding value is a list of instances.

    Data description:
        Each problem instance is a dictionary with the following entries, which describe the instance and the given solution:
            proc_times [np.array (int)]: Numpy array of shape (J, O) with the processing time
                for each operation. Processing time is an integer between 1 and 99.
            ops_machines [np.array (int)]: Numpy array of shape (J, O) with the index of the
                mmachine the operation has to run on.
            title [str]: Optional title of the instance to identify it (used for instances from
                the literature)
            obj [int]: Makespan of the solution.
            job_seq [List[int]]: Solution of the given instance, represented by the
                ordered sequence of jobs whose next operation should be scheduled.
            op_mask_by_machines [np.array (bool)]: Boolean array of shape (J*O, J*O) which can serve as an attention
                mask where an operation can only attend to another operation if it runs on the same machine. Regarding indexing,
                the first O operations belong to the first job, operations O+1,..., 2*O belong to the second job and so on.
                In particular, we have in the resulting boolean matrix that the value at ij is False iff operation i and j run
                on the same machine.
    """

    def __init__(self, expert_pickle_file: str, batch_size: int,
                 custom_num_instances: Optional[int] = None,
                 custom_num_batches: Optional[Tuple[str, int]] = None):
        """
        Parameters:
            expert_pickle_file [str]: Path to file with expert trajectories.
            batch_size [int]: Number of items to return in __getitem__
            custom_num_instances [int]: If given, only the first num instances are taken.
            custom_num_batches [int]: If given, the length of the dataset is set to this value.
        """
        self.expert_pickle_file = expert_pickle_file
        self.batch_size = batch_size
        self.instances = dict()
        with open(expert_pickle_file, "rb") as f:
            self.instances = pickle.load(f)

        num_instances = sum([len(self.instances[k]) for k in self.instances])

        if custom_num_instances is not None:
            for key in self.instances:
                self.instances[key] = self.instances[key][:custom_num_instances]

        print("Loaded dataset. Num items:")
        for key in self.instances:
            print(f"{key}: {len(self.instances[key])}")
        # One instance corresponds to one random subschedule, so
        # length of dataset corresponds to the length of one epoch.
        if custom_num_batches is None:
            self.length = num_instances // self.batch_size
        elif custom_num_batches[0] == "absolute":
            self.length = custom_num_batches[1]
        elif custom_num_batches[0] == "multiplier":
            self.length = custom_num_batches[1] * num_instances
        elif custom_num_batches[0] == "multiplier_max":
            self.length = min(custom_num_batches[1] * num_instances, custom_num_batches[2])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns a minibatch of size `batch_size` of random subschedules of random length within
        [3, num_jobs*num_operations], where the upper bound corresponds to the full instance.
        Subschedules where only 2 or less operations need to be scheduled are trivial.
        :param idx: Is not used, as we directly randomly sample from the tours here.

        A subschedule is sampled by randomly choosing an integer k in [0, num_jobs*num_operations - 3) and then
        taking the job subsequence in the solution starting from index k until the end.

        A single subschedule is then represented for the network as a dictionary with the following keys:
            "operations": (torch.FloatTensor (J, O, 2)) Each operation has two entries: Its processing
                time and the earliest time that the next operation of its parent job can start (so the second entry
                is the same for all operations of the same job). The second entry is shifted by subtracting the minimum
                and both entries are divided by 100 to scale operation times between (0,1).
            "job_ops_mask": (torch.BoolTensor (J, O)) Boolean tensor with `True` iff the operation
                has already been scheduled. Is used as key_padding_mask to mask out already scheduled operations when
                computing attention between job-wise operations.
            "ops_machines_mask": (torch.BoolTensor (J*O, J*O)) As `op_mask_by_machines` in the dataset above. Here,
                we also mask out (i.e. set to `True`) all rows and columns corresponding to already scheduled operations.
            "jobs_next_op_idx": (torch.LongTensor (J,)) Indicating for each tensor the index of the next operation to schedule,
                so this can be used to gather logits.
            "action_mask": (torch.BoolTensor (J,)) `True` for all jobs which are already finished.
            "next_action_idx": (int) Target index of next job (within the solution) from which next operation should be
                scheduled.
        """
        to_stack_operations = []
        to_stack_job_ops_mask = []
        to_stack_ops_machines_mask = []
        to_stack_jobs_next_op_idx = []
        to_stack_action_mask = []
        next_action_idx_list = []

        # We first sample one size for this batch.
        size_to_use = random.choice(list(self.instances.keys()))
        instances = self.instances[size_to_use]
        num_jobs, num_operations = instances[0]["proc_times"].shape
        num_machines = num_operations

        for _ in range(self.batch_size):
            instance_idx = random.randint(0, len(instances) - 1)
            # copy the instance
            instance = copy.deepcopy(instances[instance_idx])
            subschedule_from_idx = random.randint(0, num_jobs*num_operations - 3 - 1)
            already_scheduled_job_idcs = instance["job_seq"][:subschedule_from_idx]

            # We will now set everything as if the schedule would be started, iterate over all already scheduled idcs
            # and iteratively update the schedule state.
            # i-th entry is the time at which the last scheduled operation of i-th job has finished
            job_availability = [0] * num_jobs
            # i-th entry is the time at which the last operation on the i-th machine has finished
            machine_availability = [0] * num_machines
            # i-th entry is the last operation which has been scheduled of the i-th job
            jobs_last_op_scheduled = [-1] * num_jobs

            operations = np.zeros((num_jobs, num_operations, 2))
            operations[:, :, 0] = instance["proc_times"]

            jobs_ops_mask = ALiBiPositionalEncoding.get_position_matrix(num_operations)[None, :, :]\
                .repeat((num_jobs, 1, 1))  # (J, O, O)
            ops_machines_mask = instance["op_mask_by_machines"]
            jobs_next_op_idx = np.zeros(num_jobs, dtype=int)
            action_mask = np.zeros(num_jobs, dtype=bool)

            for job_idx in already_scheduled_job_idcs:
                op_idx = jobs_last_op_scheduled[job_idx] + 1
                jobs_last_op_scheduled[job_idx] += 1
                m_idx = instance["ops_machines"][job_idx, op_idx]
                end_time = max(job_availability[job_idx], machine_availability[m_idx]) + instance["proc_times"][job_idx, op_idx]
                job_availability[job_idx] = end_time
                machine_availability[m_idx] = end_time

                # Set this operation as scheduled in the masks. Do not mask the last operation of a job to avoid NaNs
                jobs_ops_mask[job_idx, :, op_idx] = float("-inf")  # this operation should not attend to any other
                jobs_ops_mask[job_idx, op_idx, op_idx] = 0.  # hack to avoid NaNs
                _i = job_idx * num_operations + op_idx
                ops_machines_mask[_i, :] = True  # no operation should attend to this
                ops_machines_mask[:, _i] = True  # this operation should not attend to any other
                ops_machines_mask[_i, _i] = False  # hack to avoid NaNs

                # Increase the next operation index to schedule, or mask the job if it is finished
                if jobs_next_op_idx[job_idx] < num_operations - 1:
                    jobs_next_op_idx[job_idx] += 1
                else:
                    # job is finished
                    action_mask[job_idx] = True

                # Finally, update the next earliest time for _each_ unfinished job
                for _j in range(num_jobs):
                    if not action_mask[_j]:
                        next_op_idx = jobs_last_op_scheduled[_j] + 1
                        _m = instance["ops_machines"][_j, next_op_idx]
                        operations[_j, :, 1] = max(job_availability[_j], machine_availability[_m])

            # Shift and scale operations
            operations[:, :, 1] = operations[:, :, 1] - np.min(operations[:, :, 1])
            operations = operations / 100

            to_stack_operations.append(operations)
            to_stack_job_ops_mask.append(jobs_ops_mask)
            to_stack_ops_machines_mask.append(ops_machines_mask)
            to_stack_jobs_next_op_idx.append(np.array(jobs_next_op_idx, dtype=int))
            to_stack_action_mask.append(action_mask)
            next_action_idx_list.append(instance["job_seq"][subschedule_from_idx])

        return dict(
            operations=torch.from_numpy(np.stack(to_stack_operations, axis=0)).float(),  # (B, J, O, 2)
            job_ops_mask=torch.stack(to_stack_job_ops_mask, dim=0).float(),  # (B, J, O, O)
            ops_machines_mask=torch.from_numpy(np.stack(to_stack_ops_machines_mask, axis=0)).bool(),  # (B, J*O, J*O)
            jobs_next_op_idx=torch.from_numpy(np.stack(to_stack_jobs_next_op_idx, axis=0)).long()[:, :, None],  # (B, J, 1)
            action_mask=torch.from_numpy(np.stack(to_stack_action_mask, axis=0)).bool(),  # (B, J)
            next_action_idx=torch.tensor(next_action_idx_list, dtype=torch.long)  # (B,)
        )
