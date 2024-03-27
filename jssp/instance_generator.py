import numpy as np
import random
from typing import Tuple, List, Dict


class JSSPInstanceGenerator:
    """
    Helper class to generate job shop scheduling instances as described by Taillard in "Benchmarks for basic scheduling
    problems".
    """
    @staticmethod
    def random_instance_taillard(num_jobs: int, num_machines: int) -> np.array:
        """
        Docstring uses Taillard notation.

        Returns
        -------
            [np.array] of shape (J, M, M), where entry (j, i, k) = 0 if k != M_ij else d_ij.
        """
        num_operations = num_machines  # num operations for each job equal number of machines for Taillard.

        # d_ij matrix of processing times. ij entry is processing time of j-th operation of i-th job, which is
        # an integer between 1 and 99.
        processing_time_matrix = np.random.randint(low=1, high=100, size=(num_jobs, num_operations))

        instance = np.zeros((num_jobs, num_operations, num_machines))

        # Step 0 in Taillard (p. 281-282):
        for j in range(num_jobs):
            for i in range(num_operations):
                instance[j, i, i] = processing_time_matrix[j, i]

        # Step 1 in Taillard:
        for j in range(num_jobs):
            for i in range(num_operations - 1):
                random_op = random.randint(i + 1, num_machines - 1)
                M_ij = np.nonzero(instance[j, i])[0][0]
                M_Uj = np.nonzero(instance[j, random_op])[0][0]
                _temp = instance[j, i, M_ij]
                instance[j, i, M_Uj] = instance[j, i, M_ij]
                instance[j, i, M_ij] = 0
                instance[j, random_op, M_ij] = instance[j, random_op, M_Uj]
                instance[j, random_op, M_Uj] = 0

        return instance

    @staticmethod
    def random_instance(num_jobs: int, num_machines: int) -> np.array:
        """
        Docstring uses Taillard notation.

        Returns
        -------
            Dict with keys:
                proc_times: [np.array (int)] of shape (J, O) with processing times of each operation.
                ops_machines: [np.array (int)] of shape (J, O) with machine index of each operation.
        """
        num_operations = num_machines  # num operations for each job equal number of machines for Taillard.

        # d_ij matrix of processing times. ij entry is processing time of j-th operation of i-th job, which is
        # an integer between 1 and 99.
        proc_times = np.random.randint(low=1, high=100, size=(num_jobs, num_operations), dtype=int)
        ops_machines = np.zeros((num_jobs, num_operations), dtype=int)

        for j in range(num_jobs):
            ops_machines[j] = np.random.permutation(num_machines)

        return dict(
            proc_times=proc_times,
            ops_machines=ops_machines,
            op_mask_by_machines=JSSPInstanceGenerator.make_operation_mask_by_machines(ops_machines)
        )

    @staticmethod
    def read_taillard_instance(file_path: str) -> np.array:
        """
        Reads instance in format directly from Taillard website.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        # first line is number of jobs and number of machines
        first = lines[0].split()
        num_jobs = int(first[0])
        num_machines = int(first[1])
        num_operations = num_machines

        # Setup empty instance
        instance = np.zeros((num_jobs, num_operations, num_machines))

        for j in range(num_jobs):
            line = lines[j + 1].split()

            for i in range(num_operations):
                # we always have pairs in the line, where first entry is the machine, and the second entry is the
                # processing time. Machines are indexed from 0
                machine = int(line[i*2])
                time = int(line[i*2 + 1])
                instance[j, i, machine] = time

        return instance

    @staticmethod
    def read_taillard_solution_and_convert_to_job_sequence(file_path_instance: str, file_path_solution: str) -> Tuple[int, List[int]]:
        """
        Reads in a solution to a Taillard problem and converts it into a single sequence of jobs.

        Quote: "In the solutions row i,column k gives the start time
        of job i on machine k."

        Returns:
            [int] Full makespan of solution
            [List[int]] Sequence of job indices resulting in this solution
        """
        instance = JSSPInstanceGenerator.read_taillard_instance(file_path_instance)
        num_jobs = instance.shape[0]
        num_machines = num_operations = instance.shape[2]

        with open(file_path_solution, "r") as f:
            lines = f.readlines()

        makespan = int(lines[0])

        machine_schedules = [[] for _ in range(num_machines)]

        for j in range(num_jobs):
            machine_start_times = lines[j + 1].split()
            for m in range(num_machines):
                i = np.nonzero(instance[j, :, m])[0][0]  # operation index

                start_time = int(machine_start_times[m])
                machine_schedule = machine_schedules[m]
                is_scheduled = False
                for k, (_, _, t) in enumerate(machine_schedule):
                    if start_time < t:
                        is_scheduled = True
                        # prepend current operation
                        machine_schedules[m] = machine_schedule[:k] + [(j, i, start_time)] + machine_schedule[k:]
                        break
                if not is_scheduled:
                    machine_schedule.append((j, i, start_time))

        # From the machine schedules construct a single job sequence now
        job_sequence = []

        job_ops_scheduled = [-1] * num_jobs

        while sum([len(schedule) for schedule in machine_schedules]) > 0:
            for m in range(num_machines):
                schedule = machine_schedules[m]
                if len(schedule):
                    j, i, _ = schedule[0]
                    if job_ops_scheduled[j] == i-1:
                        job_sequence.append(j)
                        job_ops_scheduled[j] = i
                        machine_schedules[m] = schedule[1:]

        return makespan, job_sequence

    @staticmethod
    def read_instance_in_taillard_format(file_path: str):
        """
        Reads an instance which is given in the Taillard specification as explained in
        http://jobshop.jjvh.nl/explanation.php, i.e.:
        On the first line are two numbers, the first is the number of jobs and the second the number of machines.
        Following there are two matrices. The first with a line for each job containing the processing times for each
        operation the second with the order for visiting the machines. The numbering of the machines starts at 1.

        Returns:
            [Dict] of the form {"proc_times": np.array (int), "ops_machines": np.array (int) }
            Where both arrays are of shape (num jobs, num operations) and "proc_times" gives the processing time
            of the operation and "ops_machines" the index of the machine it needs to run (starting at index 0).
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        # first line is number of jobs and number of machines
        first = lines[0].split()
        num_jobs = int(first[0])
        num_machines = int(first[1])
        num_operations = num_machines

        proc_times = np.zeros((num_jobs, num_operations), dtype=int)
        ops_machines = np.zeros((num_jobs, num_operations), dtype=int)

        for j in range(num_jobs):
            job_ops_proc_times = lines[1 + j].split()
            job_ops_machines = lines[1 + num_jobs + j].split()

            for o in range(num_operations):
                proc_times[j, o] = int(job_ops_proc_times[o])
                ops_machines[j, o] = int(job_ops_machines[o]) - 1

        return {
            "proc_times": proc_times,
            "ops_machines": ops_machines
        }

    @staticmethod
    def read_solution_jjvhl(file_path: str, num_jobs: int):
        """
        Reads a single solution from a text file in the following format:
        First line consists of the optimal value.
        Second line (and subsequent lines for other solutions, if present) consists of operations ordered
        in the way they are processed. Here, operations are numbered as follows: The first n operations refer
        to the first operation of each job (according to order of the jobs), operations n+1,...,2n regard the
        second operation of the n jobs, and so on. Numbering of operations starts at 1.
        So operation i is the k'th operation of job j, where k = ceil(i/n) and j = i mod n.

        Returns:
            List[int] Representation of the solution consisting of the indices of the jobs (starting to count from 0)
             of which to schedule the next operation.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        # first line is objective
        objective = int(lines[0])
        # Second line is the solution
        op_numbers_ordered = list(map(int, lines[1].split()))
        job_idx_list_ordered = []
        for op_number in op_numbers_ordered:
            if op_number % num_jobs == 0:
                job_idx = 9
            else:
                job_idx = op_number % num_jobs - 1
            job_idx_list_ordered.append(job_idx)

        return objective, job_idx_list_ordered

    @staticmethod
    def check_validity_of_instance(instance: Dict, objective: int, job_sequence: List[int]) -> bool:
        """
        Given an instance and a job sequence together with the desired objective, checks if the
        job sequence does indeed lead to the objective.
        """
        proc_times = instance["proc_times"]
        ops_machines = instance["ops_machines"]
        num_jobs = proc_times.shape[0]
        num_machines = num_operations = proc_times.shape[1]

        # i-th entry is the time at which the last scheduled operation of i-th job has finished
        job_availability = [0] * num_jobs
        # i-th entry is the time at which the last operation on the i-th machine has finished
        machine_availability = [0] * num_machines
        # i-th entry is the last operation which has been scheduled of the i-th job
        jobs_last_op_scheduled = [-1] * num_jobs

        for job_idx in job_sequence:
            op_idx = jobs_last_op_scheduled[job_idx] + 1
            jobs_last_op_scheduled[job_idx] += 1
            m_idx = ops_machines[job_idx, op_idx]
            end_time = max(job_availability[job_idx], machine_availability[m_idx]) + proc_times[job_idx, op_idx]
            job_availability[job_idx] = end_time
            machine_availability[m_idx] = end_time

        return max(machine_availability) == objective

    @staticmethod
    def make_operation_mask_by_machines(ops_machines: np.array) -> np.array:
        """
        Given np.array (int) `ops_machines` of shape (J, O) which gives the index of the machine
        that an operation must run on, returns a boolean array of shape (J*O, J*O) which can serve as an attention
        mask where an operation can only attend to another operation if it runs on the same machine. Regarding indexing,
        the first O operations belong to the first job, operations O+1,..., 2*O belong to the second job and so on.
        In particular, we have in the resulting boolean matrix that the value at ij is False iff operation i and j run
        on the same machine.
        """
        num_jobs = ops_machines.shape[0]
        num_operations = num_machines = ops_machines.shape[1]

        mask = np.ones((num_jobs*num_operations, num_jobs*num_operations), dtype=bool)

        # keep for each machine (job_idx, op_idx)-tuples of the operations which run on it
        machine_ops_list = [[] for _ in range(num_machines)]
        for j in range(num_jobs):
            for o in range(num_operations):
                machine_ops_list[ops_machines[j, o]].append((j, o))

        for machine_ops in machine_ops_list:
            for j1, o1 in machine_ops:
                idx1 = j1 * num_operations + o1
                for j2, o2 in machine_ops:
                    idx2 = j2 * num_operations + o2
                    # job2, op2 may attend to job1, op1
                    mask[idx1, idx2] = False

        return mask
