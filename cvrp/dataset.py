import copy
import time
from typing import Optional, List, Tuple

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from tsp.tsp_geometry import TSPGeometry


class RandomCVRPDataset(Dataset):
    """
    Dataset for supervised training of the Capacitated Vehice Routing Problem given expert solutions (e.g., optimal solutions).
    Data description:
        Each problem instance is a dictionary with entries:
            capacity [float]: Capacity of the vehicle.
            nodes [np.array]: Shape (1 + num_customers, 2). Coordinates of depot (first entry) and customers.
            demands [np.array]: Shape (1 + num_customers). Demands, where first entry is 0 ('demand of depot').
            tour_length [float]: Total tour length of the expert solution.
            tour_node_idcs [[int]]: List of length num_nodes with the order of the customers.
            tour_node_flags [[int]]: List of length num_nodes where each entry is a 1 if the customer is reached
                via the depot in the expert solution, or a 0 if it is reached directly from previous customer.
                Note that the first entry in the list is always 1, as the first customer is always reached via the depot.

    For performance reasons, our model only takes batches with subtours of equal length (so no padding is required).
    In particular, one call to __getitem__ already returns a batch with subtours of equal length.
    Each sample corresponds to a subtour of the final tour where the next node must be chosen and it must be decided
    whether the node is reached via the depot or not. Also, we make the restriction that a subtour must always end
    at the depot, so after the last node of the subtour, the vehicle returns to the depot in the expert solution.

    The action distribution predicted by the network is of length 2*num_nodes and of the form
    [node_0 not via depot, node_0 via depot, node_1 not via depot, node_1 via depot, ...]. Hence, in a subtour, the
    next action to predict is either 0 or 1. If the subtour starts at the depot, we mask the 0th entry.
    Furthermore, we also mask all actions where reaching the customer would exceed the vehicle's capacity (this of course
    only affects the actions where customers are reached not via the depot).

    One point in the dataset represents all the above by a dictionary of the form:
    {
        "current_capacity": (torch.Tensor: (1,)) The current normalized capacity (normalized by full vehicle capacity).
        "demands": (torch.Tensor: (1 + <num_nodes_in_subtour>, 1) Normalized demands of the nodes in the subtour. The
            demand of the depot is assigned a value of 0.
        "nodes": (torch.Tensor: (1 + <num_nodes_in_subtour>, 2)) All nodes of the subtour, where the
            first node is the depot and the last node is the final node in the subtour before returning to depot.
        "start_node_mask": (torch.Tensor: (2, )) Tensor of length 2, where we have a 1 at the 0th position if the subtour
            starts at the depot and a 1 at the 1st position if subtour starts at first node. This is used to later
            add the learnable start node embedding to the correct position.
        "action_mask": (torch.Tensor: (<num_nodes_in_subtour * 2,>) Boolean Tensor where True indicates that the logit
            should be masked, and False else. If we do not start at the depot, then the starting node is
            masked.
        "next_action_idx": Index of target node to choose in subtour (when removing the depot node), and if it
            should be reached via depot or not. This will either correspond directly to the node at position 0 (if
            we start at the depot) or the node at position 1.
    }
    """
    def __init__(self, model_type: str, expert_pickle_file: str, batch_size: int,
                 data_augmentation: bool = False, data_augmentation_linear_scale: bool = False,
                 augment_direction: bool = False, augment_subtour_order: Optional[str] = None,
                 custom_num_instances: Optional[int] = None,
                 custom_num_batches: Tuple[str, int] = None):
        """
        Parameters:
            model_type [str]: Either "BQ" or "LEHD"
            expert_pickle_file [str]: Path to file with expert trajectories.
            batch_size [int]: Number of items to return in __getitem__
            data_augmentation [bool]: If True, a geometric augmentation is performed on the instance
                before choosing subtour.
            data_augmentation_linear_scale [bool]: If True, linear scaling is performed in geometric augmentation.
                Not that this changes the distribution of the points.
            augment_direction [bool]: If True, direction of the subtour is randomly swapped.
            augment_subtour_order [bool]: If True, the subtours of a CVRP solution is randomly permuted.
            custom_num_instances [int]: If given, only the first num instances are taken.
            custom_num_batches [Tuple[str, int]]: If the first entry is "multiplier", the number of batches (i.e. length of dataset)
                is equal to the given value multiplied with the number of instances in the dataset.
                If the first entry is "absolute" the given value is explicitly taken as the number of samples.
        """
        self.model_type = model_type
        self.expert_pickle_file = expert_pickle_file
        self.data_augmentation = data_augmentation
        self.data_augmentation_linear_scale = data_augmentation_linear_scale
        self.augment_direction = augment_direction
        self.augment_subtour_order = augment_subtour_order
        self.batch_size = batch_size
        with open(expert_pickle_file, "rb") as f:
            self.instances = pickle.load(f)

        if custom_num_instances is not None:
            self.instances = self.instances[:custom_num_instances]

        print(f"Loaded dataset. Num items: {len(self.instances)}")

        self.num_nodes = len(self.instances[0]["tour_node_idcs"])
        # One instance corresponds to one random subtour, so
        # length of dataset corresponds to the length of one epoch.
        if custom_num_batches is None:
            self.length = len(self.instances) // self.batch_size
        elif custom_num_batches[0] == "absolute":
            self.length = custom_num_batches[1]
        elif custom_num_batches[0] == "multiplier":
            self.length = custom_num_batches[1] * len(self.instances)
        elif custom_num_batches[0] == "multiplier_max":
            self.length = min(int(custom_num_batches[1] * len(self.instances)), custom_num_batches[2])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        This is for both architectures LEHD and BQ.
        Returns a minibatch of size `batch_size` of random subtours of random length within
        [3, `num_nodes`], where the upper bound corresponds to the full instance.
        :param idx: Is not used, as we directly randomly sample from the tours here.

        A subtour of length N >= 3 is sampled as follows:
            1.) The instance is augmented (random geometrically, random direction)
            2.) As the endpoint of the partial solution must end at the depot, we
                randomly shift the tour such that the flag of the first node is a 1.
            3.) Take N nodes from the right of the sequence (so that the subtour hits the end
                of the full shifted tour).
            4.) If the flag of the first node in the sequence is a 1 (so is reached via depot),
                we randomly decide if the sequence should start at the depot or not.
            5.) Compute the current capacity.

        Returns:
            A batched dictionary corresponding to the main description of the class.
        """
        # Prepare all needed lists
        current_capacities: List[float] = []
        to_stack_demands: List[np.array] = []
        to_stack_nodes: List[np.array] = []
        to_stack_start_node_mask: List[np.array] = []
        to_stack_action_mask: List[np.array] = []
        next_action_idx: List[int] = []

        # Get a random subtour length. We start at subtours of length 3.
        subtour_len = random.randint(3, self.num_nodes)
        for _ in range(self.batch_size):
            # Get random instance and augment
            instance_idx = random.randint(0, len(self.instances) - 1)
            # --- IMPORTANT: We need to copy the instance as we are making several augmentation in-place operations.
            instance = copy.deepcopy(self.instances[instance_idx])
            total_capacity = instance["capacity"]
            # get all starts of the subtours, i.e. nodes where the flags are 1
            subtours_start_idcs = [i for i, flag in enumerate(instance["tour_node_flags"]) if flag == 1]

            if self.augment_direction:
                instance = self._randomly_reverse_subtour_direction(instance, subtours_start_idcs)
            if self.augment_subtour_order is not None:
                instance = self._permute_subtours(instance, permutation_type=self.augment_subtour_order,
                                                  subtours_start_idcs=subtours_start_idcs)
            _nodes = instance["nodes"]
            if self.data_augmentation:
                _nodes = TSPGeometry.random_state_augmentation(_nodes, do_linear_scale=self.data_augmentation_linear_scale)

            # Recompute start idcs
            subtours_start_idcs = [i for i, flag in enumerate(instance["tour_node_flags"]) if flag == 1]

            # Randomly cut the whole tour at a completed subtour so that our desired subtour length still fits in
            all_nodes_with_one = [i for i in subtours_start_idcs if i >= subtour_len]
            j = random.choice(all_nodes_with_one) if len(all_nodes_with_one) else len(instance["tour_node_idcs"])
            tour_node_idcs = instance["tour_node_idcs"][:j]
            tour_node_flags = instance["tour_node_flags"][:j]

            # Get the start index of the subtour to which the first node belongs
            subtour_start_idx = [i for i in subtours_start_idcs if i <= j - subtour_len][-1]

            # Get the subtour to train on
            subtour_node_idcs = tour_node_idcs[-subtour_len:]
            subtour_node_flags = tour_node_flags[-subtour_len:]

            nodes = _nodes[[0] + subtour_node_idcs]  # note that we may not forget the depot
            demands = instance["demands"][[0] + subtour_node_idcs]  # also normalize
            start_at_depot = False
            if subtour_node_flags[0] == 1 and random.randint(0, 1) == 0:
                start_at_depot = True

            # Compute the current capacity when starting the subtour (in the original tour!)
            remaining_capacity = total_capacity
            if not start_at_depot:
                # we only need to do this if we do not start at the depot
                for i in range(subtour_start_idx, len(tour_node_idcs)):
                    idx = tour_node_idcs[i]
                    remaining_capacity = remaining_capacity - instance["demands"][idx]
                    if idx == subtour_node_idcs[0]:
                        break

            start_node_mask = np.zeros(2, dtype=np.uint8)
            start_node_mask[0 if start_at_depot else 1] = 1
            # Prepare action mask and next action to choose.
            action_mask = np.zeros(2 * subtour_len, dtype=bool)
            # If we start at the depot, all possible nodes must be reached via depot obviously
            if start_at_depot:
                next_action = 1  # reach first node via depot
                for i in range(subtour_len):
                    action_mask[i * 2] = True
            else:
                next_action = 2 + subtour_node_flags[1]  # Use actual flag in solution
                # We start at the first node, so we first completely mask that
                action_mask[:2] = True
                # Now iterate over every possible node in the subtour and mask the action
                # corresponding to directly visiting the node (not via depot) if the demand
                # exceeds the vehicle's current capacity
                feasible = remaining_capacity - demands
                for i in range(1, subtour_len):
                    if feasible[i + 1] < 0:  # +1 in index to skip depot
                        action_mask[i * 2] = True

            # Prepare everything for the batch
            current_capacities.append(remaining_capacity / total_capacity)  # normalize
            to_stack_demands.append(demands / total_capacity)  # normalize
            to_stack_nodes.append(nodes)
            to_stack_start_node_mask.append(start_node_mask)
            to_stack_action_mask.append(action_mask)
            next_action_idx.append(next_action)

        return dict(
            current_capacity=torch.tensor(current_capacities, dtype=torch.float)[:, None],  # (B, 1)
            demands=torch.from_numpy(np.stack(to_stack_demands, axis=0)).float(),  # (B, N + 1)
            nodes=torch.from_numpy(np.stack(to_stack_nodes, axis=0)).float(),  # (B, N + 1)
            start_node_mask=torch.from_numpy(np.stack(to_stack_start_node_mask, axis=0)).float(),  # (B, 2)
            action_mask=torch.from_numpy(np.stack(to_stack_action_mask, axis=0)).bool(),
            next_action_idx=torch.LongTensor(next_action_idx)
        )

    @staticmethod
    def _reverse_direction(instance: dict):
        """
        Given a CVRP instance with solution, reverses the direction of the complete solution and returns a non-copied (!) instance
        with the reversed direction.
        """
        rev_instance = instance
        # Everything stays the same except the tour_node_idcs which we can just reverse, and the tour_node_flags
        rev_instance["tour_node_idcs"] = rev_instance["tour_node_idcs"][::-1]
        # For the flags, we simply shift everything one position to the front, and then reverse it (if a node is reached
        # via the depot, this means that the previous node (in the original tour) is reached
        # via the depot in the reversed instance. As the first node is always reached via the depot, we can simply shift
        # the first flag to the last position
        rev_instance["tour_node_flags"] = rev_instance["tour_node_flags"][1:] + [rev_instance["tour_node_flags"][0]]
        rev_instance["tour_node_flags"] = rev_instance["tour_node_flags"][::-1]

        return rev_instance

    @staticmethod
    def _randomly_reverse_subtour_direction(instance: dict, subtours_start_idcs: List[int]):
        """
        Given a CVRP instance with a solution, randomly reverses the direction of the SUBTOURS and returns a non-copied(!)
        instance with the new subtours.

        `subtours_start_idcs` [List[int]]: List of all indices in `tour_node_flags` which are equal to 1.
        """
        new_idcs = []
        for i, start_idx in enumerate(subtours_start_idcs):
            to_idx = len(instance["tour_node_flags"]) if i == len(subtours_start_idcs) - 1 else \
                subtours_start_idcs[i + 1]
            if random.randint(0, 1) == 0:
                new_idcs.extend(instance["tour_node_idcs"][start_idx: to_idx][::-1])
            else:
                new_idcs.extend(instance["tour_node_idcs"][start_idx: to_idx])
        instance["tour_node_idcs"] = new_idcs
        return instance


    @staticmethod
    def _permute_subtours(instance: dict, permutation_type: str, subtours_start_idcs: List[int]):
        """
        Given an instance, returns a non-copied (!) instance where
        the subtours are permuted (which start at the depot and return to the depot).

        `permutation_type` can be either "random" (random permutation) or "sort" (sort subtours
        by their remaining capacity in ascending order).

        `subtours_start_idcs` [List[int]]: List of all indices in `tour_node_flags` which are equal to 1.
        """
        perm_instance = instance
        if permutation_type == "random":
            permutation = np.random.permutation(len(subtours_start_idcs))
        elif permutation_type == "sort":
            # get remaining capacity for each subtour
            remaining_capacities = []
            for i, start_idx in enumerate(subtours_start_idcs):
                to_idx = len(perm_instance["tour_node_flags"]) if i == len(subtours_start_idcs) - 1 else \
                    subtours_start_idcs[i + 1]
                remaining_capacity = perm_instance["capacity"]
                for j in range(start_idx, to_idx):
                    remaining_capacity = remaining_capacity - perm_instance["demands"][j]
                remaining_capacities.append(remaining_capacity)
            # Sort the remaining capacities
            permutation = list(np.argsort(remaining_capacities))
        else:
            raise ValueError(f"Permutation type {permutation_type} unknown.")
        permuted_node_idcs = []
        permuted_node_flags = []
        for i in permutation:
            from_idx = subtours_start_idcs[i]
            to_idx = len(perm_instance["tour_node_flags"]) if i == len(permutation) - 1 else subtours_start_idcs[i + 1]
            permuted_node_idcs = permuted_node_idcs + perm_instance["tour_node_idcs"][from_idx: to_idx]
            permuted_node_flags = permuted_node_flags + perm_instance["tour_node_flags"][from_idx: to_idx]

        # Set everything back
        perm_instance["tour_node_idcs"] = permuted_node_idcs
        perm_instance["tour_node_flags"] = permuted_node_flags
        return perm_instance

