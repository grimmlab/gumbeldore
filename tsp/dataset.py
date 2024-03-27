from typing import Optional, Tuple

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from tsp.tsp_geometry import TSPGeometry


class RandomTSPDataset(Dataset):
    """
    Dataset for supervised training of TSP given expert solutions (i.e., optimal solutions).
    Each problem instance is a dictionary of the form { "inst": np.array: (<num_nodes>,2), "tour": list (<num_nodes),
    "sol": float }.
    For performance reasons, our model only takes batches with subtours of equal length (so no padding is required).
    In particular, one call to __getitem__ already returns a batch with subtours of equal length.
    Each sample corresponds to a subtour of the final tour where the next node must be chosen.
    The dataset represents it by a dictionary of the form:
    {
        "nodes": (torch.Tensor: (<num_nodes_in subtour>, 2)) All nodes of the subtour, where the
            first node is the starting point and the last node is the destination. In case where
            only the subtour corresponds to the situation of the expert tour where only the first node
            has been chosen, the first node is equal to the last node.
        "next_node_idx": Index of target node to choose in subtour. This is always 0 during training.
    }
    """
    def __init__(self, model_type: str, expert_pickle_file: str, batch_size: int,
                 data_augmentation: bool = False, data_augmentation_linear_scale: bool = False,
                 augment_direction: bool = False, custom_num_instances: Optional[int] = None,
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
        self.batch_size = batch_size
        with open(expert_pickle_file, "rb") as f:
            self.instances = pickle.load(f)

        if custom_num_instances is not None:
            self.instances = self.instances[:custom_num_instances]

        print(f"Loaded dataset. Num items: {len(self.instances)}")

        self.num_nodes = self.instances[0]["inst"].shape[0]
        # One instance corresponds to one random subtour, so
        # length of dataset corresponds to the length of one epoch.
        if custom_num_batches is None:
            self.length = len(self.instances) // self.batch_size
        elif custom_num_batches[0] == "absolute":
            self.length = custom_num_batches[1]
        elif custom_num_batches[0] == "multiplier":
            self.length = custom_num_batches[1] * len(self.instances)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.model_type == "BQ":
            return self.getitem_for_bq()
        elif self.model_type == "LEHD":
            return self.getitem_for_lehd()
        else:
            raise ValueError(f"Model type {self.model_type} is unknown.")

    def getitem_for_bq(self):
        """
        Returns a minibatch of size `batch_size` of random subtours of random length within
        [4, `num_nodes + 1`], where the upper bound corresponds to returning to the first node.
        :param idx: Is not used, as we directly randomly sample from the tours here.

        Returns:
            For BQ: A dictionary of the form
            {
                "nodes": torch.FloatTensor of shape (batch, length of subtour, 2).
                    Note that we only return the nodes of the sampled subtour.
                "next_node_idx": torch.LongTensor of shape (batch, 1) corresponding to the target index of the next
                    node to choose. As we return the nodes in the order of the subtour, the idx is 0 everywhere
                    (as we discard the first and last node).
            }
        """
        to_stack_nodes: list[
            np.array] = []  # list of nodes of shape (subtour length, 2). Each entry corresponds to one item in batch.

        # Get a random subtour length. We start at subtours of length 4 (everything below is trivial).
        subtour_len = random.randint(4, self.num_nodes + 1)
        for _ in range(self.batch_size):
            # Get random instance
            idx = random.randint(0, len(self.instances) - 1)
            nodes = self.instances[idx]["inst"]
            tour = list(self.instances[idx]["tour"])

            # Get random start point of subtour
            start_idx = random.randint(0, self.num_nodes - 1)
            subtour = tour[start_idx: start_idx + subtour_len]
            if len(subtour) < subtour_len:
                subtour = subtour + tour[0: subtour_len - len(subtour)]
            assert len(subtour) == subtour_len
            if self.augment_direction:
                if random.randint(0, 1) == 0:
                    # reverse order of tour
                    subtour = subtour[::-1]

            _nodes = nodes
            if self.data_augmentation:
                _nodes = np.copy(nodes)
                _nodes = TSPGeometry.random_state_augmentation(_nodes)

            to_stack_nodes.append(_nodes[subtour])

        batch_nodes = np.stack(to_stack_nodes, axis=0)
        # The target idx to choose is - when discarding the first and last node - always 0!
        batch_next_node_idx = torch.LongTensor([0] * self.batch_size)
        return {
            "nodes": torch.from_numpy(batch_nodes).float(),
            "next_node_idx": batch_next_node_idx
        }

    def getitem_for_lehd(self):
        """
        Returns a minibatch of size `batch_size` of random subtours of random length within
        [4, `num_nodes + 1`], where the upper bound corresponds to returning to the first node.
        :param idx: Is not used, as we directly randomly sample from the tours here.

        The key difference to BQ is that the idcs of the subtour are the same for _all_
        entries in the minibatch to enable efficient indexing (otherwise, we would need to
        use gather).

        For LEHD: A dictionary of the form
        {
            "nodes": torch.FloatTensor of shape (batch, num nuodes, 2),
                corresponding to _all_ nodes in the full problem instance, in
                each instance with a random start point and (if givne in config)
                random direction.
            "subtour_length": Length of subtour. In the model, the nodes of the subtour are taken
                as the first `subtour_length` nodes.
            "next_node_idx": As in BQ, with ones everywhere.
        }
        """
        to_stack_nodes: list[np.array] = []  # list of nodes of shape (subtour length, 2). Each entry corresponds to one item in batch.

        # Get a random subtour length. We start at subtours of length 4 (everything below is trivial).
        subtour_len = random.randint(4, self.num_nodes + 1)
        for _ in range(self.batch_size):
            # Get random instance
            idx = random.randint(0, len(self.instances) - 1)
            tour = list(self.instances[idx]["tour"])
            nodes = self.instances[idx]["inst"][tour]  # nodes in tour order

            # Get random start point of subtour
            start_idx = random.randint(0, self.num_nodes - 1)

            nodes = np.concatenate((nodes[start_idx:], nodes[:start_idx]), axis=0)
            if self.augment_direction:
                if random.randint(0, 1) == 0:
                    nodes = np.flip(nodes, axis=0)
            assert nodes.shape[0] == self.num_nodes

            _nodes = nodes
            if self.data_augmentation:
                _nodes = np.copy(nodes)
                _nodes = TSPGeometry.random_state_augmentation(_nodes, self.data_augmentation_linear_scale)

            to_stack_nodes.append(_nodes)

        batch_nodes = np.stack(to_stack_nodes, axis=0)
        # The target idx to choose is - when discarding the first and last node - always 0!
        batch_next_node_idx = torch.LongTensor([0] * self.batch_size)
        return {
            "nodes": torch.from_numpy(batch_nodes).float(),
            "subtour_length": subtour_len,
            "next_node_idx": batch_next_node_idx,
            "start_node_is_end_node": subtour_len == (self.num_nodes + 1)  # Subtour is equal to the full tour, so we should arrive at the start node again
        }