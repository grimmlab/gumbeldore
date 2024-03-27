from typing import Optional, List

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from gomoku.env.gomoku_env import PlayerMove
"""
DISCLAIMER: Experimental
"""


class RandomGomokuDataset(Dataset):
    """
    Dataset for training of Gomoku given (observation, action)-pairs.
    """
    def __init__(self, move_list: List[PlayerMove], batch_size: int, num_batches: int,
                 data_augmentation: bool = False):

        print(f"Initializing dataset with {len(move_list)} moves.")
        self.move_list = move_list
        self.data_augmentation = data_augmentation  # no augmentation implemented yet

        self.batch_size = batch_size
        self.length = num_batches

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Sample player moves without replacement (if possible) from the dataset.
        choice_fn = random.choices if self.batch_size > len(self.move_list) else random.sample
        moves = choice_fn(self.move_list, self.batch_size)

        return {
            "obs": torch.stack([
                torch.from_numpy(move.observation[:2])
                for move in moves
            ], dim=0),
            "action_mask": torch.stack([
                torch.from_numpy(move.action_mask)
                for move in moves
            ], dim=0),
            "action": torch.LongTensor([move.action_taken for move in moves])
        }
