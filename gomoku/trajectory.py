from typing import Tuple, List, Union

import numpy as np
import torch
from gomoku.config import GomokuConfig
from core.abstracts import BaseTrajectory, Instance
from gomoku.env.gomoku_env import GomokuEnv
"""
DISCLAIMER: Experimental
"""


class Trajectory(BaseTrajectory):

    def __init__(self, env: GomokuEnv):
        self.env = env

    @staticmethod
    def init_batch_from_instance_list(instances: List[np.array], network: torch.nn.Module, device: torch.device):
        """
        `instances` is a list of (board_size, board_size) arrays consisting of starting boards
        where the first black stone has been placed.
        """
        trajectories = []
        for board in instances:
            config = GomokuConfig()
            traj = Trajectory(GomokuEnv(cfg=config.game_cfg))
            if config.game_cfg.bot_action_type == "alpha_beta_pruning":
                first_move = np.argwhere(board != 0)[0]
                traj.env.reset_for_alpha_beta_pruning((int(first_move[0]), int(first_move[1])))
            else:
                traj.env.reset(start_player_index=1, init_state=board)
            trajectories.append(traj)
        return trajectories

    @staticmethod
    def log_probability_fn(trajectories: List['Trajectory'], network: torch.nn.Module, to_numpy: bool) -> Union[torch.Tensor, List[np.array]]:
        device = network.device
        with torch.no_grad():
            _batch = torch.stack([
                torch.from_numpy(traj.env.current_timestep.obs["observation"])[:2]
                # only take the views of the board, not which player's turn it is
                for traj in trajectories
            ], dim=0).to(device)

            policy_logits, _ = network(_batch)
            # mask illegal actions
            action_mask = torch.stack([
                torch.from_numpy(traj.env.current_timestep.obs["action_mask"])
                for traj in trajectories
            ], dim=0).to(device)
            policy_logits[action_mask == 0] = float("-inf")
            batch_log_probs = torch.log_softmax(policy_logits, dim=1)
        if not to_numpy:
            return batch_log_probs

        batch_log_probs = batch_log_probs.cpu().numpy()
        return [batch_log_probs[i] for i in range(len(trajectories))]

    def transition_fn(self, action: int) -> Tuple['Trajectory', bool]:
        new_traj = Trajectory(self.env.clone())
        new_traj.env.step(action)
        return new_traj, new_traj.env.done

    def to_max_evaluation_fn(self) -> float:
        if self.env.winner == -1:
            # draw
            return 0
        if self.env.winner == 0:
            # player 1 (bot) won
            return -1
        if self.env.winner == 1:
            # player 2 (agent) won
            return 1

    def num_actions(self) -> int:
        # not needed for Gomoku
        pass