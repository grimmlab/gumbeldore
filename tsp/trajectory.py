import torch
import numpy as np
from typing import Optional, List, Union
from tsp.lehd_network import LEHDPolicyNetwork
from tsp.bq_network import BQPolicyNetwork
from core.abstracts import BaseTrajectory


class Trajectory(BaseTrajectory):
    """
    Represents a partial tour used for beam search/rolling out policy
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.objective: Optional[float] = None  # tour length (should be set when trajectory is finished)
        self.nodes: Optional[torch.FloatTensor] = None  # shape (num_nodes, 2 (for BQ) or latent_dim (for LEHD))
        self.distance_matrix: Optional[torch.FloatTensor] = None  # shape (num_nodes, num_nodes)
        self.remaining_nodes: Optional[torch.FloatTensor] = None  # subset of nodes corresponding to unscheduled nodes.
        self.remaining_idcs: Optional[List[int]] = None  # keeps ids corresponding to remaining nodes
        self.start_node: Optional[torch.FloatTensor] = None  # start of current partial tour
        self.dest_node: Optional[torch.FloatTensor] = None  # end of current partial tour
        self.num_nodes: int = -1
        self.partial_tour: list[int] = []  # use first node in problem instance as start if needed
        self.remaining_idcs: list[int] = []

    @staticmethod
    def init_from_nodes(nodes: torch.FloatTensor, distance_matrix: torch.FloatTensor):
        traj = Trajectory()
        traj.nodes = nodes
        traj.distance_matrix = distance_matrix
        traj.num_nodes = nodes.shape[0]
        traj.remaining_idcs = list(range(0, traj.num_nodes))
        traj.remaining_nodes = traj.nodes
        # we set the start node manually to the first node in instance. This is the default behaviour.
        traj.partial_tour.append(0)
        traj.start_node = traj.nodes[0:1]
        traj.dest_node = traj.start_node
        traj.remaining_idcs = list(range(1, traj.num_nodes))
        traj.remaining_nodes = traj.nodes[1:]

        return traj

    @staticmethod
    def init_batch_from_instance_list(instances, network, device: torch.device):

        nodes_batch = torch.cat([
            torch.from_numpy(instance["inst"]).float().to(device)[None, :, :]
            for instance in instances], dim=0)  # (B, num_nodes, 2)
        dist_mats_batch = torch.cdist(nodes_batch, nodes_batch, p=2.)

        if isinstance(network, LEHDPolicyNetwork):
            nodes_batch = network.encode(nodes_batch).detach().cpu()

        return [Trajectory.init_from_nodes(
            nodes=nodes_batch[i],
            distance_matrix=dist_mats_batch[i]
        ) for i, instance in enumerate(instances)]

    @staticmethod
    def log_probability_fn(trajectories: List['Trajectory'], network: Union[BQPolicyNetwork, LEHDPolicyNetwork],
                           to_numpy: bool) -> Union[torch.Tensor, List[np.array]]:
        with torch.no_grad():
            batch = Trajectory.trajectories_to_batch(trajectories)
            if isinstance(network, LEHDPolicyNetwork):
                policy_logits = network.decode(batch["nodes"])
            else:
                policy_logits = network(batch)  # (num_trajectories, num_actions)
            batch_log_probs = torch.log_softmax(policy_logits, dim=1)
        if not to_numpy:
            return batch_log_probs

        batch_log_probs = batch_log_probs.cpu().numpy()
        return [batch_log_probs[i] for i in range(len(trajectories))]

    def transition_fn(self, action: int):
        new_traj = self.add_remaining_node_idx_to_tour(action)
        is_leaf = len(new_traj.remaining_idcs) == 0
        return new_traj, is_leaf

    def to_max_evaluation_fn(self) -> float:
        return -1. * self.objective

    def num_actions(self) -> int:
        return len(self.remaining_idcs)

    def add_remaining_node_idx_to_tour(self, remaining_idx: int):
        """
        Returns a new trajectory with idx added to the partial tour.
        If the tour can be finished automatically (this is the case if only one index is left afterward), it does so.
        """
        idx = self.remaining_idcs[remaining_idx]
        if self.debug:
            assert idx not in self.partial_tour, f"Node idx {idx} already present in tour."
        new_partial_tour = self.partial_tour + [idx]

         # remove from remaining
        new_remaining_idcs = self.remaining_idcs.copy()
        del new_remaining_idcs[remaining_idx]
        new_remaining_nodes = torch.cat((self.remaining_nodes[:remaining_idx], self.remaining_nodes[remaining_idx + 1:]), dim=0)
        new_objective = self.objective
        if len(new_remaining_idcs) == 1:
            # we can autofinish the tour
            # get the missing index
            idx = new_remaining_idcs[0]
            new_partial_tour.append(idx)
            new_remaining_idcs = []
            new_remaining_nodes = None
            # Compute tour length
            tour_length = 0.
            for i in range(self.num_nodes):
                node_idx_1 = new_partial_tour[i]
                node_idx_2 = new_partial_tour[(i + 1) % self.num_nodes]
                tour_length = tour_length + self.distance_matrix[node_idx_1, node_idx_2]
            new_objective = tour_length.item()
        traj = Trajectory()
        traj.objective = new_objective
        traj.nodes = self.nodes
        traj.distance_matrix = self.distance_matrix
        traj.num_nodes = self.num_nodes
        traj.remaining_nodes = new_remaining_nodes
        traj.partial_tour = new_partial_tour
        traj.remaining_idcs = new_remaining_idcs
        traj.start_node = self.nodes[idx: idx+1]
        traj.dest_node = traj.start_node if self.dest_node is None else self.dest_node
        return traj

    def finished(self):
        return len(self.remaining_idcs) == 0

    @staticmethod
    def trajectories_to_batch(trajectories):
        """
        Assumes all trajectories to have same number of remaining idcs.
        """
        return {
            "nodes": torch.stack([
                torch.cat((traj.start_node, traj.remaining_nodes, traj.dest_node), dim=0)
                if traj.start_node is not None else traj.remaining_nodes
                for traj in trajectories
            ], dim=0).float().to(trajectories[0].distance_matrix.device)
        }
