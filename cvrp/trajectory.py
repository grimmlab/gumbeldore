import torch
import numpy as np
from typing import Optional, Union, List
from cvrp.lehd_network import LEHDPolicyNetwork
from cvrp.bq_network import BQPolicyNetwork
from core.abstracts import BaseTrajectory


class Trajectory(BaseTrajectory):
    """
    Represents a partial CVRP solution used for beam search/rolling out policy/incremental SBS.
    """
    def __init__(self, debug: bool = False, nearest_k: int = 250):
        """
        If `nearest_k` is given, when batching states for the model, the trajectory restricts itself
        to the k nodes which are nearest to the current start position.
        """
        self.debug = debug
        self.nearest_k = nearest_k  # If the number of nodes exceeds k, in each step we only consider the closest k remaining nodes to the current start node.
        self.objective: Optional[float] = None  # tour length
        self.nodes: Optional[torch.FloatTensor] = None  # shape (num_nodes + 1, 2) for the coordinates, including depot as first element. For LEHD, last dimension has size latent_dim
        self.remaining_nodes: Optional[torch.FloatTensor] = None  # subset of unscheduled nodes (without depot and start node)
        self.remaining_idcs: Optional[np.array] = None  # keeps ids corresponding to remaining nodes
        self.nearest_idcs_of_remaining: Optional[np.array] = None   # keeps indices of the closest remaining idcs.
        self.start_node: Optional[torch.FloatTensor] = None  # current start node (if None, we start at depot)
        self.start_node_idx: int = 0  # Index of the current start node (over all nodes, not just remaining).
        self.distance_matrix: Optional[torch.FloatTensor] = None  # Current distance matrix of shape (num nodes + 1, num nodes + 1)

        self.vehicle_capacity_unnormed: float = 0.  # This is fixed after setup from instance
        self.current_capacity: float = 0.  # Keeps current capacity (unnormalized)
        self.demands: Optional[torch.FloatTensor] = None  # Keeps demands of customers (and depot) (unnormalized)
        self.start_node_demand: Optional[torch.FloatTensor] = None
        self.remaining_demands: Optional[torch.FloatTensor] = None  # Keeps remaining demands

        self.num_nodes: int = -1
        self.partial_tour_idcs: List[int] = []  # Keeps indices of the nodes in the tour.
        self.partial_tour_flags: List[int] = []  # Keeps the flags of the nodes visited so far.

    @staticmethod
    def init_from_instance(instance: dict, nodes: Optional[torch.FloatTensor],
                           distance_matrix: torch.FloatTensor,
                           device: Optional[Union[torch.device, str]] = None):
        """
        Initializes a Trajectory from an instance as given in CVRPDataset.
        If `nodes` is given (e.g., for LEHD) it must be in the shape (num_nodes + 1, latent dim), and then this tensor
        is used instead of taking it from instance.
        Already sends all tensors to the given device, if specified.
        """
        device = "cpu" if device is None else device
        traj = Trajectory()
        traj.nodes = nodes.to(device)
        traj.distance_matrix = distance_matrix
        traj.num_nodes = traj.nodes.shape[0] - 1  # We do not count depot
        traj.remaining_nodes = traj.nodes[1:]
        traj.remaining_idcs = np.array(list(range(1, traj.num_nodes + 1)), dtype=int)  # In the remaining idcs, we do not count the depot
        traj.start_node_idx = 0  # we start from depot
        traj.set_nearest_idcs_of_remaining()
        traj.vehicle_capacity_unnormed = instance["capacity"]
        traj.current_capacity = traj.vehicle_capacity_unnormed
        traj.demands = torch.from_numpy(instance["demands"]).float().to(device)
        traj.remaining_demands = traj.demands[1:]

        return traj

    @staticmethod
    def init_batch_from_instance_list(instances, network, device: torch.device):

        nodes_batch = torch.cat([
                    torch.from_numpy(instance["nodes"]).float().to(device)[None, :, :]
                    for instance in instances], dim=0)  # (B, num_nodes, 2)
        dist_mats_batch = torch.cdist(nodes_batch, nodes_batch, p=2.).cpu()

        if isinstance(network, LEHDPolicyNetwork):
            nodes_batch = network.encode(
                nodes=nodes_batch,
                demands=torch.cat([
                    torch.from_numpy(instance["demands"]).float().to(device)[None, :] / instance["capacity"]
                    for instance in instances], dim=0)
            ).detach().cpu()

        return [Trajectory.init_from_instance(
            instance=instance,
            nodes=nodes_batch[i],
            distance_matrix=dist_mats_batch[i],
            device=device
        ) for i, instance in enumerate(instances)]

    @staticmethod
    def log_probability_fn(trajectories: List['Trajectory'], network: Union[BQPolicyNetwork, LEHDPolicyNetwork], to_numpy: bool) -> Union[
        torch.Tensor, List[np.array]]:
        with torch.no_grad():
            batch = Trajectory.trajectories_to_batch(trajectories)
            if isinstance(network, LEHDPolicyNetwork):
                policy_logits = network.decode(encoded_node_seq=batch["nodes"], x=batch)
            else:
                policy_logits = network(batch)  # (num_trajectories, num_actions)
            batch_log_probs = torch.log_softmax(policy_logits, dim=1)
        if not to_numpy:
            return batch_log_probs

        batch_log_probs = batch_log_probs.cpu().numpy()
        return [batch_log_probs[i] for i in range(len(trajectories))]

    def transition_fn(self, action: int):
        new_traj = self.add_remaining_node_with_flag_to_tour(action)
        is_leaf = len(new_traj.remaining_idcs) == 0
        return new_traj, is_leaf

    def to_max_evaluation_fn(self) -> float:
        return -1. * self.objective

    def set_nearest_idcs_of_remaining(self):
        if len(self.remaining_idcs) <= self.nearest_k:
            # no need to do anything
            self.nearest_idcs_of_remaining = None
        else:
            # get top k distance of start node to all nodes
            dist = self.distance_matrix[self.start_node_idx].numpy()
            dist = dist[self.remaining_idcs]
            self.nearest_idcs_of_remaining = np.argpartition(-dist, -self.nearest_k)[-self.nearest_k:]

    def compute_objective(self):
        """
        Assumes that the trajectory is finished and computes the final tour length of
        the full tour. At the same time asserts that the tour is feasible and no constraints
        are violated.
        """
        if self.debug:
            assert self.num_nodes == len(self.partial_tour_idcs), f"Tour is not finished. Required scheduled nodes: {self.num_nodes}, Actual: {len(self.partial_tour_idcs)}"
            assert self.partial_tour_flags[0] == 1, "First flag should always be 1."

        tour_length = 0.
        current_node_idx = 0
        for i, node_idx in enumerate(self.partial_tour_idcs):
            flag = self.partial_tour_flags[i]

            if flag == 0:
                tour_length = tour_length + self.distance_matrix[current_node_idx, node_idx]
            else:
                if self.debug:
                    current_capacity = self.vehicle_capacity_unnormed
                tour_length = tour_length + self.distance_matrix[current_node_idx, 0] + self.distance_matrix[0, node_idx]
            current_node_idx = node_idx
            if self.debug:
                current_capacity = current_capacity - self.demands[node_idx]
                assert current_capacity >= 0., f"Capacity exceeded ({current_capacity})!"
        # Add final route from last node to depot
        tour_length = tour_length + self.distance_matrix[current_node_idx, 0]
        self.objective = tour_length

    def num_actions(self) -> int:
        return len(self.nearest_idcs_of_remaining) if self.nearest_idcs_of_remaining is not None else len(self.remaining_idcs)

    def add_remaining_node_with_flag_to_tour(self, action: int):
        """
        Given an action, we get the remaining node idx it corresponds to (and flag).
        Returns a new trajectory with the node added to the partial tour. `log_prob` is the log probability
        of the new tour (i.e., log prob of adding idx + accumulated log prob of existing tour).
        If the tour can be finished automatically (this is the case if only one index is left afterward), it does so.
        """
        # Note that the action indexing starts at the current start node, so we need to
        # account for that (subtract 1), as if we start in the beginning at the depot, the first node in the
        # sequence is still part of remaining nodes.
        remaining_idx = action // 2 - (0 if self.start_node is None else 1)

        # We now need to check whether we have narrowed down the nodes to the nearest k. If so, we need to
        # account for that and `remaining_idx` actually points to the index within `nearest_idcs_of_remaining`.
        if self.nearest_idcs_of_remaining is not None:
            remaining_idx = self.nearest_idcs_of_remaining[remaining_idx]

        flag = action % 2  # if we reach the node via the depot or not
        idx = self.remaining_idcs[remaining_idx]

        if self.debug:
            assert idx not in self.partial_tour_idcs, f"Node idx {idx} already present in tour."

        new_partial_tour_idcs = self.partial_tour_idcs + [idx]
        new_partial_tour_flags = self.partial_tour_flags + [flag]
        # Remove from remaining nodes
        new_remaining_idcs = np.delete(self.remaining_idcs, remaining_idx)

        new_remaining_nodes = torch.cat(
            (self.remaining_nodes[:remaining_idx], self.remaining_nodes[remaining_idx + 1:]),
            dim=0)
        new_start_node = self.nodes[idx][None, :]
        new_start_idx = idx
        new_start_node_demand = self.demands[idx: idx+1]
        # Update the current capacity
        new_capacity = self.vehicle_capacity_unnormed if flag == 1 else self.current_capacity
        new_capacity = new_capacity - new_start_node_demand[0]#.item()

        if self.debug:
            assert new_capacity >= 0., f"New capacity {new_capacity} is less than zero. Infeasible solution!"

        new_remaining_demands = torch.cat(
            (self.remaining_demands[:remaining_idx], self.remaining_demands[remaining_idx + 1:]),
            dim=0)

        if len(new_remaining_idcs) == 1:
            # We can autofinish the tour
            # get the missing index
            idx = new_remaining_idcs[0]
            demand = self.demands[idx].item()
            if new_capacity - demand < 0:
                new_capacity = self.vehicle_capacity_unnormed - demand
                last_flag = 1
            else:
                new_capacity = new_capacity - demand
                last_flag = 0
            new_partial_tour_idcs.append(idx)
            new_partial_tour_flags.append(last_flag)
            new_remaining_demands = None
            new_remaining_nodes = None
            new_remaining_idcs = []
            new_start_node = None
            new_start_idx = 0
            new_start_node_demand = None

        # Set up new trajectory
        traj = Trajectory()
        traj.nodes = self.nodes
        traj.distance_matrix = self.distance_matrix
        traj.remaining_nodes = new_remaining_nodes
        traj.remaining_idcs = new_remaining_idcs
        traj.start_node = new_start_node
        traj.start_node_idx = new_start_idx
        traj.vehicle_capacity_unnormed = self.vehicle_capacity_unnormed
        traj.current_capacity = new_capacity
        traj.demands = self.demands
        traj.start_node_demand = new_start_node_demand
        traj.remaining_demands = new_remaining_demands
        traj.num_nodes = self.num_nodes
        traj.partial_tour_idcs = new_partial_tour_idcs
        traj.partial_tour_flags = new_partial_tour_flags
        traj.set_nearest_idcs_of_remaining()

        if len(traj.remaining_idcs) == 0:
            traj.compute_objective()

        return traj

    @staticmethod
    def trajectories_to_batch(trajectories: List):
        """
        Given a list of trajectories, returns a dict which can be passed through
        the policy neural network. Expects all trajectories to be at the same level, i.e.,
        same number of unscheduled nodes.
        """
        device = trajectories[0].nodes.device
        is_start = trajectories[0].start_node is None
        vehicle_capacity_unnormed = torch.tensor(
            [traj.vehicle_capacity_unnormed for traj in trajectories],
            dtype=torch.float, device=device
        )[:, None]  # (B, 1)

        # capacity and demands will be normalized later
        current_capacity = torch.tensor(
            [traj.current_capacity for traj in trajectories],
            dtype=torch.float, device=device
        )[:, None]  # (B, 1)

        demands = torch.cat([
            (
            torch.cat([traj.start_node_demand, traj.remaining_demands], dim=0) if not is_start else traj.remaining_demands
            )[None, :]
            for traj in trajectories
        ], dim=0)

        demands = torch.cat((torch.zeros((len(trajectories), 1), dtype=torch.float, device=device), demands), dim=1)  # (B, remaining + 1)

        nodes = torch.cat([
            torch.cat([
                traj.nodes[0:1],
                traj.start_node,
                traj.remaining_nodes
            ] if not is_start else [traj.nodes[0:1], traj.remaining_nodes],
                dim=0)[None, :]
            for traj in trajectories
        ], dim=0)

        # A bit dirty: If we restrict to the nearest k neighbors, we choose them in the demands
        # and nodes tensor
        if trajectories[0].nearest_idcs_of_remaining is not None:
            nearest_demands = []
            nearest_nodes = []
            for i, traj in enumerate(trajectories):
                idcs_to_take = [0, 1] if not is_start else [0]  # depot and start node
                idcs_to_take = idcs_to_take + [j + len(idcs_to_take) for j in traj.nearest_idcs_of_remaining]
                nearest_demands.append(demands[i][idcs_to_take])
                nearest_nodes.append(torch.stack(
                    [nodes[i, j] for j in idcs_to_take]
                , dim=0))
            demands = torch.stack(nearest_demands, dim=0)
            nodes = torch.stack(nearest_nodes, dim=0)

        start_node_mask = torch.zeros((len(trajectories), 2), dtype=torch.float, device=device)

        if is_start:
            start_node_mask[:, 0] = 1
        else:
            start_node_mask[:, 1] = 1

        if is_start:
            remaining_length = len(trajectories[0].remaining_idcs) if trajectories[0].nearest_idcs_of_remaining is None else trajectories[0].nearest_k
            action_mask = torch.zeros((len(trajectories), 2 * remaining_length), dtype=torch.bool, device=device)
            action_mask[:, ::2] = True
        else:
            # demands are of shape (batch, 1 (for start demand which we do not care about) + num remaining)
            exceeds_capacity = (current_capacity - demands[:, 1:] < 0)  # cut off depot demand (which is zero) from beginning of `demands`
            action_mask = exceeds_capacity[:, :, None].repeat((1, 1, 2))  # make an additional dimension for "can be reached via depot"
            action_mask[:, :, 1] = False   # they all can be reached via depot
            action_mask = action_mask.view(len(trajectories), -1)

            action_mask[:, :2] = True

        return dict(
            current_capacity=current_capacity / vehicle_capacity_unnormed,
            demands=demands / vehicle_capacity_unnormed,  # (B, N + 1)
            nodes=nodes.to(device),  # (B, N + 1, 2) (BQ) resp. (B, N+1, latent dim) (LEHD)
            start_node_mask=start_node_mask,
            action_mask=action_mask
        )


