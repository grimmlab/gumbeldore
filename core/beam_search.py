import torch
import numpy as np
from typing import List

from core.abstracts import BaseTrajectory


def beam_search(trajectories: List[BaseTrajectory], network: torch.nn.Module, beam_width: int):
    """
    Fast(er than SBS in deterministic mode) beam search implementation in torch. Only applicable to CVRP and TSP, as
    this implementation requires all trajectories to end at the same time and have the same number of actions at each
    step.

    Parameters:
        trajectories: (List[BaseTrajectory]) List of instances transformed into initial trajectories.
        beam_width: (int) Beam width (=k)
    """
    Trajectory = trajectories[0].__class__  # get the class
    network.eval()
    device = network.device
    with torch.no_grad():
        for traj in trajectories:
            traj.log_prob = 0.
        # `beams` is a list where the i-th entry is the current beam (i.e., list of trajectories) of i-th instance
        beams: List[List[BaseTrajectory]] = [
            [traj] for traj in trajectories
        ]

        # inner beam search loop. We evaluate all trajectories in the current beam, and for each trajectory
        # expanding only the top k actions. Finally, prune back to k trajectories.
        while True:
            num_actions = beams[0][0].num_actions()  # number of actions in all trajectories
            batch_log_probs = torch.tensor([[traj.log_prob] for beam in beams for traj in beam]).float().to(device)
            # batch_log_probs are of shape (num_trajectories, 1)
            policy_log_probs = Trajectory.log_probability_fn(
                trajectories=[traj for beam in beams for traj in beam],
                network=network,
                to_numpy=False
            )

            # update the log probs of all expanded trajectories present now
            batch_log_probs = policy_log_probs + batch_log_probs
            # get the top k actions for each expanded trajectory
            top_actions = torch.topk(batch_log_probs, k=min(num_actions, beam_width), dim=1)  # (all trajectories, num top actions)
            top_actions_value = top_actions.values.cpu()
            top_actions_indices = top_actions.indices.cpu()

            i = 0
            new_beams = []
            for beam in beams:
                new_beam = []
                # for each trajectory in beam, get its top k actions  (num traj in beam, num top actions)
                beam_top_actions_values = top_actions_value[i: i + len(beam)]
                beam_top_actions_idcs = top_actions_indices[i: i + len(beam)]

                # flatten to obtain the top beam_width actions across all trajectories
                beam_top_actions_flat = beam_top_actions_values.flatten()
                _, top_expansions = torch.topk(beam_top_actions_flat,
                                               min(beam_top_actions_flat.shape[0], beam_width))
                # is of shape (num top actions, 2) and indicates the index of the trajectory and action for each top expansion
                top_idcs = np.array(np.unravel_index(top_expansions.numpy(), beam_top_actions_values.shape)).T
                finished = False
                for top_idx in top_idcs:
                    trajectory = beam[top_idx[0]]
                    action = beam_top_actions_idcs[top_idx[0], top_idx[1]].item()
                    log_prob = beam_top_actions_values[top_idx[0], top_idx[1]].item()
                    new_trajectory, finished = trajectory.transition_fn(action)
                    new_trajectory.log_prob = log_prob
                    new_beam.append(new_trajectory)

                i += len(beam)
                new_beams.append(new_beam)

            beams = new_beams
            if finished:
                break

    # For each instance, return the trajectory with the minimum tour length
    return [
        max([traj for traj in beam], key=lambda x: x.to_max_evaluation_fn())
    for beam in beams]