import copy
import random
import pickle
import torch.optim
import numpy as np
import time

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from core.gumbeldore_dataset import GumbeldoreDataset
from core.train import main_train_cycle

from gomoku.config import GomokuConfig
from gomoku.dataset import RandomGomokuDataset
from gomoku.env.gomoku_env import PlayerMove
from gomoku.trajectory import Trajectory as GomokuTrajectory
from modules.alphazero.alphazero_model import AlphaZeroModel

from tqdm import tqdm

from typing import Tuple, List, Optional

"""
Gomoku. Disclaimer: EXPERIMENTAL 
===========================
Aim is to beat expert strategies in a greedy fashion as fast as possible.
Expert strategies, environment and model are taken from
LightZero (https://github.com/opendilab/LightZero). Thanks to them!
Train only with Gumbeldore (no supervised training with expert trajectories).
"""


def get_network(config: GomokuConfig, device: torch.device) -> AlphaZeroModel:
    network = AlphaZeroModel(**config.az_model_cfg)
    network.device = device
    return network


def generate_instances(config: GomokuConfig):
    """
    No active search here. The generated instances are all possible starting positions of a black stone
    on the board.
    """
    # We generate all possible boards where the bot (black) has put a first stone
    board_size = config.game_cfg.board_size
    problem_instances = []
    for i in range(board_size):
        for j in range(board_size):
            board = np.zeros((board_size, board_size), dtype="int32")
            board[i, j] = 1  # set black stone
            problem_instances.append(board)

    return problem_instances, config.gumbeldore_config["batch_size_per_worker"], config.gumbeldore_config["batch_size_per_cpu_worker"]


def beam_leaves_to_result(trajectories: List[GomokuTrajectory]):
    allow_using_opponent_trajectories = GomokuConfig().gumbeldore_config["allow_using_opponent_trajectories"]
    # Return the winner if agent has won
    winners = [traj.env.winner for traj in trajectories]
    trajectory_to_return = None
    if 1 in winners:
        for traj in trajectories:
            if traj.env.winner == 1:
                trajectory_to_return = traj.env.player_trajectories[1]
                break
        marker = "w"
    elif (not allow_using_opponent_trajectories or not len([traj.env.player_trajectories[0] for traj in trajectories if traj.env.winner == 0])) and -1 in winners:
        # There's a draw, let's take this one
        agent_trajs = [traj.env.player_trajectories[1] for traj in trajectories if traj.env.winner == -1]
        trajectory_to_return = random.choice(agent_trajs)
        marker = "d"
    elif allow_using_opponent_trajectories:
        # Take a random opponent trajectory
        opponent_trajs = [traj.env.player_trajectories[0] for traj in trajectories if traj.env.winner == 0]
        trajectory_to_return = random.choice(opponent_trajs)
        marker = "l"
    else:
        # Take a random agent trajectory
        agent_trajs = [traj.env.player_trajectories[1] for traj in trajectories]
        trajectory_to_return = random.choice(agent_trajs)
        marker = "l"

    return trajectory_to_return, marker


def save_search_results_to_dataset(destination_path: str, problem_instances, results, append_to_dataset):
    """
    As the dataset is very small, we just keep it in memory and return it to be used for training.
    Hence, `destination_path` and `append_to_dataset` is not used.
    """
    # Each result in `results` is a tuple (player moves, marker ['d'raw, 'w'in or 'l'ose]) (see above)
    total_wins = sum([1 if marker == "w" else 0 for _, marker in results])
    total_losses = sum([1 if marker == "l" else 0 for _, marker in results])
    total_draws = sum([1 if marker == "d" else 0 for _, marker in results])
    total_num_trajectories = len(results)

    # flatten all trajectories into single moves
    dataset: List[PlayerMove] = [copy.deepcopy(player_move) for trajectory, _ in results for player_move in
                                 trajectory]
    print(
        f"Num trajectories: {total_num_trajectories} / Num wins: {total_wins} / Num draws: {total_draws} / Num losses: {total_losses}")
    return dataset, dict(num_wins=total_wins, num_draws=total_draws, num_losses=total_losses)


# EVALUATION
def evaluate(eval_type: str, config: GomokuConfig, network: AlphaZeroModel, to_evaluate_path: str, num_instances: Optional[int] = None):
    def process_search_results(destination_path: str, problem_instances, results, append_to_dataset):
        # `destination_path` and `append_to_dataset` are not needed
        # Return mean score
        scores = [dict(d=0, w=1, l=-1)[marker] for _, marker in results]
        return {"mean_score": np.array(scores).mean()}

    if not config.gumbeldore_eval:
        loggable_results = dict()
        metric = None
        for beam_width, batch_size in config.beams_with_batch_sizes.items():
            print(f"Evaluating with beam search (k={beam_width})")
            _config = copy.deepcopy(config)
            _config.gumbeldore_config["search_type"] = "beam_search"
            _config.gumbeldore_config["beam_width"] = beam_width
            _config.gumbeldore_config["devices_for_workers"] = _config.devices_for_eval_workers
            _config.gumbeldore_config["batch_size_per_worker"] = batch_size
            _config.gumbeldore_config["batch_size_per_cpu_worker"] = batch_size
            beam_width_results = GumbeldoreDataset(
                config=_config, trajectory_cls=GomokuTrajectory, generate_instances_fn=generate_instances,
                get_network_fn=get_network, beam_leaves_to_result_fn=beam_leaves_to_result, process_search_results_fn=process_search_results
            ).generate_dataset(copy.deepcopy(network.get_weights()), False)

            loggable_results[f"{eval_type} beam width {beam_width}. Mean score"] = float(beam_width_results["mean_score"])
            if beam_width == config.validation_relevant_beam_width:
                # get metric used to decide whether network improved or not
                metric = beam_width_results["mean_score"]
        return metric, loggable_results
    else:
        results = GumbeldoreDataset(
            config=config, trajectory_cls=GomokuTrajectory, generate_instances_fn=generate_instances,
            get_network_fn=get_network, beam_leaves_to_result_fn=beam_leaves_to_result,
            process_search_results_fn=process_search_results
        ).generate_dataset(copy.deepcopy(network.get_weights()), False)
        metric = results["mean_score"]
        return metric, {
            f"{eval_type} Gumbelore. Mena score": results["mean_obj"]
        }


def validate(config: GomokuConfig, network: AlphaZeroModel):
    return evaluate("Validation", config, network, None, None)


def test(config: GomokuConfig, network: AlphaZeroModel):
    _, loggable_test_metrics = evaluate("Test", config, network, None, None)
    return loggable_test_metrics


# TRAINING
def train_for_one_epoch_gumbeldore(config: GomokuConfig, network: AlphaZeroModel, network_weights: dict,
                                   optimizer: torch.optim.Optimizer, append_to_dataset: bool) -> Tuple[float, dict]:
    list_moves, generated_game_statistics = GumbeldoreDataset(
        config=config,
        trajectory_cls=GomokuTrajectory,
        generate_instances_fn=generate_instances,
        get_network_fn=get_network,
        beam_leaves_to_result_fn=beam_leaves_to_result,
        process_search_results_fn=save_search_results_to_dataset
    ).generate_dataset(network_weights, append_to_dataset)
    print("Training with generated data.")

    torch.cuda.empty_cache()

    # Load dataset.
    dataset = RandomGomokuDataset(
        move_list=list_moves,
        batch_size=config.batch_size_training,
        num_batches=config.num_batches_per_epoch,
        data_augmentation=config.data_augmentation,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=config.num_dataloader_workers, pin_memory=True,
                            persistent_workers=True)

    network.train()

    accumulated_loss = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(data_iter)
        data["obs"] = data["obs"][0].to(config.training_device)
        data["action_mask"] = data["action_mask"][0].to(config.training_device)
        data["action"] = data["action"][0].to(config.training_device)

        logits, _ = network(data["obs"])
        # mask the logits
        logits = logits + (1. - data["action_mask"]) * -10000.
        criterion = CrossEntropyLoss(reduction='mean')
        loss = criterion(logits, data["action"])

        # Optimization step
        optimizer.zero_grad()
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()

        batch_loss = loss.item()
        accumulated_loss += batch_loss

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    avg_loss = accumulated_loss / num_batches
    return avg_loss, {
                "Generated num wins": generated_game_statistics["num_wins"],
                "Generated num draws": generated_game_statistics["num_draws"],
                "Generated num losses": generated_game_statistics["num_losses"],
            }


if __name__ == '__main__':
    print(">> Gomoku Gumbeldore <<")
    config = GomokuConfig()

    main_train_cycle(
        learning_type="gumbeldore",
        config=config,
        get_network_fn=get_network,
        validation_fn=validate,
        test_fn=test,
        get_supervised_dataloader=None,
        train_for_one_epoch_supervised_fn=None,
        train_for_one_epoch_gumbeldore_fn=train_for_one_epoch_gumbeldore
    )