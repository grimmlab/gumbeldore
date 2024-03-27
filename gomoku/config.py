import os
import datetime
from easydict import EasyDict
"""
DISCLAIMER: Experimental
"""


class GomokuConfig:
    def __init__(self):
        self.architecture = "LightZero"

        self.game_cfg = EasyDict(
            board_size=6,
            battle_mode='play_with_bot_mode',
            bot_action_type='v1',
            prob_random_action_in_bot=0.,
            channel_last=False,
            manager=dict(shared_memory=False, ),
            # ==============================================================
            # for the creation of simulation env
            agent_vs_human=False,
            prob_random_agent=0.,
            prob_expert_agent=0.,
            scale=False,
            screen_scaling=9,
            render_mode=None,
            alphazero_mcts_ctree=False,
            # ==============================================================
        )
        self.eval_bot_action_type = "alpha_beta_pruning"
        self.az_model_cfg = dict(
            observation_shape=(2, self.game_cfg.board_size, self.game_cfg.board_size),
            action_space_size=int(1 * self.game_cfg.board_size * self.game_cfg.board_size),
            num_res_blocks=1,
            num_channels=32
        )

        self.load_checkpoint_from_path = None  # If given, model checkpoint is loaded from this path.
        self.load_optimizer_state = False  # If True, the optimizer state is also loaded.
        self.reset_best_validation = True

        # For training
        self.seed = 41
        self.data_augmentation = True  # TODO unclear if needed, maybe for flipping/rotating boards
        self.num_dataloader_workers = 3  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0,1,2,3"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = "cpu"
        self.num_epochs = 300  # Number of epochs (i.e. passes through training set) to train
        self.validation_every_n_epochs = 1  # After how many epochs to always perform validation
        self.batch_size_training = 32
        self.num_batches_per_epoch = 100

        self.optimizer = {
            "lr": 2e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1.
            }
        }

        self.gumbeldore_config = {
            "active_search": None,
            "use_best_model_for_generation": False,
            "append_if_not_new_best": False,
            "devices_for_workers": ["cpu"] * 4,
            "num_instances_to_generate": 36,
            "allow_using_opponent_trajectories": False,
            "destination_path": "./data/gomoku_gumbeldore_dataset.pickle",
            "batch_size_per_worker": 9,
            "batch_size_per_cpu_worker": 9,  # different batch size for CPU workers, should be smaller
            "search_type": "gumbeldore",
            "beam_width": 32,
            "num_rounds": 10,
            "pin_workers_to_core": False,
            "advantage_constant": 1,
            "min_max_normalize_advantage": False,
            "expected_value_use_simple_mean": False,
            "use_pure_outcomes": True,
            "normalize_advantage_by_visit_count": True,
            "perform_first_round_deterministic": False,
            "min_nucleus_top_p": 1.,
        }

        # For evaluation
        self.gumbeldore_eval = False
        self.num_eval_workers = 4               # Number of workers for beam search evaluation
        self.devices_for_eval_workers = ["cpu"] * self.num_eval_workers  # length should match number of workers
        self.beams_with_batch_sizes = {  # Keys: Beam widths with which to perform evaluation, values: corresponding batch sizes
            1: 9
        }
        self.validation_relevant_beam_width = 1  # Output from this beam width is used as the relevant metric to decide if a model has improved or not

        # Results and logging
        self.results_path = os.path.join("./results",
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights
        self.log_to_file = True
        self.log_to_mlflow = False
        # Optional MLFlow logging
        self.mlflow_server_uri = "<mlflow_server_uri>"
        os.environ["AWS_ACCESS_KEY_ID"] = "<AWS_ACCESS_KEY_ID>"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "<AWS_SECRET_ACCESS_KEY>"
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
        os.environ[
            "MLFLOW_S3_ENDPOINT_URL"] = "<MLFLOW_S3_ENDPOINT_URL>"  # How to reach MinIO server. should be the host ip and S3_PORT in .env file
        # Mlflow username and password. Should be MLFLOW_USER and MLFLOW_PASSWORD in .env
        os.environ['MLFLOW_TRACKING_USERNAME'] = "<uname>"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "<pw>"

