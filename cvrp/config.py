import os
import datetime


class CVRPConfig:
    def __init__(self):
        self.learning_type = "gumbeldore"  # 'gumbeldore' or 'supervised'
        self.num_nodes = 100  # number of customers
        # Vehicle capacity to use when generating data. Can be a single integer (always choose this capacity)
        # or a tuple (lower and upper bound from which to sample an integer)
        # or a list (sample capacity from list)
        self.vehicle_capacity = 50
        # "BQ" for https://arxiv.org/abs/2301.03313 or "LEHD" for https://arxiv.org/abs/2310.07985
        self.architecture = "BQ"

        # For network
        self.latent_dimension = 192
        self.num_transformer_blocks = 9
        self.num_attention_heads = 12
        self.feedforward_dimension = 512
        self.dropout = 0.0
        self.use_rezero_transformer = True  # This is only relevant for BQ, for LEHD no normalization is used.

        self.load_checkpoint_from_path = None  # If given, model checkpoint is loaded from this path.
        self.load_optimizer_state = True       # If True, the optimizer state is also loaded.
        self.reset_best_validation = True      # If True, the current best validation metric of the current best model is reset.

        # For training and supervised data loading
        self.seed = 42  # Random seed
        self.augment_direction = True  # If True, order of subtours is randomly reversed.
        self.augment_subtour_order = "sort"   # "random": Subtours are permuted randomly, "sort": Subtours are sorted (ASC) by their final remaining capacity.
        self.data_augmentation = True  # Use geometrical augmentation as in TSP (rotation, reflection, etc.)
        self.data_augmentation_linear_scale = False  # If this is True, then during augmentation we linearly scale an instance with a value between [0.5, 1]
        self.num_dataloader_workers = 1  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0,1"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = "cpu"  # Device on which supervised training is performed.
        self.num_epochs = 100  # Number of epochs to train
        self.validation_every_n_epochs = 1  # After how many epochs to always perform validation
        self.batch_size_training = 1024

        self.optimizer = {
            "lr": 2e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 0,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {  # Learning rate scheduling.
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }

        # Config for Gumbeldore sampling and data generation.
        self.gumbeldore_config = {
            "active_search": None,  # If a path is given here, we don't randomly generate instances but load them from here.
            "use_best_model_for_generation": True,  # if this is True, we do not take the current model checkpoint to generate data, but the best one so far.
            "append_if_not_new_best": True,  # if this is True, and use_best_model_for_generation is True, we do not reset the dataset after each epoch when the model has not improved, but append to the current.
            "devices_for_workers": ["cpu"] * 1,  # List of devices where the i-th data generation worker uses the i-th device.
            "num_instances_to_generate": 1000,  # Number of instances to generate in each epoch.
            "destination_path": "./data/cvrp_gumbeldore_dataset.pickle",  # Path to save the generated dataset to in each epoch. (gets overwritten)
            "batch_size_per_worker": 4,  # Batch size for each worker on the GPU
            "batch_size_per_cpu_worker": 1,  # Different batch size for CPU workers, should be smaller
            # Must be "wor" (sample without replacement), "gumbeldore" (our method), or
            # "theory_gumbeldore" (locally estimated advantage with theoretical policy improvement)
            "search_type": "gumbeldore",
            "beam_width": 32,  # Beam width k for SBS in each round
            "num_rounds": 4,  # Number of rounds
            "pin_workers_to_core": False,  # If True, workers are pinned to a single CPU thread
            # For all the following parameters, see the docstring of
            # `IncrementalSBS.perform_incremental_sbs` in `core/incremental_sbs.py`.
            "advantage_constant": 3.0,
            "min_max_normalize_advantage": False,
            "expected_value_use_simple_mean": False,
            "use_pure_outcomes": False,
            "normalize_advantage_by_visit_count": False,
            "perform_first_round_deterministic": False,
            "min_nucleus_top_p": 1.
        }

        # For evaluation. This is done with beam search, except when `gumbeldore_eval` is True, then the gumbeldore settings above are used.
        self.gumbeldore_eval = False  # If True, evaluate with Gumbeldore at inference time, using the config above.
        self.devices_for_eval_workers = ["cpu"] * 4  # List of devices on which evaluation is performed. Here,  the i-th worker uses the i-th device.
        # We perform beam search for non-gumbeldore evaluation. This dict specifies the beam widths with which to evaluate.
        # Keys: Beam widths with which to perform evaluation, values: corresponding batch size for each eval worker.
        self.beams_with_batch_sizes = {
            1: 16
        }
        self.validation_relevant_beam_width = 1  # Output from this beam width is used as the relevant metric to decide if a model has improved or not
        # Path to training data for supervised training. NOTE: Due to size constraints, the given file here is not the
        # real training data file with one million instances. See README.
        self.training_set_path = "./data/cvrp/cvrp_100_10k_test.pickle"
        self.custom_num_instances = None  # If integer n is given, only the first n instances are used from training set.
        # Number of batches for each of supervised training.
        # Given as [Tuple[str, int]]: If the first entry is "multiplier", the number of batches
        # is equal to the given value multiplied with the number of instances in the dataset.
        # If the first entry is "absolute" the given value is explicitly taken as the number of batches.
        self.custom_num_batches = ("absolute", 1000)  # See `cvrp/dataset.py`
        self.validation_set_path = "./data/cvrp/cvrp_100_10k_validation.pickle"
        self.validation_custom_num_instances = 5000  # If integer n is given, only the first n instances are used from validation set.
        self.test_set_path = "./data/cvrp/cvrp_100_10k_test.pickle"

        # Results and logging
        self.results_path = os.path.join("./results",
                                         datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights
        self.log_to_file = True
        self.log_to_mlflow = False
        # Optional MLFlow logging
        self.mlflow_server_uri = "<mlflow_server_uri>"
        os.environ["AWS_ACCESS_KEY_ID"] = "<AWS_ACCESS_KEY_ID>"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "<AWS_SECRET_ACCESS_KEY>"
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "<MLFLOW_S3_ENDPOINT_URL>"  # How to reach MinIO server. should be the host ip and S3_PORT in .env file
        # Mlflow username and password. Should be MLFLOW_USER and MLFLOW_PASSWORD in .env
        os.environ['MLFLOW_TRACKING_USERNAME'] = "<uname>"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "<pw>"

