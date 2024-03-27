import os
import datetime


class JSSPConfig:
    def __init__(self):
        self.learning_type = "gumbeldore"  # 'gumbeldore' or 'supervised'
        self.architecture = "JobTransformer"  # Only 'JobTransformer' possible here.

        # Gumbeldore samples from these sizes (num jobs, num machines) to generate training data.
        self.problem_sizes_to_generate = [(15, 10), (15, 15), (15, 20)]

        # For network
        self.latent_dimension = 64
        self.num_transformer_blocks = 6
        self.num_attention_heads = 8
        self.feedforward_dimension = 256
        self.dropout = 0.0
        self.use_rezero_transformer = True  # Use rezero normalization.

        self.load_checkpoint_from_path = None  # If path given, model checkpoint is loaded from this path.
        self.load_optimizer_state = False  # If True, the optimizer state is also loaded.
        self.reset_best_validation = False  # If True, the current best validation metric of the current best model is reset.

        # For training and supervised data loading
        self.seed = 42
        self.num_dataloader_workers = 1  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0,1"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = "cpu"  # Device on which supervised training is performed.
        self.num_epochs = 100  # Number of epochs to train.
        self.validation_every_n_epochs = 1  # After how many epochs to always perform validation
        self.batch_size_training = 512

        self.optimizer = {
            "lr": 2e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 0,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {  # Learning rate scheduling.
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }

        self.gumbeldore_config = {
            "active_search": None,  # If a path is given here, we don't randomly generate instances but load them from here.
            "use_best_model_for_generation": True,  # if this is True, we not take the current model checkpoint to generate data, but the best one so far.
            "append_if_not_new_best": True,  # if this is True, and use_best_model_for_generation is True, we do not reset the dataset after each epoch when the model has not improved, but append to the current.
            "devices_for_workers": ["cpu"] * 3,  # List of devices where the i-th data generation worker uses the i-th device.
            # A list where the i-th entry defines how many instances of the i-th size in `problem_sizes_to_generate` should be
            # generated.
            "num_instances_to_generate": [512, 512, 512],
            "destination_path": "./data/jssp_gumbeldore_dataset.pickle",  # Path so save generated data.
            # A list with the batch size for each problem size to sample from (see above). If `gumbeldore_eval` is True,
            # this should be a single integer.
            "batch_size_per_worker": [4, 4, 4],
            "batch_size_per_cpu_worker": [1, 1, 1],  # Different batch size for CPU workers, specified as `batch_size_per_worker`.
            # Must be "wor" (sample without replacement), "gumbeldore" (our method), or
            # "theory_gumbeldore" (locally estimated advantage with theoretical policy improvement)
            "search_type": "gumbeldore",
            "beam_width": 32,  # Beam width k for SBS in each round
            "num_rounds": 4,  # Number of rounds
            "pin_workers_to_core": False,  # If True, workers are pinned to a single CPU thread
            # For all the following parameters, see the docstring of
            # `IncrementalSBS.perform_incremental_sbs` in `core/incremental_sbs.py`.
            "advantage_constant": .05,
            "min_max_normalize_advantage": False,
            "expected_value_use_simple_mean": False,
            "use_pure_outcomes": False,
            "normalize_advantage_by_visit_count": False,
            "perform_first_round_deterministic": False,
            "min_nucleus_top_p": 1.0,
        }

        # For evaluation. This is done with beam search, except when `gumbeldore_eval` is True, then the gumbeldore settings above are used.
        self.gumbeldore_eval = False  # If True, evaluate with Gumbeldore at inference time, using the config above.
        self.devices_for_eval_workers = ["cpu"] * 4  # List of devices on which evaluation is performed. Here,  the i-th worker uses the i-th device.
        # We perform beam search for non-gumbeldore evaluation. This dict specifies the beam widths with which to evaluate.
        # Keys: Beam widths with which to perform evaluation, values: corresponding batch size for each eval worker.
        self.beams_with_batch_sizes = {
            1: 3
        }
        self.validation_relevant_beam_width = 1  # Output from this beam width is used as the relevant metric to decide if a model has improved or not
        self.training_set_path = None
        self.custom_num_instances = None
        # Number of batches for each of supervised training.
        # Given as [Tuple[str, int]]: If the first entry is "multiplier", the number of batches
        # is equal to the given value multiplied with the number of instances in the dataset.
        # If the first entry is "absolute" the given value is explicitly taken as the number of batches.
        self.custom_num_batches = ("absolute", 1000)
        self.validation_set_path = "./data/jssp/l2d/l2d_generatedData20_20_Seed200.pickle"
        self.validation_custom_num_instances = 100  # If integer n is given, only the first n instances are used from validation set.
        self.test_set_path = "./data/jssp/literature/jssp_taillard_15_15.pickle"

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

