import os
import mlflow
import argparse
import inspect
import json

from typing import Optional


class Logger:
    def __init__(self, results_path: str, log_to_file: bool, log_to_mlflow: bool, mlflow_server_uri: Optional[str] = None):
        self.results_path = results_path
        self.log_to_file = log_to_file
        self.log_to_mlflow = log_to_mlflow
        self.mlflow_server_uri = mlflow_server_uri

        self.file_log_path = os.path.join(self.results_path, "log.txt")
        if self.log_to_file:
            os.makedirs(self.results_path, exist_ok=True)

        # Check if we should log to MLflow
        parser = argparse.ArgumentParser(description='Experiment.')
        parser.add_argument('--debug', help="debug flag to turn off server logging", action="store_true")
        parser.add_argument('--run-name', type=str, help="give a descriptive run name so we can keep track of results", default="Default run")
        parser.add_argument('--exp-name', type=str, help="MLflow Experiment name to group run into", default="Default experiment")
        args = parser.parse_args()

        if args.debug:
            self.log_to_mlflow = False
        if self.log_to_mlflow:
            print(
                f"MLFlow: Tracking run '{args.run_name}' in experiment '{args.exp_name}'")
            mlflow.set_tracking_uri(self.mlflow_server_uri)
            mlflow.set_experiment(args.exp_name)
            mlflow.set_tag("mlflow.runName", args.run_name)

    def log_hyperparams(self, config_object):
        attributes = inspect.getmembers(config_object, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
        attribute_dict = {}

        def add_to_attribute_dict(a):
            for key, value in a:
                if isinstance(value, dict):
                    add_to_attribute_dict([(f"{key}.{k}", v) for k, v in value.items()])
                else:
                    if key not in ["devices_for_eval_workers"] and len(str(value)) <= 500:
                        attribute_dict[key] = value

        add_to_attribute_dict(attributes)

        if self.log_to_mlflow:
            mlflow.log_params(attribute_dict)
        if self.log_to_file:
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps({"hyperparameters": attribute_dict}))
                f.write("\n")

    def log_metrics(self, metrics: dict, step: Optional[int] = None, step_desc: Optional[str] = "epoch"):
        if self.log_to_mlflow:
            mlflow.log_metrics(metrics, step=step)
        if self.log_to_file:
            if step is not None:
                metrics[step_desc] = step
            with open(self.file_log_path, "a+") as f:
                f.write(json.dumps(metrics))
                f.write("\n")
