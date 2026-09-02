"""Loader and validation for complete experiment YAML files."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_experiment_config(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Experiment config does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping: {}".format(path))
    if not config.get("dataset"):
        raise ValueError("Experiment config requires 'dataset': {}".format(path))
    if isinstance(config["dataset"], str):
        config["dataset"] = [config["dataset"]]
    if not isinstance(config["dataset"], list) or not all(isinstance(v, str) for v in config["dataset"]):
        raise ValueError("Experiment config 'dataset' must be a string or list of strings: {}".format(path))
    for section in ("training", "vision", "protocol", "output"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError("Experiment config '{}' must be a mapping: {}".format(section, path))
    seeds = config.get("protocol", {}).get("seeds")
    if seeds is not None and (not isinstance(seeds, list) or not all(isinstance(v, int) for v in seeds)):
        raise ValueError("Experiment config protocol.seeds must be a list of integers: {}".format(path))
    if "backbone" in config and not isinstance(config["backbone"], dict):
        raise ValueError("Experiment config 'backbone' must be a mapping: {}".format(path))
    if "backbone_name" in config and not isinstance(config["backbone_name"], str):
        raise ValueError("Experiment config 'backbone_name' must be a string: {}".format(path))
    if "model_arch" in config and not isinstance(config["model_arch"], str):
        raise ValueError("Experiment config 'model_arch' must be a string: {}".format(path))
    config["_path"] = str(path.resolve())
    return deepcopy(config)
