"""Backbone experiment profiles.

Profiles are intentionally data-only YAML files.  Dataset shape/storage
settings stay in :mod:`configs.downstream`; a profile owns the model name,
backbone-wide training overrides, and result bookkeeping for that model.
"""

from copy import deepcopy
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
BACKBONE_CONFIG_ROOT = ROOT / "backbones"


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_.]+", "_", str(name)).strip("_").lower()


def available_backbones():
    """Return profile directory names that contain ``default.yaml``."""
    if not BACKBONE_CONFIG_ROOT.exists():
        return ()
    return tuple(sorted(
        path.parent.name
        for path in BACKBONE_CONFIG_ROOT.glob("*/default.yaml")
    ))


def _dataset_candidates(directory, dataset):
    if dataset is None:
        return ()
    safe = safe_name(dataset)
    return (
        directory / (str(dataset) + ".yaml"),
        directory / (safe + ".yaml"),
    )


def _profile_directory(value):
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    if not path.is_absolute():
        candidate = BACKBONE_CONFIG_ROOT / path
        if candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_backbone_config(value, dataset=None):
    """Resolve a profile name or YAML path, optionally for one dataset."""
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return path.resolve()
    profile_dir = _profile_directory(value)
    if profile_dir is None:
        candidates = [ROOT / path, BACKBONE_CONFIG_ROOT / (safe_name(path.name) + ".yaml")]
    else:
        candidates = list(_dataset_candidates(profile_dir, dataset)) + [
            profile_dir / "default.yaml",
            profile_dir / "config.yaml",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Unknown backbone profile {!r}; expected a YAML path or one of: {}".format(
            value, ", ".join(available_backbones()) or "<none>"
        )
    )


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_backbone_config(value, dataset=None):
    if value is None:
        return {}
    value_path = Path(value)
    profile_dir = _profile_directory(value)
    explicit_file = value_path.is_file()

    if explicit_file:
        path = value_path.resolve()
        config = _read_yaml(path)
        if dataset is not None and path.name not in {
            str(dataset) + ".yaml", safe_name(dataset) + ".yaml",
            "default.yaml", "config.yaml",
        }:
            # A complete experiment YAML is also passed to the child process
            # as its profile source. In that case validate the declared
            # dataset instead of requiring a particular filename.
            if path.name not in {"default.yaml", "config.yaml"}:
                declared = config.get("dataset")
                if declared is None:
                    raise ValueError(
                        "Backbone dataset profile {} cannot be used for dataset {}; "
                        "pass the profile directory/name instead.".format(path.name, dataset)
                    )
                declared = [declared] if isinstance(declared, str) else declared
                if not isinstance(declared, list) or dataset not in declared:
                    raise ValueError(
                        "Experiment profile {} does not declare dataset {}.".format(path, dataset)
                    )
        # Explicit default.yaml may still be used as a base for one dataset.
        if dataset is not None and path.name in {"default.yaml", "config.yaml"}:
            dataset_path = next((candidate for candidate in _dataset_candidates(path.parent, dataset)
                                 if candidate.is_file()), None)
            if dataset_path is not None:
                config = _deep_merge(config, _read_yaml(dataset_path))
    else:
        path = resolve_backbone_config(value, dataset=None)
        if path is None:
            return {}
        profile_dir = profile_dir or path.parent
        base_path = next((candidate for candidate in (
            profile_dir / "default.yaml", profile_dir / "config.yaml"
        ) if candidate.is_file()), None)
        config = _read_yaml(base_path) if base_path else {}
        if dataset is not None:
            dataset_path = next((candidate for candidate in _dataset_candidates(profile_dir, dataset)
                                 if candidate.is_file()), None)
            if dataset_path is not None:
                path = dataset_path.resolve()
                config = _deep_merge(config, _read_yaml(path))

    _validate_profile(config, path)
    config["_path"] = str(path)
    config["_dataset"] = dataset
    return config


def _read_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Backbone profile must contain a YAML mapping: {}".format(path))
    if "extends" in config:
        parent = (path.parent / str(config.pop("extends"))).resolve()
        if not parent.is_file():
            raise FileNotFoundError("Profile parent does not exist: {}".format(parent))
        config = _deep_merge(_read_yaml(parent), config)
    return config


def _validate_profile(config, path):
    if "backbone_name" in config and not isinstance(config["backbone_name"], str):
        raise ValueError("'backbone_name' must be a string: {}".format(path))
    for section in ("training", "vision", "protocol", "results"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError("'{}' must be a YAML mapping: {}".format(section, path))
    training = config.get("training", {})
    invalid_training = sorted(set(training) - {
        "lr", "backbone_lr_scale", "batch_size", "epochs", "weight_decay", "min_lr",
        "warmup_epochs", "warmup_start_factor", "clip_value", "ema_decay", "num_workers",
        "optimizer", "label_smoothing", "binary_pos_weight", "dropout", "drop_path_rate",
        "early_stop", "frozen", "multi_lr", "use_pretrained_weights", "balanced_sampling",
        "balanced_sampling_power", "balanced_sampling_min_share", "mirror_augmentation",
        "mirror_prob", "time_roll_augmentation", "time_roll_prob", "time_roll_max_fraction",
        "amplitude_scale_augmentation", "amplitude_scale_prob", "amplitude_scale_min",
        "amplitude_scale_max", "amplitude_scale_distribution", "mixup_augmentation",
        "mixup_prob", "mixup_alpha", "amp", "amp_dtype", "mental_scale",
        "shu_clip_limit", "shu_scale", "shu_bandpass_low", "shu_bandpass_high",
        "shu_filter_order", "physio_lowpass_hz", "physio_filter_order", "mumtaz_lowpass_hz",
        "mumtaz_filter_order", "faced_input_norm", "faced_robust_clip", "test_each_epoch",
        "run_final_test", "selection_metric",
    })
    if invalid_training:
        raise ValueError("Unknown training keys in {}: {}".format(path, ", ".join(invalid_training)))
    seeds = config.get("protocol", {}).get("seeds")
    if seeds is not None and (not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds)):
        raise ValueError("protocol.seeds must be a list of integers: {}".format(path))


def backbone_name_for(config, fallback=None):
    return config.get("backbone_name") or config.get("backbone", {}).get("name") or fallback


def profile_training_overrides(config):
    values = config.get("training", {})
    if not isinstance(values, dict):
        raise ValueError("Backbone profile 'training' must be a mapping")
    return deepcopy(values)


def profile_result_summary(config):
    """Return bookkeeping fields without mutating the loaded profile."""
    return {
        "seeds": tuple(config.get("protocol", {}).get("seeds", ())),
        "results": deepcopy(config.get("results", {})),
        "notes": config.get("notes", ""),
    }
