"""
Configuration loading and management.

Loads YAML configuration files with support for inheritance (default.yaml
is loaded first, then overridden by the specific config).
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str, default_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file, optionally merging with defaults.

    Args:
        config_path: Path to the specific config file.
        default_path: Path to the default config file. If None, attempts to
                      find 'default.yaml' in the same directory.

    Returns:
        Merged configuration dictionary.
    """
    # Load default config
    if default_path is None:
        config_dir = os.path.dirname(config_path)
        default_path = os.path.join(config_dir, "default.yaml")

    default_config = {}
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            default_config = yaml.safe_load(f) or {}

    # Load specific config
    with open(config_path, "r") as f:
        specific_config = yaml.safe_load(f) or {}

    # Merge: specific overrides default
    merged = merge_configs(default_config, specific_config)
    return merged


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two configuration dictionaries.

    Values in 'override' take precedence over 'base'.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration.
    """
    merged = base.copy()
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_config(config: Dict, path: str):
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_to_str(config: Dict, indent: int = 0) -> str:
    """Pretty-print a configuration dictionary."""
    lines = []
    prefix = "  " * indent
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(config_to_str(value, indent + 1))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)
