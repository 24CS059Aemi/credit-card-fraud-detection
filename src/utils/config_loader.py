import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
