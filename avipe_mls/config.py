from pathlib import Path
import yaml

from .types import FullConfig

def load(path: str | Path) -> FullConfig:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    
    # Validate and parse with pydantic
    config = FullConfig(**raw)
    return config