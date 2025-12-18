from pathlib import Path
import yaml
import datetime
from .types import FullConfig

def load(path: str | Path) -> FullConfig:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    
    # Validate and parse with pydantic
    config = FullConfig(**raw)
    # Last modified tiemstamp
    config.timestamp = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return config