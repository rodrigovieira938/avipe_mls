from .types import DatasetConfig
import kagglehub
import huggingface_hub
import importlib.util
from pathlib import Path

class DatasetDownloader:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def download(self):
        source_type = self.config.source.type
        path = ""
        if source_type == "kaggle":
            path = self._download_kaggle()
        elif source_type == "huggingface":
            path = self._download_huggingface()
        else:
            raise ValueError(f"Unknown dataset source type: {source_type}")
        print("Provided dataset is located at:", path)
        self._run_post_download_scripts(path)
    def _download_kaggle(self):
        return kagglehub.download_competition(self.config.source.competition) # type: ignore

    def _download_huggingface(self):
        return huggingface_hub.snapshot_download(repo_id=self.config.source.repo, repo_type=self.config.source.repo_type) # type: ignore
    def _run_post_download_scripts(self, dataset_path: str):
        for script_path in self.config.post_download_scripts:
            script_path_obj = Path(script_path)
            if not script_path_obj.is_file():
                raise FileNotFoundError(f"Post-download script not found: {script_path}")
            spec = importlib.util.spec_from_file_location("post_download_script", script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load script: {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not hasattr(module, "run"):
                raise AttributeError(f"Script {script_path} does not have a 'run' function")
            print(f"Running post-download script: {script_path}")
            module.run(dataset_path)