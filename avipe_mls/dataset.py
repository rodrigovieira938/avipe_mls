from .types import DatasetConfig
import kagglehub
import huggingface_hub

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
    def _download_kaggle(self):
        return kagglehub.download_competition(self.config.source.competition) # type: ignore

    def _download_huggingface(self):
        return huggingface_hub.snapshot_download(repo_id=self.config.source.repo, repo_type=self.config.source.repo_type) # type: ignore