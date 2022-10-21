
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import debug, info
from pathlib import Path
from typing import Optional

import numpy as np
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import \
    get_artifact_repository
from torch.utils.data import IterableDataset, get_worker_info

from .models.math_functions import *
from .tools import *


def get_worker_id():
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info else 0
    return worker_id


@dataclass
class FileInfo:
    """Descriptor for a file containing one or more episodes."""
    path: str
    episode_from: int
    episode_to: int
    steps: int
    artifact_repo: ArtifactRepository

    def load_data(self) -> Dict[str, np.ndarray]:
        data = mlflow_load_npz(self.path, self.artifact_repo)
        return data

    def __repr__(self):
        return f'{self.path}'


class EpisodeRepository(ABC):

    @abstractmethod
    def save_data(self, data: Dict[str, np.ndarray], episode_from: int, episode_to: int):
        ...

    @abstractmethod
    def list_files(self) -> List[FileInfo]:
        ...


class MlflowEpisodeRepository(EpisodeRepository):

    def __init__(self, artifact_uris: Union[str, List[str]]):
        super().__init__()
        self.artifact_uris = [artifact_uris] if isinstance(artifact_uris, str) else artifact_uris
        self.read_repos: List[ArtifactRepository] = [get_artifact_repository(uri) for uri in self.artifact_uris]
        self.write_repo = self.read_repos[0]

    def save_data(self, data: Dict[str, np.ndarray], episode_from: int, episode_to: int, chunk_seq: Optional[int] = None):
        n_episodes = data['reset'].sum()