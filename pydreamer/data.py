
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
        n_steps = len(data['reset']) - n_episodes
        reward = data['reward'].sum()
        fname = self.build_episode_name(episode_from, episode_to, reward, n_steps, chunk_seq=chunk_seq)
        print_once(f'Saving episode data ({chunk_seq}):', self.write_repo.artifact_uri + '/' + fname)
        mlflow_log_npz(data, fname, repository=self.write_repo)

    def list_files(self) -> List[FileInfo]:
        while True:
            try:
                return self._list_files()
            except:
                exception('Error listing files - will retry.')
                time.sleep(10)

    def _list_files(self) -> List[FileInfo]:
        files = []
        for repo in self.read_repos:
            for f in repo.list_artifacts(''):  # type: ignore
                if f.path.endswith('.npz') and not f.is_dir:
                    (ep_from, ep_to, steps) = self.parse_episode_name(f.path)
                    files.append(FileInfo(path=f.path,
                                          episode_from=ep_from,
                                          episode_to=ep_to,
                                          steps=steps,
                                          artifact_repo=repo))
        return files

    def count_steps(self):
        files = self.list_files()
        steps = sum(f.steps for f in files)
        episodes = (max(f.episode_to for f in files) + 1) if files else 0
        return len(files), steps, episodes

    
    def build_episode_name(self, episode_from, episode, reward, steps, chunk_seq=None):
        if chunk_seq is None:
            return f'ep{episode_from:06}_{episode:06}-r{reward:.0f}-{steps:04}.npz'
        else:
            return f'ep{episode_from:06}_{episode:06}-{chunk_seq}-r{reward:.0f}-{steps:04}.npz'


    def parse_episode_name(self, fname):
        # TODO: regex
        fname = fname.split('/')[-1].split('.')[0]
        if fname.startswith('ep'):
            # fname = 'ep{epfrom}_{episode}-r{reward}-{steps}.npz'
            #       | 'ep{episode}-r{reward}-{steps}.npz'
            steps = fname.split('-')[-1]
            steps = int(steps) if steps.isnumeric() else 0
            ep_from = fname.split('ep')[1].split('-')[0].split('_')[0]
            ep_from = int(ep_from) if ep_from.isnumeric() else 0
            ep_to = fname.split('ep')[1].split('-')[0].split('_')[-1]
            ep_to = int(ep_to) if ep_to.isnumeric() else 0
            return (ep_from, ep_to, steps)
        else:
            # fname = '{timestamp}-{steps}
            steps = fname.split('-')[-1]
            steps = int(steps) if steps.isnumeric() else 0
            return (0, 0, steps)

    def __repr__(self):
        return f'{self.artifact_uris}'


class DataSequential(IterableDataset):
    """Dataset which processes episodes sequentially"""

    def __init__(self,
                 repository: EpisodeRepository, 
                 batch_length, 
                 batch_size, 
                 skip_first=True, 
                 reload_interval=0, 
                 buffer_size=0, 
                 reset_interval=0,
                 allow_mid_reset=False,
                 check_nonempty=True,
                 ):
        super().__init__()
        self.repository = repository
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.skip_first = skip_first
        self.reload_interval = reload_interval
        self.buffer_size = buffer_size
        self.reset_interval = reset_interval
        self.allow_mid_reset = allow_mid_reset
        self.reload_files(True)
        if check_nonempty:
            assert len(self.files) > 0, 'No data found'

    def reload_files(self, is_first=False):
        verbose = get_worker_id() == 0
        if is_first and verbose:
            debug(f'Reading files from {self.repository}...')

        files_all = self.repository.list_files()
        files_all.sort(key=lambda e: -e.episode_to)  # TODO: prefill should be sorted first. can we use timestamp?

        files = []
        steps_total = 0
        steps_filtered = 0
        for f in files_all:
            steps_total += f.steps
            if steps_total < self.buffer_size or not self.buffer_size:
                files.append(f)
                steps_filtered += f.steps

        self.files: List[FileInfo] = files
        self.last_reload = time.time()
        self.stats_steps = steps_total

        if verbose:
            debug(f'Found total files|steps: {len(files_all)}|{steps_total}, filtered: {len(self.files)}|{steps_filtered}')

    def should_reload_files(self):
        return self.reload_interval and (time.time() - self.last_reload > self.reload_interval)

    def __iter__(self):
        # Parallel iteration over (batch_size) iterators
        # Iterates forever
        iters = [self.iter_single(ix) for ix in range(self.batch_size)]
        for batches in zip(*iters):
            batch = stack_structure_np(batches)
            batch = map_structure(batch, lambda d: d.swapaxes(0, 1))
            yield batch

    def iter_single(self, ix):
        # Iterates "single thread" forever
        skip_random = self.skip_first
        last_partial_batch = None

        for file in self.iter_shuffled_files():
            if last_partial_batch: