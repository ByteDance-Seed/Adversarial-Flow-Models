"""
Persistence manager
"""

from multiprocessing.pool import ThreadPool
from os.path import basename, join, splitext
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from torch.distributed import ProcessGroup
from torch.distributed._state_dict_utils import _gather_state_dict as gather_state_dict

from common.decorators import global_rank_zero_only
from common.distributed import get_global_rank
from common.fs import exists, listdir

from .dataclass import (
    PersistedConfig,
    PersistedMetric,
    PersistedModel,
    PersistedOptimizer,
    PersistedStates,
    PersistedTrainingState,
)


class PersistenceManager:
    """
    Persistence manager handles config and checkpoint saving and resuming.
    """

    def __init__(
        self,
        path: str,
    ):
        self.path = path
        self.pool = ThreadPool(processes=1)

    # ---------------- Saving ----------------

    @global_rank_zero_only
    def save_config(
        self,
        config: DictConfig,
        override: False,
    ):
        """
        Save trainer config. Call by all ranks!
        """
        # Save config.
        persist = self._get_config()
        if override or not exists(persist.path):
            persist.save(config)
        else:
            new_config_yaml = OmegaConf.to_yaml(config, resolve=True)
            old_config_yaml = OmegaConf.to_yaml(persist.load(distributed=False), resolve=True)
            if old_config_yaml != new_config_yaml:
                new_cfg = OmegaConf.to_container(config, resolve=True)
                old_cfg = OmegaConf.to_container(persist.load(distributed=False), resolve=True)
                raise ValueError(
                    f"The persistence path already exists with a different config. "
                    f"The difference is {DeepDiff(old_cfg, new_cfg)}."
                )

    def save_model(
        self,
        *,
        step: int,
        name: str = "model",
        config: Optional[DictConfig],
        states: Any,
        dtype: torch.dtype = None,
        blocking: bool = True,
        transform: Callable = lambda x: x,
        process_group: Optional[ProcessGroup] = None,
    ):
        """
        Save model checkpoint. Save trainer config. Call by all ranks!
        Config is the model config, states is the model state_dict.
        Support saving multiple models by assigning different names.
        Support dtype conversion if needed.
        If process_group is not None, need to gather sharded state_dict before calling torch.save.

        Args:
            blocking: True if saving FSDP, False if DDP and others.
                It refers to whether concurrently run torch.save.
                HDFS copy is always non-blocking.
        """

        def _save_model():
            if get_global_rank() == 0:
                model = self._get_model(step, name)
                model.states.save(transform(states), dtype=dtype)
                if config is not None:
                    model.config.save(config)

        # DTensor dosen't support async gather_state_dict.
        # TODO(Jiashi): support async gather_state_dict with DTensor.
        if process_group is not None:
            states = gather_state_dict(
                state_dict=states,
                pg=process_group,
                cpu_offload=True,
                ranks_only=(0,),
            )
        if blocking:
            _save_model()
        else:
            self.pool.apply_async(_save_model)

    def save_optimizer(
        self,
        *,
        step: int,
        name: str = "optimizer",
        states: Any,
        dtype: torch.dtype = None,
        blocking: bool = True,
        process_group: Optional[ProcessGroup] = None,
    ):
        """
        Save optimizer checkpoint. Call by all ranks!
        States is the optimizer state_dict.
        Support saving multiple optimizers by assigning different names.
        Support dtype conversion if needed.
        If process_group is not None, need to gather sharded state_dict before calling torch.save.

        Args:
            blocking: True if saving FSDP, False if DDP and others.
                It refers to whether concurrently run torch.save.
                HDFS copy is always non-blocking.
        """

        def _save_optimizer():
            if get_global_rank() == 0:
                optimizer = self._get_optimizer(step, name)
                optimizer.states.save(states, dtype=dtype)

        # DTensor dosen't support async gather_state_dict.
        # TODO(Jiashi): support async gather_state_dict with DTensor.
        if process_group is not None:
            states = gather_state_dict(
                state_dict=states,
                pg=process_group,
                cpu_offload=True,
                ranks_only=(0,),
            )
        if blocking:
            _save_optimizer()
        else:
            self.pool.apply_async(_save_optimizer)

    @global_rank_zero_only
    def save_metric(self, *, step: int, metric: Dict[str, Any]):
        """
        Save metric. Called by all ranks.
        """
        metrics = self._get_metrics()
        metrics.save(step, metric)

    # ---------------- Loading ----------------

    def load_last_step(self) -> Optional[PersistedTrainingState]:
        """
        Load the last step, or return None if not found.
        Call this method by all ranks at the start of training to resume.
        """
        return self.load_step(step=None)

    def load_step(self, step: Optional[int]) -> Optional[PersistedTrainingState]:
        """
        Load a specific step, or return the last step.
        Return None if no content found.
        """
        if step is None or step == -1:
            # Find last step.
            steps = self.list_steps()
            if not len(steps):
                return None
            step = steps[-1]

        if not exists(join(self.path, f"states/{step:010}")):
            return None

        return PersistedTrainingState(
            step=step,
            models=self._get_models(step),
            optimizers=self._get_optimizers(step),
        )

    def load_config(self) -> Optional[PersistedConfig]:
        """
        Load the trainer config.
        """
        config = self._get_config()
        return config if exists(config.path) else None

    def list_steps(self) -> List[int]:
        """
        List all the saved steps.
        """
        states_dir = join(self.path, "states")
        if not exists(states_dir):
            return []
        return sorted([int(basename(path)) for path in listdir(states_dir)])

    def list_unevaluated_step(self, metric_names: Union[Sequence[str], str]) -> List[int]:
        """
        List all the unevaluated steps.
        """
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        all_steps = self.list_steps()
        existing_record = self._get_metrics().load()
        if existing_record is not None:
            existing_record = pd.DataFrame(existing_record)
            evaluated_steps = set(existing_record["step"])
            target_steps = set(all_steps) - evaluated_steps
            for metric_name in metric_names:
                if metric_name not in existing_record:
                    return all_steps
                if existing_record[metric_name].isna().any():
                    index = np.where(existing_record[metric_name].isna())[0]
                    target_steps.update(existing_record["step"].iloc[index])
            return sorted(list(target_steps))
        else:
            return all_steps

    # ---------------- Internal ----------------

    def _get_models(self, step) -> Dict[str, PersistedModel]:
        path = join(self.path, f"states/{step:010}/models")
        if not exists(path):
            return {}
        names = [splitext(basename(path))[0] for path in listdir(path)]
        return {name: self._get_model(step, name) for name in names}

    def _get_optimizers(self, step) -> Dict[str, PersistedOptimizer]:
        path = join(self.path, f"states/{step:010}/optimizers")
        if not exists(path):
            return {}
        names = [splitext(basename(path))[0] for path in listdir(path)]
        return {name: self._get_optimizer(step, name) for name in names}

    def _get_model(self, step: int, name: str) -> PersistedModel:
        return PersistedModel(
            config=PersistedConfig(join(self.path, f"configs/models/{name}.yaml")),
            states=PersistedStates(join(self.path, f"states/{step:010}/models/{name}.pth")),
        )

    def _get_optimizer(self, step: int, name: str) -> PersistedOptimizer:
        return PersistedOptimizer(
            states=PersistedStates(join(self.path, f"states/{step:010}/optimizers/{name}.pth")),
        )

    def _get_config(self) -> PersistedConfig:
        return PersistedConfig(join(self.path, "configs/main.yaml"))

    def _get_metrics(self) -> PersistedMetric:
        return PersistedMetric(join(self.path, "metrics/main.csv"))
