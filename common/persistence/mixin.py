from typing import Optional
from omegaconf import OmegaConf

from common.entrypoint import Entrypoint

from .dataclass import PersistedTrainingState
from .manager import PersistenceManager


class PersistenceMixin:
    """
    Provide persistence capability.
    Config must contain a "persistence" key.
    """

    # ----------------- Example Config -----------------
    # persistence:
    #   path: hdfs://path/to/location  (required)
    #   override: False                (default False)
    # --------------------------------------------------

    persistence: PersistenceManager
    resume: Optional[PersistedTrainingState]

    def configure_persistence(self, mode: str = "train"):
        assert mode in ["train", "eval"]
        assert isinstance(self, Entrypoint)
        self.persistence = PersistenceManager(path=self.config.persistence.path)
        if mode == "train":
            config = self.config.copy()
            OmegaConf.set_readonly(config, False)
            override = config.persistence.pop("override", False)
            self.persistence.save_config(config, override=override)
            self.resume = self.persistence.load_last_step()
