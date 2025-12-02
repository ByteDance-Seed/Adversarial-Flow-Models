"""
Entrypoint for launching train jobs.
The first argument must be a yaml training configuration file path.
The additional arguments support commandline override.
"""

from sys import argv

from common.config import create_object, load_config
from common.entrypoint import Entrypoint

# Load config.
config = load_config(argv[1], argv[2:])

# Load trainer.
entrypoint = create_object(config)
assert isinstance(entrypoint, Entrypoint)

# Start trainer.
entrypoint.entrypoint()
