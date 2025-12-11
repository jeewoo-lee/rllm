"""
Ray runtime environment setup and initialization for SkyRL backend.

This module abstracts away direct imports from skyrl_train and provides
a clean interface for initializing Ray and validating configs.
"""
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

# Add skyrl-train to Python path (it's a git submodule)
_skyrl_train_path = Path(__file__).parent.parent.parent.parent / "skyrl" / "skyrl-train"
if _skyrl_train_path.exists():
    sys.path.insert(0, str(_skyrl_train_path))


def validate_cfg(cfg: "DictConfig") -> None:
    """
    Validate SkyRL training configuration.
    
    Args:
        cfg: Training configuration to validate
    """
    from skyrl_train.entrypoints.main_base import validate_cfg as _validate_cfg
    _validate_cfg(cfg)


def initialize_ray(cfg: "DictConfig") -> None:
    """
    Initialize Ray cluster for SkyRL training.
    
    This function handles:
    - Shutting down existing Ray cluster if needed
    - Initializing Ray with SkyRL's runtime environment
    - GPU detection and configuration
    
    Args:
        cfg: Training configuration
    """
    import ray
    from skyrl_train.utils import initialize_ray as _initialize_ray
    
    # If Ray is already initialized, shut it down first so SkyRL can initialize it properly
    if ray.is_initialized():
        ray.shutdown()
    
    _initialize_ray(cfg)


