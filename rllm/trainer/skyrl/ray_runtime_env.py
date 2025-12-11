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


def prepare_config(
    cfg: "DictConfig",
    workflow_class: type | None = None,
    workflow_args: dict | None = None,
) -> None:
    """
    Prepare SkyRL config by serializing workflow_class.
    
    This function:
    - Converts workflow_class to a string path (for Ray serialization)
    
    Note: workflow_args are NOT put into config here. They are passed directly
    to skyrl_entrypoint() as a parameter and merged with config values in
    RLLMPPOExp.get_generator(). This allows functions to be included (Ray
    serializes them with cloudpickle).
    
    Args:
        cfg: Training configuration to modify in-place
        workflow_class: Workflow class to serialize into config
        workflow_args: Workflow arguments (unused here, kept for API consistency)
    """
    # Set workflow class in config if provided
    # Convert to string path because:
    # 1. Config needs to be serializable for Ray remote calls
    # 2. Class objects can't be serialized in DictConfig
    # 3. RLLMPPOExp.get_generator() will deserialize it using import_module
    if workflow_class is not None:
        workflow_class_str = f"{workflow_class.__module__}.{workflow_class.__name__}"
        cfg.generator.workflow_class = workflow_class_str
    
    # Note: workflow_args are NOT put into config here because:
    # 1. Functions can't be serialized in DictConfig
    # 2. workflow_args are passed directly to skyrl_entrypoint() as a parameter
    # 3. They will be merged with config values in RLLMPPOExp.get_generator()


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


def prep_ray_env(
    cfg: "DictConfig",
    workflow_class: type | None = None,
    workflow_args: dict | None = None,
) -> None:
    """
    Prepare Ray environment for SkyRL training.
    
    This is a convenience function that encapsulates all setup steps:
    1. Prepare config (serialize workflow_class and workflow_args)
    2. Validate config
    3. Initialize Ray cluster
    
    Args:
        cfg: Training configuration to modify in-place
        workflow_class: Workflow class to serialize into config
        workflow_args: Workflow arguments to merge into config (only serializable values)
    """
    prepare_config(cfg=cfg, workflow_class=workflow_class, workflow_args=workflow_args)
    validate_cfg(cfg)
    initialize_ray(cfg)
