"""
SkyRL entrypoint for training rLLM workflows.
"""

import hydra
import asyncio
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.entrypoints.main_base import BasePPOExp
from pathlib import Path
from skyrl_train.generators.utils import get_rollout_metrics
import ray

from rllm.engine.rollout.skyrl_engine import SkyRLEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from .agent_workflow_trainer_skyrl import AgentWorkflowPPOTrainer


class RLLMGenerator(GeneratorInterface):
    """rLLM generator that uses rLLM workflows for SkyRL training.

    This generator:
    1. Receives GeneratorInput from SkyRL trainer
    2. Uses AgentWorkflowEngine to run rLLM workflows and generate trajectories
    3. Returns GeneratorOutput for SkyRL training
    """

    def __init__(
        self,
        config,
        tokenizer,
        inference_engine_client: InferenceEngineClient,
        workflow_class=None,
        workflow_args=None,
    ):
        """Initialize the rLLM generator.

        Args:
            config: Training configuration
            tokenizer: Tokenizer instance
            inference_engine_client: SkyRL's InferenceEngineClient
            workflow_class: rLLM workflow class to use
            workflow_args: Arguments for workflow initialization
        """
        self.config = config
        self.tokenizer = tokenizer
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self.max_response_length = config.generator.sampling_params.get("max_tokens", 4096)

        # Create the SkyRL engine (wraps InferenceEngineClient to RolloutEngine interface)
        self.rollout_engine = SkyRLEngine(
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )
        
        # Create AgentWorkflowEngine that uses SkyRLEngine
        n_parallel_tasks = config.generator.get("n_parallel_tasks", 128)
        retry_limit = config.generator.get("retry_limit", 3)
        
        self.agent_workflow_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=self.rollout_engine,
            config=self.config,
            n_parallel_tasks=n_parallel_tasks,
            retry_limit=retry_limit,
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories using rLLM workflows via AgentWorkflowEngine.

        Args:
            input_batch: SkyRL's GeneratorInput with prompts and environment info

        Returns:
            GeneratorOutput: Tensor data ready for SkyRL training
        """
        # Use AgentWorkflowEngine to execute tasks and transform results
        return await self.agent_workflow_engine.execute_tasks_skyrl(input_batch)


class RLLMPPOExp(BasePPOExp):
    def __init__(self, cfg: DictConfig, workflow_class: type | None = None, workflow_args: dict | None = None):
        """Initialize RLLM PPO experiment.
        
        Args:
            cfg: Training configuration
            workflow_class: Workflow class (required, must be passed as parameter)
            workflow_args: Optional workflow arguments (can include functions)
        """
        super().__init__(cfg)
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
    
    def get_generator(self, cfg, tokenizer, llm_endpoint_client):
        # workflow_class must be provided as parameter (no config fallback)
        if self.workflow_class is None:
            raise ValueError("workflow_class must be provided as parameter to RLLMPPOExp")
        
        workflow_class = self.workflow_class
        
        # Merge workflow_args from parameter with config (matching verl's pattern)
        # Start with parameter, then merge config values into it (parameter takes precedence)
        workflow_args = self.workflow_args or {}
        if cfg.generator.get("workflow_args") is not None:
            for key, value in cfg.generator.get("workflow_args").items():
                if value is not None:
                    if key in workflow_args and isinstance(workflow_args[key], dict) and isinstance(value, dict):
                        workflow_args[key].update(value)
                    else:
                        # Only set config value if parameter doesn't have this key
                        if key not in workflow_args:
                            workflow_args[key] = value
        
        generator = RLLMGenerator(
            config=cfg,
            tokenizer=tokenizer,
            inference_engine_client=llm_endpoint_client,
            workflow_class=workflow_class,
            workflow_args=workflow_args,
        )
        return generator

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator: GeneratorInterface,
        colocate_pg,
    ):
        """Initializes the trainer.

        Returns:
            RayPPOTrainer: The trainer.
        """
        return AgentWorkflowPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig, workflow_class: type | None = None, workflow_args: dict | None = None):
    """SkyRL entrypoint for training rLLM workflows.
    
    Args:
        cfg: Training configuration
        workflow_class: Workflow class (required, must be passed as parameter)
        workflow_args: Optional workflow arguments (can include functions)
    """
    from rllm.trainer.skyrl.ray_runtime_env import validate_cfg
    
    # Validate config (matching verl's pattern where validation happens in remote function)
    validate_cfg(cfg)
    
    # make sure that the training loop is not run on the head node.
    exp = RLLMPPOExp(cfg, workflow_class=workflow_class, workflow_args=workflow_args)
    exp.run()


def run_skyrl(cfg: DictConfig, workflow_class: type | None = None, workflow_args: dict | None = None) -> None:
    """Run SkyRL training.
    
    Args:
        cfg: Training configuration
        workflow_class: Workflow class (required, must be passed as parameter)
        workflow_args: Optional workflow arguments (can include functions)
    """
    import ray
    
    from rllm.trainer.skyrl.ray_runtime_env import initialize_ray
    
    # Initialize Ray (matching verl's pattern - inline initialization)
    initialize_ray(cfg)
    
    task = skyrl_entrypoint.remote(cfg, workflow_class=workflow_class, workflow_args=workflow_args)
    try:
        ray.get(task)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
        ray.cancel(task)
        raise


# Use rLLM's config directory for rLLM-specific configs
rllm_config_dir = str(Path(__file__).parent.parent / "config")

@hydra.main(config_path=rllm_config_dir, config_name="rllm_skyrl_ppo_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_skyrl(cfg)


if __name__ == "__main__":
    main()
