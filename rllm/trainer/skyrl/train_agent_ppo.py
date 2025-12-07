# Copyright under Agentica Project.
"""
SkyRL entrypoint for training with rLLM workflows.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf, DictConfig

from rllm.trainer.env_agent_mappings import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING, WORKFLOW_CLASS_MAPPING
from rllm.engine.rollout.skyrl_engine import SkyRLEngine

# SkyRL imports
from skyrl_train.trainer.main_base import BasePPOExp
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.generators.base import GeneratorInterface


@hydra.main(config_path="../config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config):
    """Main entry point for SkyRL PPO training."""
    # Validate config
    validate_cfg(config)
    
    # Initialize Ray
    if not ray.is_initialized():
        initialize_ray(config)
    
    # Create remote task runner
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
    
    # Optional timeline
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks with SkyRL."""

    def run(self, config, workflow_class=None, workflow_args=None):
        """Execute the main PPO training workflow.
        
        Args:
            config: Training configuration object
            workflow_class: Optional workflow class to use
            workflow_args: Optional workflow arguments
        """
        from pprint import pprint
        from transformers import AutoTokenizer

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))
        
        # Load tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = AutoTokenizer.from_pretrained(
            config.trainer.policy.model.path,
            trust_remote_code=trust_remote_code,
            use_fast=not config.trainer.get("disable_fast_tokenizer", False),
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Get workflow class if using workflows
        if config.rllm.workflow.use_workflow:
            if workflow_class is None:
                workflow_class = WORKFLOW_CLASS_MAPPING[config.rllm.workflow.name]
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value
        
        # Create SkyRL PPO experiment
        exp = RLLMWorkflowPPOExp(
            cfg=config,
            workflow_class=workflow_class,
            workflow_args=workflow_args,
        )
        exp.run()


class RLLMWorkflowPPOExp(BasePPOExp):
    """SkyRL PPO experiment that uses rLLM workflows via SkyRLEngine."""
    
    def __init__(self, cfg: DictConfig, workflow_class=None, workflow_args=None):
        """Initialize the experiment with workflow configuration."""
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        super().__init__(cfg)
    
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Returns the SkyRLEngine generator that uses rLLM workflows.
        
        Args:
            cfg: Configuration object
            tokenizer: Tokenizer instance
            inference_engine_client: Inference engine client
            
        Returns:
            GeneratorInterface: The SkyRLEngine generator
        """
        return SkyRLEngine(
            config=cfg,
            tokenizer=tokenizer,
            inference_engine_client=inference_engine_client,
            workflow_class=self.workflow_class,
            workflow_args=self.workflow_args,
        )


if __name__ == "__main__":
    main()
