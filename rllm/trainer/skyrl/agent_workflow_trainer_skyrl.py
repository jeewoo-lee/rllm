"""
SkyRL Trainer for rLLM workflows integration.

This trainer uses rLLM's AgentWorkflowEngine to execute workflows with SkyRL's training infrastructure.
The key integration point is the generate() method which sets up episode logging metadata.
"""

import asyncio
import torch
import numpy as np

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from skyrl_train.utils.trainer_utils import validate_generator_output
from skyrl_train.utils.ppo_utils import register_advantage_estimator


@register_advantage_estimator("loop")
def compute_advantages_and_returns_loop(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    values: torch.Tensor,
    config,
    gamma,
    lambd,
    grpo_norm_by_std,
    **kwargs,
):
    from collections import defaultdict

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    id2samples = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2samples[index[i]].append((i, scores[i]))
        for group in id2samples.values():
            group_size = len(group)
            total_score = sum(score for _, score in group)
            for i, score in group:  # i is original index
                loo_baseline = 0
                if group_size == 1:
                    print("Cannot compute LOO advantage using 1 sample. 0 baseline is used")
                else:
                    loo_baseline = (total_score - score) / (group_size - 1)
                scores[i] = score - loo_baseline
        scores = scores.unsqueeze(-1) * response_mask
        return scores, scores


class AgentWorkflowPPOTrainer(RayPPOTrainer):
    """PPO Trainer for rLLM workflows using SkyRL infrastructure.
    
    This trainer integrates rLLM's AgentWorkflowEngine with SkyRL's training pipeline.
    The generator (RLLMGenerator) uses AgentWorkflowEngine to execute workflows and
    transform results to SkyRL's GeneratorOutput format.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: Base class already builds dataloaders, but we can override if needed for reproducibility
        # For now, we use the base class dataloaders

    @torch.no_grad()
    async def generate(
        self,
        input_batch: GeneratorInput,
    ) -> GeneratorOutput:
        """
        Generate rollouts.

        If colocate_all is enabled:
        - before calling this method, the policy model should be on CPU and inference engine should
            be awake (i.e. on GPU).
        - after calling this method, the same model placement still holds.
        """
        # Initialize AgentWorkflowEngine pool if generator uses it and pool is not initialized
        if hasattr(self.generator, 'agent_workflow_engine'):
            if self.generator.agent_workflow_engine.workflow_queue is None:
                await self.generator.agent_workflow_engine.initialize_pool()
            
            # Set training step for episode logging (rLLM abstraction)
            # Calculate epoch from global_step and dataloader length
            batch_metadata = input_batch.get("batch_metadata")
            if batch_metadata:
                global_step = batch_metadata.global_step if hasattr(batch_metadata, 'global_step') else self.global_step
                training_phase = batch_metadata.training_phase if hasattr(batch_metadata, 'training_phase') else "train"
            else:
                global_step = self.global_step
                training_phase = "train"
            
            # Calculate epoch: epoch = global_step // steps_per_epoch
            # Note: global_step starts at 1, so we subtract 1 before dividing
            steps_per_epoch = len(self.train_dataloader) if self.train_dataloader else 1
            epoch = (global_step - 1) // steps_per_epoch if global_step > 0 else 0
            
            self.generator.agent_workflow_engine.set_training_step(global_step, mode=training_phase, epoch=epoch)
        
        # NOTE: we assume that .generate returns samples in the same order as passed in
        # Here SkyAgent would return a repeated output (n_samples_per_prompt times)
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])

        # Validate output - base function takes num_prompts, not input_batch
        validate_generator_output(len(input_batch["prompts"]), generator_output)

        return generator_output
