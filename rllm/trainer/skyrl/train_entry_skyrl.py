"""
SkyRL entrypoint for training rLLM workflows.
"""

import hydra
import asyncio
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
from skyrl_train.generators.utils import get_rollout_metrics, get_response_ids_and_loss_mask_from_messages
import ray

from rllm.engine.rollout.skyrl_engine import SkyRLEngine
from .agent_workflow_trainer_skyrl import AgentWorkflowPPOTrainer


class RLLMGenerator(GeneratorInterface):
    """rLLM generator that uses rLLM workflows for SkyRL training.

    This generator:
    1. Receives GeneratorInput from SkyRL trainer
    2. Runs rLLM workflows to generate trajectories
    3. Converts results to GeneratorOutput for SkyRL training
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

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories using rLLM workflows.

        Args:
            input_batch: SkyRL's GeneratorInput with prompts and environment info

        Returns:
            GeneratorOutput: Tensor data ready for SkyRL training
        """
        # Extract prompts and environment info
        prompts = input_batch["prompts"]
        env_extras = input_batch.get("env_extras", [{}] * len(prompts))

        # Run all workflows in parallel to get episodes
        tasks = []
        for prompt, env_extra in zip(prompts, env_extras):
            tasks.append(self._run_workflow(prompt, env_extra))

        episodes = await asyncio.gather(*tasks)

        # Transform episodes to GeneratorOutput
        return self._transform_episodes_to_generator_output(episodes)

    async def _run_workflow(self, prompt, env_extra):
        """Run a single rLLM workflow and return Episode.

        Args:
            prompt: ConversationType (list of messages)
            env_extra: Environment configuration/task data

        Returns:
            Episode: Result from workflow execution
        """
        # Extract task from env_extra
        task = env_extra.get("task", {"prompt": prompt})

        # Create workflow instance
        executor = ThreadPoolExecutor(max_workers=1)
        workflow = self.workflow_class(
            rollout_engine=self.rollout_engine,
            executor=executor,
            **self.workflow_args
        )

        # Run workflow to get episode
        episode = await workflow.run(task=task, uid=f"workflow_{id(prompt)}")

        return episode

    def _transform_episodes_to_generator_output(self, episodes) -> GeneratorOutput:
        """Transform Episodes to GeneratorOutput.

        This method is episode-facing: it processes all trajectories from all episodes.

        Args:
            episodes: List of Episode objects from workflow execution

        Returns:
            GeneratorOutput: Formatted data for SkyRL training
        """
        prompt_token_ids = []
        response_ids = []
        rewards = []
        loss_masks = []
        stop_reasons = []
        rollout_logprobs = []

        # Episode-facing loop
        for episode in episodes:
            if not episode.trajectories or len(episode.trajectories) == 0:
                continue

            # Process ALL trajectories in the episode
            for trajectory in episode.trajectories:
                if len(trajectory.steps) == 0:
                    continue

                # Build chat history from all steps
                chat_history = []
                for step in trajectory.steps:
                    if step.chat_completions:
                        chat_history.extend(step.chat_completions)

                if len(chat_history) == 0:
                    continue

                # Extract prompt (first user message)
                assert chat_history[0]["role"] == "user", "First message should be user"
                prompt_messages = [chat_history[0]]
                prompt_ids = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=False,
                    tokenize=True,
                )

                # Process response messages
                response_messages = chat_history[1:]
                assistant_logprobs = None  # TODO: Extract if available from ModelOutput
                resp_ids, loss_mask, resp_logprobs = get_response_ids_and_loss_mask_from_messages(
                    response_messages, self.tokenizer, assistant_logprobs
                )

                # Determine stop reason
                stop_reason = "complete"
                if len(resp_ids) > self.max_response_length:
                    stop_reason = "length"

                # Truncate to maximum allowed length
                resp_ids = resp_ids[:self.max_response_length]
                loss_mask = loss_mask[:self.max_response_length]
                resp_logprobs = resp_logprobs[:self.max_response_length] if resp_logprobs else None

                # Append to batch
                prompt_token_ids.append(prompt_ids)
                response_ids.append(resp_ids)
                rewards.append(trajectory.reward)
                loss_masks.append(loss_mask)
                stop_reasons.append(stop_reason)
                rollout_logprobs.append(resp_logprobs)

        # Compute rollout metrics
        rollout_metrics = get_rollout_metrics(response_ids, rewards)

        # Return GeneratorOutput
        return GeneratorOutput(
            prompt_token_ids=prompt_token_ids,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=stop_reasons,
            rollout_metrics=rollout_metrics,
            rollout_logprobs=rollout_logprobs,
        )


class SkyRLAgentPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, llm_endpoint_client):
        # Get workflow class and args from config
        workflow_class = cfg.generator.get("workflow_class", None)
        workflow_args = cfg.generator.get("workflow_args", {})
        
        if workflow_class is None:
            raise ValueError("workflow_class must be specified in cfg.generator")
        
        # Import the workflow class
        from importlib import import_module
        module_path, class_name = workflow_class.rsplit(".", 1)
        module = import_module(module_path)
        workflow_cls = getattr(module, class_name)
        
        generator = RLLMGenerator(
            config=cfg,
            tokenizer=tokenizer,
            inference_engine_client=llm_endpoint_client,
            workflow_class=workflow_cls,
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
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = SkyRLAgentPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    task = skyrl_entrypoint.remote(cfg)
    try:
        ray.get(task)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
        ray.cancel(task)
        raise


if __name__ == "__main__":
    main()
