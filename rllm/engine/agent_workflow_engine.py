from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from tqdm import tqdm

from rllm.agents.agent import Episode
from rllm.engine.rollout import RolloutEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason, Workflow

# Avoid hard dependency on verl at import time; only for typing
if TYPE_CHECKING:
    from verl import DataProto

logger = logging.getLogger(__name__)


class AgentWorkflowEngine:
    def __init__(self, workflow_cls: type[Workflow], workflow_args: dict, rollout_engine: RolloutEngine, config=None, n_parallel_tasks: int = 128, retry_limit: int = 3, raise_on_error: bool = True, episode_logger=None, **kwargs):
        """Initialize the AgentWorkflowEngine.

        Args:
            workflow_cls: The workflow class to instantiate for each task.
            workflow_args: Arguments to pass to workflow instances.
            rollout_engine: Engine for model inference and rollout.
            config: Optional configuration object for training.
            n_parallel_tasks: Number of parallel workflow instances to maintain.
            retry_limit: Maximum number of retry attempts for failed tasks.
            raise_on_error: Whether to raise exceptions on permanent failures.
            episode_logger: Optional logger for saving episode data to files.
            **kwargs: Additional keyword arguments.
        """
        self.workflow_cls = workflow_cls
        self.workflow_args = workflow_args or {}

        self.rollout_engine = rollout_engine
        self.config = config  # if training

        self.retry_limit = retry_limit  # number of attempts to retry a task
        self.raise_on_error = raise_on_error
        self.kwargs = kwargs

        self.n_parallel_tasks = n_parallel_tasks
        self.executor = ThreadPoolExecutor(max_workers=self.n_parallel_tasks)
        self.workflow_queue = None

        # Episode logging support
        self.episode_logger = episode_logger
        self.current_step = 0
        self.current_epoch = 0
        self.current_mode = "train"  # "train" or "val"

    def set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
        """Set current training step for episode logging.

        Args:
            step: Current training step number
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        self.current_step = step
        self.current_mode = mode
        self.current_epoch = epoch

    async def initialize_pool(self):
        """Initialize the workflow pool with parallel workflow instances.

        Creates and populates the workflow queue with workflow instances
        for parallel task processing. This method is idempotent and will
        not recreate the pool if it already exists.
        """
        assert self.executor is not None, "executor is not initialized"
        if self.workflow_queue is not None:
            return
        self.workflow_queue = asyncio.Queue(maxsize=self.n_parallel_tasks)
        for i in range(self.n_parallel_tasks):
            workflow = self.workflow_cls(rollout_engine=self.rollout_engine, executor=self.executor, **self.workflow_args)
            assert workflow.is_multithread_safe(), "Workflows must contain only thread-save environments"
            self.workflow_queue.put_nowait(workflow)

    async def process_task_with_retry(self, task: dict, task_id: str, rollout_idx: int, **kwargs) -> tuple[str, int, Episode]:
        """Process a single task rollout with retry logic based on termination reasons.

        Args:
            task: Task dictionary containing the task specification.
            task_id: Unique identifier for the task.
            rollout_idx: Index of this rollout attempt for the task.
            **kwargs: Additional arguments passed to the workflow.

        Returns:
            tuple[str, int, Episode]: Task ID, rollout index, and completed episode.

        Raises:
            Exception: If task fails permanently after retry_limit attempts and raise_on_error is True.
        """
        assert self.workflow_queue is not None, "workflow_queue is not initialized"
        workflow = await self.workflow_queue.get()
        try:
            for retry_attempt in range(1, self.retry_limit + 1):
                uid = f"{task_id}:{rollout_idx}"
                episode = await workflow.run_with_termination_handling(task=task, uid=uid, **kwargs)

                # Display rewards for all trajectories. Fallback to last step reward if trajectory reward is not set.
                reward_strs = []
                for traj in episode.trajectories:
                    reward = "N/A"
                    if traj.reward is not None:
                        reward = f"{traj.reward:.1f}"
                    elif len(traj.steps) > 0:
                        reward = f"{traj.steps[-1].reward:.1f}"
                    reward_strs.append(f"{traj.name}: {reward}")
                colorful_print(f"[{uid}] Rollout completed. Rewards: [{', '.join(reward_strs)}], Termination: {episode.termination_reason}", fg="green" if episode.is_correct else "yellow")

                if episode.termination_reason != TerminationReason.ERROR:
                    return task_id, rollout_idx, episode

                error_tb = episode.info.get("error", {}).get("traceback")
                if error_tb:
                    print(error_tb)

                if retry_attempt < self.retry_limit:
                    print(f"[{uid}] Rollout failed on attempt {retry_attempt}/{self.retry_limit}, retrying...")
                    continue

            if not self.raise_on_error:
                print(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts.")
            else:
                raise Exception(f"[{uid}] Rollout failed permanently after {self.retry_limit} attempts.")

            return task_id, rollout_idx, episode

        finally:
            await self.workflow_queue.put(workflow)

    async def execute_tasks(self, tasks: list[dict], task_ids: list[str] | None = None, **kwargs) -> list[Episode]:
        """Run asynchronous workflow execution with retry logic for multiple tasks.
        TODO(listar2000): refactor this function to get rid of the ugly `task_states` dictionary.
        Args:
            tasks: List of task dictionaries to process.
            task_ids: Optional list of task identifiers. If None, UUIDs are generated.
            **kwargs: Additional arguments passed to individual task processing.

        Returns:
            list[Episode]: List of completed episodes from all tasks.
        """
        if self.workflow_queue is None:
            await self.initialize_pool()

        if task_ids is None:
            task_ids = [str(uuid.uuid4()) for _ in tasks]

        task_states = defaultdict(lambda: {"idx": None, "task": None, "episodes": [], "completed": 0, "total_rollouts": 0, "is_complete": False})

        futures = []
        idx_counter = 0
        for task, task_id in zip(tasks, task_ids, strict=True):
            state = task_states[task_id]
            if state["idx"] is None:  # First time seeing this task_id
                state["idx"] = idx_counter
                state["task"] = task
                idx_counter += 1
            rollout_idx = state["total_rollouts"]
            futures.append(self.process_task_with_retry(task, task_id, rollout_idx, **kwargs))
            state["total_rollouts"] += 1

        with tqdm(total=len(tasks), desc="Generating trajectories") as pbar:
            for future in asyncio.as_completed(futures):
                task_id, rollout_idx, episode = await future

                state = task_states[task_id]
                state["episodes"].append(episode)
                state["completed"] += 1
                pbar.update(1)

        results = []
        sorted_tasks = sorted(task_states.keys(), key=lambda task_id: task_states[task_id]["idx"])
        for task_id in sorted_tasks:
            results.extend(task_states[task_id]["episodes"])

        # Log episodes if logger is provided
        if self.episode_logger is not None:
            try:
                logger.info(f"Logging {len(results)} episodes to step={self.current_step}, mode={self.current_mode}, epoch={self.current_epoch}")
                self.episode_logger.log_episodes_batch(results, self.current_step, self.current_mode, self.current_epoch)
            except Exception as e:
                logger.error(f"Failed to log episodes: {e}")
                import traceback

                traceback.print_exc()

        return results

    async def execute_tasks_verl(self, batch: DataProto, **kwargs) -> list[Episode]:
        """Execute tasks from a Verl DataProto batch and return results.

        Args:
            batch: Verl DataProto containing tasks and metadata.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            DataProto: Transformed results compatible with Verl training.
        """
        assert isinstance(self.rollout_engine, VerlEngine), "Rollout engine must be a VerlEngine to invoke execute_tasks_verl"
        await self.rollout_engine.wake_up()

        is_validation = batch.meta_info.get("validate", False)
        if is_validation:
            self.rollout_engine.validate = True
            self.current_mode = "val"
        else:
            self.current_mode = "train"
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        episodes = await self.execute_tasks(tasks, task_ids, **kwargs)  # list of Episodes
        # handle data sources in the input dataproto
        if "data_source" in batch.non_tensor_batch:
            data_sources = batch.non_tensor_batch["data_source"].tolist()
            for episode, data_source in zip(episodes, data_sources, strict=True):
                episode.info["data_source"] = data_source

        self.rollout_engine.validate = False

        await self.rollout_engine.sleep()

        self.current_mode = "train"
        return episodes

    def shutdown(self):
        """Shutdown the workflow engine and cleanup resources."""
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
            
    async def execute_tasks_skyrl(self, generator_input, **kwargs):
        """Execute tasks from a SkyRL GeneratorInput and return GeneratorOutput.

        Args:
            generator_input: SkyRL GeneratorInput with prompts, env_classes, env_extras, etc.
            **kwargs: Additional arguments passed to execute_tasks.

        Returns:
            GeneratorOutput: Transformed results compatible with SkyRL training.
        """
        # Wake up the rollout engine if it's a SkyRLEngine
        if hasattr(self.rollout_engine, 'wake_up'):
            await self.rollout_engine.wake_up()
        
        try:
            # Extract tasks from generator_input
            prompts = generator_input["prompts"]
            env_extras = generator_input.get("env_extras", [{}] * len(prompts))
            env_classes = generator_input.get("env_classes", [None] * len(prompts))
            trajectory_ids = generator_input.get("trajectory_ids", None)
            
            # Convert prompts and env_extras to task dictionaries
            tasks = []
            task_ids = []
            for i, (prompt, env_extra, env_class) in enumerate(zip(prompts, env_extras, env_classes)):
                # Create task dict from prompt and env_extra
                task = env_extra.copy() if env_extra else {}
                if "task" not in task:
                    task["task"] = {"prompt": prompt}
                if "prompt" not in task:
                    task["prompt"] = prompt
                tasks.append(task)
                
                # Use trajectory_id if available, otherwise generate one
                if trajectory_ids and i < len(trajectory_ids):
                    traj_id = trajectory_ids[i]
                    if hasattr(traj_id, 'instance_id'):
                        task_id = traj_id.instance_id
                    else:
                        task_id = str(traj_id)
                else:
                    task_id = str(uuid.uuid4())
                task_ids.append(task_id)
            
            # Set validation mode if needed
            batch_metadata = generator_input.get("batch_metadata")
            if batch_metadata and hasattr(batch_metadata, 'training_phase'):
                if batch_metadata.training_phase == "eval":
                    self.current_mode = "val"
                    if hasattr(self.rollout_engine, 'validate'):
                        self.rollout_engine.validate = True
                else:
                    self.current_mode = "train"
            
            # Execute tasks
            results = await self.execute_tasks(tasks, task_ids, **kwargs)  # list of Episodes
            
            # Reset validation mode
            if hasattr(self.rollout_engine, 'validate'):
                self.rollout_engine.validate = False
            self.current_mode = "train"
            
            return self.transform_results_for_skyrl(results, generator_input)
        finally:
            # Sleep the rollout engine if it's a SkyRLEngine
            if hasattr(self.rollout_engine, 'sleep'):
                await self.rollout_engine.sleep()

    def transform_results_for_skyrl(self, episodes: list[Episode], generator_input):
        """Transform episode results into SkyRL-compatible GeneratorOutput format.

        Args:
            episodes: List of completed episodes from workflow execution.
            generator_input: Original GeneratorInput for reference.

        Returns:
            GeneratorOutput: Formatted data ready for SkyRL training pipeline.
        """
        from skyrl_train.generators.base import GeneratorOutput
        from skyrl_train.generators.utils import get_response_ids_and_loss_mask_from_messages, get_rollout_metrics
        
        prompt_token_ids = []
        response_ids = []
        rewards = []
        loss_masks = []
        stop_reasons = []
        rollout_logprobs = []
        
        max_response_length = generator_input.get("sampling_params", {}).get("max_tokens", 4096)
        if hasattr(self, 'config') and self.config is not None:
            max_response_length = getattr(self.config.generator.sampling_params, 'max_generate_length', max_response_length)
        
        tokenizer = self.rollout_engine.tokenizer
        
        # Check if stepwise advantage is enabled
        stepwise_enabled = False
        if hasattr(self, 'config') and self.config is not None:
            try:
                stepwise_enabled = self.config.rllm.stepwise_advantage.enable
            except (AttributeError, KeyError):
                # rllm.stepwise_advantage not in config, default to False
                stepwise_enabled = False
        
        # Episode-facing loop
        for episode in episodes:
            if episode is None:
                continue
                
            if not episode.trajectories or len(episode.trajectories) == 0:
                continue

            # Process ALL trajectories in the episode
            for trajectory in episode.trajectories:
                if len(trajectory.steps) == 0:
                    continue

                if not stepwise_enabled:
                    # Trajectory-level mode: use cumulative chat history and trajectory reward
                    # Build chat history from all steps
                    chat_history = []
                    for step in trajectory.steps:
                        if step.chat_completions:
                            chat_history.extend(step.chat_completions)

                    if len(chat_history) == 0:
                        continue

                    # Extract prompt (first user message)
                    if chat_history[0]["role"] != "user":
                        logger.warning(f"First message is not user, skipping trajectory")
                        continue
                        
                    prompt_messages = [chat_history[0]]
                    prompt_ids = tokenizer.apply_chat_template(
                        prompt_messages,
                        add_generation_prompt=False,
                        tokenize=True,
                    )

                    # Process response messages
                    response_messages = chat_history[1:]
                    assistant_logprobs = None  # TODO: Extract if available from ModelOutput
                    resp_ids, loss_mask, resp_logprobs = get_response_ids_and_loss_mask_from_messages(
                        response_messages, tokenizer, assistant_logprobs
                    )

                    # Determine stop reason
                    stop_reason = "complete"
                    if len(resp_ids) > max_response_length:
                        stop_reason = "length"

                    # Truncate to maximum allowed length
                    resp_ids = resp_ids[:max_response_length]
                    loss_mask = loss_mask[:max_response_length]
                    resp_logprobs = resp_logprobs[:max_response_length] if resp_logprobs else None

                    # Append to batch - use trajectory-level reward
                    prompt_token_ids.append(prompt_ids)
                    response_ids.append(resp_ids)
                    rewards.append(trajectory.reward)  # Trajectory-level reward
                    loss_masks.append(loss_mask)
                    stop_reasons.append(stop_reason)
                    rollout_logprobs.append(resp_logprobs)

                else:
                    # Stepwise mode: process each step separately with step-level rewards
                    # Similar to Verl: each step's chat_completions contains full conversation up to that step
                    for step_idx, step in enumerate(trajectory.steps):
                        # Use step.chat_completions directly (contains full conversation up to this step)
                        if not step.chat_completions:
                            continue

                        chat_history = step.chat_completions

                        if len(chat_history) == 0:
                            continue

                        # Extract prompt (first user message)
                        if chat_history[0]["role"] != "user":
                            logger.warning(f"First message is not user, skipping step {step_idx}")
                            continue
                            
                        prompt_messages = [chat_history[0]]
                        prompt_ids = tokenizer.apply_chat_template(
                            prompt_messages,
                            add_generation_prompt=False,
                            tokenize=True,
                        )

                        # Process response messages (from this step onwards)
                        response_messages = chat_history[1:]
                        assistant_logprobs = None  # TODO: Extract if available from ModelOutput
                        resp_ids, loss_mask, resp_logprobs = get_response_ids_and_loss_mask_from_messages(
                            response_messages, tokenizer, assistant_logprobs
                        )

                        # Determine stop reason
                        stop_reason = "complete"
                        if len(resp_ids) > max_response_length:
                            stop_reason = "length"

                        # Truncate to maximum allowed length
                        resp_ids = resp_ids[:max_response_length]
                        loss_mask = loss_mask[:max_response_length]
                        resp_logprobs = resp_logprobs[:max_response_length] if resp_logprobs else None

                        # Append to batch - use step-level reward
                        prompt_token_ids.append(prompt_ids)
                        response_ids.append(resp_ids)
                        rewards.append(step.reward)  # Step-level reward
                        loss_masks.append(loss_mask)
                        stop_reasons.append(stop_reason)
                        rollout_logprobs.append(resp_logprobs)

        # Compute rollout metrics
        rollout_metrics = get_rollout_metrics(response_ids, rewards) if response_ids else None

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
