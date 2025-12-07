"""
SkyRL integration for rLLM:
1. SkyRLInferenceWrapper: Adapts SkyRL's InferenceEngineClient to rLLM's RolloutEngine interface
2. SkyRLEngine: Main generator that uses rLLM workflows and returns SkyRL's GeneratorOutput
"""
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add skyrl-train to Python path
skyrl_train_path = Path(__file__).parent.parent.parent.parent / "skyrl" / "skyrl-train"
if skyrl_train_path.exists():
    sys.path.insert(0, str(skyrl_train_path))

# Now you can import from skyrl_train
from skyrl_train.generators import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
)

from rllm.engine.rollout import RolloutEngine, ModelOutput

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


class SkyRLInferenceWrapper(RolloutEngine):
    """Adapts SkyRL's InferenceEngineClient to rLLM's RolloutEngine interface.

    This adapter allows rLLM workflows to use SkyRL's inference backends
    (vLLM, SGLang, etc.) transparently during trajectory generation.
    """

    def __init__(
        self,
        inference_engine_client: "InferenceEngineClient",
        tokenizer,
        max_prompt_length: int = 4096,
        max_response_length: int = 4096,
    ):
        """Initialize the wrapper.

        Args:
            inference_engine_client: SkyRL's InferenceEngineClient
            tokenizer: Tokenizer instance
            max_prompt_length: Maximum prompt length in tokens
            max_response_length: Maximum response length in tokens
        """
        super().__init__()
        self.inference_engine = inference_engine_client
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        """Get model response using SkyRL's inference engine.

        Args:
            messages: List of chat messages in OpenAI format
            **kwargs: Additional parameters including:
                - sampling_params: Dict with temperature, top_p, max_tokens, etc.
                - validate: Whether this is validation (for greedy decoding)
                - enforce_max_prompt_length: Whether to enforce max prompt length

        Returns:
            ModelOutput: Structured model response with token IDs and metadata
        """
        from skyrl_train.inference_engines.base import InferenceEngineInput
        from rllm.workflows import TerminationEvent, TerminationReason

        # Extract parameters
        sampling_params = kwargs.get("sampling_params", {})
        validate = kwargs.get("validate", False)
        enforce_max_prompt_length = kwargs.get("enforce_max_prompt_length", True)

        # Convert messages to token IDs
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text)
        prompt_length = len(prompt_ids)

        # Enforce prompt length limit
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        # Prepare SkyRL inference input
        inference_input: InferenceEngineInput = {
            "prompts": [messages],
            "prompt_token_ids": None,
            "sampling_params": {
                "max_tokens": sampling_params.get("max_tokens", self.max_response_length),
                "temperature": 0.0 if validate else sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 1.0),
                **{k: v for k, v in sampling_params.items() if k not in ["max_tokens", "temperature", "top_p"]}
            },
            "session_ids": None,
        }

        # Call SkyRL's inference engine
        output = await self.inference_engine.generate(inference_input)

        # Extract response
        response_text = output["responses"][0]
        response_ids = output["response_ids"][0]
        stop_reason = output["stop_reasons"][0]
        logprobs = output.get("response_logprobs", [None])[0] if output.get("response_logprobs") else None

        # Parse response for structured content
        # TODO: Implement parsing logic for reasoning and tool calls if needed
        content = response_text
        reasoning = ""
        tool_calls = []

        # Determine finish reason
        finish_reason = stop_reason
        if len(response_ids) >= sampling_params.get("max_tokens", self.max_response_length):
            finish_reason = "length"

        return ModelOutput(
            text=response_text,
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            prompt_ids=prompt_ids,
            completion_ids=response_ids,
            prompt_length=prompt_length,
            completion_length=len(response_ids),
            finish_reason=finish_reason,
        )

    async def wake_up(self):
        """Wake up the inference engine (for colocated training)."""
        await self.inference_engine.wake_up()

    async def sleep(self):
        """Put the inference engine to sleep (for colocated training)."""
        await self.inference_engine.sleep()


class SkyRLEngine(GeneratorInterface):
    """Main SkyRL generator that uses rLLM workflows.

    This generator:
    1. Receives GeneratorInput from SkyRL trainer
    2. Runs rLLM workflows to generate trajectories
    3. Converts results to GeneratorOutput for SkyRL training
    """

    def __init__(
        self,
        config,
        tokenizer,
        inference_engine_client: "InferenceEngineClient",
        workflow_class=None,
        workflow_args=None,
    ):
        """Initialize the SkyRL generator.

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

        # Create the inference wrapper for workflows
        self.rollout_engine = SkyRLInferenceWrapper(
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
        import asyncio
        from skyrl_train.generators.utils import get_rollout_metrics

        # Extract prompts and environment info
        prompts = input_batch["prompts"]
        env_extras = input_batch.get("env_extras", [{}] * len(prompts))

        # Run all workflow loops in parallel
        tasks = []
        for prompt, env_extra in zip(prompts, env_extras):
            tasks.append(self._workflow_agent_loop(prompt, env_extra))

        all_outputs = await asyncio.gather(*tasks)

        # Collect results
        responses = [output.response_ids for output in all_outputs]
        rewards = [output.reward for output in all_outputs]
        rollout_metrics = get_rollout_metrics(responses, rewards)

        # Format as GeneratorOutput
        generator_output: GeneratorOutput = {
            "prompt_token_ids": [output.prompt_ids for output in all_outputs],
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": [output.loss_mask for output in all_outputs],
            "stop_reasons": [output.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": [output.rollout_logprobs for output in all_outputs],
        }

        return generator_output

    async def _workflow_agent_loop(self, prompt, env_extra):
        """Run a single rLLM workflow (equivalent to terminal_bench_agent_loop).

        Args:
            prompt: ConversationType (list of messages)
            env_extra: Environment configuration/task data

        Returns:
            WorkflowAgentOutput: Results from workflow execution
        """
        from dataclasses import dataclass
        from typing import List, Optional
        from skyrl_train.generators.utils import get_response_ids_and_loss_mask_from_messages
        from concurrent.futures import ThreadPoolExecutor

        @dataclass
        class WorkflowAgentOutput:
            response_ids: List[int]
            reward: float
            stop_reason: str
            loss_mask: List[int]
            prompt_ids: List[int]
            rollout_logprobs: Optional[List[float]]

        # Extract task from env_extra
        task = env_extra.get("task", {"prompt": prompt})

        # Create workflow instance
        executor = ThreadPoolExecutor(max_workers=1)
        workflow = self.workflow_class(
            rollout_engine=self.rollout_engine,  # Uses SkyRLInferenceWrapper
            executor=executor,
            **self.workflow_args
        )

        # Run workflow to get episode
        episode = await workflow.run(task=task, uid=f"workflow_{id(prompt)}")

        # Extract chat history from episode
        # Assume we want the primary trajectory (first one)
        if not episode.trajectories or len(episode.trajectories) == 0:
            raise ValueError("Workflow returned no trajectories")

        trajectory = episode.trajectories[0]
        if len(trajectory.steps) == 0:
            raise ValueError("Trajectory has no steps")

        # Build chat history from steps
        chat_history = []
        for step in trajectory.steps:
            if step.chat_completions:
                chat_history.extend(step.chat_completions)

        # Extract prompt (first user message)
        assert chat_history[0]["role"] == "user", "First message should be user"
        prompt_messages = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=False,
            tokenize=True,
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages
        response_messages = chat_history[1:]
        assistant_logprobs = None  # TODO: Extract if available from ModelOutput
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            response_messages, self.tokenizer, assistant_logprobs
        )

        # Get reward from trajectory
        reward = trajectory.reward

        # Determine stop reason
        max_response_tokens = self.max_response_length
        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        rollout_logprobs = rollout_logprobs[:max_response_tokens] if rollout_logprobs else None

        return WorkflowAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=rollout_logprobs,
        )
