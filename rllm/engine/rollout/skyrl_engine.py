"""
SkyRL integration for rLLM:
SkyRLEngine: Adapts SkyRL's InferenceEngineClient to rLLM's RolloutEngine interface.

This adapter allows rLLM workflows to use SkyRL's inference backends
(vLLM, SGLang, etc.) transparently during trajectory generation.
"""
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add skyrl-train to Python path
skyrl_train_path = Path(__file__).parent.parent.parent.parent / "skyrl" / "skyrl-train"
if skyrl_train_path.exists():
    sys.path.insert(0, str(skyrl_train_path))

from rllm.engine.rollout import RolloutEngine, ModelOutput

if TYPE_CHECKING:
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


class SkyRLEngine(RolloutEngine):
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
