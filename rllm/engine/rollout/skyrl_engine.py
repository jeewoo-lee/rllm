"""
TODO: Import skyrl generator from skyrl_train sdk and implement the custom generator logic to be fed into agent_workflow_engine.py
"""
import sys
from pathlib import Path

# Add skyrl-train to Python path
skyrl_train_path = Path(__file__).parent.parent.parent.parent / "skyrl" / "skyrl-train"
if skyrl_train_path.exists():
    sys.path.insert(0, str(skyrl_train_path))

# Now you can import from skyrl_train
from skyrl_train.generators import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
    SkyRLGymGenerator,
)


class SkyRLEngine(GeneratorInterface):
    """Custom generator that wraps SkyRL for rLLM agent workflow."""

    def __init__(self):
        # TODO: Initialize your custom logic
        pass

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch."""
        # TODO: Implement custom generator logic
        raise NotImplementedError("Implement your custom SkyRL generator logic here")
