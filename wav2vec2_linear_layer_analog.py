import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets import IdealizedPreset

# Debug mode flag
DEBUG = True

def debug_print(*args, **kwargs):
    """Print debug information only if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

# Define RPU configuration for analog inference
# This configuration determines how the analog hardware behaves
rpu_config = InferenceRPUConfig()
debug_print("Initialized RPU configuration for analog inference")

class AnalogWav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config, rpu_config=None, debug=False):
        super().__init__(config)
        global DEBUG
        DEBUG = debug
        debug_print("Initializing AnalogWav2Vec2ForCTC model...")
        
        # Count the number of linear layers to be replaced
        linear_count = sum(1 for _ in self.named_modules() if isinstance(_[1], nn.Linear))
        debug_print(f"Found {linear_count} linear layers to convert to analog")
        
        # Replace all linear layers with analog linear layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                debug_print(f"Converting linear layer '{name}' to analog")
                # Create analog layer with same dimensions and bias configuration
                analog_layer = AnalogLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    rpu_config=rpu_config
                )
                # Navigate to the parent module to replace the layer
                parent = self
                for n in name.split('.')[:-1]:
                    parent = getattr(parent, n)
                setattr(parent, name.split('.')[-1], analog_layer)
        
        debug_print("All linear layers converted to analog successfully")

    def transfer_digital_weights(self, digital_model):
        """
        Transfer weights from digital model to analog model using from_digital method
        This ensures proper weight transfer considering analog hardware characteristics
        """
        debug_print("Starting weight transfer from digital to analog model...")
        
        # Count the number of analog layers to transfer weights to
        analog_count = sum(1 for _ in self.named_modules() if isinstance(_[1], AnalogLinear))
        debug_print(f"Found {analog_count} analog layers to transfer weights to")
        
        for name, module in self.named_modules():
            if isinstance(module, AnalogLinear):
                debug_print(f"Transferring weights for analog layer '{name}'")
                # Get the corresponding digital module
                digital_module = digital_model
                for n in name.split('.'):
                    digital_module = getattr(digital_module, n)
                # Transfer weights using from_digital method
                module.from_digital(digital_module, rpu_config=rpu_config)
        
        debug_print("Weight transfer completed successfully")

# Example: Load pretrained model and create analog inference model
debug_print("\nLoading pretrained Wav2Vec2 model...")
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

# Load pretrained model configuration and weights
debug_print("Loading model configuration...")
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
debug_print("Loading pretrained weights...")
digital_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

# Initialize our analog model
debug_print("\nCreating analog model...")
analog_model = AnalogWav2Vec2ForCTC(config, rpu_config=rpu_config, debug=True)  # Enable debug mode here

# Transfer weights from digital to analog model
debug_print("\nTransferring weights from digital to analog model...")
analog_model.transfer_digital_weights(digital_model)

# Move model to CUDA
debug_print("\nMoving model to CUDA...")
analog_model = analog_model.to('cuda')

# Switch to evaluation mode for inference
debug_print("\nSwitching to evaluation mode...")
analog_model.eval()

# Example input for testing
debug_print("\nPreparing test input...")
dummy_input = torch.randn(1, 16000)  # 1 second of 16kHz audio
dummy_input = dummy_input.to('cuda')  # Move input to CUDA
debug_print(f"Input shape: {dummy_input.shape}")

# Run inference
debug_print("\nRunning inference...")
with torch.no_grad():
    outputs = analog_model(dummy_input)
    logits = outputs.logits
    debug_print(f"Output logits shape: {logits.shape}")
    debug_print(f"First few logits: {logits[0, 0, :5]}")  # Print first 5 logits for verification
