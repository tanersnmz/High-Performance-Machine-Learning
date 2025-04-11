import torch
import torch.nn as nn
import copy

class Quantizer:
    def __init__(self, weight_bits, activation_bits):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_scales = {}
        self.weight_zeros = {}
        self.activation_scales = {}
        self.activation_zeros = {}

    def ptq(self, model, calibration_data, num_batches=100):
        """
        Perform Post-training Quantization (PTQ) on the model.
        
        Args:
            model: The model to quantize
            calibration_data: Data used for calibration
            num_batches: Number of batches to use for calibration
            
        Returns:
            Quantized model
        """
        # Set model to evaluation mode
        model.eval()
        
        # Initialize quantization parameters
        self._initialize_quantization_params(model)
        
        # Calibrate quantization parameters
        self._calibrate(model, calibration_data, num_batches)
        
        # Quantize model weights
        self._quantize_weights(model)
        
        # Create quantized model
        quantized_model = self._create_quantized_model(model)
        
        return quantized_model
        
    def _initialize_quantization_params(self, model):
        """Initialize quantization parameters for each layer."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Initialize weight quantization parameters
                self.weight_scales[name] = torch.max(torch.abs(module.weight)) / (2**(self.weight_bits-1)-1)
                self.weight_zeros[name] = torch.zeros_like(module.weight)
                
                # Initialize activation quantization parameters
                self.activation_scales[name] = torch.tensor(1.0)
                self.activation_zeros[name] = torch.tensor(0.0)
                
    def _calibrate(self, model, calibration_data, num_batches):
        """Calibrate quantization parameters using calibration data."""
        # Create a dictionary to store activation outputs for each module
        activation_outputs = {}
        
        # Define hooks to collect activation outputs
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activation_outputs:
                    activation_outputs[name] = []
                # Store output tensor for later analysis
                activation_outputs[name].append(output.detach().cpu())
            return hook
            
        # Register hooks to collect activation outputs
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
                
        with torch.no_grad():
            # Process calibration data to collect statistics
            for i, (inputs, _) in enumerate(calibration_data):
                if i >= num_batches:
                    break
                    
                # Forward pass to collect activation statistics
                model(inputs)
                
        # Calculate and update activation scales based on collected statistics
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in activation_outputs:
                # Concatenate all collected outputs if possible
                try:
                    layer_outputs = activation_outputs[name]
                    if layer_outputs:
                        # Compute robust statistics over all collected outputs
                        all_values = torch.cat([tensor.flatten() for tensor in layer_outputs])
                        
                        # Use percentile instead of max for more robust quantization
                        q_min = torch.quantile(all_values, 0.001)
                        q_max = torch.quantile(all_values, 0.999)
                        
                        # Compute scale based on the range
                        max_range = max(abs(q_min), abs(q_max))
                        self.activation_scales[name] = max_range / (2**(self.activation_bits-1) - 1)
                except:
                    # Fallback to original method if concatenation fails
                    max_abs = torch.tensor(0.0)
                    for tensor in activation_outputs[name]:
                        max_abs = torch.max(max_abs, torch.max(torch.abs(tensor)))
                    
                    self.activation_scales[name] = max_abs / (2**(self.activation_bits-1) - 1)
        
        # Remove all hooks
        for hook in hooks:
            hook.remove()
                        
    def _quantize_weights(self, model):
        """Quantize model weights."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Quantize weights
                quantized_weights = torch.round(module.weight / self.weight_scales[name])
                quantized_weights = torch.clamp(
                    quantized_weights,
                    -2**(self.weight_bits-1),
                    2**(self.weight_bits-1)-1
                )
                module.weight.data = quantized_weights * self.weight_scales[name]
                
    def _create_quantized_model(self, model):
        """Create a quantized version of the model."""
        # Create a copy of the model
        quantized_model = copy.deepcopy(model)
        
        # Replace layers with quantized versions
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Create quantized layer
                quantized_layer = QuantizedLayer(
                    module,
                    self.weight_scales[name],
                    self.weight_zeros[name],
                    self.activation_scales[name],
                    self.activation_zeros[name],
                    self.weight_bits,
                    self.activation_bits
                )
                
                # Replace original layer with quantized version
                parent_name = '.'.join(name.split('.')[:-1])
                if parent_name:
                    parent = quantized_model.get_submodule(parent_name)
                    setattr(parent, name.split('.')[-1], quantized_layer)
                else:
                    setattr(quantized_model, name, quantized_layer)
                    
        return quantized_model 
        
# Add QuantizedLayer class definition
class QuantizedLayer(nn.Module):
    def __init__(
        self,
        orig_layer,
        weight_scale,
        weight_zero,
        activation_scale,
        activation_zero,
        weight_bits,
        activation_bits
    ):
        super().__init__()
        self.orig_layer = orig_layer
        self.weight_scale = weight_scale
        self.weight_zero = weight_zero
        self.activation_scale = activation_scale
        self.activation_zero = activation_zero
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Apply quantization to weights (store quantized weights)
        self.quantized_weight = None
        self._quantize_weights()
        
    def _quantize_weights(self):
        """Pre-quantize weights for better performance"""
        # This is just a soft-quantization - we're not actually using integers
        # but we're restricting the values to what would be representable in integers
        w = self.orig_layer.weight
        w_scaled = w / self.weight_scale
        w_quant = torch.round(w_scaled)
        w_quant = torch.clamp(
            w_quant,
            -2**(self.weight_bits-1),
            2**(self.weight_bits-1)-1
        )
        self.quantized_weight = w_quant * self.weight_scale
        
    def forward(self, x):
        # Use quantized weights instead of originals
        original_weight = self.orig_layer.weight.data.clone()
        self.orig_layer.weight.data = self.quantized_weight
        
        # Skip activation quantization for higher bits (>8) to reduce error
        if self.activation_bits <= 8:
            # Quantize input activations
            x_scaled = x / self.activation_scale
            x_quant = torch.round(x_scaled)
            x_quant = torch.clamp(
                x_quant,
                -2**(self.activation_bits-1),
                2**(self.activation_bits-1)-1
            )
            x_dequant = x_quant * self.activation_scale
            
            # Use original layer for computation with quantized inputs
            output = self.orig_layer(x_dequant)
        else:
            # For higher precision, skip activation quantization
            output = self.orig_layer(x)
        
        # Restore original weights
        self.orig_layer.weight.data = original_weight
        
        return output 