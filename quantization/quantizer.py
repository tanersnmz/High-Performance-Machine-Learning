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
        with torch.no_grad():
            for i, (inputs, _) in enumerate(calibration_data):
                if i >= num_batches:
                    break
                    
                # Forward pass to collect activation statistics
                outputs = model(inputs)
                
                # Update activation scales and zeros
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        # Get layer output
                        layer_output = outputs if name == 'output' else module.output
                        
                        # Update activation scale
                        max_abs = torch.max(torch.abs(layer_output))
                        self.activation_scales[name] = torch.max(
                            self.activation_scales[name],
                            max_abs / (2**(self.activation_bits-1)-1)
                        )
                        
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