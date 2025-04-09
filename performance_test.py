import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List, Tuple
import logging
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets import IdealizedPreset, GokmenVlasovPreset
from wav2vec2_linear_layer_analog import AnalogWav2Vec2ForCTC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        dataset_name: str = "librispeech_asr",
        split: str = "test.clean",
        batch_size: int = 4,
        debug: bool = False,
        use_mini_dataset: bool = True  # Add flag for mini dataset
    ):
        """
        Initialize the performance evaluator
        
        Args:
            model_name: Name of the pretrained model
            dataset_name: Name of the dataset to use
            split: Dataset split to use
            batch_size: Batch size for evaluation
            debug: Whether to enable debug mode
            use_mini_dataset: Whether to use mini dataset for testing
        """
        self.debug = debug
        self.batch_size = batch_size
        
        # Load processor and model
        logger.info("Loading processor and model...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        # Load digital model
        self.digital_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.digital_model.eval()
        
        # Load analog model
        rpu_config = InferenceRPUConfig()
        self.analog_model = AnalogWav2Vec2ForCTC(
            self.digital_model.config,
            rpu_config=rpu_config,
            debug=debug
        )
        self.analog_model.transfer_digital_weights(self.digital_model)
        self.analog_model.eval()
        
        # Load dataset
        if use_mini_dataset:
            logger.info("Loading Mini LibriSpeech dataset...")
            self.dataset = load_dataset("librispeech_asr", "clean", split="validation")
            # Take a small subset for testing
            self.dataset = self.dataset.select(range(50))  # Use only 50 samples
        else:
            logger.info(f"Loading dataset {dataset_name}...")
            self.dataset = load_dataset(dataset_name, split=split)
        
        logger.info(f"Dataset size: {len(self.dataset)} samples")
        
    def preprocess_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a batch of audio data
        
        Args:
            batch: Batch of audio data
            
        Returns:
            Tuple of (input_values, attention_mask)
        """
        # Process audio
        input_values = self.processor(
            batch["audio"]["array"],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True
        ).input_values
        
        return input_values
    
    def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate Word Error Rate
        
        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            
        Returns:
            Word Error Rate
        """
        # Simple WER implementation - can be replaced with more sophisticated version
        total_words = sum(len(ref.split()) for ref in references)
        errors = 0
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            # Simple word-level comparison
            errors += abs(len(pred_words) - len(ref_words))
            for p, r in zip(pred_words, ref_words):
                if p != r:
                    errors += 1
        return errors / total_words if total_words > 0 else 0.0
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cuda"
    ) -> Dict:
        """
        Evaluate a model on the dataset
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for the dataset
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model = model.to(device)
        predictions = []
        references = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Preprocess batch
                input_values = self.preprocess_batch(batch).to(device)
                
                # Measure inference time
                start_time = time.time()
                logits = model(input_values).logits
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)
                
                predictions.extend(transcription)
                references.extend(batch["text"])
        
        # Calculate metrics
        wer = self.calculate_wer(predictions, references)
        avg_inference_time = np.mean(inference_times)
        
        return {
            "wer": wer,
            "avg_inference_time": avg_inference_time,
            "total_samples": len(predictions)
        }
    
    def run_comparison(self, num_samples: int = 100) -> Dict:
        """
        Run comparison between digital and analog models
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary containing comparison results
        """
        # Create dataloader
        dataloader = DataLoader(
            self.dataset.select(range(num_samples)),
            batch_size=self.batch_size
        )
        
        # Evaluate digital model
        logger.info("Evaluating digital model...")
        digital_results = self.evaluate_model(self.digital_model, dataloader)
        
        # Evaluate analog model
        logger.info("Evaluating analog model...")
        analog_results = self.evaluate_model(self.analog_model, dataloader)
        
        # Calculate relative differences
        wer_diff = analog_results["wer"] - digital_results["wer"]
        time_diff = analog_results["avg_inference_time"] - digital_results["avg_inference_time"]
        
        return {
            "digital": digital_results,
            "analog": analog_results,
            "wer_difference": wer_diff,
            "time_difference": time_diff
        }
    
    def analyze_noise_sensitivity(
        self,
        noise_levels: List[float],
        num_samples: int = 50
    ) -> Dict:
        """
        Analyze model sensitivity to different noise levels
        
        Args:
            noise_levels: List of noise levels to test
            num_samples: Number of samples to evaluate per noise level
            
        Returns:
            Dictionary containing noise sensitivity analysis results
        """
        results = {}
        
        for noise_level in noise_levels:
            logger.info(f"Testing noise level: {noise_level}")
            
            # Configure RPU with noise
            rpu_config = InferenceRPUConfig()
            rpu_config.noise_model = GokmenVlasovPreset()
            rpu_config.noise_model.prog_noise_scale = noise_level
            
            # Create analog model with noise
            noisy_model = AnalogWav2Vec2ForCTC(
                self.digital_model.config,
                rpu_config=rpu_config,
                debug=self.debug
            )
            noisy_model.transfer_digital_weights(self.digital_model)
            noisy_model.eval()
            
            # Evaluate
            dataloader = DataLoader(
                self.dataset.select(range(num_samples)),
                batch_size=self.batch_size
            )
            results[noise_level] = self.evaluate_model(noisy_model, dataloader)
        
        return results

def main():
    # Initialize evaluator with mini dataset
    evaluator = PerformanceEvaluator(debug=True, use_mini_dataset=True)
    
    # Run basic comparison with smaller number of samples
    logger.info("Running basic comparison...")
    comparison_results = evaluator.run_comparison(num_samples=20)  # Reduced sample size
    logger.info(f"Comparison results: {comparison_results}")
    
    # Run noise sensitivity analysis with fewer samples
    logger.info("Running noise sensitivity analysis...")
    noise_levels = [0.0, 0.1, 0.2, 0.3]  # Reduced number of noise levels
    noise_results = evaluator.analyze_noise_sensitivity(noise_levels, num_samples=10)
    logger.info(f"Noise sensitivity results: {noise_results}")

if __name__ == "__main__":
    main() 