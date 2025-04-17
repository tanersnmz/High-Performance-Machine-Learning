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
from aihwkit.exceptions import TileModuleError
from wav2vec2_linear_layer_analog import AnalogWav2Vec2ForCTC
from itertools import islice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA availability
if not torch.cuda.is_available():
    logger.warning("CUDA is not available. Falling back to CPU.")
    DEVICE = "cpu"
else:
    DEVICE = "cuda"
    # Initialize CUDA
    torch.cuda.init()
    logger.info(f"CUDA initialized. Using device: {torch.cuda.get_device_name(0)}")

class PerformanceEvaluator:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        dataset_name: str = "librispeech_asr",
        split: str = "test.clean",
        batch_size: int = 4,
        debug: bool = False,
        use_mini_dataset: bool = True
    ):
        self.debug = debug
        self.batch_size = batch_size
        self.device = DEVICE

        # Load processor and model
        logger.info("Loading processor and model...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        # Load digital model
        self.digital_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.digital_model.eval()
        self.digital_model = self.digital_model.to(self.device)

        # Load analog model
        rpu_config = InferenceRPUConfig()
        self.analog_model = AnalogWav2Vec2ForCTC(
            self.digital_model.config,
            rpu_config=rpu_config,
            debug=debug
        )
        self.analog_model.transfer_digital_weights(self.digital_model)
        self.analog_model.eval()
        self.analog_model = self.analog_model.to(self.device)

        # Load dataset
        if use_mini_dataset:
            logger.info("Loading Mini LibriSpeech dataset...")
            self.dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
            self.dataset = list(islice(self.dataset, 50))
        else:
            logger.info(f"Loading dataset {dataset_name}...")
            self.dataset = load_dataset(dataset_name, split=split)

        logger.info(f"Dataset size: {len(self.dataset)} samples")

    def custom_collate_fn(self, batch):
        # Just return the batch as is, we'll handle the processing in preprocess_batch
        return batch

    def preprocess_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extract audio arrays from the batch
        audio_arrays = [item["audio"]["array"] for item in batch]
        
        # Process the batch with padding
        input_values = self.processor(
            audio_arrays,
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            pad_to_multiple_of=16000  # Pad to nearest second
        ).input_values

        return input_values

    def calculate_wer(self, predictions: List[str], references: List[str]) -> float:
        total_words = sum(len(ref.split()) for ref in references)
        errors = 0
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
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
        model = model.to(device)
        predictions = []
        references = []
        inference_times = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_values = self.preprocess_batch(batch).to(device)

                start_time = time.time()
                logits = model(input_values).logits
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)

                predictions.extend(transcription)
                # Extract text from each item in the batch
                references.extend([item["text"] for item in batch])

        wer = self.calculate_wer(predictions, references)
        avg_inference_time = np.mean(inference_times)

        return {
            "wer": wer,
            "avg_inference_time": avg_inference_time,
            "total_samples": len(predictions)
        }

    def run_comparison(self, num_samples: int = 100) -> Dict:
        from itertools import islice
        data = list(islice(self.dataset, num_samples))
        dataloader = DataLoader(
            data,
            batch_size=self.batch_size,
            collate_fn=self.custom_collate_fn
        )

        logger.info("Evaluating digital model...")
        digital_results = self.evaluate_model(self.digital_model, dataloader)

        logger.info("Evaluating analog model...")
        analog_results = self.evaluate_model(self.analog_model, dataloader)

        wer_diff = analog_results["wer"] - digital_results["wer"]
        time_diff = analog_results["avg_inference_time"] - digital_results["avg_inference_time"]

        return {
            "digital": digital_results,
            "analog": analog_results,
            "wer_difference": wer_diff,
            "time_difference": time_diff
        }

    # def analyze_noise_sensitivity(
    #     self,
    #     noise_levels: List[float],
    #     num_samples: int = 50
    # ) -> Dict:
    #     results = {}

    #     for noise_level in noise_levels:
    #         logger.info(f"Testing noise level: {noise_level}")

    #         rpu_config = InferenceRPUConfig()
    #         rpu_config.noise_model = GokmenVlasovPreset()
    #         rpu_config.noise_model.prog_noise_scale = noise_level

    #         noisy_model = AnalogWav2Vec2ForCTC(
    #             self.digital_model.config,
    #             rpu_config=rpu_config,
    #             debug=self.debug
    #         )
    #         noisy_model.transfer_digital_weights(self.digital_model)
    #         noisy_model.eval()

    #         try:
    #             noisy_model = noisy_model.to(self.device)
    #         except TileModuleError as e:
    #             logger.warning(f"Analog model with noise could not use CUDA. Falling back to CPU. Reason: {e}")
    #             self.device = "cpu"
    #             noisy_model = noisy_model.to(self.device)

    #         data = list(islice(self.dataset, num_samples))
    #         dataloader = DataLoader(
    #             data,
    #             batch_size=self.batch_size
    #         )
    #         results[noise_level] = self.evaluate_model(noisy_model, dataloader)

    #     return results

def main():
    evaluator = PerformanceEvaluator(debug=False, use_mini_dataset=True)

    logger.info("Running basic comparison...")
    comparison_results = evaluator.run_comparison(num_samples=20)
    logger.info(f"Comparison results: {comparison_results}")

    # logger.info("Running noise sensitivity analysis...")
    # noise_levels = [0.0, 0.1, 0.2, 0.3]
    # noise_results = evaluator.analyze_noise_sensitivity(noise_levels, num_samples=10)
    # logger.info(f"Noise sensitivity results: {noise_results}")

if __name__ == "__main__":
    main()
