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
from quantizer import Quantizer
from itertools import islice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantizationEvaluator:
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        dataset_name: str = "librispeech_asr",
        split: str = "test.clean",
        batch_size: int = 4,
        use_mini_dataset: bool = True
    ):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor and model
        logger.info("Loading processor and model...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        self.model = self.model.to(self.device)

        # Load dataset
        if use_mini_dataset:
            logger.info("Loading Mini LibriSpeech dataset...")
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
            self.dataset = list(islice(self.dataset, 50))
        else:
            logger.info(f"Loading dataset {dataset_name}...")
            self.dataset = load_dataset(dataset_name, split=split)

        logger.info(f"Dataset size: {len(self.dataset)} samples")

    def custom_collate_fn(self, batch):
        return batch

    def preprocess_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_arrays = [item["audio"]["array"] for item in batch]
        input_values = self.processor(
            audio_arrays,
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            pad_to_multiple_of=16000
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
    ) -> Dict:
        model = model.to(self.device)
        predictions = []
        references = []
        inference_times = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_values = self.preprocess_batch(batch).to(self.device)

                start_time = time.time()
                logits = model(input_values).logits
                inference_time = time.time() - start_time
                inference_times.append(inference_time)

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)

                predictions.extend(transcription)
                references.extend([item["text"] for item in batch])

        wer = self.calculate_wer(predictions, references)
        avg_inference_time = np.mean(inference_times)

        return {
            "wer": wer,
            "avg_inference_time": avg_inference_time,
            "total_samples": len(predictions)
        }

    def run_comparison(self, num_samples: int = 100) -> Dict:
        data = list(islice(self.dataset, num_samples))
        dataloader = DataLoader(
            data,
            batch_size=self.batch_size,
            collate_fn=self.custom_collate_fn
        )

        logger.info("Evaluating original model...")
        original_results = self.evaluate_model(self.model, dataloader)

        # Quantize model
        logger.info("Quantizing model...")
        quantizer = Quantizer(weight_bits=8, activation_bits=8)
        calibration_data = [(self.preprocess_batch([item]).to(self.device), None) for item in data[:10]]
        quantized_model = quantizer.ptq(self.model, calibration_data, num_batches=10)

        logger.info("Evaluating quantized model...")
        quantized_results = self.evaluate_model(quantized_model, dataloader)

        wer_diff = quantized_results["wer"] - original_results["wer"]
        time_diff = quantized_results["avg_inference_time"] - original_results["avg_inference_time"]

        return {
            "original": original_results,
            "quantized": quantized_results,
            "wer_difference": wer_diff,
            "time_difference": time_diff
        }

def main():
    evaluator = QuantizationEvaluator(use_mini_dataset=True)
    
    logger.info("Running comparison...")
    comparison_results = evaluator.run_comparison(num_samples=20)
    
    logger.info("\nResults:")
    logger.info(f"Original Model:")
    logger.info(f"  WER: {comparison_results['original']['wer']:.4f}")
    logger.info(f"  Avg Inference Time: {comparison_results['original']['avg_inference_time']:.4f}s")
    
    logger.info(f"\nQuantized Model:")
    logger.info(f"  WER: {comparison_results['quantized']['wer']:.4f}")
    logger.info(f"  Avg Inference Time: {comparison_results['quantized']['avg_inference_time']:.4f}s")
    
    logger.info(f"\nDifferences:")
    logger.info(f"  WER Difference: {comparison_results['wer_difference']:.4f}")
    logger.info(f"  Time Difference: {comparison_results['time_difference']:.4f}s")
    logger.info(f"  Speedup: {comparison_results['original']['avg_inference_time'] / comparison_results['quantized']['avg_inference_time']:.2f}x")

if __name__ == "__main__":
    main() 