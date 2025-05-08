import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import wandb
import json
import os
from typing import Dict, List
import logging
import time
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(
        self,
        gpt2_model_name: str = "gpt2",
        gpt2_medium_model_name: str = "gpt2-medium",
        distilled_model_path: str = "distilled_model/epoch_30",
        batch_size: int = 4,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load models and tokenizer
        logger.info("Loading models and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
        self.gpt2.eval()
        
        # Load GPT-2 Medium
        self.gpt2_medium = GPT2LMHeadModel.from_pretrained(gpt2_medium_model_name).to(device)
        self.gpt2_medium.eval()
        
        # Load distilled model
        self.distilled = GPT2LMHeadModel.from_pretrained(distilled_model_path).to(device)
        self.distilled.eval()
        
        # Load dataset
        logger.info("Loading dataset...")
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
    def prepare_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare batch for model input"""
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encodings.items()}
    
    def perform_warmup(self, model: GPT2LMHeadModel, dataloader: torch.utils.data.DataLoader, num_warmup_batches: int = 3):
        """Perform warmup iterations to avoid cold start timing issues"""
        logger.info(f"Performing {num_warmup_batches} warmup batches...")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_warmup_batches:
                    break
                    
                texts = [text for text in batch if text.strip()]
                if not texts:
                    continue
                    
                inputs = self.prepare_batch(texts)
                _ = model(**inputs, labels=inputs["input_ids"])
                
        # Clear CUDA cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Warmup completed")
    
    def evaluate_model(
        self,
        model: GPT2LMHeadModel,
        dataloader: torch.utils.data.DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        metrics = {
            "loss": [],
            "perplexity": [],
            "inference_time": [],
            "memory_usage": []
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                # Convert batch to list of strings
                texts = [text for text in batch if text.strip()]
                if not texts:  # Skip empty batches
                    continue
                    
                inputs = self.prepare_batch(texts)
                
                # Measure memory usage
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
                
                # Measure inference time
                start_time = time.time()
                
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                inference_time = time.time() - start_time
                peak_memory = torch.cuda.max_memory_allocated() - start_memory
                
                metrics["loss"].append(loss.item())
                metrics["perplexity"].append(perplexity.item())
                metrics["inference_time"].append(inference_time)
                metrics["memory_usage"].append(peak_memory / 1024**2)  # Convert to MB
        
        # Calculate statistics
        results = {
            "avg_loss": np.mean(metrics["loss"]),
            "std_loss": np.std(metrics["loss"]),
            "avg_perplexity": np.mean(metrics["perplexity"]),
            "std_perplexity": np.std(metrics["perplexity"]),
            "avg_inference_time": np.mean(metrics["inference_time"]),
            "std_inference_time": np.std(metrics["inference_time"]),
            "avg_memory_usage": np.mean(metrics["memory_usage"]),
            "std_memory_usage": np.std(metrics["memory_usage"])
        }
        
        return results
    
    def run_comparison(self, num_samples: int = 100) -> Dict[str, Dict[str, float]]:
        """Run comparison between all models"""
        # Prepare data
        test_data = self.dataset["test"]["text"]
        test_data = [text for text in test_data if len(text.strip()) > 0][:num_samples]
        
        # Create a custom dataset class
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts):
                self.texts = texts
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                return self.texts[idx]
        
        dataset = TextDataset(test_data)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Create separate dataloaders for warmup and evaluation
        warmup_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True  # Use different samples for warmup
        )
        
        # Initialize wandb
        wandb.init(
            project="gpt2-model-comparison",
            config={
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "num_samples": num_samples
            }
        )
        
        # Perform a general GPU warmup before any evaluations
        logger.info("Performing general GPU warmup...")
        # Run a few batches through any model to warm up the GPU
        self.perform_warmup(self.gpt2, warmup_dataloader)
        
        # Evaluate each model with specific warmup for each
        logger.info("Evaluating GPT-2...")
        self.perform_warmup(self.gpt2, warmup_dataloader)
        gpt2_results = self.evaluate_model(self.gpt2, dataloader, "GPT-2")
        
        logger.info("Evaluating GPT-2 Medium...")
        self.perform_warmup(self.gpt2_medium, warmup_dataloader)
        gpt2_medium_results = self.evaluate_model(self.gpt2_medium, dataloader, "GPT-2 Medium")
        
        logger.info("Evaluating Distilled Model...")
        self.perform_warmup(self.distilled, warmup_dataloader)
        distilled_results = self.evaluate_model(self.distilled, dataloader, "Distilled")
        
        # Log results to wandb in a comparative way
        # Create comparison tables and charts
        comparison_data = {
            "Model": ["GPT-2", "GPT-2 Medium", "Distilled"],
            "Loss": [
                gpt2_results["avg_loss"],
                gpt2_medium_results["avg_loss"],
                distilled_results["avg_loss"]
            ],
            "Perplexity": [
                gpt2_results["avg_perplexity"],
                gpt2_medium_results["avg_perplexity"],
                distilled_results["avg_perplexity"]
            ],
            "Inference Time (s)": [
                gpt2_results["avg_inference_time"],
                gpt2_medium_results["avg_inference_time"],
                distilled_results["avg_inference_time"]
            ],
            "Memory Usage (MB)": [
                gpt2_results["avg_memory_usage"],
                gpt2_medium_results["avg_memory_usage"],
                distilled_results["avg_memory_usage"]
            ]
        }
        
        # Log each metric individually for comparison charts
        for i, model_name in enumerate(comparison_data["Model"]):
            wandb.log({
                "loss_comparison": wandb.Table(
                    columns=["Model", "Loss"],
                    data=[[model, loss] for model, loss in zip(comparison_data["Model"], comparison_data["Loss"])]
                ),
                "perplexity_comparison": wandb.Table(
                    columns=["Model", "Perplexity"],
                    data=[[model, ppl] for model, ppl in zip(comparison_data["Model"], comparison_data["Perplexity"])]
                ),
                "inference_time_comparison": wandb.Table(
                    columns=["Model", "Inference Time (s)"],
                    data=[[model, time] for model, time in zip(comparison_data["Model"], comparison_data["Inference Time (s)"])]
                ),
                "memory_usage_comparison": wandb.Table(
                    columns=["Model", "Memory Usage (MB)"],
                    data=[[model, mem] for model, mem in zip(comparison_data["Model"], comparison_data["Memory Usage (MB)"])]
                ),
            })
        
        # Create a summary table with all metrics
        wandb.log({
            "model_comparison_summary": wandb.Table(
                columns=["Model", "Loss", "Perplexity", "Inference Time (s)", "Memory Usage (MB)"],
                data=[
                    [
                        comparison_data["Model"][i],
                        comparison_data["Loss"][i],
                        comparison_data["Perplexity"][i],
                        comparison_data["Inference Time (s)"][i],
                        comparison_data["Memory Usage (MB)"][i]
                    ] for i in range(len(comparison_data["Model"]))
                ]
            )
        })
        
        # Log bar charts for easy comparison
        for metric in ["Loss", "Perplexity", "Inference Time (s)", "Memory Usage (MB)"]:
            wandb.log({
                f"{metric}_bar_chart": wandb.plot.bar(
                    wandb.Table(
                        columns=["Model", metric],
                        data=[[model, value] for model, value in zip(comparison_data["Model"], comparison_data[metric])]
                    ),
                    "Model",
                    metric,
                    title=f"Comparison of {metric} Across Models"
                )
            })
        
        # Log relative performance to GPT-2 (as percentages)
        relative_performance = {
            "Model": ["GPT-2", "GPT-2 Medium", "Distilled"],
            "Relative Loss (%)": [
                100.0,  # GPT-2 baseline (100%)
                (gpt2_medium_results["avg_loss"] / gpt2_results["avg_loss"]) * 100,
                (distilled_results["avg_loss"] / gpt2_results["avg_loss"]) * 100
            ],
            "Relative Perplexity (%)": [
                100.0,  # GPT-2 baseline (100%)
                (gpt2_medium_results["avg_perplexity"] / gpt2_results["avg_perplexity"]) * 100,
                (distilled_results["avg_perplexity"] / gpt2_results["avg_perplexity"]) * 100
            ],
            "Relative Inference Time (%)": [
                100.0,  # GPT-2 baseline (100%)
                (gpt2_medium_results["avg_inference_time"] / gpt2_results["avg_inference_time"]) * 100,
                (distilled_results["avg_inference_time"] / gpt2_results["avg_inference_time"]) * 100
            ],
            "Relative Memory Usage (%)": [
                100.0,  # GPT-2 baseline (100%)
                (gpt2_medium_results["avg_memory_usage"] / gpt2_results["avg_memory_usage"]) * 100,
                (distilled_results["avg_memory_usage"] / gpt2_results["avg_memory_usage"]) * 100
            ]
        }
        
        wandb.log({
            "relative_performance": wandb.Table(
                columns=[
                    "Model", 
                    "Relative Loss (%)", 
                    "Relative Perplexity (%)", 
                    "Relative Inference Time (%)", 
                    "Relative Memory Usage (%)"
                ],
                data=[
                    [
                        relative_performance["Model"][i],
                        relative_performance["Relative Loss (%)"][i],
                        relative_performance["Relative Perplexity (%)"][i],
                        relative_performance["Relative Inference Time (%)"][i],
                        relative_performance["Relative Memory Usage (%)"][i]
                    ] for i in range(len(relative_performance["Model"]))
                ]
            )
        })
        
        # Print comparison table
        logger.info("\nModel Comparison Results:")
        logger.info("=" * 100)
        logger.info(f"{'Model':<15} {'Loss':<15} {'Perplexity':<15} {'Inference Time (s)':<20} {'Memory Usage (MB)':<20}")
        logger.info("-" * 100)
        
        for model_name, results in [
            ("GPT-2", gpt2_results),
            ("GPT-2 Medium", gpt2_medium_results),
            ("Distilled", distilled_results)
        ]:
            logger.info(
                f"{model_name:<15} "
                f"{results['avg_loss']:.4f} ± {results['std_loss']:.4f} "
                f"{results['avg_perplexity']:.4f} ± {results['std_perplexity']:.4f} "
                f"{results['avg_inference_time']:.4f} ± {results['std_inference_time']:.4f} "
                f"{results['avg_memory_usage']:.2f} ± {results['std_memory_usage']:.2f}"
            )
        
        wandb.finish()
        
        return {
            "gpt2": gpt2_results,
            "gpt2_medium": gpt2_medium_results,
            "distilled": distilled_results
        }

def main():
    comparator = ModelComparator()
    results = comparator.run_comparison(num_samples=100)
    
    # Save results to file
    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info("Results saved to model_comparison_results.json")

if __name__ == "__main__":
    main() 