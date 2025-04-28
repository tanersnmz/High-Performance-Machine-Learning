import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import wandb
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import itertools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name: str
    use_flash_attention: bool = False
    use_quantization: bool = False
    batch_size: int = 8
    max_length: int = 128

class ModelBenchmark:
    def __init__(self, config: ModelConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Initialize model and tokenizer
        logger.info(f"Loading {config.model_name} model and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate configuration
        model_kwargs = {}
        if config.use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        ).to(self.device)
        
        if config.use_quantization:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        # Load dataset
        logger.info("Loading dataset...")
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
    def prepare_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of texts for model input"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)
        
    def benchmark_training(self, num_batches: int = 100) -> Dict[str, List[float]]:
        """Benchmark training performance"""
        logger.info("Starting training benchmark...")
        self.model.train()
        
        # Prepare data
        train_data = self.dataset["train"]["text"]
        train_data = [text for text in train_data if len(text.strip()) > 0]
        
        metrics = {
            "training_time": [],
            "memory_usage": [],
            "throughput": [],
            "loss": [],
            "perplexity": []
        }
        
        for i in tqdm(range(num_batches)):
            batch_texts = train_data[i*self.config.batch_size:(i+1)*self.config.batch_size]
            if not batch_texts:
                continue
                
            inputs = self.prepare_batch(batch_texts)
            
            start_time = time.time()
            
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
            
            loss.backward()
            
            batch_time = time.time() - start_time
            memory_usage = torch.cuda.memory_allocated() / 1024**2
            
            # Log batch metrics with configuration details
            wandb.log({
                "batch_training_time": batch_time,
                "batch_memory_usage": memory_usage,
                "batch_throughput": self.config.batch_size / batch_time,
                "batch_loss": loss.item(),
                "batch_perplexity": perplexity.item(),
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "use_flash_attention": self.config.use_flash_attention,
                "use_quantization": self.config.use_quantization
            })
            
            metrics["training_time"].append(batch_time)
            metrics["memory_usage"].append(memory_usage)
            metrics["throughput"].append(self.config.batch_size / batch_time)
            metrics["loss"].append(loss.item())
            metrics["perplexity"].append(perplexity.item())
            
            self.model.zero_grad()
            
        return metrics
    
    def benchmark_inference(self, num_batches: int = 100) -> Dict[str, List[float]]:
        """Benchmark inference performance"""
        logger.info("Starting inference benchmark...")
        self.model.eval()
        
        test_data = self.dataset["test"]["text"]
        test_data = [text for text in test_data if len(text.strip()) > 0]
        
        metrics = {
            "inference_time": [],
            "memory_usage": [],
            "throughput": [],
            "perplexity": []
        }
        
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                batch_texts = test_data[i*self.config.batch_size:(i+1)*self.config.batch_size]
                if not batch_texts:
                    continue
                    
                inputs = self.prepare_batch(batch_texts)
                
                start_time = time.time()
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                batch_time = time.time() - start_time
                memory_usage = torch.cuda.memory_allocated() / 1024**2
                
                # Log batch metrics with configuration details
                wandb.log({
                    "batch_inference_time": batch_time,
                    "batch_inference_memory": memory_usage,
                    "batch_inference_throughput": self.config.batch_size / batch_time,
                    "batch_inference_perplexity": perplexity.item(),
                    "batch_size": self.config.batch_size,
                    "max_length": self.config.max_length,
                    "use_flash_attention": self.config.use_flash_attention,
                    "use_quantization": self.config.use_quantization
                })
                
                metrics["inference_time"].append(batch_time)
                metrics["memory_usage"].append(memory_usage)
                metrics["throughput"].append(self.config.batch_size / batch_time)
                metrics["perplexity"].append(perplexity.item())
        
        return metrics
    
    def run_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks and log results"""
        logger.info("Starting comprehensive benchmark...")
        
        # Initialize wandb with detailed configuration
        wandb.init(
            project="model-benchmark",
            config={
                "model_name": self.config.model_name,
                "use_flash_attention": self.config.use_flash_attention,
                "use_quantization": self.config.use_quantization,
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length
            },
            name=f"{self.config.model_name}_bs{self.config.batch_size}_ml{self.config.max_length}_"
                f"flash{self.config.use_flash_attention}_quant{self.config.use_quantization}"
        )
        
        training_metrics = self.benchmark_training()
        inference_metrics = self.benchmark_inference()
        
        results = {
            "training": {
                "avg_time": np.mean(training_metrics["training_time"]),
                "avg_memory": np.mean(training_metrics["memory_usage"]),
                "avg_throughput": np.mean(training_metrics["throughput"]),
                "avg_loss": np.mean(training_metrics["loss"]),
                "avg_perplexity": np.mean(training_metrics["perplexity"]),
                "std_time": np.std(training_metrics["training_time"]),
                "std_memory": np.std(training_metrics["memory_usage"]),
                "std_throughput": np.std(training_metrics["throughput"]),
                "std_loss": np.std(training_metrics["loss"]),
                "std_perplexity": np.std(training_metrics["perplexity"])
            },
            "inference": {
                "avg_time": np.mean(inference_metrics["inference_time"]),
                "avg_memory": np.mean(inference_metrics["memory_usage"]),
                "avg_throughput": np.mean(inference_metrics["throughput"]),
                "avg_perplexity": np.mean(inference_metrics["perplexity"]),
                "std_time": np.std(inference_metrics["inference_time"]),
                "std_memory": np.std(inference_metrics["memory_usage"]),
                "std_throughput": np.std(inference_metrics["throughput"]),
                "std_perplexity": np.std(inference_metrics["perplexity"])
            }
        }
        
        # Log final results with configuration details
        wandb.log({
            **results,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "use_flash_attention": self.config.use_flash_attention,
            "use_quantization": self.config.use_quantization
        })
        
        logger.info("\nBenchmark Results:")
        logger.info("Training:")
        logger.info(f"Average time per batch: {results['training']['avg_time']:.4f} ± {results['training']['std_time']:.4f} seconds")
        logger.info(f"Average memory usage: {results['training']['avg_memory']:.2f} ± {results['training']['std_memory']:.2f} MB")
        logger.info(f"Average throughput: {results['training']['avg_throughput']:.2f} ± {results['training']['std_throughput']:.2f} samples/second")
        logger.info(f"Average loss: {results['training']['avg_loss']:.4f} ± {results['training']['std_loss']:.4f}")
        logger.info(f"Average perplexity: {results['training']['avg_perplexity']:.4f} ± {results['training']['std_perplexity']:.4f}")
        
        logger.info("\nInference:")
        logger.info(f"Average time per batch: {results['inference']['avg_time']:.4f} ± {results['inference']['std_time']:.4f} seconds")
        logger.info(f"Average memory usage: {results['inference']['avg_memory']:.2f} ± {results['inference']['std_memory']:.2f} MB")
        logger.info(f"Average throughput: {results['inference']['avg_throughput']:.2f} ± {results['inference']['std_throughput']:.2f} samples/second")
        logger.info(f"Average perplexity: {results['inference']['avg_perplexity']:.4f} ± {results['inference']['std_perplexity']:.4f}")
        
        wandb.finish()
        return results

def generate_configurations() -> List[ModelConfig]:
    """Generate all possible configurations for benchmarking"""
    model_names = ["gpt2"]
    batch_sizes = [4, 8, 16, 32]
    max_lengths = [64, 128, 256, 512]
    flash_options = [False, True]
    quant_options = [False, True]
    
    configs = []
    for model_name, batch_size, max_length, use_flash, use_quant in itertools.product(
        model_names, batch_sizes, max_lengths, flash_options, quant_options
    ):
        configs.append(ModelConfig(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            use_flash_attention=use_flash,
            use_quantization=use_quant
        ))
    
    return configs

def run_all_benchmarks():
    """Run benchmarks for all model configurations"""
    configs = generate_configurations()
    all_results = {}
    
    for config in configs:
        logger.info(f"\nRunning benchmark for config: {config}")
        benchmark = ModelBenchmark(config)
        results = benchmark.run_benchmarks()
        config_key = (
            f"{config.model_name}_bs{config.batch_size}_ml{config.max_length}_"
            f"flash{config.use_flash_attention}_quant{config.use_quantization}"
        )
        all_results[config_key] = results
    
    return all_results

if __name__ == "__main__":
    all_results = run_all_benchmarks() 