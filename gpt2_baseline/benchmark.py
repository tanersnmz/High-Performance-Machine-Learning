import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import time
import wandb
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2Benchmark:
    def __init__(self, model_name="gpt2", batch_size=8, max_length=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize model and tokenizer
        logger.info(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        # Load dataset
        logger.info("Loading dataset...")
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
    def benchmark_training(self, num_batches=100):
        """Benchmark training performance"""
        logger.info("Starting training benchmark...")
        self.model.train()
        
        # Prepare data
        train_data = self.dataset["train"]["text"]
        train_data = [text for text in train_data if len(text) > 0]
        
        metrics = {
            "training_time": [],
            "memory_usage": [],
            "throughput": []
        }
        
        for i in tqdm(range(num_batches)):
            # Prepare batch
            batch_texts = train_data[i*self.batch_size:(i+1)*self.batch_size]
            inputs = self.tokenizer(batch_texts, 
                                 padding=True, 
                                 truncation=True, 
                                 max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            
            # Start timing
            start_time = time.time()
            
            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Record metrics
            batch_time = time.time() - start_time
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            
            metrics["training_time"].append(batch_time)
            metrics["memory_usage"].append(memory_usage)
            metrics["throughput"].append(self.batch_size / batch_time)
            
            # Clear gradients
            self.model.zero_grad()
            
        return metrics
    
    def benchmark_inference(self, num_batches=100):
        """Benchmark inference performance"""
        logger.info("Starting inference benchmark...")
        self.model.eval()
        
        # Prepare data
        test_data = self.dataset["test"]["text"]
        test_data = [text for text in test_data if len(text) > 0]
        
        metrics = {
            "inference_time": [],
            "memory_usage": [],
            "throughput": []
        }
        
        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                # Prepare batch
                batch_texts = test_data[i*self.batch_size:(i+1)*self.batch_size]
                inputs = self.tokenizer(batch_texts, 
                                     padding=True, 
                                     truncation=True, 
                                     max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
                
                # Start timing
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Record metrics
                batch_time = time.time() - start_time
                memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
                
                metrics["inference_time"].append(batch_time)
                metrics["memory_usage"].append(memory_usage)
                metrics["throughput"].append(self.batch_size / batch_time)
        
        return metrics
    
    def run_benchmarks(self):
        """Run all benchmarks and log results"""
        logger.info("Starting comprehensive benchmark...")
        
        # Initialize wandb
        wandb.init(project="gpt2-benchmark", 
                  config={
                      "model_name": self.model_name,
                      "batch_size": self.batch_size,
                      "max_length": self.max_length
                  })
        
        # Run benchmarks
        training_metrics = self.benchmark_training()
        inference_metrics = self.benchmark_inference()
        
        # Calculate and log statistics
        results = {
            "training": {
                "avg_time": np.mean(training_metrics["training_time"]),
                "avg_memory": np.mean(training_metrics["memory_usage"]),
                "avg_throughput": np.mean(training_metrics["throughput"]),
                "std_time": np.std(training_metrics["training_time"]),
                "std_memory": np.std(training_metrics["memory_usage"]),
                "std_throughput": np.std(training_metrics["throughput"])
            },
            "inference": {
                "avg_time": np.mean(inference_metrics["inference_time"]),
                "avg_memory": np.mean(inference_metrics["memory_usage"]),
                "avg_throughput": np.mean(inference_metrics["throughput"]),
                "std_time": np.std(inference_metrics["inference_time"]),
                "std_memory": np.std(inference_metrics["memory_usage"]),
                "std_throughput": np.std(inference_metrics["throughput"])
            }
        }
        
        # Log to wandb
        wandb.log(results)
        
        # Print results
        logger.info("\nBenchmark Results:")
        logger.info("Training:")
        logger.info(f"Average time per batch: {results['training']['avg_time']:.4f} ± {results['training']['std_time']:.4f} seconds")
        logger.info(f"Average memory usage: {results['training']['avg_memory']:.2f} ± {results['training']['std_memory']:.2f} MB")
        logger.info(f"Average throughput: {results['training']['avg_throughput']:.2f} ± {results['training']['std_throughput']:.2f} samples/second")
        
        logger.info("\nInference:")
        logger.info(f"Average time per batch: {results['inference']['avg_time']:.4f} ± {results['inference']['std_time']:.4f} seconds")
        logger.info(f"Average memory usage: {results['inference']['avg_memory']:.2f} ± {results['inference']['std_memory']:.2f} MB")
        logger.info(f"Average throughput: {results['inference']['avg_throughput']:.2f} ± {results['inference']['std_throughput']:.2f} samples/second")
        
        wandb.finish()
        return results

if __name__ == "__main__":
    benchmark = GPT2Benchmark()
    results = benchmark.run_benchmarks() 