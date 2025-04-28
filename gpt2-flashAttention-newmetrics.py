import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
import time
import numpy as np
from tqdm import tqdm
import logging
import psutil
import os
import platform
import GPUtil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2FlashBenchmark:
    def __init__(self, model_name="gpt2", batch_size=8, max_length=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info(f"Loading {model_name} model and tokenizer with flash attention enabled...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        config = GPT2Config.from_pretrained(model_name)
        config.use_flash_attention = True

        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(self.device)

        logger.info("Loading dataset...")
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def get_system_metrics(self):
        gpu = GPUtil.getGPUs()[0]
        return {
            "cpu_percent": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().used / 1e9,
            "gpu_load": gpu.load,
            "gpu_mem_used": gpu.memoryUsed,
            "gpu_temp": gpu.temperature,
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cuda_version": torch.version.cuda
        }

    def benchmark_training(self, num_batches=100):
        logger.info("Starting training benchmark...")
        self.model.train()

        train_data = self.dataset["train"]["text"]
        train_data = [text for text in train_data if len(text.strip()) > 0]

        metrics = {
            "training_time": [],
            "memory_usage": [],
            "throughput": [],
            "loss": [],
            "perplexity": [],
            "gpu_temp": [],
            "gpu_util": [],
            "cpu_percent": [],
            "ram_usage": []
        }

        for i in tqdm(range(num_batches)):
            batch_texts = train_data[i*self.batch_size:(i+1)*self.batch_size]
            if not batch_texts:
                continue

            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)

            start_time = time.time()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
            loss.backward()
            batch_time = time.time() - start_time
            memory_usage = torch.cuda.memory_allocated() / 1024**2

            sys_metrics = self.get_system_metrics()
            metrics["training_time"].append(batch_time)
            metrics["memory_usage"].append(memory_usage)
            metrics["throughput"].append(self.batch_size / batch_time)
            metrics["loss"].append(loss.item())
            metrics["perplexity"].append(perplexity.item())
            metrics["gpu_temp"].append(sys_metrics["gpu_temp"])
            metrics["gpu_util"].append(sys_metrics["gpu_load"])
            metrics["cpu_percent"].append(sys_metrics["cpu_percent"])
            metrics["ram_usage"].append(sys_metrics["ram_usage"])

            self.model.zero_grad()

        return metrics

    def benchmark_inference(self, num_batches=100):
        logger.info("Starting inference benchmark...")
        self.model.eval()

        test_data = self.dataset["test"]["text"]
        test_data = [text for text in test_data if len(text.strip()) > 0]

        metrics = {
            "inference_time": [],
            "memory_usage": [],
            "throughput": [],
            "perplexity": [],
            "gpu_temp": [],
            "gpu_util": [],
            "cpu_percent": [],
            "ram_usage": []
        }

        with torch.no_grad():
            for i in tqdm(range(num_batches)):
                batch_texts = test_data[i*self.batch_size:(i+1)*self.batch_size]
                if not batch_texts:
                    continue

                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)

                start_time = time.time()
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss)
                batch_time = time.time() - start_time
                memory_usage = torch.cuda.memory_allocated() / 1024**2

                sys_metrics = self.get_system_metrics()
                metrics["inference_time"].append(batch_time)
                metrics["memory_usage"].append(memory_usage)
                metrics["throughput"].append(self.batch_size / batch_time)
                metrics["perplexity"].append(perplexity.item())
                metrics["gpu_temp"].append(sys_metrics["gpu_temp"])
                metrics["gpu_util"].append(sys_metrics["gpu_load"])
                metrics["cpu_percent"].append(sys_metrics["cpu_percent"])
                metrics["ram_usage"].append(sys_metrics["ram_usage"])

        return metrics

    def run_benchmarks(self):
        logger.info("Starting comprehensive benchmark for flash attention model...")
        training_metrics = self.benchmark_training()
        inference_metrics = self.benchmark_inference()

        results = {
            "training": {metric: np.mean(values) for metric, values in training_metrics.items()},
            "training_std": {f"std_{metric}": np.std(values) for metric, values in training_metrics.items()},
            "inference": {metric: np.mean(values) for metric, values in inference_metrics.items()},
            "inference_std": {f"std_{metric}": np.std(values) for metric, values in inference_metrics.items()}
        }

        logger.info("\nBenchmark Results (Flash Attention Model):")
        logger.info("Training:")
        for k, v in results["training"].items():
            logger.info(f"{k}: {v:.4f} ± {results['training_std']['std_' + k]:.4f}")

        logger.info("\nInference:")
        for k, v in results["inference"].items():
            logger.info(f"{k}: {v:.4f} ± {results['inference_std']['std_' + k]:.4f}")

        return results

if __name__ == "__main__":
    benchmark = GPT2FlashBenchmark(batch_size=32, max_length=128)
    results = benchmark.run_benchmarks()
