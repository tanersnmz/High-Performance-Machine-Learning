#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
import torch
import platform
import time
#import wandb 
import numpy as np
from tqdm import tqdm
import logging


# In[ ]:


print(f"Python version: {platform.python_version()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Warning: FlashAttention requires CUDA GPU support. CPU execution will not work.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[ ]:


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2FlashBenchmark:
    def __init__(self, model_name="gpt2", batch_size=16, max_length=2048):
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
        self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        
    def benchmark_training(self, num_batches=100):
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
            batch_texts = train_data[i*self.batch_size:(i+1)*self.batch_size]
            if not batch_texts:  
                continue
                
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
            
            # Calculate perplexity
            perplexity = torch.exp(loss)
            
            # Backward pass
            loss.backward()
            
            # Record metrics
            batch_time = time.time() - start_time
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            
            """
            wandb.log({
                "batch_training_time": batch_time,
                "batch_memory_usage": memory_usage,
                "batch_throughput": self.batch_size / batch_time,
                "batch_loss": loss.item(),
                "batch_perplexity": perplexity.item()
            })
            """
            
            metrics["training_time"].append(batch_time)
            metrics["memory_usage"].append(memory_usage)
            metrics["throughput"].append(self.batch_size / batch_time)
            metrics["loss"].append(loss.item())
            metrics["perplexity"].append(perplexity.item())
            self.model.zero_grad()
            
        return metrics
    
    def benchmark_inference(self, num_batches=100):
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
                batch_texts = test_data[i*self.batch_size:(i+1)*self.batch_size]
                if not batch_texts: 
                    continue
                    
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
                perplexity = torch.exp(loss)
                
                # Record metrics
                batch_time = time.time() - start_time
                memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
                

                """
                wandb.log({
                    "batch_inference_time": batch_time,
                    "batch_inference_memory": memory_usage,
                    "batch_inference_throughput": self.batch_size / batch_time,
                    "batch_inference_perplexity": perplexity.item()
                })
                """
                
                metrics["inference_time"].append(batch_time)
                metrics["memory_usage"].append(memory_usage)
                metrics["throughput"].append(self.batch_size / batch_time)
                metrics["perplexity"].append(perplexity.item())
        
        return metrics
    
    def run_benchmarks(self, num_batches=100):
        """Run all benchmarks and log results"""
        logger.info("Starting comprehensive benchmark for flash attention model...")
        

        """
        wandb.init(project="gpt2-benchmark-flash", 
                   config={
                       "model_name": self.model_name,
                       "batch_size": self.batch_size,
                       "max_length": self.max_length
                   })
        """
        
        training_metrics = self.benchmark_training(num_batches=num_batches)
        inference_metrics = self.benchmark_inference(num_batches=num_batches)
        
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
        
        """
        wandb.log(results)
        logger.info("Benchmark Results:")
        """
        
        logger.info("\nBenchmark Results (Flash Attention Model):")
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
        
        """
        wandb.finish()
        """
        return results



# In[ ]:


benchmark = GPT2FlashBenchmark()
results = benchmark.run_benchmarks(100)


# # Lora

# In[1]:


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[3]:


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
print("Loaded GPT-2 base model and tokenizer")


# In[4]:


for param in model.parameters():
    param.requires_grad = False

print("All GPT-2 parameters frozen because we'll only train LoRA adapters")


# In[7]:


# LoRA adapters
lora_config = LoraConfig(
    r=8,                        
    lora_alpha=32,             
    target_modules=["c_attn"],  
    lora_dropout=0.05,          
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("LoRA adapters injected")


# In[8]:


# Example testing
example_texts = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog."
]
inputs = tokenizer(
    example_texts,
    padding=True,
    truncation=True,
    max_length=32,
    return_tensors="pt"
).to(device)

outputs = model(**inputs, labels=inputs.input_ids)
loss = outputs.loss
loss.backward()   

print(f"Smoke test passed! Loss = {loss.item():.4f}")


# # Lora Benchmark

# In[12]:


import torch
import time
import numpy as np
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Loading wikitext-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


# In[13]:


# LoRA adapters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
base_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(base_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained(base_name).to(device)
for p in base_model.parameters(): 
    p.requires_grad = False
logger.info("Loaded & froze base GPT-2")

# LoRA
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)
logger.info("Injected LoRA adapters")


# In[16]:


def benchmark_loRA(model, tokenizer, dataset, batch_size, max_length=128,
                   num_train_batches=100, num_infer_batches=100):
    import time, torch, numpy as np

    device = next(model.parameters()).device
    train_texts = [t for t in dataset["train"]["text"] if t.strip()]
    test_texts  = [t for t in dataset["test"]["text"]  if t.strip()]

    # ---- TRAINING BENCHMARK ----
    model.train()
    train_times, train_mems = [], []
    train_thrpts, train_losses, train_perps = [], [], []

    for i in range(num_train_batches):
        batch = train_texts[i*batch_size:(i+1)*batch_size]
        if not batch:
            break

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        start = time.time()
        outputs = model(**inputs, labels=inputs.input_ids)
        loss    = outputs.loss
        perp    = torch.exp(loss).item()
        loss.backward()
        elapsed = time.time() - start
        mem     = torch.cuda.memory_allocated(device) / 1024**2

        train_times .append(elapsed)
        train_mems  .append(mem)
        train_thrpts.append(batch_size / elapsed)
        train_losses.append(loss.item())
        train_perps .append(perp)

        model.zero_grad()

    train_stats = {
        "time":       (np.mean(train_times),   np.std(train_times)),
        "memory":     (np.mean(train_mems),    np.std(train_mems)),
        "throughput": (np.mean(train_thrpts),  np.std(train_thrpts)),
        "loss":       (np.mean(train_losses),  np.std(train_losses)),
        "perplexity": (np.mean(train_perps),   np.std(train_perps))
    }

    # ---- INFERENCE BENCHMARK ----
    model.eval()
    infer_times, infer_mems = [], []
    infer_thrpts, infer_perps = [], []

    with torch.no_grad():
        for i in range(num_infer_batches):
            batch = test_texts[i*batch_size:(i+1)*batch_size]
            if not batch:
                break

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            start = time.time()
            outputs = model(**inputs, labels=inputs.input_ids)
            loss    = outputs.loss
            perp    = torch.exp(loss).item()
            elapsed = time.time() - start
            mem     = torch.cuda.memory_allocated(device) / 1024**2

            infer_times .append(elapsed)
            infer_mems  .append(mem)
            infer_thrpts.append(batch_size / elapsed)
            infer_perps .append(perp)

    infer_stats = {
        "time":       (np.mean(infer_times),   np.std(infer_times)),
        "memory":     (np.mean(infer_mems),    np.std(infer_mems)),
        "throughput": (np.mean(infer_thrpts),  np.std(infer_thrpts)),
        "perplexity": (np.mean(infer_perps),   np.std(infer_perps))
    }

    return train_stats, infer_stats


# In[17]:


# Running benchmarks
max_length = 128

for bs in [8, 16, 32]:
    logger.info(f"\nRunning benchmark for LoRA model with batch_size={bs}, max_length={max_length}")
    logger.info("Configuration:")
    logger.info(f"  • Model:  LoRA on GPT-2")
    logger.info(f"  • Batch size:   {bs}")
    logger.info(f"  • Max length:   {max_length}")
    
    train_stats, infer_stats = benchmark_loRA(
        model, tokenizer, dataset,
        batch_size=bs, max_length=max_length,
        num_train_batches=100, num_infer_batches=100
    )

    # Training results
    t = train_stats
    logger.info("\nTraining:")
    logger.info(f"  Average time per batch:    {t['time'][0]:.4f} ± {t['time'][1]:.4f} seconds")
    logger.info(f"  Average memory usage:      {t['memory'][0]:.2f} ± {t['memory'][1]:.2f} MB")
    logger.info(f"  Average throughput:        {t['throughput'][0]:.2f} ± {t['throughput'][1]:.2f} samples/second")
    logger.info(f"  Average loss:              {t['loss'][0]:.4f} ± {t['loss'][1]:.4f}")
    logger.info(f"  Average perplexity:        {t['perplexity'][0]:.4f} ± {t['perplexity'][1]:.4f}")
    
    # Inference results
    i = infer_stats
    logger.info("\nInference:")
    logger.info(f"  Average time per batch:    {i['time'][0]:.4f} ± {i['time'][1]:.4f} seconds")
    logger.info(f"  Average memory usage:      {i['memory'][0]:.2f} ± {i['memory'][1]:.2f} MB")
    logger.info(f"  Average throughput:        {i['throughput'][0]:.2f} ± {i['throughput'][1]:.2f} samples/second")
    logger.info(f"  Average perplexity:        {i['perplexity'][0]:.4f} ± {i['perplexity'][1]:.4f}")


# # Lora apples-to-apples bench

# In[1]:


import torch
import time, math, logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW  
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# In[2]:


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

dataset   = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# In[4]:


# Helper functions
def compute_perplexity(model, tokenizer, texts, device,
                       batch_size=8, max_length=128):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            loss = model(**inputs, labels=inputs.input_ids).loss
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)

def train_until(model, tokenizer, dataset, target_ppl,
                device, lr=5e-5, batch_size=8, max_length=128,
                eval_every=500):
    optimizer   = AdamW(model.parameters(), lr=lr)
    train_texts = [t for t in dataset["train"]["text"]      if t.strip()]
    val_texts   = [t for t in dataset["validation"]["text"] if t.strip()]

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    step = 0

    while True:
        model.train()
        for i in range(0, len(train_texts), batch_size):
            batch  = train_texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)

            loss = model(**inputs, labels=inputs.input_ids).loss
            loss.backward()
            optimizer.step()
            model.zero_grad()

            step += 1
            if step % eval_every == 0:
                val_ppl = compute_perplexity(model, tokenizer, val_texts, device,
                                             batch_size=batch_size, max_length=max_length)
                elapsed_h = (time.time() - start_time) / 3600
                peak_mem  = torch.cuda.memory_allocated(device) / 1024**2
                logger.info(f"[step {step}] val_ppl={val_ppl:.2f} time={elapsed_h:.2f}h peak_mem={peak_mem:.1f}MB")
                if val_ppl <= target_ppl:
                    total_h = (time.time() - start_time) / 3600
                    final_mem = torch.cuda.memory_allocated(device) / 1024**2
                    return {
                        "steps":       step,
                        "total_hours": total_h,
                        "peak_mem_mb": final_mem,
                        "final_ppl":   val_ppl
                    }


# In[5]:


#Baseline full fine-tune benchmark
baseline_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
for p in baseline_model.parameters():
    p.requires_grad = True

baseline_res = train_until(
    baseline_model, tokenizer, dataset, target_ppl=15,
    device=device, lr=5e-5, batch_size=16, max_length=128,
    eval_every=500
)
logger.info(f"Baseline results: {baseline_res}")


# In[7]:


# after baseline finishes
del baseline_model  
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)


# In[8]:


# LoRA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

base_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(base_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = GPT2LMHeadModel.from_pretrained(base_name).to(device)
logger.info("Loaded GPT-2")

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_cfg)
logger.info("Injected LoRA adapters")
model.print_trainable_parameters()

lora_res = train_until(
    model, tokenizer, dataset, target_ppl=15,
    device=device, lr=1e-4, batch_size=16, max_length=128,
    eval_every=500
)
logger.info(f"LoRA results: {lora_res}")


# In[9]:


# Comparison
import pandas as pd

df = pd.DataFrame([baseline_res, lora_res], index=["Baseline", "LoRA"])
df.rename(columns={
    "steps":       "Steps",
    "total_hours":"Total Hours (h)",
    "peak_mem_mb":"Peak GPU Mem (MB)",
    "final_ppl":  "Final Val Perplexity"
}, inplace=True)


# In[10]:


df


# In[ ]:




