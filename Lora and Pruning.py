#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset
import torch
import platform
import time
#import wandb 
import numpy as np
from tqdm import tqdm
import logging


# In[2]:


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


# In[5]:


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


# In[6]:


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

# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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

# In[11]:


import torch
import time, math, logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW  
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# In[12]:


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

dataset   = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# In[13]:


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


# In[14]:


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


# In[15]:


# after baseline finishes
del baseline_model  
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)


# In[16]:


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


# In[17]:


# Comparison
import pandas as pd

df = pd.DataFrame([baseline_res, lora_res], index=["Baseline", "LoRA"])
df.rename(columns={
    "steps":       "Steps",
    "total_hours":"Total Hours (h)",
    "peak_mem_mb":"Peak GPU Mem (MB)",
    "final_ppl":  "Final Val Perplexity"
}, inplace=True)


# In[18]:


df


# In[19]:


# Saving Lora model
'''
save_dir = "lora_gpt2_checkpoint"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"LoRA model & tokenizer saved to {save_dir}")'''


# In[4]:


#Reload
'''
save_dir = "lora_gpt2_checkpoint"
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
base = GPT2LMHeadModel.from_pretrained("gpt2")
tok  = GPT2Tokenizer.from_pretrained(save_dir)
tok.pad_token = tok.eos_token
model = PeftModel.from_pretrained(base, save_dir).to(device)'''


# # Unstructured Pruning

# In[5]:


import torch
import torch.nn.utils.prune as prune
import peft 
import logging


# In[6]:


lora_model = model


# In[7]:


# Pruning Parameters 
pruning_amount = 0.5  # Prune 50% of the weights 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"Starting global unstructured pruning with amount: {pruning_amount}")
logger.info(f"Targeting LoRA A and LoRA B weights.")


# In[8]:


# Identifying LoRA Parameters to Prune 
parameters_to_prune = []
target_modules = []

logger.info("Identifying LoRA parameters for pruning...")
for name, module in lora_model.named_modules():
    if isinstance(module, peft.tuners.lora.Linear):
        logger.debug(f"Found LoRA linear layer: {name}")
        for param_name, param in module.named_parameters():
            if 'lora_A' in param_name and 'weight' in param_name:
                parameters_to_prune.append((module, 'weight')) # Prune the weight of lora_A
                target_modules.append(module) 
                logger.debug(f"  - Added '{param_name}' (accessed via 'weight' on module {type(module).__name__}) to pruning list.")
            elif 'lora_B' in param_name and 'weight' in param_name:
                 pass # We will handle this below more robustly


# In[9]:


# Refined approach to find correct parameter names for pruning within LoRA layers
parameters_to_prune_refined = []
target_modules_refined = []
for name, module in lora_model.named_modules():
     if isinstance(module, peft.tuners.lora.LoraLayer): 
         logger.debug(f"Found LoRA layer: {name} of type {type(module)}")
         if hasattr(module, 'lora_A') and hasattr(module.lora_A, 'default'):
             if hasattr(module.lora_A['default'], 'weight'):
                 logger.debug(f"  - Adding parameter 'lora_A.default.weight' from module {name}")
                 parameters_to_prune_refined.append((module.lora_A['default'], 'weight'))
                 target_modules_refined.append(module.lora_A['default'])
         if hasattr(module, 'lora_B') and hasattr(module.lora_B, 'default'): 
             if hasattr(module.lora_B['default'], 'weight'):
                 logger.debug(f"  - Adding parameter 'lora_B.default.weight' from module {name}")
                 parameters_to_prune_refined.append((module.lora_B['default'], 'weight'))
                 target_modules_refined.append(module.lora_B['default'])


if not parameters_to_prune_refined:
    logger.warning("Could not find any LoRA A/B weights matching expected structure.")
    logger.warning("Please inspect your model's layers and parameter names.")
else:
    logger.info(f"Identified {len(parameters_to_prune_refined)} parameter tensors to prune.")

    # Global Unstructured Pruning
    logger.info(f"Applying global unstructured pruning (L1 magnitude) with amount={pruning_amount}...")
    prune.global_unstructured(
        parameters_to_prune_refined,
        pruning_method=prune.L1Unstructured, # Prune weights with smallest L1 norm 
        amount=pruning_amount,
    )
    logger.info("Pruning mask applied.")

    # Verify Sparsity 
    def calculate_global_sparsity(parameters):
        total_params = 0
        zero_params = 0
        for module, name in parameters:
            param = getattr(module, name)
            total_params += param.nelement()
            zero_params += torch.sum(param == 0).item()
        if total_params == 0:
            return 0.0
        sparsity = 100. * float(zero_params) / float(total_params)
        return sparsity, zero_params, total_params

    # Calculate sparsity *while the mask is active*
    sparsity_before_remove, _, _ = calculate_global_sparsity(parameters_to_prune_refined)
    logger.info(f"Sparsity after applying mask: {sparsity_before_remove:.2f}%")

    # Make Pruning Permanent 
    logger.info("Making pruning permanent by removing masks and zeroing weights...")
    for module in target_modules_refined:
         if prune.is_pruned(module):
              prune.remove(module, 'weight') 
              logger.debug(f"Removed pruning mask from parameter 'weight' in module {type(module).__name__}")
         else:
             logger.debug(f"No pruning mask found on 'weight' in module {type(module).__name__} (already removed or never pruned).")


    logger.info("Pruning made permanent.")

    # Verify Sparsity After Removal 
    final_sparsity, final_zeros, final_total = calculate_global_sparsity(parameters_to_prune_refined)
    logger.info(f"Final sparsity after mask removal: {final_sparsity:.2f}%")
    logger.info(f"Total LoRA parameters considered for pruning: {final_total}")
    logger.info(f"Total zeroed parameters in LoRA layers: {final_zeros}")

    # The model `lora_model` now has its LoRA weights permanently pruned.
    logger.info("Pruning process complete. `lora_model` has been modified in-place.")


# In[10]:


# Checking a specific pruned weight matrix 
try:
 module_to_inspect = target_modules_refined[0] # Getting the first module we pruned
 weight_tensor = module_to_inspect.weight 
 logger.info(f"Sample pruned weight tensor (first 5x5 elements or less):\n {weight_tensor.data[:5,:5]}")
 sparsity_sample = 100. * float(torch.sum(weight_tensor == 0)) / float(weight_tensor.nelement())
 logger.info(f"Sparsity of this sample tensor: {sparsity_sample:.2f}%")
except IndexError:
 logger.info("Cannot show sample tensor, no parameters were identified for pruning.")
except AttributeError:
 logger.info("Cannot show sample tensor, module structure might have changed unexpectedly.")


# # Unstructured Pruning Benchmark

# In[39]:


import torch
import time
import numpy as np
import logging
from datasets import load_dataset
from transformers import GPT2Tokenizer 

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("\n" + "="*50)
logger.info("Starting Benchmarks for PRUNED LoRA Model")
logger.info(f"Pruned Model Sparsity: {final_sparsity:.2f}%")
logger.info("="*50 + "\n")


# Re-running the exact same benchmark loop on the PRUNED model 
for bs in [8, 16, 32]:
    logger.info(f"\nRunning benchmark for PRUNED LoRA model with batch_size={bs}, max_length={max_length}")
    logger.info("Configuration:")
    logger.info(f"  • Model:      Pruned LoRA on GPT-2 ({final_sparsity:.1f}% sparse)")
    logger.info(f"  • Batch size: {bs}")
    logger.info(f"  • Max length: {max_length}")
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_stats, infer_stats = benchmark_loRA(
        lora_model, tokenizer, dataset, # Passing the pruned lora_model
        batch_size=bs, max_length=max_length,
        num_train_batches=100, num_infer_batches=100
    )

    # Training results for the pruned model
    t = train_stats
    logger.info("\nTraining (Pruned Model):")
    logger.info(f"  Average time per batch:    {t['time'][0]:.4f} ± {t['time'][1]:.4f} seconds")
    logger.info(f"  Average memory usage:      {t['memory'][0]:.2f} ± {t['memory'][1]:.2f} MB")
    logger.info(f"  Average throughput:        {t['throughput'][0]:.2f} ± {t['throughput'][1]:.2f} samples/second")
    logger.info(f"  Average loss:              {t['loss'][0]:.4f} ± {t['loss'][1]:.4f}")
    logger.info(f"  Average perplexity:        {t['perplexity'][0]:.4f} ± {t['perplexity'][1]:.4f}")

    # Inference results for the pruned model
    i = infer_stats
    logger.info("\nInference (Pruned Model):")
    logger.info(f"  Average time per batch:    {i['time'][0]:.4f} ± {i['time'][1]:.4f} seconds")
    logger.info(f"  Average memory usage:      {i['memory'][0]:.2f} ± {i['memory'][1]:.2f} MB")
    logger.info(f"  Average throughput:        {i['throughput'][0]:.2f} ± {i['throughput'][1]:.2f} samples/second")
    logger.info(f"  Average perplexity:        {i['perplexity'][0]:.4f} ± {i['perplexity'][1]:.4f}")

logger.info("\n" + "="*50)
logger.info("Benchmarking for PRUNED LoRA Model Complete")
logger.info("="*50 + "\n")


# # Final Comparison

# In[12]:


import torch
import time, math, logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from datasets import load_dataset
from peft import LoraConfig, get_peft_model 
from peft.tuners.lora import LoraLayer
import torch.nn.utils.prune as prune
import pandas as pd
import gc

# setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model_name = "gpt2"
max_length = 128
target_ppl = 15
eval_every = 500
batch_size = 16
pruning_amount = 0.8 # 80% sparsity

# Dataset and Tokenizer
logger.info("Loading wikitext-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
logger.info("Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_texts_full = [t for t in dataset["train"]["text"] if t.strip()]
val_texts_full = [t for t in dataset["validation"]["text"] if t.strip()]
logger.info("Dataset and tokenizer loaded.")


# Helper functions 
def compute_perplexity(model, tokenizer, texts, device,
                       batch_size=8, max_length=128):
    model.eval()
    losses = []
    total_evaluated = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if not batch: continue
            total_evaluated += len(batch)
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs.input_ids)
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                 losses.append(outputs.loss.item() * len(batch)) 
                 pass 

    if not losses or total_evaluated == 0:
         return float('inf')

    avg_loss = sum(losses) / total_evaluated
    if avg_loss <= 0:
         return float('inf')
    return math.exp(avg_loss)

def train_until(model, tokenizer, train_texts, val_texts, target_ppl,
                device, lr=5e-5, batch_size=8, max_length=128,
                eval_every=500, run_label="Training"):
    optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    start_time = time.time()
    if device == torch.device("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    step = 0
    max_steps = (len(train_texts) // batch_size) * 20
    logger.info(f"--- Starting {run_label} ---")
    logger.info(f"Target PPL: {target_ppl}, LR: {lr}, Batch Size: {batch_size}, Eval Every: {eval_every}")
    logger.info(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")
    model.train()

    best_ppl = float('inf')

    while True:
        model.train()
        for i in range(0, len(train_texts), batch_size):
            batch  = train_texts[i:i+batch_size]
            if not batch: continue

            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)

            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss

            if loss is None:
                step += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            model.zero_grad(set_to_none=True)

            step += 1

            if step % eval_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                val_ppl = compute_perplexity(model, tokenizer, val_texts, device,
                                             batch_size=batch_size, max_length=max_length)
                elapsed_h = (time.time() - start_time) / 3600
                if device == torch.device("cuda"):
                    peak_mem = torch.cuda.memory_allocated(device) / 1024**2
                else:
                    peak_mem = 0

                logger.info(f"[{run_label} Step {step}/{max_steps}] val_ppl={val_ppl:.2f} (Best: {best_ppl:.2f}) loss={loss.item():.3f} lr={current_lr:.1e} time={elapsed_h:.2f}h peak_mem={peak_mem:.1f}MB")

                if val_ppl <= target_ppl:
                    total_h = (time.time() - start_time) / 3600
                    final_mem = peak_mem
                    logger.info(f"--- Target PPL {target_ppl} reached for {run_label} ---")
                    return {
                        "steps":       step,
                        "total_hours": total_h,
                        "peak_mem_mb": final_mem,
                        "final_ppl":   val_ppl
                    }

                best_ppl = min(best_ppl, val_ppl)
                model.train()

            if step >= max_steps:
                 logger.warning(f"{run_label} did not reach target PPL within {max_steps} steps. Returning current state.")
                 total_h = (time.time() - start_time) / 3600
                 final_mem = torch.cuda.memory_allocated(device) / 1024**2 if device == torch.device("cuda") else 0
                 return {
                     "steps": step,
                     "total_hours": total_h,
                     "peak_mem_mb": final_mem,
                     "final_ppl": best_ppl 
                 }


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
#  Baseline full fine-tune benchmark 
logger.info("\n===== Starting Baseline Full Fine-tune Benchmark =====")
baseline_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for p in baseline_model.parameters():
    p.requires_grad = True

baseline_res = train_until(
    baseline_model, tokenizer, train_texts_full, val_texts_full, target_ppl=target_ppl,
    device=device, lr=5e-5, batch_size=batch_size, max_length=max_length,
    eval_every=eval_every, run_label="Baseline"
)
logger.info(f"Baseline results: {baseline_res}")

# after baseline finishes
del baseline_model
gc.collect()
if device == torch.device("cuda"):
    torch.cuda.empty_cache()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
#  LoRA fine-tune benchmark 
logger.info("\n===== Starting LoRA Fine-tune Benchmark =====")
base_model_lora = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for param in base_model_lora.parameters():
    param.requires_grad = False

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(base_model_lora, lora_cfg)
logger.info("Injected LoRA adapters for LoRA run.")
lora_model.print_trainable_parameters()

lora_res = train_until(
    lora_model, tokenizer, train_texts_full, val_texts_full, target_ppl=target_ppl,
    device=device, lr=1e-4, batch_size=batch_size, max_length=max_length,
    eval_every=eval_every, run_label="LoRA"
)
logger.info(f"LoRA results: {lora_res}")

# after LoRA finishes
del base_model_lora, lora_model
gc.collect()
if device == torch.device("cuda"):
    torch.cuda.empty_cache()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
#  Pruned LoRA Fine-tune Benchmark 
logger.info("\n===== Starting Pruned LoRA Fine-tune Benchmark =====")
# Load base model again
base_model_pruned = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for param in base_model_pruned.parameters():
    param.requires_grad = False

# Apply LoRA config again
lora_model_pruned = get_peft_model(base_model_pruned, lora_cfg)
logger.info("Injected LoRA adapters for Pruned LoRA run.")
lora_model_pruned.print_trainable_parameters()

# --- Apply Pruning BEFORE Training ---
logger.info(f"Applying {pruning_amount:.1%} global unstructured pruning to LoRA weights...")
parameters_to_prune = []
target_modules = set()
for name, module in lora_model_pruned.named_modules():
     if isinstance(module, LoraLayer):
         for key in ['lora_A', 'lora_B']:
            if hasattr(module, key):
                sub_module_dict = getattr(module, key)
                if isinstance(sub_module_dict, torch.nn.ModuleDict):
                     for adapter_name, sub_module in sub_module_dict.items():
                         if hasattr(sub_module, 'weight'):
                            parameters_to_prune.append((sub_module, 'weight'))
                            target_modules.add(sub_module)
                elif isinstance(sub_module_dict, torch.nn.Module):
                    if hasattr(sub_module_dict, 'weight'):
                        parameters_to_prune.append((sub_module_dict, 'weight'))
                        target_modules.add(sub_module_dict)

if not parameters_to_prune:
    logger.warning("Could not find any LoRA parameters to prune!")
    pruned_lora_res = {"steps": -1, "total_hours": -1, "peak_mem_mb": -1, "final_ppl": -1}
else:
    logger.info(f"Identified {len(parameters_to_prune)} parameter tensors for pruning.")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount,
    )
    logger.info("Pruning mask applied. Making pruning permanent...")
    final_total_params = 0
    final_zero_params = 0
    for module in target_modules:
        if prune.is_pruned(module):
             param = getattr(module, 'weight')
             final_total_params += param.nelement()
             final_zero_params += torch.sum(param == 0).item()
             prune.remove(module, 'weight')
        else:
             param = getattr(module, 'weight')
             final_total_params += param.nelement()


    if final_total_params > 0:
        final_sparsity = 100. * float(final_zero_params) / float(final_total_params)
        logger.info(f"Pruning made permanent. Achieved sparsity: {final_sparsity:.2f}%")
    else:
        logger.info("Pruning made permanent (no parameters found).")


    #  Train the Pruned LoRA Model 
    pruned_lora_res = train_until(
        lora_model_pruned, tokenizer, train_texts_full, val_texts_full, target_ppl=target_ppl,
        device=device, lr=1e-4, batch_size=batch_size, max_length=max_length,
        eval_every=eval_every, run_label="Pruned LoRA"
    )
    logger.info(f"Pruned LoRA results: {pruned_lora_res}")

# after Pruned LoRA finishes
del base_model_pruned, lora_model_pruned
gc.collect()
if device == torch.device("cuda"):
    torch.cuda.empty_cache()

#  Comparison 
logger.info("\n===== Benchmark Comparison =====")

results_list = []
indices = []
if 'steps' in baseline_res and baseline_res['steps'] != -1:
    results_list.append(baseline_res)
    indices.append("Baseline")

if 'steps' in lora_res and lora_res['steps'] != -1:
    results_list.append(lora_res)
    indices.append("LoRA")

# Ensure pruned_lora_res exists even if pruning failed
if 'pruned_lora_res' not in locals():
     pruned_lora_res = {"steps": -1, "total_hours": -1, "peak_mem_mb": -1, "final_ppl": -1}


if 'steps' in pruned_lora_res and pruned_lora_res['steps'] != -1:
    results_list.append(pruned_lora_res)
    indices.append(f"Pruned LoRA ({pruning_amount:.0%})")


if results_list:
    df = pd.DataFrame(results_list, index=indices)
    df.rename(columns={
        "steps": "Steps",
        "total_hours": "Total Hours (h)",
        "peak_mem_mb": "Peak GPU Mem (MB)",
        "final_ppl": "Final Val Perplexity"
    }, inplace=True)

    # Format the DataFrame
    df['Steps'] = df['Steps'].map('{:,.0f}'.format)
    df['Total Hours (h)'] = df['Total Hours (h)'].map('{:.2f}'.format)
    df['Peak GPU Mem (MB)'] = df['Peak GPU Mem (MB)'].map('{:,.1f}'.format)
    df['Final Val Perplexity'] = df['Final Val Perplexity'].map('{:.2f}'.format)

    logger.info("\nComparison DataFrame:\n%s", df.to_string())
else:
    logger.error("No successful benchmark runs to compare.")

logger.info("\n===== Script Finished =====")


# In[20]:


import wandb
wandb.login()


# In[21]:


import wandb
import pandas as pd
import math
import logging

#  Configuration 
config_manual = {
    "model_name": "gpt2",
    "max_length": 128,
    "num_train_epochs_intended": 1,
    "batch_size": 16,
    "pruning_amount": 0.8,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_target_modules": ["c_attn"],
    "baseline_lr": 5e-5,
    "lora_lr": 1e-4,
    "eval_every_steps": 500,
    "target_ppl_set": 15,
    "device": "cuda",
    "memory_metric": "torch.cuda.memory_allocated"
}

#  Results 
baseline_res_manual = {
    "steps": 500, "total_hours": 0.08, "peak_mem_mb": 2554.6, "final_ppl": 8.07
}
lora_res_manual = {
    "steps": 500, "total_hours": 0.06, "peak_mem_mb": 1606.7, "final_ppl": 8.86
}
pruned_lora_res_manual = {
    "steps": 500, "total_hours": 0.06, "peak_mem_mb": 1605.7, "final_ppl": 8.89
}

# Wandb 
logging.basicConfig(level=logging.INFO) 
run = None
try:
    run = wandb.init(
        project="lora-pruning-comparison",
        config=config_manual,
        name="Manual Log - Run ending PPL ~8-9 (80% Pruning)",
        job_type="logging"
    )
    print("Wandb initialized.")
    pruning_label = f"({config_manual['pruning_amount']:.0%})"
    if 'final_ppl' in baseline_res_manual and not math.isnan(baseline_res_manual['final_ppl']):
        wandb.log({f"Summary/Baseline/{k}": v for k, v in baseline_res_manual.items()})
    if 'final_ppl' in lora_res_manual and not math.isnan(lora_res_manual['final_ppl']):
         wandb.log({f"Summary/LoRA/{k}": v for k, v in lora_res_manual.items()})
    if 'final_ppl' in pruned_lora_res_manual and not math.isnan(pruned_lora_res_manual['final_ppl']):
         wandb.log({f"Summary/Pruned LoRA {pruning_label}/{k}": v for k, v in pruned_lora_res_manual.items()})
    print("Summary metrics logged.")

    # Log DataFrame
    results_list = [baseline_res_manual, lora_res_manual, pruned_lora_res_manual]
    indices = ["Baseline", "LoRA", f"Pruned LoRA {pruning_label}"]
    valid_results = []
    valid_indices = []
    for res, idx in zip(results_list, indices):
         if res and 'final_ppl' in res and isinstance(res['final_ppl'], (int, float)) and not math.isnan(res['final_ppl']) and not math.isinf(res['final_ppl']):
              valid_results.append(res)
              valid_indices.append(idx)
    if valid_results:
        df = pd.DataFrame(valid_results, index=valid_indices)
        df.rename(columns={"steps": "Steps", "total_hours": "Total Hours (h)", "peak_mem_mb": "Peak GPU Mem (MB)", "final_ppl": "Final Val Perplexity"}, inplace=True)
        if 'Steps' in df.columns: df['Steps'] = df['Steps'].map('{:,.0f}'.format)
        if 'Total Hours (h)' in df.columns: df['Total Hours (h)'] = df['Total Hours (h)'].map('{:.2f}'.format)
        if 'Peak GPU Mem (MB)' in df.columns: df['Peak GPU Mem (MB)'] = df['Peak GPU Mem (MB)'].map('{:,.1f}'.format)
        if 'Final Val Perplexity' in df.columns: df['Final Val Perplexity'] = df['Final Val Perplexity'].map('{:.2f}'.format)
        df_log = df.reset_index().rename(columns={'index': 'Method'})
        wandb.log({"Comparison Table": wandb.Table(dataframe=df_log)})
        print("\nDat;aFrame Logged to Wandb:")
        print(df.to_string())

except Exception as e:
    print(f"An error occurred during wandb processing: {e}")
finally:
    if run:
        wandb.finish()
        print("Wandb run finished.")
    else:
        print("Wandb was not initialized successfully.")


# # Final Comparison (Baseline vs (Baseline + LORA) vs (Baseline + LORA + Pruning))

# In[ ]:


import torch
import time, math, logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig
from torch.optim import AdamW
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
import torch.nn.utils.prune as prune
import pandas as pd
import gc
import copy
import wandb
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model_name = "gpt2"
max_length = 128
num_train_epochs = 2
eval_every_steps = 200
batch_size = 16
inference_batch_size = 8
num_inference_batches = 50
pruning_amount = 0.8
lora_r = 8
lora_alpha = 32
lora_target_modules = ["c_attn"]
baseline_lr = 5e-5
lora_lr = 1e-4
run_inference_benchmark = True

try:
    run = wandb.init(
        project="lora-pruning-comparison-final",
        name=f"Baseline_vs_LoRA_vs_Pruned_{num_train_epochs}Ep_{pruning_amount:.0%}",
        config={
            "model_name": model_name, "max_length": max_length,
            "num_train_epochs": num_train_epochs, "batch_size": batch_size,
            "pruning_amount": pruning_amount, "lora_r": lora_r,
            "lora_alpha": lora_alpha, "lora_target_modules": lora_target_modules,
            "baseline_lr": baseline_lr, "lora_lr": lora_lr,
            "eval_every_steps": eval_every_steps, "device": str(device),
            "inference_batch_size": inference_batch_size,
            "num_inference_batches": num_inference_batches,
            "run_inference_benchmark": run_inference_benchmark
        }
    )
    logger.info("Weights & Biases initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Weights & Biases: {e}")
    run = None

logger.info("Loading data...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_texts_full = [t for t in dataset["train"]["text"] if t.strip()]
val_texts_full = [t for t in dataset["validation"]["text"] if t.strip()]
test_texts_full = val_texts_full[:inference_batch_size * num_inference_batches]
logger.info(f"Data loaded. Train: {len(train_texts_full)}, Val/Test: {len(val_texts_full)}")

@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device, batch_size=8, max_length=128):
    model.eval()
    losses = []
    total_evaluated = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if not batch: continue
        total_evaluated += len(batch)
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs.input_ids)
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            losses.append(outputs.loss.item() * len(batch))
    if not losses or total_evaluated == 0: return float('inf')
    avg_loss = sum(losses) / total_evaluated
    if avg_loss <= 0: return float('inf')
    return math.exp(avg_loss)

def train_fixed_duration(model, tokenizer, train_texts, val_texts, num_epochs,
                         device, lr=5e-5, batch_size=8, max_length=128,
                         eval_every=500, run_label="Training"):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    start_time = time.time()
    peak_mem_overall = 0

    if device == torch.device("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    total_expected_steps = (len(train_texts) // batch_size) * num_epochs
    step = 0
    all_val_ppl = []
    steps_log = []

    logger.info(f"--- Starting {run_label} ---")
    logger.info(f"Epochs: {num_epochs}, LR: {lr}, Batch Size: {batch_size}, Eval Every: {eval_every}")

    for epoch in range(num_epochs):
        model.train()
        logger.info(f"[{run_label} Epoch {epoch+1}/{num_epochs}] Starting...")
        progress_bar = tqdm(range(0, len(train_texts), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for i in progress_bar:
            batch = train_texts[i:i+batch_size]
            if not batch: continue

            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss

            if loss is None or torch.isnan(loss):
                logger.warning(f"[{run_label} Step {step}] Loss is None or NaN. Skipping step.")
                step += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            progress_bar.set_postfix({'loss': f"{loss.item():.3f}"})

            is_last_batch_overall = (epoch == num_epochs - 1 and i + batch_size >= len(train_texts))
            if step % eval_every == 0 or is_last_batch_overall:
                current_loss_val = loss.item() if loss is not None else float('nan')
                val_ppl = compute_perplexity(model, tokenizer, val_texts, device, batch_size=batch_size, max_length=max_length)
                if not math.isinf(val_ppl) and not math.isnan(val_ppl):
                    all_val_ppl.append(val_ppl)
                    steps_log.append(step)

                elapsed_h = (time.time() - start_time) / 3600
                current_peak_mem = 0
                if device == torch.device("cuda"):
                    current_peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
                    peak_mem_overall = max(peak_mem_overall, current_peak_mem)

                logger.info(f"[{run_label} Step {step}/{total_expected_steps}] val_ppl={val_ppl:.2f} loss={current_loss_val:.3f} time={elapsed_h:.2f}h peak_mem={current_peak_mem:.1f}MB")

                if run:
                     wandb.log({
                         f"{run_label}/Step": step, f"{run_label}/Val Perplexity": val_ppl,
                         f"{run_label}/Loss": current_loss_val, f"{run_label}/Elapsed Hours": elapsed_h,
                         f"{run_label}/Current Peak Mem MB": current_peak_mem
                     }, step=step)
                model.train()

        logger.info(f"[{run_label} Epoch {epoch+1}/{num_epochs}] Completed.")

    total_h = (time.time() - start_time) / 3600
    final_mem = peak_mem_overall
    final_ppl = all_val_ppl[-1] if all_val_ppl else float('inf')

    logger.info(f"--- Finished {run_label} ---")
    return {
        "steps": step, "total_hours": total_h, "peak_mem_mb": final_mem,
        "final_ppl": final_ppl, "steps_log": steps_log, "ppl_log": all_val_ppl
    }

def apply_pruning(model_to_prune, amount):
    parameters_to_prune = []
    target_modules = set()
    logger.debug("Starting parameter search for pruning...")
    for name, module in model_to_prune.named_modules():
         if isinstance(module, LoraLayer):
             logger.debug(f"Found LoraLayer: {name}")
             for key in ['lora_A', 'lora_B']:
                if hasattr(module, key):
                    sub_module_dict = getattr(module, key)
                    if isinstance(sub_module_dict, torch.nn.ModuleDict):
                         for adapter_name, sub_module in sub_module_dict.items():
                             if hasattr(sub_module, 'weight'):
                                logger.debug(f"   Found weight in {key}.{adapter_name}. Adding.")
                                parameters_to_prune.append((sub_module, 'weight'))
                                target_modules.add(sub_module)
                    elif isinstance(sub_module_dict, torch.nn.Module):
                        if hasattr(sub_module_dict, 'weight'):
                            logger.debug(f"   Found weight directly in {key}. Adding.")
                            parameters_to_prune.append((sub_module_dict, 'weight'))
                            target_modules.add(sub_module_dict)

    if not parameters_to_prune:
        logger.warning("Could not find any LoRA parameters to prune!")
        return 0.0
    else:
        logger.info(f"Identified {len(parameters_to_prune)} parameter tensors for pruning.")
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
        logger.info("Pruning mask applied. Making pruning permanent...")
        final_total_params = 0; final_zero_params = 0; pruned_module_count = 0
        for module in target_modules:
            is_currently_pruned = prune.is_pruned(module)
            if is_currently_pruned:
                 try:
                     mask = getattr(module, 'weight_mask', None); orig_param = getattr(module, 'weight_orig', None)
                     if mask is not None and orig_param is not None:
                          param_element_count = orig_param.nelement(); param_zero_count = torch.sum(mask == 0).item()
                          final_total_params += param_element_count; final_zero_params += param_zero_count
                     elif hasattr(module, 'weight'):
                          param = getattr(module, 'weight'); final_total_params += param.nelement(); final_zero_params += torch.sum(param == 0).item()
                 except AttributeError:
                       if hasattr(module, 'weight'):
                           param = getattr(module, 'weight'); final_total_params += param.nelement(); final_zero_params += torch.sum(param == 0).item()
                 prune.remove(module, 'weight'); pruned_module_count += 1
            elif hasattr(module, 'weight'):
                 param = getattr(module, 'weight'); final_total_params += param.nelement()
        logger.info(f"Removed pruning buffer from {pruned_module_count} modules.")
        if final_total_params > 0:
            final_sparsity = 100. * float(final_zero_params) / float(final_total_params)
            logger.info(f"Pruning made permanent. Calculated sparsity: {final_sparsity:.2f}%")
            return final_sparsity
        else:
            logger.info("Pruning made permanent (no parameters found or counted)."); return 0.0

@torch.no_grad()
def benchmark_inference(model, tokenizer, texts, device, batch_size=8, max_length=128, num_batches=50, generation=False):
    model.eval()
    latencies = []
    total_samples = 0
    logger.info(f"--- Starting Inference Benchmark (Generation: {generation}) ---")
    generation_config = GenerationConfig(max_new_tokens=5, pad_token_id=tokenizer.pad_token_id) if generation else None

    for i in tqdm(range(0, min(len(texts), batch_size * num_batches), batch_size), desc="Inference", leave=False):
        batch = texts[i:i+batch_size]
        if not batch: continue
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        batch_samples = inputs['input_ids'].shape[0]
        total_samples += batch_samples

        start_time = time.perf_counter()
        if generation:
            _ = model.generate(**inputs, generation_config=generation_config)
        else:
            _ = model(**inputs)
        if device == torch.device("cuda"): torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    avg_latency_batch = np.mean(latencies) if latencies else 0
    throughput_samples_sec = total_samples / sum(latencies) if latencies else 0
    avg_latency_sample = (sum(latencies) / total_samples) * 1000 if total_samples > 0 else 0

    logger.info(f"--- Finished Inference Benchmark ---")
    return {
        "avg_inference_latency_ms_per_sample": avg_latency_sample,
        "avg_inference_throughput_samples_sec": throughput_samples_sec
    }

lora_cfg = LoraConfig(
    r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules,
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)

all_results = {}
trained_models = {}

logger.info("\n===== Starting Baseline Full Fine-tune Training =====")
model_baseline = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for p in model_baseline.parameters(): p.requires_grad = True
logger.info(f"Trainable params (Baseline): {sum(p.numel() for p in model_baseline.parameters() if p.requires_grad):,}")

baseline_res = train_fixed_duration(
    model_baseline, tokenizer, train_texts_full, val_texts_full, num_epochs=num_train_epochs,
    device=device, lr=baseline_lr, batch_size=batch_size, max_length=max_length,
    eval_every=eval_every_steps, run_label="Baseline"
)
logger.info(f"Baseline results: {baseline_res}")
if 'final_ppl' in baseline_res: all_results["Baseline"] = baseline_res
if run and 'final_ppl' in baseline_res: wandb.log({f"Summary/Baseline/{k}": v for k, v in baseline_res.items() if 'log' not in k})
if 'final_ppl' in baseline_res: trained_models["Baseline"] = copy.deepcopy(model_baseline)

del model_baseline; gc.collect(); torch.cuda.empty_cache() if device == torch.device("cuda") else None

logger.info("\n===== Starting Standard LoRA Training =====")
base_model_std = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for param in base_model_std.parameters(): param.requires_grad = False
lora_model_std = get_peft_model(base_model_std, lora_cfg)
lora_model_std.print_trainable_parameters()

lora_std_res = train_fixed_duration(
    lora_model_std, tokenizer, train_texts_full, val_texts_full, num_epochs=num_train_epochs,
    device=device, lr=lora_lr, batch_size=batch_size, max_length=max_length,
    eval_every=eval_every_steps, run_label="LoRA Standard"
)
logger.info(f"Standard LoRA results: {lora_std_res}")
if 'final_ppl' in lora_std_res: all_results["LoRA Standard"] = lora_std_res
if run and 'final_ppl' in lora_std_res: wandb.log({f"Summary/Standard LoRA/{k}": v for k, v in lora_std_res.items() if 'log' not in k})
if 'final_ppl' in lora_std_res: trained_models["LoRA Standard"] = copy.deepcopy(lora_model_std)

del base_model_std, lora_model_std; gc.collect(); torch.cuda.empty_cache() if device == torch.device("cuda") else None

logger.info("\n===== Starting Pre-Training Pruned LoRA Training =====")
base_model_pre_prune = GPT2LMHeadModel.from_pretrained(model_name).to(device)
for param in base_model_pre_prune.parameters(): param.requires_grad = False
lora_model_pre_prune = get_peft_model(base_model_pre_prune, lora_cfg)
lora_model_pre_prune.print_trainable_parameters()

logger.info(f"Applying {pruning_amount:.1%} pruning BEFORE training...")
apply_pruning(lora_model_pre_prune, pruning_amount)
logger.info("Model pruned before training.")

lora_pre_prune_res = train_fixed_duration(
    lora_model_pre_prune, tokenizer, train_texts_full, val_texts_full, num_epochs=num_train_epochs,
    device=device, lr=lora_lr, batch_size=batch_size, max_length=max_length,
    eval_every=eval_every_steps, run_label="LoRA Pre-Pruned"
)
logger.info(f"Pre-Training Pruned LoRA results: {lora_pre_prune_res}")
if 'final_ppl' in lora_pre_prune_res: all_results["LoRA Pre-Pruned"] = lora_pre_prune_res
if run and 'final_ppl' in lora_pre_prune_res: wandb.log({f"Summary/Pre-Pruned LoRA ({pruning_amount:.0%})/{k}": v for k, v in lora_pre_prune_res.items() if 'log' not in k})
if 'final_ppl' in lora_pre_prune_res: trained_models["LoRA Pre-Pruned"] = copy.deepcopy(lora_model_pre_prune)

del base_model_pre_prune, lora_model_pre_prune; gc.collect(); torch.cuda.empty_cache() if device == torch.device("cuda") else None

if run_inference_benchmark:
    logger.info("\n===== Starting Inference Benchmarks =====")
    for label, model in trained_models.items():
        if model is not None:
            logger.info(f"--- Benchmarking Inference for: {label} ---")
            inference_res_fwd = benchmark_inference(
                model, tokenizer, test_texts_full, device,
                batch_size=inference_batch_size, max_length=max_length,
                num_batches=num_inference_batches, generation=False
            )
            inference_res_gen = benchmark_inference(
                model, tokenizer, test_texts_full, device,
                batch_size=inference_batch_size, max_length=max_length,
                num_batches=num_inference_batches // 2,
                generation=True
            )
            all_results[label].update({
                 "fwd_pass_latency_ms": inference_res_fwd["avg_inference_latency_ms_per_sample"],
                 "fwd_pass_throughput": inference_res_fwd["avg_inference_throughput_samples_sec"],
                 "gen_latency_ms": inference_res_gen["avg_inference_latency_ms_per_sample"],
                 "gen_throughput": inference_res_gen["avg_inference_throughput_samples_sec"]
            })
            logger.info(f"Inference Results for {label}: Fwd Latency={inference_res_fwd['avg_inference_latency_ms_per_sample']:.2f}ms, Gen Latency={inference_res_gen['avg_inference_latency_ms_per_sample']:.2f}ms")
            if run:
                wandb.log({
                    f"Summary/{label}/Fwd Pass Latency ms": inference_res_fwd["avg_inference_latency_ms_per_sample"],
                    f"Summary/{label}/Fwd Pass Throughput": inference_res_fwd["avg_inference_throughput_samples_sec"],
                    f"Summary/{label}/Gen Latency ms": inference_res_gen["avg_inference_latency_ms_per_sample"],
                    f"Summary/{label}/Gen Throughput": inference_res_gen["avg_inference_throughput_samples_sec"]
                })
        else:
            logger.warning(f"Skipping inference benchmark for {label} as training failed.")
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache() if device == torch.device("cuda") else None
    trained_models = {}

logger.info("\n===== Generating Learning Curve Plot =====")
plt.figure(figsize=(10, 6))
plot_data_logged = False
for label, results in all_results.items():
    if "steps_log" in results and "ppl_log" in results and results["steps_log"] and results["ppl_log"]:
        steps_axis = np.array(results["steps_log"])
        ppl_values = np.array(results["ppl_log"])
        mask = np.isfinite(ppl_values)
        if np.any(mask):
             plt.plot(steps_axis[mask], ppl_values[mask], marker='.', linestyle='-', label=label)
             plot_data_logged = True
        else:
             logger.warning(f"No valid PPL data points to plot for {label}.")

if plot_data_logged:
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Perplexity")
    plt.title("Perplexity vs. Training Steps")
    plt.legend()
    plt.grid(True)
    min_ppl_overall = min(min(r["ppl_log"]) for r in all_results.values() if r and r.get("ppl_log")) if any(r and r.get("ppl_log") for r in all_results.values()) else 7
    plt.ylim(bottom=max(5, min(min_ppl_overall - 0.5, 7))) 

    if run:
        try:
            wandb.log({"Learning Curve": wandb.Image(plt)})
            logger.info("Learning curve plot logged to Weights & Biases.")
        except Exception as e:
            logger.error(f"Failed to log plot to wandb: {e}")
    plt.close()
else:
     logger.warning("No valid data found to generate learning curve plot.")


logger.info("\n===== Final Benchmark Comparison =====")
results_list = []
indices = []

if "Baseline" in all_results:
    results_list.append(all_results["Baseline"])
    indices.append("Baseline")
if "LoRA Standard" in all_results:
    results_list.append(all_results["LoRA Standard"])
    indices.append("LoRA Standard")
if "LoRA Pre-Pruned" in all_results:
    results_list.append(all_results["LoRA Pre-Pruned"])
    indices.append(f"LoRA Pre-Pruned ({pruning_amount:.0%})")

if results_list:
    df = pd.DataFrame(results_list, index=indices)
    df_display = df.drop(columns=['ppl_log', 'steps_log'], errors='ignore')

    cols_to_rename = {
        "steps": "Total Steps", "total_hours": "Total Hours (h)",
        "peak_mem_mb": "Peak GPU Mem (MB)", "final_ppl": "Final Val PPL"
    }
    if run_inference_benchmark:
        cols_to_rename.update({
            "fwd_pass_latency_ms": "Fwd Latency (ms)", "fwd_pass_throughput": "Fwd TP (samples/s)",
            "gen_latency_ms": "Gen Latency (ms)", "gen_throughput": "Gen TP (samples/s)"
        })

    df_display.rename(columns=cols_to_rename, inplace=True)

    format_map = {
        "Total Steps": '{:,.0f}', "Total Hours (h)": '{:.2f}',
        "Peak GPU Mem (MB)": '{:,.1f}', "Final Val PPL": '{:.2f}',
        "Fwd Latency (ms)": '{:.1f}', "Fwd TP (samples/s)": '{:.1f}',
        "Gen Latency (ms)": '{:.1f}', "Gen TP (samples/s)": '{:.1f}'
    }
    for col, fmt in format_map.items():
        if col in df_display.columns:
            try:
                df_display[col] = df_display[col].map(lambda x: fmt.format(x) if pd.notnull(x) else 'N/A')
            except (TypeError, ValueError):
                 logger.warning(f"Could not format column {col}. Skipping formatting.")


    logger.info("\nComparison DataFrame:\n%s", df_display.to_string())

    if run:
        try:
            df_log = df_display.reset_index().rename(columns={'index': 'Method'})
            wandb.log({"Comparison Table": wandb.Table(dataframe=df_log)})
            logger.info("Comparison table logged to Weights & Biases.")
        except Exception as e:
            logger.error(f"Failed to log DataFrame to Weights & Biases: {e}")

else:
    logger.error("No successful benchmark runs to compare.")

if run:
    wandb.finish()
    logger.info("Weights & Biases run finished.")

logger.info("\n===== Script Finished =====")

