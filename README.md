# High-Performance GPT2 Optimization

This repository provides a comprehensive optimization framework for the GPT-2 language model, integrating multiple model compression and acceleration techniques including Knowledge Distillation, LoRA (Low-Rank Adaptation), and Weight Pruning, alongside FlashAttention and quantization support. It systematically measures training and inference performance across different configurations.

## 🧪 Experimental Setup

* **Python version**: `3.10.15`
* **PyTorch version**: `2.4.1+cu121`
* **CUDA available**: `True`
* **CUDA version**: `12.1`
* **GPU model**: `NVIDIA Tesla T4`
* **Number of GPUs**: `1`
* **Available GPU memory**: `15.64 GB`
* **Device used**: `cuda`

## 📖 Project Overview

This repository investigates **how far we can push GPT-2 on affordable GPUs** by *stacking* several optimization methods into a single training & inference pipeline.

| Stage                              | What we do                                                              | Why it helps                                                           |
| ---------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **1 · Knowledge Distillation**     | Train a compact *student* GPT-2 on the soft-labels of a larger teacher. | Shrinks the base model while keeping perplexity close to the original. |
| **2 · LoRA (Low-Rank Adaptation)** | Add \~0.25 % trainable rank-8 adapters to frozen weights.               | Fine-tunes with **≈100× fewer** updated parameters & lower VRAM.       |
| **3 · 80 % Adapter Pruning**       | L1-prune the LoRA weights before fine-tuning.                           | Extra sparsity ⇒ smaller checkpoint; no extra compute at runtime.      |
| **4 · INT4 NF4 Quantization**      | Load weights in 4-bit (bitsandbytes) with double-quant + FP16 compute.  | Cuts model memory by **4-5×** and speeds up GEMM kernels.              |
| **5 · FlashAttention 2**           | Replace vanilla soft-max attention with bandwidth-optimal kernels.      | 2-3× faster attention + lower activation RAM.                          |

### End-to-end Gains (Distilled → LoRA → Prune → INT4 + Flash)

* **Training VRAM:** ↓ 66 %
* **Inference VRAM:** ↓ 27 %
* **Inference Forward throughput:** ↑ 3× | **Inference Generation throughput:** ↑ 1.5×
* **Inference Latency:** -62 % (fwd) | -30 % (gen)
* **Quality hit:** Δ PPL ≈ +0.9 (7.57 → 8.46)

> **Goal:** put *full-pipeline* GPT-2 fine-tuning & serving on 6 GB laptop-class GPUs without noticeable quality loss.

## 🔧 Setup

1. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Login to Weights & Biases:**

   ```bash
   wandb login
   ```

## 📁 Project Structure

~~~plaintext
├── Distillation-with-Lora-and-Pruning.ipynb  # Knowledge Distillation, LoRA and Pruning experiments
├── gpt2base-lora and pruning.ipynb          # LoRA and Pruning experiments for base model GPT2, just for experiment purposes, not used in final code
├── Quantization.ipynb                       # Quantization experiments
├── Quantization-with-Flash-Attention.ipynb  # Quantization with FlashAttention
├── gpt2-flashattention.py                   # FlashAttention implementation, just for experiment purposes, not used in final code
├── gpt2-flashAttention-newmetrics.py       # Enhanced metrics for FlashAttention, just for experiment purposes, not used in final code
├── distillation/                           # Knowledge distillation implementations
│   ├── model_comparison.py                 # Compare different model configurations
│   ├── run_distillation_experiments.py     # Run distillation experiments
│   └── distillation_eval.py                # Evaluate distilled models
├── old_experiment/                         # Legacy experiments on aihwkit (ignore it)
├── requirements.txt                        # Project dependencies
├── LICENSE                                 # License information
└── README.md   
~~~

## 🚀 Usage

### Execution Sequence

1. First, run the distillation experiments to create the distilled model:

   ~~~bash
   python distillation/run\_distillation\_experiments.py
   ~~~

   This script performs knowledge distillation from GPT-2-medium (teacher) to GPT-2 (student), creating checkpoints at specified epochs (5, 10, 15, 20, 25, 30, 35, 40, 45, 50). The distilled model will be saved in the distilled\_model directory.
2. Next, compare the distilled model with GPT-2 and GPT-2-medium:

   ~~~bash
   python distillation/model_comparison.py
   ~~~

   This will evaluate the performance of all three models (GPT-2, GPT-2-medium, and the distilled model) and generate comparison metrics including loss, perplexity, inference time, and memory usage.
3. For LoRA and pruning experiments, run the Jupyter notebook:

   ~~~bash
   jupyter notebook Distillation-with-Lora-and-Pruning.ipynb
   ~~~

   This notebook demonstrates how to apply LoRA and pruning techniques to the distilled model for further optimization.
4. FlashAttention and quantization experiments:

   ```bash

   jupyter notebook Quantization-with-Flash-Attention.ipynb
   ```

## 📊 Metrics Tracked

| Phase         | Metric              | What it Measures                                 | Unit           | Example WandB Key            |
| ------------- | ------------------- | ------------------------------------------------ | -------------- | ---------------------------- |
| **Training**  | **Time / batch**    | Wall-clock duration for one optimisation step    | seconds        | `batch_training_time`        |
|               | **GPU VRAM**        | Allocated device memory during the step *(peak)* | MB             | `batch_memory_usage`         |
|               | **Throughput**      | Samples processed per second                     | samples / s    | `batch_throughput`           |
|               | **Loss**            | Cross-entropy training loss                      | –              | `batch_loss`                 |
|               | **Perplexity**      | exp(loss) – readability of model outputs         | –              | `batch_perplexity`           |
|               | **± StdDev**        | Variation across all recorded batches            | same as metric | calculated offline           |
| **Inference** | **Latency / batch** | End-to-end forward (or generate) time            | seconds        | `batch_inference_time`       |
|               | **GPU VRAM**        | Allocated device memory during the pass          | MB             | `batch_inference_memory`     |
|               | **Throughput**      | Samples served per second                        | samples / s    | `batch_inference_throughput` |
|               | **Perplexity**      | exp(loss) on validation set                      | –              | `batch_inference_perplexity` |
|               | **± StdDev**        | Variation across all recorded batches            | same as metric | calculated offline           |


## 📈 Experimental Results

### View Results Online

All experiment results are logged to Weights & Biases for detailed analysis and visualization. You can access the results at the following URLs:

1. The impact about Batch Size and Sequnence Length: https://wandb.ai/hpml_final_project/model-benchmark
2. Distillation Experiments: https://wandb.ai/hpml_final_project/gpt2-progressive-distillation
3. Distilled Model, GPT2 and GPT2 Medium Comparison: https://wandb.ai/hpml_final_project/gpt2-model-comparison
4. LoRA and Pruning Experiments: https://wandb.ai/hpml_final_project/lora-pruning-comparison-dstill-3
5. Flash-Attention Experiments：

## 💾 Model Weights

### Pre-trained Models

All our optimized models are available for download:

1. Distilled Model (GPT2 distilled from GPT2-Medium): https://drive.google.com/drive/folders/1Uf_C71Goa9yB8zThMvuhkkE2AU11EUTX?usp=drive_link
2. Distilled Model with LoRA and Pruning:
