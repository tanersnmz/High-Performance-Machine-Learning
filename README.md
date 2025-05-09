# High-Performance GPT2 Optimization

This repository provides a comprehensive optimization framework for the GPT-2 language model, integrating multiple model compression and acceleration techniques including Knowledge Distillation, LoRA (Low-Rank Adaptation), and Weight Pruning, alongside FlashAttention and quantization support. It systematically measures training and inference performance across different configurations.This repository provides a comprehensive optimization framework for the GPT-2 language model, integrating multiple model compression and acceleration techniques including Knowledge Distillation, LoRA (Low-Rank Adaptation), and Weight Pruning, alongside FlashAttention and quantization support. It systematically measures training and inference performance across different configurations.This repository provides a comprehensive benchmarking pipeline for the GPT-2 language model, supporting optional FlashAttention, quantization (FP16), and knowledge distillation (planned). It measures training and inference performance across multiple configurations.

## 📖 Project Overview

This project researches and implements the combined effects of the following model optimization techniques:

1. Knowledge Distillation: Transferring knowledge from larger "teacher" models to smaller "student" models
2. LoRA (Low-Rank Adaptation): Inserting a small number of trainable parameters through low-rank matrix decomposition while keeping most model weights frozen
3. Weight Pruning: Identifying and removing unimportant parameter connections to reduce model size and computational complexity
4. FlashAttention: Optimized implementation to accelerate attention mechanism computations
5. Quantization: Reducing weight precision from FP32/ FP16 to INT4

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
├── Lora and Pruning.ipynb                   # LoRA and Pruning experiments
├── Quantization.ipynb                       # Quantization experiments
├── Quantization-with-Flash-Attention.ipynb  # Quantization with FlashAttention
├── gpt2-lora.py                            # LoRA implementation for GPT-2
├── gp2-flashattention.py                   # FlashAttention implementation
├── gpt2-flashAttention-newmetrics.py       # Enhanced metrics for FlashAttention
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

   jupyte notebook Quantization-with-Flash-Attention.ipynb
   ```

## 📊 Metrics Tracked

### Training

- Time per batch (seconds)
- GPU memory usage (MB)
- Throughput (samples/second)
- Standard deviation for all metrics

### Inference

- Latency per batch (seconds)
- GPU memory usage (MB)
- Throughput (samples/second)
- Standard deviation for all metrics

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
