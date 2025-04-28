# GPT-2 Benchmarking Suite

This repository provides a comprehensive benchmarking pipeline for the GPT-2 language model, supporting optional FlashAttention, quantization (FP16), and knowledge distillation (planned). It measures training and inference performance across multiple configurations.

---

## 🔧 Setup

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Login to Weights & Biases:**
   ```bash
   wandb login
   ```

---

## 🚀 Usage

To run the benchmark:
```bash
python benchmark.py
```

The script will:
- Load the GPT-2 model and tokenizer
- Load the WikiText-2 dataset
- Run training and inference benchmarks
- Log metrics to Weights & Biases (if enabled)
- Print detailed performance summaries to console

---

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

---

## ⚙️ Configuration

You can modify benchmarking parameters in the `ModelConfig` class:

```python
ModelConfig(
    model_name="gpt2",
    batch_size=8,
    max_length=128,
    use_flash_attention=False,
    use_quantization=False
)
```

---

## 📁 Output

Results are logged to:
- **Console**: Clean summary of each experiment
- **WandB Dashboard** (optional): Rich tracking with graphs and tables

---

## 🧠 Features

- ✅ Supports FlashAttention (Hugging Face native config)
- ✅ FP16 quantization (`model.half()`)
- 🔜 Knowledge distillation (coming after mid-point)
- 📈 Full profiling with wandb + PyTorch memory tracking

---

## 📌 Notes

- Current quantization is a naive FP16 cast using `model.half()`. We plan to implement proper post-training or QAT methods later.
- FlashAttention shows stronger benefits at higher batch sizes and longer sequences.
