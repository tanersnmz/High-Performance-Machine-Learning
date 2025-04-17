# GPT-2 Baseline Benchmarking

This directory contains code for benchmarking the standard GPT-2 model with vanilla attention mechanism. The benchmarking system measures various performance metrics including training time, inference latency, memory usage, and throughput.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Weights & Biases (optional but recommended for tracking experiments):
```bash
wandb login
```

## Usage

Run the benchmark:
```bash
python benchmark.py
```

The script will:
1. Load the GPT-2 model and tokenizer
2. Load the Wikitext-2 dataset
3. Run training and inference benchmarks
4. Log metrics to Weights & Biases (if configured)
5. Print detailed performance statistics

## Metrics Tracked

### Training Metrics
- Training time per batch
- GPU memory usage
- Throughput (samples/second)
- Standard deviations for all metrics

### Inference Metrics
- Inference latency per batch
- GPU memory usage
- Throughput (samples/second)
- Standard deviations for all metrics

## Configuration

The benchmark can be configured by modifying the `GPT2Benchmark` class initialization parameters:
- `model_name`: GPT-2 model variant (default: "gpt2")
- `batch_size`: Batch size for training/inference (default: 8)
- `max_length`: Maximum sequence length (default: 128)

## Results

Results are logged to:
1. Console output with detailed statistics
2. Weights & Biases dashboard (if configured)

The results include mean and standard deviation for all metrics, providing a comprehensive view of the model's performance characteristics. 