# High-Performance-Learning

## wav2vec

### aihwkit
PerformanceEvaluator.run_comparison(num_samples=20) can run successfully. However, the analog model is slower than original model. ~

### Quantization
The project implements Post-Training Quantization (PTQ) for speech recognition models to improve inference efficiency.

#### Quantization Implementation
- **Method**: Post-Training Quantization (PTQ)
- **Framework**: Custom implementation with PyTorch hooks
- **Quantization Bits**: 16-bit weights and 16-bit activations
- **Calibration**: Uses a subset of the dataset to determine optimal quantization parameters

#### Performance Results
Quantizing a Wav2Vec2 model achieves:
- **Accuracy**: Minimal WER (Word Error Rate) increase compared to full-precision model
- **Model Size**: ~50% reduction in model size (from 32-bit to 16-bit precision)
- **Inference Speed**: Comparable to the original model, with minimal overhead

#### Text Prediction Comparison
Examples showing the accuracy of quantized model compared to the original:

| Sample | Reference | Original Model | Quantized Model (16-bit) |
|--------|-----------|---------------|--------------------------|
| 1 | "THE OLD MAN WAS BENT INTO A CAPITAL C" | "THE OLD MAN WAS BENT INTO A CAPITAL C" | "THE OLD MAN WAS BENT INTO A CAPITAL C" |
| 2 | "HE STRUCK A MATCH AND LIGHTED HIS CIGARETTE" | "HE STRUCK A MATCH AND LIGHTED HIS CIGARETTE" | "HE STRUCK A MATCH AND LIGHTED HIS CIGARETTE" |

INFO:__main__:Original Model:
INFO:__main__:  WER: 0.1610
INFO:__main__:  Avg Inference Time: 0.1284s
INFO:__main__:
Quantized Model:
INFO:__main__:  WER: 0.1610
INFO:__main__:  Avg Inference Time: 0.0132s
INFO:__main__:
Differences:
INFO:__main__:  WER Difference: 0.0000
INFO:__main__:  Time Difference: -0.1151s
INFO:__main__:  Speedup: 9.70x