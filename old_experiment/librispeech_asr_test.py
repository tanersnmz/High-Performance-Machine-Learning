# from datasets import load_dataset


# lib = load_dataset(
#                 "librispeech_asr",
#                 "clean",
#                 split="test",
#                 streaming=True
#             )

# from itertools import islice

# for example in islice(lib, 5):
#     print("Audio path:", example["file"])
#     print("Transcript:", example["text"])
#     print("Speaker ID:", example["speaker_id"])
#     print("===")

# from IPython.display import Audio
# Audio(example["audio"]["array"], rate=16000)
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# Step 1: Load pretrained Wav2Vec2 model + processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# Step 2: Load a sample from LibriSpeech (use streaming for memory efficiency)
dataset = load_dataset("ç", "clean", split="test", streaming=True)
sample = next(iter(dataset))

# Step 3: Preprocess audio
inputs = processor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True)

# Step 4: Inference
with torch.no_grad():
    logits = model(**inputs).logits

# Step 5: Decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("📢 原始文本:", sample["text"])
print("🧠 模型转录:", transcription)
