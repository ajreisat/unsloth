# DeepSeek OCR Finetuning on Google Colab

Complete guide for training DeepSeek OCR on Google Colab using your dataset prepared on M1 Mac.

## 🚀 Quick Start

### Step 1: Open the Official Colab Notebook

Click here to open Unsloth's DeepSeek OCR notebook:
```
https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Deepseek_OCR_(3B).ipynb
```

**OR** Create a new notebook and copy the code below.

### Step 2: Enable GPU

1. In Colab, go to: **Runtime** → **Change runtime type**
2. Select: **T4 GPU** (free tier) or **A100** (if you have Colab Pro)
3. Click **Save**

### Step 3: Upload Your Dataset

Choose one of these methods:

#### Option A: Upload from Google Drive (Recommended)
```python
from google.colab import drive
drive.mount('/content/drive')

# If your dataset is in Google Drive at: My Drive/ocr_dataset/
dataset_path = "/content/drive/MyDrive/ocr_dataset"
```

#### Option B: Upload ZIP file directly
```python
from google.colab import files
import zipfile

# Upload your dataset.zip
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')

dataset_path = "./ocr_dataset"  # or your extracted folder name
```

#### Option C: Download from HuggingFace
```python
from datasets import load_dataset

dataset = load_dataset("your-username/your-dataset-name")
```

---

## 📝 Complete Training Code for Colab

Copy this entire code into a new Colab notebook:

```python
# ============================================================================
# CELL 1: Install Unsloth
# ============================================================================
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# ============================================================================
# CELL 2: Import Libraries and Check GPU
# ============================================================================
import torch
from unsloth import FastVisionModel
from datasets import load_dataset, Dataset
from PIL import Image
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
import json
from pathlib import Path

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CELL 3: Upload and Load Your Dataset
# ============================================================================

# METHOD 1: If you uploaded a zip file
from google.colab import files
import zipfile

print("Upload your ocr_dataset.zip file:")
uploaded = files.upload()

for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"Extracted: {filename}")

dataset_path = "./ocr_dataset"  # Change if needed

# OR METHOD 2: From Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# dataset_path = "/content/drive/MyDrive/ocr_dataset"

# ============================================================================
# CELL 4: Load and Format Dataset
# ============================================================================

# Load your prepared dataset
with open(f"{dataset_path}/train.json") as f:
    train_data = json.load(f)

# Optional: Load validation set
try:
    with open(f"{dataset_path}/validation.json") as f:
        val_data = json.load(f)
    has_validation = True
except:
    has_validation = False
    print("No validation set found, using only training data")

print(f"Loaded {len(train_data)} training samples")
if has_validation:
    print(f"Loaded {len(val_data)} validation samples")

# Format for Unsloth
def format_sample(sample):
    """Convert your dataset format to Unsloth format"""
    # Load image
    image = Image.open(sample["image_path"]).convert("RGB")

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert OCR system."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["question"]},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ],
    }

# Create HuggingFace datasets
train_dataset = Dataset.from_list([format_sample(s) for s in train_data])

if has_validation:
    val_dataset = Dataset.from_list([format_sample(s) for s in val_data])

print("✅ Dataset formatted successfully!")

# ============================================================================
# CELL 5: Load DeepSeek OCR Model with 4-bit Quantization
# ============================================================================

max_seq_length = 2048

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "deepseek-ai/deepseek-ocr-3b",
    max_seq_length = max_seq_length,
    load_in_4bit = True,        # 4-bit quantization for memory efficiency
    dtype = None,               # Auto-detect best dtype
)

print("✅ Model loaded successfully!")

# ============================================================================
# CELL 6: Add LoRA Adapters
# ============================================================================

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,      # Finetune vision encoder
    finetune_language_layers = True,    # Finetune language decoder
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 16,                             # LoRA rank (8, 16, 32, 64)
    lora_alpha = 32,
    lora_dropout = 0,                   # 0 is optimized
    bias = "none",                      # "none" is optimized
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state = 3407,
)

print("✅ LoRA adapters added!")

# ============================================================================
# CELL 7: Configure Training
# ============================================================================

FastVisionModel.for_training(model)

# Adjust these settings based on your dataset size and GPU
training_args = SFTConfig(
    # Dataset
    per_device_train_batch_size = 2,      # Increase if you have more VRAM
    gradient_accumulation_steps = 4,       # Effective batch size = 2 * 4 = 8

    # Training duration
    num_train_epochs = 3,                  # Or use max_steps = 100

    # Optimization
    warmup_ratio = 0.03,
    learning_rate = 2e-4,
    weight_decay = 0.01,
    optim = "adamw_8bit",
    lr_scheduler_type = "linear",

    # Performance
    fp16 = not is_bf16_supported(),
    bf16 = is_bf16_supported(),
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},

    # Logging
    logging_steps = 5,

    # Saving
    output_dir = "outputs",
    save_strategy = "epoch",               # Save after each epoch
    save_total_limit = 2,                  # Keep only 2 checkpoints

    # Random seed
    seed = 3407,

    # REQUIRED for vision models
    remove_unused_columns = False,
    dataset_text_field = "",
    dataset_kwargs = {"skip_prepare_dataset": True},
    max_seq_length = max_seq_length,
)

# Create trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = train_dataset,
    args = training_args,
)

# ============================================================================
# CELL 8: Train!
# ============================================================================

print("🚀 Starting training...")
print("=" * 50)

trainer_stats = trainer.train()

print("=" * 50)
print("✅ Training completed!")
print(f"Final loss: {trainer_stats.training_loss:.4f}")

# ============================================================================
# CELL 9: Save Model
# ============================================================================

# Save LoRA adapters (small, ~100-500MB)
output_dir = "deepseek-ocr-finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")

# Optional: Save merged model (full model, ~6GB)
# model.save_pretrained_merged(
#     save_directory = "deepseek-ocr-merged",
#     tokenizer = tokenizer,
# )

# ============================================================================
# CELL 10: Test Inference
# ============================================================================

FastVisionModel.for_inference(model)

# Test on a sample from your validation set
if has_validation:
    test_sample = val_data[0]
    test_image = Image.open(test_sample["image_path"])
    test_question = test_sample["question"]
    ground_truth = test_sample["answer"]
else:
    test_sample = train_data[0]
    test_image = Image.open(test_sample["image_path"])
    test_question = test_sample["question"]
    ground_truth = test_sample["answer"]

# Display test image
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(test_image)
plt.axis('off')
plt.title("Test Image")
plt.show()

# Run inference
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": test_question},
            {"type": "image", "image": test_image},
        ],
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=1.5,
    min_p=0.1,
    use_cache=True,
)

prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=" * 70)
print("GROUND TRUTH:")
print(ground_truth)
print("\n" + "=" * 70)
print("MODEL PREDICTION:")
print(prediction)
print("=" * 70)

# ============================================================================
# CELL 11: Push to HuggingFace Hub (Optional)
# ============================================================================

# First, login to HuggingFace
# !pip install -U huggingface_hub
# from huggingface_hub import login
# login()  # This will prompt for your token

# # Push LoRA adapters
# model.push_to_hub(
#     "your-username/deepseek-ocr-finetuned",
#     token = "hf_...",  # Or omit if you logged in above
# )
# tokenizer.push_to_hub(
#     "your-username/deepseek-ocr-finetuned",
#     token = "hf_...",
# )

# # Or push merged model
# model.push_to_hub_merged(
#     "your-username/deepseek-ocr-finetuned-merged",
#     tokenizer = tokenizer,
#     token = "hf_...",
# )

# ============================================================================
# CELL 12: Download Model to Your Computer
# ============================================================================

# Zip the model folder
!zip -r deepseek-ocr-finetuned.zip {output_dir}

# Download to your M1 Mac
from google.colab import files
files.download('deepseek-ocr-finetuned.zip')

print("✅ Download started! Check your browser's download folder.")
```

---

## 📊 Expected Training Time

On **Free T4 GPU**:
- Small dataset (100 samples, 3 epochs): ~10-15 minutes
- Medium dataset (1000 samples, 3 epochs): ~30-60 minutes
- Large dataset (5000+ samples, 3 epochs): ~2-4 hours

On **Colab Pro A100 GPU**:
- 2-3x faster than T4

---

## 💾 Saving Your Work

### Option 1: Download to M1 Mac
```python
# In Colab
!zip -r my-model.zip deepseek-ocr-finetuned
from google.colab import files
files.download('my-model.zip')
```

### Option 2: Save to Google Drive
```python
!cp -r deepseek-ocr-finetuned /content/drive/MyDrive/
```

### Option 3: Push to HuggingFace (Best for sharing)
```python
from huggingface_hub import login
login()

model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")
```

---

## ⚙️ Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

# Or reduce LoRA rank
r = 8  # instead of 16

# Or reduce max sequence length
max_seq_length = 1024  # instead of 2048
```

### Session Timeout
Google Colab free tier disconnects after ~90 minutes of inactivity:
- Save checkpoints regularly
- Use Google Drive to persist data
- Or upgrade to Colab Pro for longer sessions

### Images Not Loading
Make sure image paths in your JSON are relative to the dataset folder:
```python
# Fix paths if needed
for sample in train_data:
    sample["image_path"] = f"{dataset_path}/{Path(sample['image_path']).name}"
```

---

## 🎯 Next Steps

After training in Colab:
1. Download your finetuned model
2. Use the inference script on your M1 Mac
3. Test on real OCR tasks
4. Share on HuggingFace if you want!

---

## 📚 Additional Resources

- Official Colab Notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Deepseek_OCR_(3B).ipynb
- Unsloth Docs: https://docs.unsloth.ai/new/deepseek-ocr-run-and-fine-tune
- HuggingFace Hub: https://huggingface.co/deepseek-ai/deepseek-ocr-3b
