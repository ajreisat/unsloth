# DeepSeek OCR Finetuning Scripts for M1 Mac + Google Colab

Complete workflow for finetuning DeepSeek OCR using your M1 MacBook Pro for dataset preparation and Google Colab for training.

## 📋 Overview

Since Unsloth doesn't support M1 Macs, this workflow splits the process:
1. **M1 Mac**: Prepare dataset from Transkribus
2. **Google Colab**: Train the model with Unsloth (fast & free GPU)
3. **M1 Mac**: Run inference with your finetuned model

## 🗂️ Files in this Folder

| File | Purpose |
|------|---------|
| `export_from_transkribus.py` | Export your Transkribus data to training format |
| `prepare_ocr_dataset_local.py` | Alternative: Prepare dataset from images + text files |
| `colab_training_guide.md` | Complete guide for training on Google Colab |
| `run_inference_m1.py` | Run your finetuned model locally on M1 Mac |

---

## 🚀 Complete Workflow

### Step 1: Export from Transkribus (On M1 Mac)

#### Option A: Export from Transkribus Web/Desktop

1. Open your document collection in Transkribus
2. Go to: **Tools** → **Export Document**
3. Select format: **PAGE XML** (recommended) or **ALTO XML**
4. **✅ Check "Include Images"**
5. Download the export (usually a ZIP file)

#### Option B: Use the Export Script

```bash
# Install dependencies
pip install pillow

# Process Transkribus ZIP export
python export_from_transkribus.py transkribus_export.zip page my_dataset

# Or process an extracted folder
python export_from_transkribus.py /path/to/export_folder page my_dataset
```

This will create:
```
my_dataset/
├── images/
│   ├── image_00000.jpg
│   ├── image_00001.jpg
│   └── ...
├── train.json
├── validation.json
└── dataset_info.json
```

### Step 2: Compress and Upload (On M1 Mac)

```bash
# Compress the dataset
zip -r my_dataset.zip my_dataset

# Upload to Google Drive
# You can use Google Drive desktop app or web interface
# Or upload directly in Colab (next step)
```

### Step 3: Train on Google Colab (Free GPU!)

1. **Open the Colab notebook:**
   - Official: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Deepseek_OCR_(3B).ipynb
   - Or create new notebook and follow `colab_training_guide.md`

2. **Enable GPU:**
   - Runtime → Change runtime type → T4 GPU → Save

3. **Upload your dataset:**
   ```python
   # In Colab, upload your ZIP
   from google.colab import files
   import zipfile

   uploaded = files.upload()  # Select my_dataset.zip

   for filename in uploaded.keys():
       with zipfile.ZipFile(filename, 'r') as zip_ref:
           zip_ref.extractall('.')
   ```

4. **Run the training** (see `colab_training_guide.md` for full code)

5. **Download your finetuned model:**
   ```python
   # In Colab
   !zip -r deepseek-ocr-finetuned.zip deepseek-ocr-finetuned
   from google.colab import files
   files.download('deepseek-ocr-finetuned.zip')
   ```

### Step 4: Run Inference Locally (On M1 Mac)

```bash
# Install dependencies
pip install torch torchvision transformers pillow peft accelerate

# Extract your finetuned model
unzip deepseek-ocr-finetuned.zip

# Run OCR on a single image
python run_inference_m1.py \
  --model ./deepseek-ocr-finetuned \
  --image test_receipt.jpg

# Or batch process a folder
python run_inference_m1.py \
  --model ./deepseek-ocr-finetuned \
  --folder ./my_images/ \
  --output results.txt
```

---

## 📊 Expected Timeline

| Step | Time | Where |
|------|------|-------|
| Export from Transkribus | 5-10 min | M1 Mac |
| Prepare dataset with script | 5-15 min | M1 Mac |
| Upload to Colab | 5-20 min | Upload |
| Training (100 samples, 3 epochs) | 15-30 min | Colab (T4 GPU) |
| Training (1000 samples, 3 epochs) | 30-60 min | Colab (T4 GPU) |
| Download finetuned model | 5-10 min | Download |
| Run inference locally | 10-30s per image | M1 Mac |

**Total for small dataset**: ~1-2 hours
**Total for large dataset**: ~2-3 hours

---

## 💡 Tips and Tricks

### For Better Training Results

1. **More data is better**: Aim for 500-1000+ training samples
2. **Quality over quantity**: Clean, accurate transcriptions matter
3. **Diverse examples**: Include various handwriting styles, layouts, etc.
4. **Validation set**: Always split 10-20% for validation
5. **Multiple epochs**: Try 3-5 epochs for better learning

### For Faster Inference on M1

1. **Use MPS (Metal)**: The script uses it by default
2. **Batch similar-sized images**: Reduces memory overhead
3. **Lower temperature**: Use `--temperature 1.0` for more deterministic output

### Saving Money on Colab

1. **Free T4 GPU**: Sufficient for most tasks
2. **Save checkpoints**: In case of disconnection
3. **Use Google Drive**: Persist data across sessions
4. **Colab Pro ($10/month)**: Get A100 GPU for 2-3x faster training

---

## 🔧 Troubleshooting

### "Out of Memory" on Colab
```python
# Reduce batch size
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

# Or reduce LoRA rank
r = 8  # instead of 16
```

### "Module not found" on M1 Mac
```bash
# Install dependencies
pip install transformers pillow torch peft accelerate
```

### Images not loading in Colab
```python
# Fix image paths in JSON
import json
from pathlib import Path

with open('train.json') as f:
    data = json.load(f)

for sample in data:
    # Make paths relative
    sample['image_path'] = f"./my_dataset/images/{Path(sample['image_path']).name}"

with open('train_fixed.json', 'w') as f:
    json.dump(data, f, indent=2)
```

### Slow inference on M1
This is expected - M1 runs on CPU/MPS which is slower than CUDA GPUs. For production use, consider:
- Using cloud inference (Replicate, HuggingFace Inference API)
- Quantizing the model further
- Processing in batches

---

## 📚 Additional Resources

- **Transkribus**: https://readcoop.eu/transkribus/
- **Unsloth Docs**: https://docs.unsloth.ai/new/deepseek-ocr-run-and-fine-tune
- **DeepSeek OCR Model**: https://huggingface.co/deepseek-ai/deepseek-ocr-3b
- **Google Colab**: https://colab.research.google.com/
- **PyTorch on M1**: https://pytorch.org/get-started/locally/

---

## 📝 Example Dataset Formats

### Transkribus PAGE XML Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<PcGts>
  <Page imageFilename="page_001.jpg">
    <TextRegion>
      <TextLine>
        <Unicode>This is the transcribed text</Unicode>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
```

### Output Training Format (JSON)
```json
[
  {
    "image_id": 0,
    "image_path": "./my_dataset/images/image_00000.jpg",
    "question": "Extract all text from this image.",
    "answer": "This is the transcribed text from the document.",
    "width": 2048,
    "height": 1536
  }
]
```

---

## 🎯 Quick Start Commands

```bash
# 1. Export from Transkribus
python export_from_transkribus.py transkribus_export.zip page my_dataset

# 2. Compress for upload
zip -r my_dataset.zip my_dataset

# 3. Train on Colab (see colab_training_guide.md)

# 4. Run inference locally
python run_inference_m1.py --model ./deepseek-ocr-finetuned --image test.jpg
```

---

## ❓ Need Help?

1. Check `colab_training_guide.md` for detailed Colab instructions
2. Review the Troubleshooting section above
3. Check Unsloth docs: https://docs.unsloth.ai
4. Open an issue on Unsloth GitHub: https://github.com/unslothai/unsloth

---

**Happy Finetuning! 🚀**
