#!/usr/bin/env python3
"""
Dataset Preparation Script for M1 Mac
Prepare your OCR dataset locally before uploading to Colab for training.

This script:
1. Organizes your images and text labels
2. Validates the data
3. Creates a dataset file ready for Colab
4. Saves to datasets format that can be uploaded to HuggingFace or Google Drive
"""

import os
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict
import pandas as pd

class OCRDatasetPreparer:
    """Prepare OCR datasets for DeepSeek OCR finetuning"""

    def __init__(self, output_dir: str = "ocr_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data = []

    def add_sample(self, image_path: str, ground_truth_text: str,
                   question: str = "Extract all text from this image."):
        """
        Add a single OCR training sample.

        Args:
            image_path: Path to the image file
            ground_truth_text: The correct OCR output (what the model should learn)
            question: The prompt to use (default is generic OCR extraction)
        """
        image_path = Path(image_path)

        # Validate image exists and can be opened
        if not image_path.exists():
            print(f"⚠️  Warning: Image not found: {image_path}")
            return False

        try:
            img = Image.open(image_path)
            width, height = img.size

            # Copy image to dataset directory with consistent naming
            new_image_name = f"image_{len(self.data):05d}{image_path.suffix}"
            new_image_path = self.output_dir / "images" / new_image_name
            new_image_path.parent.mkdir(exist_ok=True)

            # Save image (converts to RGB if needed)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(new_image_path)

            self.data.append({
                "image_id": len(self.data),
                "image_path": str(new_image_path),
                "original_path": str(image_path),
                "question": question,
                "answer": ground_truth_text,
                "width": width,
                "height": height,
            })

            print(f"✅ Added: {image_path.name} ({width}x{height})")
            return True

        except Exception as e:
            print(f"⚠️  Error processing {image_path}: {e}")
            return False

    def add_from_folder(self, images_folder: str, texts_folder: str = None,
                       text_extension: str = ".txt"):
        """
        Batch add samples where images and text files have matching names.

        Args:
            images_folder: Folder containing images
            texts_folder: Folder containing text files (if None, uses same as images_folder)
            text_extension: Extension of text files (default: .txt)
        """
        images_folder = Path(images_folder)
        texts_folder = Path(texts_folder) if texts_folder else images_folder

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_folder.glob(f"*{ext}"))
            image_files.extend(images_folder.glob(f"*{ext.upper()}"))

        print(f"\n📁 Processing folder: {images_folder}")
        print(f"   Found {len(image_files)} images")

        added = 0
        for image_path in sorted(image_files):
            # Find corresponding text file
            text_path = texts_folder / f"{image_path.stem}{text_extension}"

            if not text_path.exists():
                print(f"⚠️  No text file for: {image_path.name}")
                continue

            try:
                ground_truth = text_path.read_text(encoding='utf-8').strip()
                if self.add_sample(image_path, ground_truth):
                    added += 1
            except Exception as e:
                print(f"⚠️  Error reading {text_path}: {e}")

        print(f"\n✅ Successfully added {added}/{len(image_files)} samples\n")
        return added

    def add_from_csv(self, csv_path: str, image_col: str = "image_path",
                     text_col: str = "text", question_col: str = None):
        """
        Add samples from a CSV file.

        Args:
            csv_path: Path to CSV file
            image_col: Column name containing image paths
            text_col: Column name containing ground truth text
            question_col: Optional column for custom questions
        """
        df = pd.read_csv(csv_path)
        print(f"\n📊 Loading from CSV: {csv_path}")
        print(f"   Found {len(df)} rows")

        added = 0
        for idx, row in df.iterrows():
            image_path = row[image_col]
            ground_truth = row[text_col]
            question = row[question_col] if question_col and question_col in df.columns else None

            if self.add_sample(image_path, ground_truth, question):
                added += 1

        print(f"\n✅ Successfully added {added}/{len(df)} samples\n")
        return added

    def add_from_json(self, json_path: str):
        """
        Add samples from a JSON file.
        Expected format: [{"image_path": "...", "text": "...", "question": "..."}, ...]
        """
        with open(json_path) as f:
            data = json.load(f)

        print(f"\n📄 Loading from JSON: {json_path}")
        print(f"   Found {len(data)} samples")

        added = 0
        for item in data:
            if self.add_sample(
                item['image_path'],
                item['text'],
                item.get('question', "Extract all text from this image.")
            ):
                added += 1

        print(f"\n✅ Successfully added {added}/{len(data)} samples\n")
        return added

    def split_dataset(self, train_ratio: float = 0.9):
        """Split dataset into train and validation sets"""
        import random
        random.seed(42)

        indices = list(range(len(self.data)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = set(indices[:split_idx])

        train_data = [self.data[i] for i in range(len(self.data)) if i in train_indices]
        val_data = [self.data[i] for i in range(len(self.data)) if i not in train_indices]

        return train_data, val_data

    def save(self, split: bool = True):
        """Save dataset to disk in HuggingFace datasets compatible format"""
        if not self.data:
            print("❌ No data to save!")
            return

        print(f"\n💾 Saving dataset...")
        print(f"   Total samples: {len(self.data)}")

        if split and len(self.data) > 10:
            train_data, val_data = self.split_dataset()

            # Save train split
            train_file = self.output_dir / "train.json"
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            print(f"   📝 Train: {len(train_data)} samples -> {train_file}")

            # Save validation split
            val_file = self.output_dir / "validation.json"
            with open(val_file, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)
            print(f"   📝 Validation: {len(val_data)} samples -> {val_file}")
        else:
            # Save all as train
            train_file = self.output_dir / "train.json"
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            print(f"   📝 All data: {len(self.data)} samples -> {train_file}")

        # Save summary
        summary = {
            "total_samples": len(self.data),
            "dataset_dir": str(self.output_dir.absolute()),
            "sample_example": self.data[0] if self.data else None
        }

        summary_file = self.output_dir / "dataset_info.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"   ℹ️  Info: {summary_file}")
        print(f"\n✅ Dataset saved to: {self.output_dir.absolute()}")
        print(f"\n📤 Next steps:")
        print(f"   1. Compress the folder: zip -r {self.output_dir.name}.zip {self.output_dir.name}")
        print(f"   2. Upload to Google Drive or HuggingFace")
        print(f"   3. Use in Colab for training!")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the dataset preparer"""

    # Initialize preparer
    preparer = OCRDatasetPreparer(output_dir="my_ocr_dataset")

    # Example 1: Add individual samples
    preparer.add_sample(
        image_path="receipts/receipt_001.jpg",
        ground_truth_text="Store: ABC Mart\nDate: 2024-01-15\nTotal: $45.99"
    )

    # Example 2: Add from folder (images and .txt files with same names)
    # preparer.add_from_folder(
    #     images_folder="my_images/",
    #     texts_folder="my_labels/",
    #     text_extension=".txt"
    # )

    # Example 3: Add from CSV
    # preparer.add_from_csv(
    #     csv_path="dataset.csv",
    #     image_col="image_path",
    #     text_col="ocr_text"
    # )

    # Example 4: Add from JSON
    # preparer.add_from_json("dataset.json")

    # Save the dataset
    preparer.save(split=True)


if __name__ == "__main__":
    print("=" * 70)
    print("OCR Dataset Preparation Tool for DeepSeek OCR Finetuning")
    print("=" * 70)
    print("\nThis script helps you prepare your OCR dataset locally on M1 Mac,")
    print("then upload to Google Colab for fast training with Unsloth.\n")

    # Run example
    print("Running example usage...\n")
    example_usage()

    print("\n" + "=" * 70)
    print("Customize the example_usage() function for your specific dataset!")
    print("=" * 70)
