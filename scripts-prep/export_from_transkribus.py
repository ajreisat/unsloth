#!/usr/bin/env python3
"""
Transkribus to DeepSeek OCR Dataset Converter

This script helps you export data from Transkribus and prepare it for
DeepSeek OCR finetuning with Unsloth.

Supports:
- Transkribus PAGE XML exports
- Transkribus ALTO XML exports
- Transkribus document exports with images
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
import zipfile

class TranskribusExporter:
    """Convert Transkribus exports to DeepSeek OCR training format"""

    def __init__(self, output_dir: str = "ocr_dataset_from_transkribus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        self.data = []

    def extract_text_from_page_xml(self, xml_path: str) -> str:
        """
        Extract text from Transkribus PAGE XML format.

        PAGE XML is the primary export format from Transkribus.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Handle namespace
            ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

            # Try to find namespace in root
            if not ns['page'] in str(root.tag):
                # Try common alternatives
                for possible_ns in [
                    'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15',
                    'http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15',
                    'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15',
                ]:
                    if possible_ns in str(root.tag):
                        ns['page'] = possible_ns
                        break
                else:
                    # No namespace
                    ns = {}

            # Extract all text regions
            text_lines = []

            if ns:
                # With namespace
                for text_region in root.findall('.//page:TextRegion', ns):
                    for text_line in text_region.findall('.//page:TextLine', ns):
                        unicode_elem = text_line.find('.//page:Unicode', ns)
                        if unicode_elem is not None and unicode_elem.text:
                            text_lines.append(unicode_elem.text.strip())
            else:
                # Without namespace
                for text_region in root.findall('.//TextRegion'):
                    for text_line in text_region.findall('.//TextLine'):
                        unicode_elem = text_line.find('.//Unicode')
                        if unicode_elem is not None and unicode_elem.text:
                            text_lines.append(unicode_elem.text.strip())

            # Join lines with newlines
            full_text = '\n'.join(text_lines)
            return full_text

        except Exception as e:
            print(f"⚠️  Error parsing PAGE XML {xml_path}: {e}")
            return ""

    def extract_text_from_alto_xml(self, xml_path: str) -> str:
        """Extract text from Transkribus ALTO XML format"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # ALTO namespace
            ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v2#'}

            # Try to detect namespace
            if not ns['alto'] in str(root.tag):
                # Try alternatives
                for possible_ns in [
                    'http://www.loc.gov/standards/alto/ns-v3#',
                    'http://www.loc.gov/standards/alto/ns-v4#',
                ]:
                    if possible_ns in str(root.tag):
                        ns['alto'] = possible_ns
                        break
                else:
                    ns = {}

            text_lines = []

            if ns:
                for text_line in root.findall('.//alto:TextLine', ns):
                    line_text = []
                    for string_elem in text_line.findall('.//alto:String', ns):
                        content = string_elem.get('CONTENT', '')
                        if content:
                            line_text.append(content)
                    if line_text:
                        text_lines.append(' '.join(line_text))
            else:
                for text_line in root.findall('.//TextLine'):
                    line_text = []
                    for string_elem in text_line.findall('.//String'):
                        content = string_elem.get('CONTENT', '')
                        if content:
                            line_text.append(content)
                    if line_text:
                        text_lines.append(' '.join(line_text))

            full_text = '\n'.join(text_lines)
            return full_text

        except Exception as e:
            print(f"⚠️  Error parsing ALTO XML {xml_path}: {e}")
            return ""

    def process_transkribus_export(self, export_folder: str,
                                   xml_format: str = 'page'):
        """
        Process a Transkribus document export folder.

        Args:
            export_folder: Path to Transkribus export folder
            xml_format: 'page' for PAGE XML or 'alto' for ALTO XML
        """
        export_folder = Path(export_folder)

        if not export_folder.exists():
            print(f"❌ Folder not found: {export_folder}")
            return 0

        print(f"\n📁 Processing Transkribus export: {export_folder}")

        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(export_folder.glob(f"**/*{ext}"))
            image_files.extend(export_folder.glob(f"**/*{ext.upper()}"))

        print(f"   Found {len(image_files)} images")

        xml_ext = '.xml'
        added = 0

        for image_path in sorted(image_files):
            # Find corresponding XML file
            # Try same name with .xml
            xml_path = image_path.with_suffix(xml_ext)

            # Try in page subfolder (common Transkribus structure)
            if not xml_path.exists():
                xml_path = image_path.parent / 'page' / f"{image_path.stem}.xml"

            if not xml_path.exists():
                print(f"⚠️  No XML found for: {image_path.name}")
                continue

            # Extract text based on format
            if xml_format.lower() == 'page':
                text = self.extract_text_from_page_xml(str(xml_path))
            elif xml_format.lower() == 'alto':
                text = self.extract_text_from_alto_xml(str(xml_path))
            else:
                print(f"❌ Unknown XML format: {xml_format}")
                return 0

            if not text:
                print(f"⚠️  No text extracted from: {xml_path.name}")
                continue

            # Copy image to output directory
            try:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                width, height = img.size

                # Save with consistent naming
                new_image_name = f"image_{len(self.data):05d}.jpg"
                new_image_path = self.output_dir / "images" / new_image_name
                img.save(new_image_path, 'JPEG', quality=95)

                self.data.append({
                    "image_id": len(self.data),
                    "image_path": str(new_image_path),
                    "original_image": str(image_path),
                    "original_xml": str(xml_path),
                    "question": "Extract all text from this image.",
                    "answer": text,
                    "width": width,
                    "height": height,
                })

                print(f"✅ Added: {image_path.name} ({len(text)} chars)")
                added += 1

            except Exception as e:
                print(f"⚠️  Error processing {image_path}: {e}")

        print(f"\n✅ Successfully processed {added}/{len(image_files)} pages\n")
        return added

    def process_transkribus_zip(self, zip_path: str, xml_format: str = 'page'):
        """
        Process a Transkribus export ZIP file directly.

        Args:
            zip_path: Path to ZIP file exported from Transkribus
            xml_format: 'page' for PAGE XML or 'alto' for ALTO XML
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            print(f"❌ ZIP file not found: {zip_path}")
            return 0

        print(f"\n📦 Extracting Transkribus ZIP: {zip_path}")

        # Extract to temp folder
        temp_dir = self.output_dir / "_temp_extract"
        temp_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        print(f"   Extracted to: {temp_dir}")

        # Process the extracted folder
        added = self.process_transkribus_export(temp_dir, xml_format)

        # Clean up temp folder
        import shutil
        shutil.rmtree(temp_dir)

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
        """Save dataset in format ready for DeepSeek OCR training"""
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
            "source": "Transkribus",
            "sample_example": self.data[0] if self.data else None
        }

        summary_file = self.output_dir / "dataset_info.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"   ℹ️  Info: {summary_file}")
        print(f"\n✅ Dataset saved to: {self.output_dir.absolute()}")
        print(f"\n📤 Next steps:")
        print(f"   1. Compress: zip -r {self.output_dir.name}.zip {self.output_dir.name}")
        print(f"   2. Upload to Google Drive or Colab")
        print(f"   3. Train with Unsloth!")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of how to use the Transkribus exporter"""

    exporter = TranskribusExporter(output_dir="my_transkribus_dataset")

    # METHOD 1: Process a Transkribus export folder
    exporter.process_transkribus_export(
        export_folder="path/to/transkribus_export_folder",
        xml_format='page'  # or 'alto'
    )

    # METHOD 2: Process a Transkribus ZIP export
    # exporter.process_transkribus_zip(
    #     zip_path="path/to/transkribus_export.zip",
    #     xml_format='page'
    # )

    # Save the dataset
    exporter.save(split=True)


if __name__ == "__main__":
    print("=" * 70)
    print("Transkribus to DeepSeek OCR Dataset Converter")
    print("=" * 70)
    print("\nHow to export from Transkribus:")
    print("1. Open your document collection in Transkribus")
    print("2. Go to: Tools → Export Document")
    print("3. Select export format: PAGE XML or ALTO XML")
    print("4. Include images in the export")
    print("5. Download the export (usually a ZIP file)")
    print("6. Use this script to convert to training format\n")
    print("=" * 70)
    print("\nExample usage:\n")

    # Interactive mode
    import sys

    if len(sys.argv) > 1:
        # Command line mode
        export_path = sys.argv[1]
        xml_format = sys.argv[2] if len(sys.argv) > 2 else 'page'
        output_dir = sys.argv[3] if len(sys.argv) > 3 else 'ocr_dataset_from_transkribus'

        exporter = TranskribusExporter(output_dir=output_dir)

        if export_path.endswith('.zip'):
            exporter.process_transkribus_zip(export_path, xml_format)
        else:
            exporter.process_transkribus_export(export_path, xml_format)

        exporter.save(split=True)
    else:
        print("Usage:")
        print("  python export_from_transkribus.py <export_folder_or_zip> [page|alto] [output_dir]")
        print("\nExample:")
        print("  python export_from_transkribus.py transkribus_export.zip page my_dataset")
        print("  python export_from_transkribus.py /path/to/export_folder alto")
