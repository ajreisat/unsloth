#!/usr/bin/env python3
"""
DeepSeek OCR Inference on M1 Mac

Run your finetuned DeepSeek OCR model locally on M1 MacBook Pro.

NOTE: This uses standard PyTorch (not Unsloth) since Unsloth doesn't support M1.
The model will run on CPU or MPS (Metal Performance Shaders).
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import argparse
from pathlib import Path
import time

class DeepSeekOCR:
    """Run DeepSeek OCR inference on M1 Mac"""

    def __init__(self, model_path: str, use_mps: bool = True):
        """
        Initialize the OCR model.

        Args:
            model_path: Path to your finetuned model or HuggingFace model ID
            use_mps: Use Apple Metal (MPS) if available, otherwise CPU
        """
        print("🚀 Loading DeepSeek OCR model...")
        print(f"   Model: {model_path}")

        # Determine device
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
            print(f"   Device: Apple Metal (MPS)")
        else:
            self.device = "cpu"
            print(f"   Device: CPU")

        # Load model and processor
        start_time = time.time()

        try:
            # Try loading as a finetuned model with adapter
            from peft import PeftModel, PeftConfig

            # Load base model
            base_model_name = "deepseek-ai/deepseek-ocr-3b"
            print(f"   Loading base model: {base_model_name}")

            self.model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                low_cpu_mem_usage=True,
            )

            # Load LoRA adapter
            print(f"   Loading LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()  # Merge adapter into base model

            self.processor = AutoProcessor.from_pretrained(model_path)
            print("   ✅ Loaded finetuned model with LoRA adapter")

        except Exception as e:
            # Try loading as a merged model
            print(f"   Loading as merged model...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("   ✅ Loaded merged model")

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        load_time = time.time() - start_time
        print(f"   ⏱️  Loaded in {load_time:.1f}s\n")

    def extract_text(self, image_path: str,
                    prompt: str = "Extract all text from this image.",
                    max_new_tokens: int = 1024,
                    temperature: float = 1.5,
                    min_p: float = 0.1) -> str:
        """
        Extract text from an image using OCR.

        Args:
            image_path: Path to image file
            prompt: Instruction prompt for the model
            max_new_tokens: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            min_p: Minimum probability threshold

        Returns:
            Extracted text as string
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        print(f"📄 Processing: {Path(image_path).name}")
        print(f"   Size: {image.size[0]}x{image.size[1]}")

        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]

        # Process with model
        start_time = time.time()

        # Create inputs
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                use_cache=True,
            )

        # Decode
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer (remove prompt)
        if "<|assistant|>" in generated_text:
            answer = generated_text.split("<|assistant|>")[-1].strip()
        else:
            answer = generated_text

        inference_time = time.time() - start_time
        print(f"   ⏱️  Processed in {inference_time:.1f}s\n")

        return answer

    def batch_process(self, image_folder: str, output_file: str = "ocr_results.txt"):
        """
        Process all images in a folder.

        Args:
            image_folder: Folder containing images
            output_file: Output text file for results
        """
        image_folder = Path(image_folder)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f"*{ext}"))
            image_files.extend(image_folder.glob(f"*{ext.upper()}"))

        print(f"📁 Found {len(image_files)} images in {image_folder}\n")

        results = []
        for i, image_path in enumerate(sorted(image_files), 1):
            print(f"[{i}/{len(image_files)}]")
            try:
                text = self.extract_text(str(image_path))
                results.append({
                    "image": image_path.name,
                    "text": text
                })
                print(f"✅ Result:\n{text}\n")
                print("-" * 70 + "\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")
                results.append({
                    "image": image_path.name,
                    "text": f"ERROR: {e}"
                })

        # Save results
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"=== {result['image']} ===\n")
                f.write(f"{result['text']}\n\n")

        print(f"💾 Results saved to: {output_path.absolute()}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run DeepSeek OCR inference on M1 Mac",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image OCR
  python run_inference_m1.py --model ./deepseek-ocr-finetuned --image receipt.jpg

  # Batch process folder
  python run_inference_m1.py --model ./deepseek-ocr-finetuned --folder ./images/

  # Use base model (no finetuning)
  python run_inference_m1.py --model deepseek-ai/deepseek-ocr-3b --image test.jpg

  # Custom prompt
  python run_inference_m1.py --model ./my-model --image form.jpg --prompt "Extract the form fields"
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to finetuned model folder or HuggingFace model ID"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image to process"
    )

    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing images to batch process"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Extract all text from this image.",
        help="Custom prompt for OCR"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ocr_results.txt",
        help="Output file for batch processing results"
    )

    parser.add_argument(
        "--no-mps",
        action="store_true",
        help="Disable MPS (Metal) and use CPU only"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.folder:
        parser.error("Must specify either --image or --folder")

    # Initialize model
    ocr = DeepSeekOCR(
        model_path=args.model,
        use_mps=not args.no_mps
    )

    # Process
    if args.image:
        # Single image
        result = ocr.extract_text(
            args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens
        )
        print("=" * 70)
        print("RESULT:")
        print("=" * 70)
        print(result)
        print("=" * 70)

    elif args.folder:
        # Batch process
        ocr.batch_process(args.folder, args.output)


if __name__ == "__main__":
    print("=" * 70)
    print("DeepSeek OCR Inference for M1 Mac")
    print("=" * 70)
    print()

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
