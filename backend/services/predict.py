#!/usr/bin/env python3
"""
============================================================
  Model Prediction Script
  Simple inference on new images using handwriting_service
  
  Usage:
  python predict.py path/to/image.png
  python predict.py path/to/folder/
============================================================
"""

import sys
from pathlib import Path
import numpy as np

# Import from handwriting service (centralized model loading)
from backend.services.handwriting_service import predict_image, ensure_loaded

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
CLASS_NAMES = ["healthy", "patient"]
CONFIDENCE_THRESHOLD = 0.5


# ══════════════════════════════════════════════════════════════════════
#  PREDICTION FROM FILE
# ══════════════════════════════════════════════════════════════════════

def predict_single_image(image_path):
    """Predict on a single image file.
    
    Uses handwriting_service.predict_image internally.
    
    Args:
        image_path: Path to image file
    
    Returns:
        dict with prediction results
    """
    try:
        # Read image file
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        # Use handwriting_service for prediction
        result = predict_image(img_bytes)
        
        return {
            'image': str(image_path),
            'class': result['label'],
            'confidence': result['prob_pos'],
            'raw_score': result['prob_pos'],
            'pred_class': result['pred']
        }
        
    except Exception as e:
        return {
            'image': str(image_path),
            'error': str(e)
        }


# ══════════════════════════════════════════════════════════════════════
#  MAIN CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main prediction pipeline."""
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <image_or_folder>\n")
        print("Examples:")
        print("  python predict.py test_image.png")
        print("  python predict.py test_images/")
        print()
        sys.exit(1)
    
    # Load model first to catch errors early
    print("\n" + "=" * 70)
    print("  PARKINSON'S DETECTION - PREDICTION")
    print("=" * 70 + "\n")
    
    print("Loading model...")
    try:
        ensure_loaded()
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}\n")
        sys.exit(1)
    
    # Get input paths
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"[ERROR] Path not found: {input_path}\n")
        sys.exit(1)
    
    # Collect image files
    image_files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in IMG_EXTENSIONS:
            image_files = [input_path]
        else:
            print(f"[ERROR] Not an image file: {input_path}\n")
            sys.exit(1)
    
    elif input_path.is_dir():
        image_files = [f for f in input_path.rglob("*") 
                      if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS]
        if not image_files:
            print(f"[ERROR] No images found in: {input_path}\n")
            sys.exit(1)
    
    # Make predictions
    print(f"Processing {len(image_files)} image(s)...\n")
    print("-" * 70)
    
    results = []
    for idx, image_file in enumerate(image_files, 1):
        result = predict_single_image(image_file)
        results.append(result)
        
        if 'error' in result:
            print(f"[{idx}] ✗ {image_file.name}")
            print(f"      Error: {result['error']}")
        else:
            confidence_pct = result['confidence'] * 100
            print(f"[{idx}] {image_file.name}")
            print(f"      Prediction: {result['class'].upper()}")
            print(f"      Confidence: {confidence_pct:.1f}%")
            print(f"      Score: {result['raw_score']:.4f}")
        print()
    
    # Summary
    print("-" * 70)
    successful = sum(1 for r in results if 'error' not in r)
    print(f"✓ Processed: {successful}/{len(image_files)} images\n")
    
    if successful == 0:
        print("[ERROR] No successful predictions\n")
        sys.exit(1)
    
    # Classification distribution
    healthy_count = sum(1 for r in results if 'error' not in r and r['class'] == 'healthy')
    patient_count = sum(1 for r in results if 'error' not in r and r['class'] == 'patient')
    
    print(f"Results Distribution:")
    print(f"  Healthy: {healthy_count}")
    print(f"  Patient: {patient_count}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
