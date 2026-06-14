"""Test script for OpenBMB MiniCPM-V optimizations."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import _image_to_base64, _preprocess_images_for_medicine, _extract_json
from PIL import Image
import io

def test_image_preprocessing():
    """Test image preprocessing for medicine labels."""
    print("Testing image preprocessing...")
    
    # Create a test image
    test_img = Image.new('RGB', (2000, 1500), color='white')
    
    # Test _image_to_base64 with resolution optimization
    b64 = _image_to_base64(test_img)
    assert len(b64) > 0, "Base64 encoding failed"
    print("[PASS] _image_to_base64 works with resolution optimization")
    
    # Test _preprocess_images_for_medicine
    processed = _preprocess_images_for_medicine([test_img])
    assert len(processed) == 1, "Preprocessing failed"
    assert processed[0].width <= 1344, "Resolution not optimized"
    print("[PASS] _preprocess_images_for_medicine works")
    
    return True

def test_json_extraction():
    """Test JSON extraction with various formats."""
    print("\nTesting JSON extraction...")
    
    # Test markdown code block
    text1 = '```json\n{"name": "test"}\n```'
    result1 = _extract_json(text1)
    assert result1 == {"name": "test"}, f"Failed: {result1}"
    print("[PASS] Markdown code block extraction works")
    
    # Test direct JSON
    text2 = '{"name": "test", "value": 123}'
    result2 = _extract_json(text2)
    assert result2 == {"name": "test", "value": 123}, f"Failed: {result2}"
    print("[PASS] Direct JSON extraction works")
    
    # Test JSON in text
    text3 = 'Here is the result: {"name": "test"} end.'
    result3 = _extract_json(text3)
    assert result3 == {"name": "test"}, f"Failed: {result3}"
    print("[PASS] JSON in text extraction works")
    
    return True

def test_backend_optimization():
    """Test backend optimization parameters."""
    print("\nTesting backend optimization...")
    
    # Read backend.py and check for optimization parameters
    with open('backend.py', 'r') as f:
        content = f.read()
    
    assert 'max_slice_nums' in content, "max_slice_nums not found in backend.py"
    assert 'use_image_id' in content, "use_image_id not found in backend.py"
    assert 'max-num-seqs' in content, "max-num-seqs not found in backend.py"
    assert 'block-size' in content, "block-size not found in backend.py"
    print("[PASS] Backend optimization parameters present")
    
    return True

if __name__ == "__main__":
    try:
        test_image_preprocessing()
        test_json_extraction()
        test_backend_optimization()
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        sys.exit(1)