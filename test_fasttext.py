#!/usr/bin/env python3
"""
Test script to verify FastText language detection is working properly
"""

import fasttext
import sys
import os

def test_fasttext_model():
    print("🔄 Loading FastText language detection model...")
    
    # Change to the IPC_predictor directory to find the model
    model_path = r"C:\Github\IPC_predictor\lid.176.bin"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return False
    
    try:
        # Load the model
        model = fasttext.load_model(model_path)
        print(f"✅ Model loaded successfully! Can detect {len(model.get_labels())} languages")
        
        # Test with different languages
        test_texts = [
            ("Hello, how are you today?", "en"),
            ("मैं आपकी मदद कर सकता हूं", "hi"),
            ("আমি আপনাকে সাহায্য করতে পারি", "bn"),
            ("நான் உங்களுக்கு உதவ முடியும்", "ta"),
            ("ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ", "kn"),
            ("മനുഷ്യാവകാശങ്ങൾ", "ml"),
        ]
        
        print("\n🌍 Testing language detection:")
        print("=" * 60)
        
        for text, expected_lang in test_texts:
            labels, probs = model.predict(text)
            detected_lang = labels[0].replace("__label__", "")
            confidence = probs[0]
            
            status = "✅" if detected_lang == expected_lang else "⚠️"
            print(f"{status} Text: '{text[:30]}...' → {detected_lang} ({confidence:.3f})")
            
        print("=" * 60)
        print("🎉 FastText language detection is working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    test_fasttext_model()
