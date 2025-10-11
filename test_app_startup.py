#!/usr/bin/env python3
"""
Test app startup and basic functionality without running the full Streamlit server.
"""

import sys
import os
import importlib.util

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import torch
        print("✅ NumPy and PyTorch imported successfully")
        
        # Test our custom modules
        from clinical_models import get_clinical_predictor
        from theme_manager import get_theme_manager
        from clinical_form import get_clinical_form
        print("✅ Custom modules imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_theme_manager():
    """Test theme manager initialization."""
    print("\nTesting theme manager...")
    
    try:
        from theme_manager import get_theme_manager
        
        # Test initialization
        theme_manager = get_theme_manager()
        print("✅ Theme manager initialized")
        
        # Test theme retrieval
        theme = theme_manager.get_current_theme()
        print(f"✅ Current theme retrieved: {theme.get('name', 'Unknown')}")
        
        # Test CSS generation
        css = theme_manager.get_css_styles()
        assert len(css) > 100, "CSS seems too short"
        print("✅ CSS generation working")
        
        return True
    except Exception as e:
        print(f"❌ Theme manager error: {str(e)}")
        return False

def test_clinical_models():
    """Test clinical models initialization."""
    print("\nTesting clinical models...")
    
    try:
        from clinical_models import get_clinical_predictor
        
        # Test predictor initialization
        predictor = get_clinical_predictor()
        print("✅ Clinical predictor initialized")
        
        # Test sample prediction structure
        sample_form_data = {
            'Age': 45,
            'Gender': 'Male',
            'Genetic_Risk': 2.5,
        }
        
        sample_tumor_info = {
            'tumor_detected': True,
            'tumor_location': 'Frontal',
            'tumor_size': 1500,
        }
        
        predictions = predictor.predict_tumor_characteristics(sample_form_data, sample_tumor_info)
        assert 'predictions_available' in predictions, "Prediction structure invalid"
        print("✅ Clinical prediction structure valid")
        
        return True
    except Exception as e:
        print(f"❌ Clinical models error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing App Startup Components")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_theme_manager,
        test_clinical_models,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 App components are ready for Streamlit!")
        print("\n💡 To start the app, run: streamlit run app.py")
        return True
    else:
        print("⚠️  Some components failed. Please fix before running the app.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)