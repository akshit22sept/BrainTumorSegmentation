#!/usr/bin/env python3
"""
Test script to validate the clinical workflow components.
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

def test_clinical_models():
    """Test the clinical models infrastructure."""
    print("Testing clinical models...")
    
    try:
        from clinical_models import ClinicalDataProcessor, get_clinical_predictor
        
        # Test data processor
        processor = ClinicalDataProcessor()
        print("✅ ClinicalDataProcessor initialized successfully")
        
        # Test categorical encoding
        gender_encoded = processor.encode_categorical('Gender', 'Female')
        assert gender_encoded == 0, "Gender encoding failed"
        print("✅ Categorical encoding working")
        
        # Test categorical decoding  
        gender_decoded = processor.decode_categorical('Gender', 0)
        assert gender_decoded == 'Female', "Gender decoding failed"
        print("✅ Categorical decoding working")
        
        # Test form data preprocessing
        sample_form_data = {
            'Age': 45,
            'Gender': 'Male',
            'Genetic_Risk': 2.5,
            'Family_History': 'No',
            'Chronic_Illness': 'No',
            'Diabetes': 'No',
            'Blood_Pressure_High': 'No',
            'Blood_Pressure_Low': 'No',
            'MRI_Findings': 'Frontal',
            'Symptom_Severity': 'Moderate',
            'Smoking_History': 'Normal',
            'Alcohol_Consumption': 'No',
            'Radiation_Exposure': 'No',
            'Head_Injury_History': 'Low'
        }
        
        sample_tumor_info = {
            'tumor_detected': True,
            'tumor_location': 'Frontal',
            'tumor_size': 1500,
            'brain_tumor_present': 1
        }
        
        features = processor.preprocess_clinical_data(sample_form_data, sample_tumor_info)
        assert features.shape == (1, 16), f"Expected (1, 16) shape, got {features.shape}"
        print("✅ Clinical data preprocessing working")
        
        # Test clinical predictor initialization
        predictor = get_clinical_predictor()
        print("✅ Clinical predictor initialized successfully")
        
        # Test prediction (will work even if H5 models are not available)
        predictions = predictor.predict_tumor_characteristics(sample_form_data, sample_tumor_info)
        assert 'predictions_available' in predictions, "Prediction structure invalid"
        print("✅ Clinical predictions structure valid")
        
    except Exception as e:
        print(f"❌ Clinical models test failed: {str(e)}")
        return False
    
    return True

def test_theme_manager():
    """Test the theme manager."""
    print("\nTesting theme manager...")
    
    try:
        from theme_manager import ThemeManager
        
        # Test theme manager initialization
        theme_manager = ThemeManager()
        print("✅ ThemeManager initialized successfully")
        
        # Test theme retrieval
        current_theme = theme_manager.get_current_theme()
        assert 'primary_color' in current_theme, "Theme structure invalid"
        print("✅ Theme retrieval working")
        
        # Test CSS generation
        css = theme_manager.get_css_styles()
        assert '<style>' in css, "CSS generation failed"
        print("✅ CSS generation working")
        
        # Test confidence class
        conf_class = theme_manager.get_confidence_class(0.9)
        assert conf_class == "confidence-high", "Confidence class mapping failed"
        print("✅ Confidence class mapping working")
        
    except Exception as e:
        print(f"❌ Theme manager test failed: {str(e)}")
        return False
    
    return True

def test_tumor_analysis():
    """Test the tumor analysis functions."""
    print("\nTesting tumor analysis...")
    
    try:
        from model import extract_tumor_location_and_size, analyze_tumor_prediction
        
        # Create sample MRI and prediction mask
        sample_mri = np.random.rand(128, 128, 64).astype(np.float32)
        sample_pred = np.zeros((128, 128, 64), dtype=np.uint8)
        
        # Add a tumor region
        sample_pred[60:80, 60:80, 30:35] = 1
        
        # Test tumor location extraction
        location, size = extract_tumor_location_and_size(sample_pred)
        assert location is not None, "Location extraction failed"
        assert size > 0, "Size calculation failed"
        print(f"✅ Tumor location extraction working: {location}, size: {size}")
        
        # Test tumor analysis
        analysis = analyze_tumor_prediction(sample_mri, sample_pred)
        assert analysis['tumor_detected'] == True, "Tumor detection failed"
        assert analysis['tumor_location'] == location, "Location mismatch"
        print("✅ Tumor analysis working")
        
    except Exception as e:
        print(f"❌ Tumor analysis test failed: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Clinical Workflow Components")
    print("=" * 50)
    
    tests = [
        test_clinical_models,
        test_theme_manager, 
        test_tumor_analysis
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
        print("🎉 All tests passed! Clinical workflow is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)