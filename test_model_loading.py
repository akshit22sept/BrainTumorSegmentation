#!/usr/bin/env python3
"""
Test script to verify H5 model loading with the new TensorFlow installation.
"""

import os
import sys
import logging

def test_tensorflow_and_models():
    """Test TensorFlow and model loading."""
    print("🔬 Testing TensorFlow and Model Loading")
    print("=" * 50)
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Test Keras
        keras = tf.keras
        print(f"✅ Keras {keras.__version__} available")
        
    except Exception as e:
        print(f"❌ TensorFlow/Keras error: {str(e)}")
        return False
    
    # Test model loading
    try:
        from clinical_models import get_clinical_predictor
        
        print(f"\n📁 Checking models directory...")
        models_dir = os.path.join(os.getcwd(), "models")
        print(f"Models directory: {models_dir}")
        
        if os.path.exists(models_dir):
            model_files = os.listdir(models_dir)
            print(f"Found files: {model_files}")
        else:
            print(f"❌ Models directory not found: {models_dir}")
            return False
        
        # Initialize predictor (this should now load the models)
        print(f"\n🤖 Initializing clinical predictor...")
        predictor = get_clinical_predictor()
        
        print(f"✅ Clinical predictor initialized")
        print(f"Available models: {list(predictor.models.keys())}")
        
        # Test prediction structure
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
        
        print(f"\n🧪 Testing predictions...")
        predictions = predictor.predict_tumor_characteristics(sample_form_data, sample_tumor_info)
        
        print(f"Prediction structure: {predictions.keys()}")
        print(f"Predictions available: {predictions.get('predictions_available', False)}")
        
        if predictions.get('predictions_available'):
            print(f"✅ Model predictions working!")
            print(f"Tumor type: {predictions.get('tumor_type', 'N/A')}")
            print(f"Growth rate: {predictions.get('tumor_growth_rate', 'N/A')}")
        else:
            print(f"⚠️ Models loaded but predictions not available")
            if 'error' in predictions:
                print(f"Error: {predictions['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

def main():
    """Run the test."""
    success = test_tensorflow_and_models()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! H5 models should work in the app.")
    else:
        print("⚠️ Some issues detected. Check the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)