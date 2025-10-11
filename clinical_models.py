import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle TensorFlow imports with compatibility
try:
    import tensorflow as tf
    # Try different ways to access keras
    try:
        keras = tf.keras
        logger.info("TensorFlow and Keras loaded successfully")
    except AttributeError:
        try:
            import keras
            logger.info("Standalone Keras loaded successfully")
        except ImportError:
            keras = None
except ImportError:
    tf = None
    keras = None
    logger.warning("TensorFlow not available. H5 model predictions will be disabled.")

class ClinicalDataProcessor:
    """
    Handles preprocessing of clinical data for H5 model input.
    """
    
    def __init__(self):
        # Define the categorical mappings as specified in requirements
        self.categorical_mappings = {
            'Gender': ['Female', 'Male', 'Other'],
            'Tumor_Location': ['Temporal', 'Frontal', 'Parietal', 'Cerebellum', 'Occipital'],
            'MRI_Findings': ['Temporal', 'Frontal', 'Parietal', 'Cerebellum', 'Occipital'],
            'Smoking_History': ['Normal', 'Abnormal', 'Severe'],
            'Alcohol_Consumption': ['Yes', 'No'],
            'Radiation_Exposure': ['Yes', 'No'],
            'Head_Injury_History': ['Low', 'Medium', 'High'],
            'Chronic_Illness': ['No', 'Yes'],
            'Diabetes': ['Yes', 'No'],
            'Family_History': ['Yes', 'No'],
            'Tumor_Type': ['Benign', 'Malignant'],
            'Tumor_Growth_Rate': ['Rapid', 'Slow', 'Moderate'],
            'Blood_Pressure_High': ['Yes', 'No'],
            'Symptom_Severity': ['Severe', 'Moderate', 'Mild'],
            'Blood_Pressure_Low': ['No', 'Yes']
        }
        
        # Expected column order for H5 model input
        self.expected_columns = [
            'Age', 'Tumor_Size', 'Genetic_Risk', 'Blood_Pressure_High',
            'Blood_Pressure_Low', 'Gender', 'Tumor_Location', 'MRI_Findings',
            'Smoking_History', 'Alcohol_Consumption', 'Radiation_Exposure',
            'Head_Injury_History', 'Chronic_Illness', 'Diabetes', 'Family_History',
            'Symptom_Severity'
        ]
    
    def encode_categorical(self, field: str, value: str) -> int:
        """
        Convert categorical value to numerical index.
        """
        if field not in self.categorical_mappings:
            raise ValueError(f"Unknown categorical field: {field}")
        
        if value not in self.categorical_mappings[field]:
            raise ValueError(f"Invalid value '{value}' for field '{field}'. Expected one of: {self.categorical_mappings[field]}")
        
        return self.categorical_mappings[field].index(value)
    
    def decode_categorical(self, field: str, index: int) -> str:
        """
        Convert numerical index back to categorical value.
        """
        if field not in self.categorical_mappings:
            raise ValueError(f"Unknown categorical field: {field}")
        
        if index >= len(self.categorical_mappings[field]):
            raise ValueError(f"Invalid index {index} for field '{field}'")
        
        return self.categorical_mappings[field][index]
    
    def preprocess_clinical_data(self, form_data: Dict, tumor_info: Dict) -> np.ndarray:
        """
        Convert form data and tumor info into model-ready numerical array.
        """
        try:
            # Create feature vector in the expected order
            features = []
            
            for column in self.expected_columns:
                if column == 'Age':
                    features.append(float(form_data.get('Age', 0)))
                elif column == 'Tumor_Size':
                    features.append(float(tumor_info.get('tumor_size', 0)))
                elif column == 'Genetic_Risk':
                    features.append(float(form_data.get('Genetic_Risk', 0)))
                elif column == 'Blood_Pressure_High':
                    # Convert Yes/No to 1/0
                    value = form_data.get('Blood_Pressure_High', 'No')
                    features.append(self.encode_categorical('Blood_Pressure_High', value))
                elif column == 'Blood_Pressure_Low':
                    value = form_data.get('Blood_Pressure_Low', 'No')
                    features.append(self.encode_categorical('Blood_Pressure_Low', value))
                elif column == 'Tumor_Location':
                    # Use tumor location from UNET model
                    tumor_location = tumor_info.get('tumor_location', 'Frontal')
                    features.append(self.encode_categorical('Tumor_Location', tumor_location))
                elif column in self.categorical_mappings:
                    # Handle other categorical fields
                    value = form_data.get(column, self.categorical_mappings[column][0])  # Use first option as default
                    features.append(self.encode_categorical(column, value))
                else:
                    # Handle any remaining numerical fields
                    features.append(float(form_data.get(column, 0)))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preprocessing clinical data: {str(e)}")
            raise


class ClinicalModelPredictor:
    """
    Handles loading and inference with H5 clinical prediction models.
    """
    
    def __init__(self, models_path: str = None):
        if models_path is None:
            # Use absolute path to models directory
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_path = os.path.join(current_dir, "models")
        else:
            self.models_path = models_path
        
        self.data_processor = ClinicalDataProcessor()
        self.models = {}
        logger.info(f"Looking for models in: {self.models_path}")
        self._load_models()
    
    def _load_models(self):
        """
        Load H5 models for tumor type and growth rate prediction.
        """
        if tf is None or keras is None:
            logger.warning("TensorFlow/Keras not available. H5 models will not be loaded.")
            return
            
        # Check if models directory exists
        if not os.path.exists(self.models_path):
            logger.error(f"Models directory not found: {self.models_path}")
            return
            
        try:
            # Load tumor type model
            tumor_type_path = os.path.join(self.models_path, "brain_tumor_type.h5")
            logger.info(f"Checking for tumor type model at: {tumor_type_path}")
            
            if os.path.exists(tumor_type_path):
                try:
                    logger.info(f"Loading tumor type model from: {tumor_type_path}")
                    self.models['tumor_type'] = keras.models.load_model(tumor_type_path)
                    logger.info("✅ Tumor type model loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Could not load tumor type model: {str(e)}")
            else:
                logger.warning(f"❌ Tumor type model not found at {tumor_type_path}")
            
            # Load tumor growth rate model
            growth_rate_path = os.path.join(self.models_path, "brain_tumor_growth.h5")
            logger.info(f"Checking for growth rate model at: {growth_rate_path}")
            
            if os.path.exists(growth_rate_path):
                try:
                    logger.info(f"Loading growth rate model from: {growth_rate_path}")
                    self.models['growth_rate'] = keras.models.load_model(growth_rate_path)
                    logger.info("✅ Tumor growth rate model loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Could not load tumor growth rate model: {str(e)}")
            else:
                logger.warning(f"❌ Tumor growth rate model not found at {growth_rate_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    def predict_tumor_characteristics(self, form_data: Dict, tumor_info: Dict) -> Dict:
        """
        Predict tumor type and growth rate based on clinical data and tumor info.
        """
        if not tumor_info.get('tumor_detected', False):
            return {
                'tumor_type': None,
                'tumor_growth_rate': None,
                'tumor_type_confidence': None,
                'growth_rate_confidence': None,
                'predictions_available': False
            }
        
        try:
            # Preprocess the input data
            input_features = self.data_processor.preprocess_clinical_data(form_data, tumor_info)
            
            results = {'predictions_available': True}
            
            # Predict tumor type
            if 'tumor_type' in self.models:
                tumor_type_pred = self.models['tumor_type'].predict(input_features, verbose=0)
                
                # Get the predicted class and confidence
                tumor_type_class = np.argmax(tumor_type_pred[0])
                tumor_type_confidence = float(np.max(tumor_type_pred[0]))
                
                # Decode the prediction
                tumor_type = self.data_processor.decode_categorical('Tumor_Type', tumor_type_class)
                
                results.update({
                    'tumor_type': tumor_type,
                    'tumor_type_confidence': tumor_type_confidence,
                    'tumor_type_probabilities': tumor_type_pred[0].tolist()
                })
            else:
                results.update({
                    'tumor_type': 'Model not available',
                    'tumor_type_confidence': None
                })
            
            # Predict tumor growth rate
            if 'growth_rate' in self.models:
                growth_rate_pred = self.models['growth_rate'].predict(input_features, verbose=0)
                
                # Get the predicted class and confidence
                growth_rate_class = np.argmax(growth_rate_pred[0])
                growth_rate_confidence = float(np.max(growth_rate_pred[0]))
                
                # Decode the prediction
                growth_rate = self.data_processor.decode_categorical('Tumor_Growth_Rate', growth_rate_class)
                
                results.update({
                    'tumor_growth_rate': growth_rate,
                    'growth_rate_confidence': growth_rate_confidence,
                    'growth_rate_probabilities': growth_rate_pred[0].tolist()
                })
            else:
                results.update({
                    'tumor_growth_rate': 'Model not available',
                    'growth_rate_confidence': None
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                'error': str(e),
                'predictions_available': False
            }
    
    def get_feature_importance_explanation(self, form_data: Dict, tumor_info: Dict) -> Dict:
        """
        Provide explanation of key factors influencing the prediction.
        """
        explanations = []
        
        # Basic explanations based on input data
        if tumor_info.get('tumor_detected'):
            explanations.append(f"Tumor detected in {tumor_info.get('tumor_location', 'unknown')} region")
            explanations.append(f"Tumor size: {tumor_info.get('tumor_size', 0)} voxels")
        
        if form_data.get('Age'):
            age = int(form_data['Age'])
            if age > 60:
                explanations.append("Advanced age is a significant risk factor")
            elif age < 40:
                explanations.append("Younger age may indicate different tumor characteristics")
        
        if form_data.get('Family_History') == 'Yes':
            explanations.append("Family history increases risk probability")
        
        if form_data.get('Radiation_Exposure') == 'Yes':
            explanations.append("Previous radiation exposure is a contributing factor")
        
        return {
            'key_factors': explanations,
            'note': 'Predictions are based on multiple clinical and imaging factors'
        }


# Global predictor instance
clinical_predictor = None

def get_clinical_predictor() -> ClinicalModelPredictor:
    """
    Get or create the global clinical predictor instance.
    """
    global clinical_predictor
    if clinical_predictor is None:
        clinical_predictor = ClinicalModelPredictor()
    return clinical_predictor