import streamlit as st
from typing import Dict, Optional, List, Tuple
from clinical_models import get_clinical_predictor
from theme_manager import get_theme_manager

class ClinicalInformationForm:
    """
    Handles the clinical information form for post-MRI tumor detection workflow.
    """
    
    def __init__(self):
        self.theme_manager = get_theme_manager()
        
        # Form field definitions as specified in requirements
        self.form_fields = {
            'patient_info': {
                'title': '👤 Patient Information',
                'fields': {
                    'Age': {
                        'type': 'number',
                        'label': 'Age',
                        'min_value': 1,
                        'max_value': 120,
                        'value': 45,
                        'help': 'Patient age in years'
                    },
                    'Gender': {
                        'type': 'selectbox',
                        'label': 'Gender',
                        'options': ['Female', 'Male', 'Other'],
                        'index': 0,
                        'help': 'Patient gender'
                    },
                    'Genetic_Risk': {
                        'type': 'number',
                        'label': 'Genetic Risk Score',
                        'min_value': 0.0,
                        'max_value': 10.0,
                        'value': 0.0,
                        'step': 0.1,
                        'help': 'Genetic risk score (0-10 scale)'
                    }
                }
            },
            'medical_history': {
                'title': '🏥 Medical History',
                'fields': {
                    'Family_History': {
                        'type': 'selectbox',
                        'label': 'Family History of Brain Tumors',
                        'options': ['Yes', 'No'],
                        'index': 1,
                        'help': 'Any family history of brain tumors'
                    },
                    'Chronic_Illness': {
                        'type': 'selectbox',
                        'label': 'Chronic Illness',
                        'options': ['No', 'Yes'],
                        'index': 0,
                        'help': 'Presence of chronic illness'
                    },
                    'Diabetes': {
                        'type': 'selectbox',
                        'label': 'Diabetes',
                        'options': ['Yes', 'No'],
                        'index': 1,
                        'help': 'Diabetes diagnosis'
                    },
                    'Blood_Pressure_High': {
                        'type': 'selectbox',
                        'label': 'High Blood Pressure',
                        'options': ['Yes', 'No'],
                        'index': 1,
                        'help': 'History of high blood pressure'
                    },
                    'Blood_Pressure_Low': {
                        'type': 'selectbox',
                        'label': 'Low Blood Pressure',
                        'options': ['No', 'Yes'],
                        'index': 0,
                        'help': 'History of low blood pressure'
                    }
                }
            },
            'mri_clinical': {
                'title': '🧠 MRI & Clinical Findings',
                'fields': {
                    'MRI_Findings': {
                        'type': 'selectbox',
                        'label': 'Primary MRI Findings Location',
                        'options': ['Temporal', 'Frontal', 'Parietal', 'Cerebellum', 'Occipital'],
                        'index': 0,
                        'help': 'Primary location of MRI findings (will be auto-filled from UNET model if tumor detected)'
                    },
                    'Symptom_Severity': {
                        'type': 'selectbox',
                        'label': 'Symptom Severity',
                        'options': ['Severe', 'Moderate', 'Mild'],
                        'index': 1,
                        'help': 'Severity of neurological symptoms'
                    }
                }
            },
            'lifestyle_factors': {
                'title': '🚭 Lifestyle & Environmental Factors',
                'fields': {
                    'Smoking_History': {
                        'type': 'selectbox',
                        'label': 'Smoking History',
                        'options': ['Normal', 'Abnormal', 'Severe'],
                        'index': 0,
                        'help': 'Smoking history classification'
                    },
                    'Alcohol_Consumption': {
                        'type': 'selectbox',
                        'label': 'Alcohol Consumption',
                        'options': ['Yes', 'No'],
                        'index': 1,
                        'help': 'Regular alcohol consumption'
                    },
                    'Radiation_Exposure': {
                        'type': 'selectbox',
                        'label': 'Previous Radiation Exposure',
                        'options': ['Yes', 'No'],
                        'index': 1,
                        'help': 'History of radiation exposure'
                    },
                    'Head_Injury_History': {
                        'type': 'selectbox',
                        'label': 'Head Injury History',
                        'options': ['Low', 'Medium', 'High'],
                        'index': 0,
                        'help': 'Severity of previous head injuries'
                    }
                }
            }
        }
        
        # Session state is initialized in main app.py
    
    def validate_form_data(self, form_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate the form data and return validation status and error messages.
        """
        errors = []
        
        # Required field validation
        required_fields = ['Age', 'Gender', 'Symptom_Severity']
        for field in required_fields:
            if field not in form_data or form_data[field] is None:
                errors.append(f"{field.replace('_', ' ')} is required")
        
        # Age validation
        if 'Age' in form_data:
            try:
                age = float(form_data['Age'])
                if age < 1 or age > 120:
                    errors.append("Age must be between 1 and 120 years")
            except (ValueError, TypeError):
                errors.append("Age must be a valid number")
        
        # Genetic risk validation
        if 'Genetic_Risk' in form_data:
            try:
                risk = float(form_data['Genetic_Risk'])
                if risk < 0 or risk > 10:
                    errors.append("Genetic Risk Score must be between 0 and 10")
            except (ValueError, TypeError):
                errors.append("Genetic Risk Score must be a valid number")
        
        return len(errors) == 0, errors
    
    def render_form_section(self, section_key: str, section_data: Dict, tumor_info: Optional[Dict] = None):
        """
        Render a single section of the form.
        """
        st.markdown(f"""
        <div class="form-section">
            <h3>{section_data['title']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(2)
        col_index = 0
        
        for field_name, field_config in section_data['fields'].items():
            with cols[col_index % 2]:
                # Auto-fill MRI findings with tumor location if available
                if field_name == 'MRI_Findings' and tumor_info and tumor_info.get('tumor_location'):
                    default_index = field_config['options'].index(tumor_info['tumor_location'])
                    field_config = field_config.copy()
                    field_config['index'] = default_index
                    st.info(f"🤖 Auto-filled from AI detection: {tumor_info['tumor_location']}")
                
                # Render field based on type
                if field_config['type'] == 'number':
                    if 'step' in field_config:
                        value = st.number_input(
                            field_config['label'],
                            min_value=field_config['min_value'],
                            max_value=field_config['max_value'],
                            value=field_config['value'],
                            step=field_config['step'],
                            help=field_config['help'],
                            key=f"form_{field_name}"
                        )
                    else:
                        value = st.number_input(
                            field_config['label'],
                            min_value=field_config['min_value'],
                            max_value=field_config['max_value'],
                            value=field_config['value'],
                            help=field_config['help'],
                            key=f"form_{field_name}"
                        )
                elif field_config['type'] == 'selectbox':
                    value = st.selectbox(
                        field_config['label'],
                        options=field_config['options'],
                        index=field_config['index'],
                        help=field_config['help'],
                        key=f"form_{field_name}"
                    )
                else:
                    st.error(f"Unknown field type: {field_config['type']}")
                    continue
                
                # Store value in session state
                st.session_state.clinical_form_data[field_name] = value
                col_index += 1
    
    def render_form(self, tumor_info: Optional[Dict] = None) -> bool:
        """
        Render the complete clinical information form.
        Returns True if form is submitted and valid.
        """
        st.markdown("""
        <div class="clinical-form">
        """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Clinical Information Form")
        st.markdown("Please provide the following clinical information to enhance the AI analysis:")
        
        # Show tumor detection status
        if tumor_info and tumor_info.get('tumor_detected'):
            st.success(f"🎯 Tumor detected in **{tumor_info.get('tumor_location', 'unknown')}** region (Size: {tumor_info.get('tumor_size', 0):,} voxels)")
            st.info("💡 The form below will help our AI models predict tumor type and growth rate based on clinical factors.")
        else:
            st.warning("⚠️ No tumor detected by the segmentation model. Clinical predictions will not be available.")
            return False
        
        # Render form sections
        for section_key, section_data in self.form_fields.items():
            self.render_form_section(section_key, section_data, tumor_info)
            st.markdown("---")
        
        # Form submission
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Submit Clinical Information", type="primary", use_container_width=True):
                # Validate form
                is_valid, errors = self.validate_form_data(st.session_state.clinical_form_data)
                
                if is_valid:
                    st.session_state.form_submitted = True
                    st.success("✅ Form submitted successfully! Generating predictions...")
                    
                    # Run clinical predictions
                    with st.spinner("🔬 Analyzing clinical data with AI models..."):
                        try:
                            predictor = get_clinical_predictor()
                            predictions = predictor.predict_tumor_characteristics(
                                st.session_state.clinical_form_data,
                                tumor_info
                            )
                            st.session_state.clinical_predictions = predictions
                            
                        except Exception as e:
                            st.error(f"❌ Error running clinical predictions: {str(e)}")
                            st.session_state.clinical_predictions = {'error': str(e)}
                    
                    return True
                else:
                    st.error("❌ Please correct the following errors:")
                    for error in errors:
                        st.error(f"• {error}")
                    return False
        
        st.markdown("</div>", unsafe_allow_html=True)
        return False
    
    def render_predictions(self) -> None:
        """
        Render the clinical predictions results.
        """
        if not st.session_state.get('clinical_predictions'):
            return
        
        predictions = st.session_state.clinical_predictions
        
        if 'error' in predictions:
            st.error(f"❌ Prediction Error: {predictions['error']}")
            return
        
        if not predictions.get('predictions_available'):
            st.info("ℹ️ Clinical predictions are not available (no tumor detected)")
            return
        
        st.markdown("""
        <div class="prediction-results">
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 AI Clinical Predictions")
        
        # Create columns for predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="result-card">
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🏷️ Tumor Type")
            if predictions.get('tumor_type'):
                st.markdown(f"""
                **Prediction:** {predictions['tumor_type']}
                """)
                
                st.success(f"AI Prediction: {predictions['tumor_type']}")
            else:
                st.info("Tumor type prediction not available")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="result-card">
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📈 Tumor Growth Rate")
            if predictions.get('tumor_growth_rate'):
                st.markdown(f"""
                **Prediction:** {predictions['tumor_growth_rate']}
                """)
                
                st.success(f"AI Prediction: {predictions['tumor_growth_rate']}")
            else:
                st.info("Growth rate prediction not available")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional information
        st.markdown("---")
        st.markdown("#### 💡 Key Contributing Factors")
        
        try:
            predictor = get_clinical_predictor()
            explanation = predictor.get_feature_importance_explanation(
                st.session_state.clinical_form_data,
                st.session_state.get('tumor_analysis', {})
            )
            
            if explanation.get('key_factors'):
                for factor in explanation['key_factors']:
                    st.markdown(f"• {factor}")
            
            if explanation.get('note'):
                st.info(f"ℹ️ {explanation['note']}")
                
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")
        
        # Clinical recommendations
        st.markdown("---")
        st.markdown("#### 🩺 Clinical Recommendations")
        
        tumor_type = predictions.get('tumor_type', '').lower()
        growth_rate = predictions.get('tumor_growth_rate', '').lower()
        
        if 'malignant' in tumor_type or 'rapid' in growth_rate:
            st.error("⚠️ **Urgent**: Results suggest potential malignancy or rapid growth. Immediate oncological consultation recommended.")
        elif 'benign' in tumor_type and 'slow' in growth_rate:
            st.success("✅ **Monitoring**: Results suggest benign tumor with slow growth. Regular monitoring advised.")
        else:
            st.warning("⚠️ **Follow-up**: Mixed indicators. Close follow-up and additional testing may be warranted.")
        
        st.info("🏥 **Important**: These AI predictions are for research and educational purposes only. All medical decisions should be made by qualified healthcare professionals based on comprehensive clinical evaluation.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def reset_form(self):
        """Reset the form and clear all data."""
        st.session_state.clinical_form_data = {}
        st.session_state.form_submitted = False
        st.session_state.clinical_predictions = None
        st.rerun()


# Global form instance
clinical_form = ClinicalInformationForm()

def get_clinical_form() -> ClinicalInformationForm:
    """Get the global clinical form instance."""
    return clinical_form