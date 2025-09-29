import requests
import json
import numpy as np
from typing import Dict, Generator, Optional

class BrainTumorAnalyzer:
    def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def analyze_prediction(self, metrics: Dict, mri_shape: tuple, pred_mask: np.ndarray) -> Generator[str, None, None]:
        tumor_locations = self._get_tumor_locations(pred_mask)
        severity = self._assess_severity(metrics, mri_shape)
        
        prompt = f"""You are a medical AI assistant analyzing brain tumor segmentation results. Focus specifically on tumor characteristics and future risk assessment.

TUMOR ANALYSIS DATA:
- Tumor volume: {metrics.get('mask_voxels', 0):,} voxels ({metrics.get('mask_fraction', 0)*100:.2f}% of brain)
- Tumor volume in mL: {metrics.get('mask_volume_mL', 'Unknown')}
- Tumor locations: {tumor_locations}
- Severity classification: {severity}
- Brain regions affected: Based on {tumor_locations}

Provide a focused analysis covering:

**TUMOR CHARACTERISTICS:**
- Size assessment and growth implications
- Location significance and affected brain functions
- Morphological features visible in the segmentation

**RISK ASSESSMENT:**
- Potential complications based on size and location
- Risk of mass effect or increased intracranial pressure
- Functional risks related to affected brain regions
- Monitoring considerations for tumor progression

**FUTURE CONSIDERATIONS:**
- Signs and symptoms to monitor
- Urgency level for medical consultation
- Follow-up imaging recommendations

Keep response concise, factual, and focused only on tumor-related information and associated risks. Avoid general medical advice."""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_predict": 800
            }
        }
        
        for attempt in range(3):
            try:
                if attempt > 0:
                    yield f"\n\nRetrying connection (attempt {attempt + 1}/3)...\n\n"
                    
                response = requests.post(self.api_url, json=payload, stream=True, timeout=30)
                response.raise_for_status()
                
                response_received = False
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                response_received = True
                                yield data['response']
                            if data.get('done', False):
                                return
                        except json.JSONDecodeError:
                            continue
                            
                if response_received:
                    return
                    
            except requests.exceptions.Timeout:
                if attempt == 2:
                    yield f"\n\n⚠️ Connection timeout. Ollama may be overloaded. Using fallback analysis...\n\n{self._get_fallback_analysis(metrics, tumor_locations, severity)}"
                    return
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    if "500" in str(e):
                        yield f"\n\n⚠️ Ollama server error (500). Using fallback analysis...\n\n{self._get_fallback_analysis(metrics, tumor_locations, severity)}"
                    else:
                        yield f"\n\n⚠️ Connection error: {str(e)}. Using fallback analysis...\n\n{self._get_fallback_analysis(metrics, tumor_locations, severity)}"
                    return
            
        yield f"\n\n⚠️ All connection attempts failed. Using fallback analysis...\n\n{self._get_fallback_analysis(metrics, tumor_locations, severity)}"
    
    def _get_tumor_locations(self, pred_mask: np.ndarray) -> str:
        if pred_mask.sum() == 0:
            return "No tumor detected"
        
        center_of_mass = np.array(np.where(pred_mask > 0)).mean(axis=1)
        x_center, y_center, z_center = center_of_mass
        x_max, y_max, z_max = pred_mask.shape
        
        regions = []
        if z_center < z_max * 0.33:
            regions.append("inferior")
        elif z_center > z_max * 0.66:
            regions.append("superior")
        else:
            regions.append("middle")
        
        if x_center < x_max * 0.33:
            regions.append("anterior")
        elif x_center > x_max * 0.66:
            regions.append("posterior")
        else:
            regions.append("central")
        
        if y_center < y_max * 0.4:
            regions.append("left hemisphere")
        elif y_center > y_max * 0.6:
            regions.append("right hemisphere")
        else:
            regions.append("midline")
        
        return f"{', '.join(regions)} regions"
    
    def _assess_severity(self, metrics: Dict, mri_shape: tuple) -> str:
        volume_fraction = metrics.get('mask_fraction', 0)
        volume_ml = metrics.get('mask_volume_mL', 0)
        
        if volume_fraction < 0.01:
            return "Small lesion"
        elif volume_fraction < 0.05:
            return "Moderate-sized lesion"
        elif volume_fraction < 0.15:
            return "Large lesion"
        else:
            return "Extensive lesion"
    
    def _get_fallback_analysis(self, metrics: Dict, tumor_locations: str, severity: str) -> str:
        tumor_vol = metrics.get('mask_voxels', 0)
        tumor_frac = metrics.get('mask_fraction', 0) * 100
        tumor_ml = metrics.get('mask_volume_mL', 'Unknown')
        
        if tumor_vol == 0:
            return """**CLINICAL SUMMARY**
No tumor detected in the current brain scan. This represents a negative finding for tumor presence based on the AI segmentation analysis.

**TUMOR CHARACTERISTICS**
No abnormal tissue masses identified. Brain tissue appears to show normal intensity patterns consistent with healthy brain matter.

**RISK ASSESSMENT**
No immediate tumor-related risks identified. However, this does not rule out microscopic changes not visible in current imaging resolution.

**FUTURE CONSIDERATIONS**
- Continue routine monitoring if clinically indicated
- Consider follow-up imaging based on clinical symptoms
- Maintain awareness of any new neurological symptoms
- Regular screening if family history or risk factors present"""
        
        analysis = f"""**CLINICAL SUMMARY**
Brain tumor segmentation analysis reveals {severity} with volume of {tumor_vol:,} voxels ({tumor_frac:.2f}% of total brain volume). 

**TUMOR CHARACTERISTICS**
• Size Assessment: {severity} occupying {tumor_frac:.2f}% of brain volume
• Volume Measurement: {tumor_ml} mL detected
• Location: Identified in {tumor_locations}
• Morphology: AI segmentation indicates defined tumor boundaries

**RISK ASSESSMENT**"""
        
        if tumor_frac < 1.0:
            analysis += """
• Low risk of immediate mass effect
• Minimal impact on surrounding brain structures
• Monitor for growth patterns over time
• Functional risks depend on specific location"""
        elif tumor_frac < 5.0:
            analysis += """
• Moderate risk of mass effect if growth continues
• Potential impact on surrounding brain functions
• Requires regular monitoring and follow-up
• Location-specific functional risks present"""
        else:
            analysis += """
• High risk of mass effect and increased intracranial pressure
• Significant impact on surrounding brain structures
• Urgent medical evaluation recommended
• Multiple functional systems potentially affected"""
            
        analysis += f"""

**FUTURE CONSIDERATIONS**
• Medical consultation recommended for treatment planning
• Follow-up MRI imaging in 3-6 months or as clinically indicated
• Monitor for symptoms: headaches, seizures, neurological changes
• Consider additional imaging modalities (contrast, spectroscopy)
• Multidisciplinary team evaluation may be beneficial

*Note: This analysis is based on AI segmentation. Clinical correlation and radiologist review are essential for definitive diagnosis and treatment planning.*"""
        
        return analysis

def get_available_models() -> list:
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except:
        pass
    return []