import streamlit as st
from typing import Dict, Optional

class ThemeManager:
    """
    Manages application themes and styling.
    """
    
    def __init__(self):
        # Check if we're running in Streamlit context
        self._in_streamlit = self._check_streamlit_context()
        
        self.themes = {
            'light': {
                'name': 'Light Mode',
                'primary_color': '#4F8BF9',
                'background_color': '#FFFFFF',
                'secondary_background': '#F8F9FA',
                'text_color': '#000000',
                'text_secondary': '#333333',
                'border_color': '#E9ECEF',
                'success_color': '#28A745',
                'warning_color': '#FFC107',
                'error_color': '#DC3545',
                'info_color': '#17A2B8',
                'card_shadow': '0 2px 4px rgba(0,0,0,0.1)',
                'hover_color': '#E3F2FD'
            },
            'dark': {
                'name': 'Dark Mode',
                'primary_color': '#5B9BD5',
                'background_color': '#1E1E1E',
                'secondary_background': '#2D2D2D',
                'text_color': '#FFFFFF',
                'text_secondary': '#CCCCCC',
                'border_color': '#404040',
                'success_color': '#4CAF50',
                'warning_color': '#FF9800',
                'error_color': '#F44336',
                'info_color': '#2196F3',
                'card_shadow': '0 2px 8px rgba(0,0,0,0.3)',
                'hover_color': '#3D3D3D'
            },
            'medical': {
                'name': 'Medical Professional',
                'primary_color': '#0D7377',
                'background_color': '#FAFAFA',
                'secondary_background': '#F0F8FF',
                'text_color': '#2C3E50',
                'text_secondary': '#5D6D7E',
                'border_color': '#BDC3C7',
                'success_color': '#27AE60',
                'warning_color': '#F39C12',
                'error_color': '#E74C3C',
                'info_color': '#3498DB',
                'card_shadow': '0 2px 6px rgba(13,115,119,0.15)',
                'hover_color': '#E8F5E8'
            }
        }
        
        # Initialize theme in session state (only if in Streamlit)
        if self._in_streamlit:
            try:
                if 'app_theme' not in st.session_state:
                    st.session_state.app_theme = 'light'
            except Exception:
                # Fallback if session state is not available
                self._current_theme = 'light'
                self._in_streamlit = False
        else:
            # Fallback for non-Streamlit contexts
            self._current_theme = 'light'
    
    def _check_streamlit_context(self) -> bool:
        """Check if we're running in Streamlit context."""
        try:
            import streamlit as st
            # Try to access session state to see if we're in Streamlit
            _ = st.session_state
            return True
        except Exception:
            return False
    
    def get_current_theme(self) -> Dict:
        """Get the current theme configuration."""
        try:
            if self._in_streamlit and hasattr(st.session_state, 'app_theme'):
                return self.themes[st.session_state.app_theme]
            else:
                return self.themes[getattr(self, '_current_theme', 'light')]
        except Exception:
            return self.themes['light']
    
    def set_theme(self, theme_name: str):
        """Set the application theme."""
        if theme_name in self.themes:
            if self._in_streamlit:
                try:
                    st.session_state.app_theme = theme_name
                except Exception:
                    self._current_theme = theme_name
            else:
                self._current_theme = theme_name
    
    def get_theme_selector(self) -> str:
        """Render theme selector widget and return selected theme."""
        theme_names = [self.themes[key]['name'] for key in self.themes.keys()]
        theme_keys = list(self.themes.keys())
        
        # Safely get current theme
        try:
            current_theme = st.session_state.app_theme if self._in_streamlit else getattr(self, '_current_theme', 'light')
        except Exception:
            current_theme = 'light'
            
        current_index = theme_keys.index(current_theme)
        
        selected_name = st.selectbox(
            "🎨 Theme",
            theme_names,
            index=current_index,
            key="theme_selector"
        )
        
        # Find the key for the selected theme name
        selected_key = None
        for key, config in self.themes.items():
            if config['name'] == selected_name:
                selected_key = key
                break
        
        if selected_key and selected_key != current_theme:
            self.set_theme(selected_key)
            if self._in_streamlit:
                st.rerun()
        
        return current_theme
    
    def get_css_styles(self) -> str:
        """Generate CSS styles based on current theme."""
        theme = self.get_current_theme()
        
        return f"""
        <style>
            /* Main App Styling */
            .main .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 95%;
                background-color: {theme['background_color']};
            }}
            
            /* Header Styling */
            .app-header {{
                background: linear-gradient(135deg, {theme['primary_color']}, {theme['primary_color']}DD);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: {theme['card_shadow']};
            }}
            
            .app-title {{
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                text-align: center;
            }}
            
            .app-subtitle {{
                font-size: 1.2rem;
                opacity: 0.9;
                text-align: center;
                margin-bottom: 0.5rem;
            }}
            
            .app-caption {{
                font-size: 1rem;
                opacity: 0.8;
                text-align: center;
            }}
            
            /* Form Styling */
            .clinical-form {{
                background-color: {theme['secondary_background']};
                border: 2px solid {theme['border_color']};
                border-radius: 12px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: {theme['card_shadow']};
            }}
            
            .form-section {{
                margin-bottom: 2rem;
                padding: 1.5rem;
                background-color: {theme['background_color']};
                border-radius: 8px;
                border-left: 4px solid {theme['primary_color']};
            }}
            
            .form-section h3 {{
                color: {theme['text_color']};
                margin-bottom: 1rem;
                font-size: 1.3rem;
                font-weight: 600;
            }}
            
            /* Metric Cards */
            div[data-testid="metric-container"] {{
                background: linear-gradient(135deg, {theme['secondary_background']}, {theme['background_color']});
                border: 1px solid {theme['border_color']};
                padding: 1.5rem;
                border-radius: 10px;
                box-shadow: {theme['card_shadow']};
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            
            div[data-testid="metric-container"]:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }}
            
            /* Results Display */
            .prediction-results {{
                background: linear-gradient(135deg, {theme['secondary_background']}, {theme['background_color']});
                border: 2px solid {theme['primary_color']};
                border-radius: 12px;
                padding: 2rem;
                margin: 2rem 0;
                box-shadow: {theme['card_shadow']};
            }}
            
            .result-card {{
                background-color: {theme['background_color']};
                border: 1px solid {theme['border_color']};
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                transition: all 0.3s ease;
            }}
            
            .result-card:hover {{
                border-color: {theme['primary_color']};
                box-shadow: {theme['card_shadow']};
                transform: translateY(-2px);
            }}
            
            /* Success/Error Messages */
            .stSuccess > div {{
                background-color: {theme['success_color']}22;
                border: 1px solid {theme['success_color']};
                color: {theme['text_color']};
                border-radius: 8px;
            }}
            
            .stError > div {{
                background-color: {theme['error_color']}22;
                border: 1px solid {theme['error_color']};
                color: {theme['text_color']};
                border-radius: 8px;
            }}
            
            .stWarning > div {{
                background-color: {theme['warning_color']}22;
                border: 1px solid {theme['warning_color']};
                color: {theme['text_color']};
                border-radius: 8px;
            }}
            
            .stInfo > div {{
                background-color: {theme['info_color']}22;
                border: 1px solid {theme['info_color']};
                color: {theme['text_color']};
                border-radius: 8px;
            }}
            
            /* Buttons */
            .stButton > button {{
                background: linear-gradient(135deg, {theme['primary_color']}, {theme['primary_color']}CC);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                background: linear-gradient(135deg, {theme['primary_color']}DD, {theme['primary_color']});
            }}
            
            /* Sidebar Styling */
            .css-1d391kg {{
                background-color: {theme['secondary_background']};
            }}
            
            /* Plotly Graphs */
            .plotly-graph-div {{
                border-radius: 10px;
                border: 1px solid {theme['border_color']};
                box-shadow: {theme['card_shadow']};
            }}
            
            /* Progress Bar */
            .stProgress > div > div > div > div {{
                background: linear-gradient(90deg, {theme['primary_color']}, {theme['primary_color']}AA);
            }}
            
            /* Input Fields */
            .stSelectbox > div > div > div {{
                background-color: {theme['background_color']};
                border-color: {theme['border_color']};
                border-radius: 6px;
            }}
            
            .stNumberInput > div > div > input {{
                background-color: {theme['background_color']};
                border-color: {theme['border_color']};
                border-radius: 6px;
                color: {theme['text_color']};
            }}
            
            .stTextInput > div > div > input {{
                background-color: {theme['background_color']};
                border-color: {theme['border_color']};
                border-radius: 6px;
                color: {theme['text_color']};
            }}
            
            /* Tab Styling */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: {theme['secondary_background']};
                border-radius: 8px 8px 0 0;
                padding: 0.75rem 1.5rem;
                color: {theme['text_secondary']};
                border: 1px solid {theme['border_color']};
                border-bottom: none;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: {theme['primary_color']};
                color: white;
                border-color: {theme['primary_color']};
            }}
            
            /* Animation for loading */
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            
            .loading-pulse {{
                animation: pulse 2s infinite;
            }}
            
            /* Custom classes for special elements */
            .tumor-detected {{
                border-left: 5px solid {theme['error_color']};
                background-color: {theme['error_color']}11;
            }}
            
            .tumor-not-detected {{
                border-left: 5px solid {theme['success_color']};
                background-color: {theme['success_color']}11;
            }}
            
            .confidence-high {{
                color: {theme['success_color']};
                font-weight: 600;
            }}
            
            .confidence-medium {{
                color: {theme['warning_color']};
                font-weight: 600;
            }}
            
            .confidence-low {{
                color: {theme['error_color']};
                font-weight: 600;
            }}
            
        </style>
        """
    
    def apply_theme(self):
        """Apply the current theme to the Streamlit app."""
        st.markdown(self.get_css_styles(), unsafe_allow_html=True)
    
    def render_header(self):
        """Render the application header with current theme."""
        st.markdown(f"""
        <div class="app-header">
            <div class="app-title">🧠 Brain Tumor Segmentation AI</div>
            <div class="app-subtitle">AI-Powered Brain Analysis</div>
            <div class="app-caption">Upload FLAIR NIfTI files for tumor segmentation and analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    def get_confidence_class(self, confidence: float) -> str:
        """Get CSS class based on confidence level."""
        if confidence is None:
            return "confidence-low"
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.6:
            return "confidence-medium"
        else:
            return "confidence-low"


# Global theme manager instance
theme_manager = ThemeManager()

def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    return theme_manager