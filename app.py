import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="Wine Quality Assessment",
    page_icon="🥂",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# Custom CSS for professional design
st.markdown("""
<style>
    /* Hide Streamlit elements */
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    .stAppDeployButton {display: none;}
    header[data-testid="stHeader"] {display: none;}
    
    /* Main container */
    .main-container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header Section */
    .header-section {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
        border-radius: 15px;
        color: white;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    /* Stats section */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .stat-box {
        background: #FAF3E0; /* Soft beige */
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        flex: 1;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #8B0000; /* Deep red */
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Input section */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Results section */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        border: 3px solid;
    }
    
    .result-good {
        background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
        border-color: #8B0000;
        color: white;
    }
    
    .result-standard {
        background: linear-gradient(135deg, #E63946 0%, #A52A2A 100%);
        border-color: #E63946;
        color: white;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .confidence-text {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #8B0000;
        box-shadow: 0 0 0 0.2rem rgba(139, 0, 0, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(139, 0, 0, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FAF3E0;
        border-radius: 10px;
        color: #8B0000;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B0000 0%, #A52A2A 100%);
        color: white;
    }
</style>

""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, feature names, and model info"""
    try:
        model = joblib.load('wine_quality_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, scaler, feature_names, model_info
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.stop()

def predict_wine_quality(features_dict, model, scaler, feature_names, model_name):
    """Make prediction using the trained model"""
    try:
        input_data = pd.DataFrame([features_dict])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        
        if model_name in ['Logistic Regression', 'SVM']:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            confidence = model.predict_proba(input_scaled)[0][1]
        else:
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][1]
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return 0, 0.5

def create_gauge_chart(confidence, is_good):
    """Create a modern gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea" if is_good else "#f5576c"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "#ffd700"},
                {'range': [80, 100], 'color': "#90EE90"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def main():
    # Load model artifacts
    model, scaler, feature_names, model_info = load_model_artifacts()
    
    # Header Section
    st.markdown("""
    <div class="header-section">
        <div class="header-title">🥂 Wine Quality Assessment System 🥂</div>
        <div class="header-subtitle">Professional Wine Quality Prediction for Boutique Wineries</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Performance Stats
    st.markdown(f"""
    <div class="stats-container">
        <div class="stat-box">
            <div class="stat-number">{model_info.get('accuracy', 0):.1%}</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{len(feature_names)}</div>
            <div class="stat-label">Chemical Parameters</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{model_info.get('model_name', 'ML')}</div>
            <div class="stat-label">Algorithm Used</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">{model_info.get('quality_threshold', 7)}</div>
            <div class="stat-label">Quality Threshold</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["🔎 Wine Analysis", "📈 Model Performance", "📑 User Guide"])
    
    with tab1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.subheader("Chemical Analysis Input")
        st.write("Enter the laboratory test results for your wine sample:")
        
        # Create input form
        with st.form("wine_analysis"):
            # Organize inputs in a clean grid
            col1, col2, col3 = st.columns(3)
            
            features_dict = {}
            
            # Default values
            defaults = {
                'fixed acidity': 7.4, 'volatile acidity': 0.7, 'citric acid': 0.0,
                'residual sugar': 1.9, 'chlorides': 0.076, 'free sulfur dioxide': 11.0,
                'total sulfur dioxide': 34.0, 'density': 0.9978, 'pH': 3.51,
                'sulphates': 0.56, 'alcohol': 9.4
            }
            
            # Create inputs
            for i, feature in enumerate(feature_names):
                col_idx = i % 3
                default_val = defaults.get(feature, 1.0)
                
                if col_idx == 0:
                    with col1:
                        features_dict[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            value=float(default_val),
                            step=0.01 if 'density' not in feature else 0.0001,
                            format="%.4f" if 'density' in feature else "%.2f"
                        )
                elif col_idx == 1:
                    with col2:
                        features_dict[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            value=float(default_val),
                            step=0.01 if 'density' not in feature else 0.0001,
                            format="%.4f" if 'density' in feature else "%.2f"
                        )
                else:
                    with col3:
                        features_dict[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            value=float(default_val),
                            step=0.01 if 'density' not in feature else 0.0001,
                            format="%.4f" if 'density' in feature else "%.2f"
                        )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analyze button
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                analyze_button = st.form_submit_button(
                    "🔎 Analyze Wine Quality",
                    use_container_width=True
                )
        
        # Results section
        if analyze_button:
            prediction, confidence = predict_wine_quality(
                features_dict, model, scaler, feature_names, 
                model_info.get('model_name', 'Unknown')
            )
            
            is_good = prediction == 1
            result_class = "result-good" if is_good else "result-standard"
            quality_text = "Premium Quality" if is_good else "Standard Quality"
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="result-box {result_class}">
                    <div class="result-title">🏆 {quality_text}</div>
                    <div class="confidence-text">Confidence: {confidence:.1%}</div>
                    <p>{'✅ Recommended for premium wine collection' if is_good 
                       else '⚠️ Suitable for standard wine offerings'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Quality interpretation
                if confidence > 0.8:
                    st.success("🎯 **Very High Confidence** - Excellent prediction reliability")
                elif confidence > 0.6:
                    st.info("👍 **Good Confidence** - Reliable prediction")
                else:
                    st.warning("🧐 **Moderate Confidence** - Consider additional testing")
            
            with col2:
                gauge_fig = create_gauge_chart(confidence, is_good)
                st.plotly_chart(gauge_fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Model Performance Metrics")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{model_info.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Model Type", model_info.get('model_name', 'Unknown'))
        with col3:
            st.metric("Features", len(feature_names))
        with col4:
            st.metric("Threshold", model_info.get('quality_threshold', 7))
        
        st.markdown("---")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("📊 Feature Importance Analysis")
            
            importance_df = pd.DataFrame({
                'Feature': [f.replace('_', ' ').title() for f in feature_names],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df.tail(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance analysis not available for this model type.")
    
    with tab3:
        st.subheader("📚 User Guide")
        
        st.markdown("""
        ### How to Use This System
        
        1. **Collect Wine Sample** - Obtain a representative sample of the wine to be tested
        2. **Laboratory Analysis** - Perform chemical analysis to get the required measurements
        3. **Input Data** - Enter all 11 chemical parameters in the analysis form
        4. **Get Results** - Click "Analyze Wine Quality" to receive prediction and confidence score
        
        ### Understanding Results
        
        **Quality Categories:**
        - ⭐⭐**Premium Quality**: Rating ≥ 7 - Suitable for premium wine collections
        - ⭐ **Standard Quality**: Rating < 7 - Good for regular wine offerings
        
        **Confidence Levels:**
        - **80-100%**: Very reliable prediction
        - **60-80%**: Good prediction reliability  
        - **40-60%**: Moderate reliability, consider additional testing
        - **Below 40%**: Low confidence, manual evaluation recommended
        
        ### Chemical Parameters Explained
        
        | Parameter | Description | Typical Range |
        |-----------|-------------|---------------|
        | Fixed Acidity | Non-volatile acids (tartaric, malic) | 4-16 g/L |
        | Volatile Acidity | Acetic acid content | 0.1-1.6 g/L |
        | Citric Acid | Adds freshness and flavor | 0-1 g/L |
        | Residual Sugar | Sugar remaining after fermentation | 1-15 g/L |
        | Chlorides | Salt content | 0.01-0.4 g/L |
        | Free SO₂ | Available sulfur dioxide | 1-80 mg/L |
        | Total SO₂ | Total sulfur dioxide content | 6-300 mg/L |
        | Density | Wine density | 0.99-1.01 g/cm³ |
        | pH | Acidity/alkalinity level | 2.7-4.0 |
        | Sulphates | Wine additive (potassium sulphate) | 0.3-2 g/L |
        | Alcohol | Ethanol percentage | 8-15% vol |
        
        ### Best Practices
        
        - Ensure accurate laboratory measurements
        - Test multiple samples for consistency
        - Use results as a screening tool alongside expert evaluation
        - Consider seasonal and batch variations
        - Keep records for quality tracking over time
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🥂 Professional Wine Quality Assessment System 🥂</p>
        <p><small>Powered by Machine Learning • Built for Boutique Wineries</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
