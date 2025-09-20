import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .good-quality {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .bad-quality {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
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
        st.error("Please ensure all .pkl files are in the repository!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def predict_wine_quality(features_dict, model, scaler, feature_names, model_name):
    """Make prediction using the trained model"""
    try:
        # Create DataFrame from input
        input_data = pd.DataFrame([features_dict])
        
        # Ensure correct column order
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        
        # Handle missing columns by filling with 0 or median values
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Make prediction based on model type
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

def create_radar_chart(features_dict, feature_names):
    """Create a radar chart for wine characteristics"""
    # Define reasonable ranges for normalization
    feature_ranges = {
        'fixed acidity': (4, 16),
        'volatile acidity': (0, 2),
        'citric acid': (0, 1),
        'residual sugar': (0, 15),
        'chlorides': (0, 0.5),
        'free sulfur dioxide': (0, 80),
        'total sulfur dioxide': (0, 300),
        'density': (0.99, 1.01),
        'pH': (2.5, 4.5),
        'sulphates': (0, 2),
        'alcohol': (8, 15)
    }
    
    values = []
    labels = []
    
    for feature in feature_names:
        if feature in features_dict:
            # Get range for normalization
            min_val, max_val = feature_ranges.get(feature, (0, 1))
            # Normalize value
            normalized_value = (features_dict[feature] - min_val) / (max_val - min_val)
            normalized_value = max(0, min(1, normalized_value))  # Clamp to [0, 1]
            values.append(normalized_value)
            labels.append(feature.replace('_', ' ').title())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]] if values else [0],
        theta=labels + [labels[0]] if labels else [''],
        fill='toself',
        name='Wine Profile',
        line_color='#722F37',
        fillcolor='rgba(114, 47, 55, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Wine Chemical Profile",
        font_size=12,
        height=400
    )
    
    return fig

def get_feature_input_widgets(feature_names):
    """Create input widgets for all features"""
    features_dict = {}
    
    # Default values for common wine features
    default_values = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }
    
    # Create columns for better layout
    cols = st.columns(3)
    
    for i, feature in enumerate(feature_names):
        col_idx = i % 3
        with cols[col_idx]:
            # Get default value
            default_val = default_values.get(feature, 1.0)
            
            # Create appropriate input widget based on feature
            if 'density' in feature.lower():
                features_dict[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=0.98,
                    max_value=1.05,
                    value=float(default_val),
                    step=0.0001,
                    format="%.4f",
                    key=feature
                )
            elif 'ph' in feature.lower():
                features_dict[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=0.0,
                    max_value=14.0,
                    value=float(default_val),
                    step=0.01,
                    key=feature
                )
            else:
                features_dict[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=0.0,
                    value=float(default_val),
                    step=0.01,
                    key=feature
                )
    
    return features_dict

def main():
    # Load model artifacts
    model, scaler, feature_names, model_info = load_model_artifacts()
    
    # Header
    st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        <h3>AI-Powered Quality Assessment for Boutique Wines</h3>
        <p>Enter the chemical properties of your wine sample to predict its quality rating.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model Type:** {model_info.get('model_name', 'Unknown')}")
        st.write(f"**Accuracy:** {model_info.get('accuracy', 0):.3f}")
        st.write(f"**Features:** {model_info.get('feature_count', len(feature_names))}")
        st.write(f"**Quality Threshold:** {model_info.get('quality_threshold', 7)}")
        
        st.markdown("---")
        st.header("🎯 Quality Standards")
        st.success("**Premium Quality:** Rating ≥ 7")
        st.warning("**Standard Quality:** Rating < 7")
        
        st.markdown("---")
        st.header("💡 Tips")
        st.info("Higher alcohol content and lower volatile acidity typically indicate better quality wines.")

    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analysis", "ℹ️ About"])
    
    with tab1:
        st.subheader("🧪 Wine Chemical Properties")
        
        # Feature input form
        with st.form("wine_features"):
            st.write("Enter the chemical analysis results for your wine sample:")
            
            # Get feature inputs
            features_dict = get_feature_input_widgets(feature_names)
            
            # Predict button
            predict_button = st.form_submit_button(
                "🔮 Predict Wine Quality",
                use_container_width=True,
                type="primary"
            )
        
        if predict_button:
            # Make prediction
            prediction, confidence = predict_wine_quality(
                features_dict, model, scaler, feature_names, model_info.get('model_name', 'Unknown')
            )
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Prediction result
                is_good = prediction == 1
                quality_text = "Premium Quality" if is_good else "Standard Quality"
                quality_class = "good-quality" if is_good else "bad-quality"
                
                st.markdown(f"""
                <div class="prediction-box {quality_class}">
                    <h2>🏆 {quality_text}</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                    <p>{'✅ Meets premium quality standards' if is_good else '⚠️ Does not meet premium quality standards'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence interpretation
                if confidence > 0.8:
                    conf_text = "Very High"
                    conf_class = "confidence-high"
                elif confidence > 0.6:
                    conf_text = "High"
                    conf_class = "confidence-high"
                elif confidence > 0.4:
                    conf_text = "Medium"
                    conf_class = "confidence-medium"
                else:
                    conf_text = "Low"
                    conf_class = "confidence-low"
                
                st.markdown(f"""
                **Confidence Level:** <span class="{conf_class}">{conf_text}</span>
                
                **Recommendation:** {
                    'Excellent for premium collection' if confidence > 0.8 and is_good else
                    'Good quality, suitable for premium line' if confidence > 0.6 and is_good else
                    'Standard quality, good for regular offerings' if not is_good else
                    'Borderline case, consider additional testing'
                }
                """, unsafe_allow_html=True)
            
            with col2:
                # Confidence gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence %"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#722F37"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Wine profile radar chart
            st.subheader("📊 Wine Profile Analysis")
            radar_fig = create_radar_chart(features_dict, feature_names)
            st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Accuracy", f"{model_info.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Features Used", model_info.get('feature_count', len(feature_names)))
        with col3:
            st.metric("Quality Threshold", model_info.get('quality_threshold', 7))
        
        st.subheader("🔬 Feature Importance")
        if hasattr(model, 'feature_importances_'):
            # Create feature importance chart
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance for Wine Quality Prediction"
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
        
        st.subheader("📋 Feature Ranges")
        feature_info = {
            'Feature': feature_names,
            'Description': [f"Chemical property: {name.replace('_', ' ')}" for name in feature_names]
        }
        st.dataframe(pd.DataFrame(feature_info), use_container_width=True)
    
    with tab3:
        st.subheader("🍷 About Wine Quality Prediction")
        
        st.markdown("""
        ### Project Overview
        This application uses machine learning to predict wine quality based on chemical analysis.
        Developed for boutique wineries to assist in quality assurance processes.
        
        ### How It Works
        1. **Input:** Enter the results of chemical analysis for your wine sample
        2. **Processing:** Our trained ML model analyzes the chemical profile
        3. **Output:** Get a quality prediction with confidence score
        
        ### Features Analyzed
        The model considers multiple chemical properties that influence wine quality:
        - **Acidity levels** (fixed, volatile, citric acid, pH)
        - **Sugar content** (residual sugar)
        - **Preservatives** (sulfur dioxide, sulphates)
        - **Physical properties** (density, alcohol content)
        - **Mineral content** (chlorides)
        
        ### Model Performance
        - Trained on comprehensive wine dataset
        - Multiple algorithms tested and best performer selected
        - Validated using industry-standard metrics
        
        ### Usage Tips
        - Ensure accurate chemical measurements
        - Higher confidence scores indicate more reliable predictions
        - Consider multiple samples for better assessment
        - Use as a screening tool alongside expert evaluation
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>🍷 Wine Quality Predictor | Built with Streamlit & Machine Learning</p>
            <p><small>Developed for boutique winery quality assurance teams</small></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
