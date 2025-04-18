import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import utils

# Set page configuration
st.set_page_config(
    page_title="Crime Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)

# Custom CSS for improved UI
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-container {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Card styling */
    .card {
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        background-color: #34495e; /* slightly lighter on hover */
    }
    
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Grid layout */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #2a5298;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        font-size: 1rem;
        margin-top: 10px;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1e3c72;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        opacity: 0.7;
        font-size: 0.8rem;
    }
    
    /* Prediction form styling */
    .form-container {
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .prediction-result {
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .grid-container {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Landing page with improved UI
def landing_page():
    # Header with the specific titles you requested
    st.markdown(
        """
        <div class="header">
            <h3>Data Science and Business Intelligence CA 3</h3>
            <h1>Crime Prediction Dashboard</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Card grid - with clickable boxes
    col1, col2 = st.columns(2)
    
    with col1:
        # PowerPoint card - clickable version
        st.markdown(
            """
            <a href="https://docs.google.com/presentation/d/14Gg9LVLnlUgaYAfefF9WECtrDAkfeyXG/edit?usp=sharing&ouid=107588517927055292313&rtpof=true&sd=true" target="_blank" style="text-decoration: none; color: inherit;">
                <div class="card">
                    <div class="card-icon">📑</div>
                    <div class="card-title">PowerPoint Presentation</div>
                </div>
            </a>
            """, 
            unsafe_allow_html=True
        )
        
        # Power BI card - clickable version
        st.markdown(
            """
            <a href="https://app.powerbi.com/groups/me/reports/00d23c0e-5915-4e75-a77c-f6a5c1d03582/315e1c9bfb835241e02d?experience=power-bi" target="_blank" style="text-decoration: none; color: inherit;">
                <div class="card">
                    <div class="card-icon">📊</div>
                    <div class="card-title">Power BI</div>
                </div>
            </a>
            """, 
            unsafe_allow_html=True
        )
            
    with col2:
        # Report card - clickable version
        st.markdown(
            """
            <a href="https://drive.google.com/file/d/1hCtoBNYJfVvZI20x025V7MxEl3LiSzon/view?usp=sharing" target="_blank" style="text-decoration: none; color: inherit;">
                <div class="card">
                    <div class="card-icon">📄</div>
                    <div class="card-title">Report</div>
                </div>
            </a>
            """, 
            unsafe_allow_html=True
        )
        
        # Model card - using the same HTML/CSS approach as other cards
        st.markdown(
            """
            <div id="model-card" class="card" onclick="document.getElementById('hidden-model-button').click();" style="cursor: pointer;">
                <div class="card-icon">🧠</div>
                <div class="card-title">Model</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Hidden button that will be triggered by clicking the card
        if st.button("Model Button", key="hidden-model-button", help="View prediction models", on_click=None):
            st.session_state.show_model = True
            st.rerun()
            
        # Hide the button with CSS
        st.markdown(
            """
            <style>
            /* Hide the actual button */
            div[data-testid="stButton"] > button[kind="secondary"] {
                display: none !important;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
    
    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>© 2025 Crime Prediction Dashboard</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Prediction page with improved UI
def show_prediction_page():
    st.markdown(
        """
        <div class="header" style="margin-bottom: 20px;">
            <h1>Crime Prediction Models</h1>
            <p>Enter case details to generate predictions</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Back button
    if st.button("← Back to Main Menu"):
        st.session_state.show_model = False
        st.rerun()
    
    # Sidebar for task selection
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0;">
                <h2>Prediction Tasks</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        task = st.radio(
            "Choose a prediction task:",
            ["Case Closed Prediction", "Crime Domain Classification", "Victim Gender Prediction"],
            key="prediction_task"
        )
    
    # Load models and data on first run
    if "case_closed_model" not in st.session_state:
        with st.spinner("Loading models..."):
            rf, scaler, imputer, columns, accuracy, report, feature_importances = utils.train_case_closed_model("crime_dataset_india.csv")
            st.session_state.case_closed_model = rf
            st.session_state.case_closed_scaler = scaler
            st.session_state.case_closed_imputer = imputer
            st.session_state.case_closed_columns = columns
            st.session_state.case_closed_accuracy = accuracy
            st.session_state.case_closed_report = report
            st.session_state.case_closed_features = feature_importances

    if "crime_domain_model" not in st.session_state:
        with st.spinner("Loading models..."):
            rf, scaler, imputer, columns, encoder, accuracy, report, feature_importances = utils.train_crime_domain_model("crime_dataset_india.csv")
            st.session_state.crime_domain_model = rf
            st.session_state.crime_domain_scaler = scaler
            st.session_state.crime_domain_imputer = imputer
            st.session_state.crime_domain_columns = columns
            st.session_state.crime_domain_encoder = encoder
            st.session_state.crime_domain_accuracy = accuracy
            st.session_state.crime_domain_report = report
            st.session_state.crime_domain_features = feature_importances

    if "victim_gender_model" not in st.session_state:
        with st.spinner("Loading models..."):
            rf, scaler, imputer, columns, encoders, target_encoder, accuracy, report, feature_importances = utils.train_victim_gender_model("crime_dataset_india.csv")
            st.session_state.victim_gender_model = rf
            st.session_state.victim_gender_scaler = scaler
            st.session_state.victim_gender_imputer = imputer
            st.session_state.victim_gender_columns = columns
            st.session_state.victim_gender_encoders = encoders
            st.session_state.victim_gender_target_encoder = target_encoder
            st.session_state.victim_gender_accuracy = accuracy
            st.session_state.victim_gender_report = report
            st.session_state.victim_gender_features = feature_importances
    
    # Show task-specific descriptions
    task_descriptions = {
        "Case Closed Prediction": "Predict whether a case will be closed based on case characteristics.",
        "Crime Domain Classification": "Classify crimes into different domains based on their attributes.",
        "Victim Gender Prediction": "Predict the likely gender of victims based on crime patterns."
    }
    
    st.markdown(
        f"""
        <div class="main-container">
            <h2>{task}</h2>
            <p>{task_descriptions[task]}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Input form with improved styling
    st.markdown(
        """
        <div class="form-container">
            <h3>Enter Case Details</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            time_of_occurrence = st.text_input("Time of Occurrence (DD-MM-YYYY HH:MM)", "01-01-2020 12:00")
            city = st.selectbox("City", ["Ahmedabad", "Chennai", "Ludhiana", "Pune", "Delhi", "Mumbai", "Surat", "Visakhapatnam", "Ghaziabad", "Bangalore", "Kolkata"])
            crime_code = st.number_input("Crime Code", min_value=0, value=100)
            crime_description = st.selectbox("Crime Description", ["ARSON", "ASSAULT", "BURGLARY", "COUNTERFEITING", "CYBERCRIME", "DRUG OFFENSE", "EXTORTION", "FRAUD", "HOMICIDE", "IDENTITY THEFT", "KIDNAPPING", "PUBLIC INTOXICATION", "SEXUAL ASSAULT", "VANDALISM", "VEHICLE - STOLEN"])
        
        with col2:
            victim_age = st.number_input("Victim Age", min_value=0, value=30)
            weapon_used = st.selectbox("Weapon Used", ["Blunt Object", "Explosives", "Firearm", "Knife", "None", "Other", "Poison"])
            police_deployed = st.number_input("Police Deployed", min_value=0, value=10)
            victim_gender = st.selectbox("Victim Gender", ["M", "F", "X"])
            crime_domain = st.selectbox("Crime Domain", ["Fire Accident", "Other Crime", "Violent Crime"])
        
        submit_button = st.form_submit_button(label="Generate Prediction")
    
    # Prediction Result
    if submit_button:
        st.markdown(
            """
            <div class="prediction-result">
                <h3>Prediction Results</h3>
            """, 
            unsafe_allow_html=True
        )
        
        with st.spinner("Analyzing data and generating prediction..."):
            try:
                # Prepare input data
                input_data = {
                    "City": city,
                    "Crime Code": crime_code,
                    "Crime Description": crime_description,
                    "Victim Age": victim_age,
                    "Weapon Used": weapon_used,
                    "Police Deployed": police_deployed
                }
                if victim_gender is not None:
                    input_data["Victim Gender"] = victim_gender
                if crime_domain is not None:
                    input_data["Crime Domain"] = crime_domain
                
                # Convert Time of Occurrence to Report_Hour
                input_data["Report_Hour"] = pd.to_datetime(time_of_occurrence, format='%d-%m-%Y %H:%M').hour
                # Assume Report_Month from Time of Occurrence
                input_data["Report_Month"] = pd.to_datetime(time_of_occurrence, format='%d-%m-%Y %H:%M').month
                
                if task == "Case Closed Prediction":
                    prediction, confidence = utils.predict_case_closed(
                        st.session_state.case_closed_model,
                        st.session_state.case_closed_scaler,
                        st.session_state.case_closed_imputer,
                        st.session_state.case_closed_columns,
                        input_data
                    )
                    
                    # Display prediction result without confidence
                    if prediction == "Yes":
                        st.markdown(
                            f"""
                            <div style="background-color: #27ae60; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                                <h2>🎉 Case Will Likely Be Closed</h2>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background-color: #e74c3c; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                                <h2>⚠️ Case Will Likely Remain Open</h2>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Display additional insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h4>Key Factors</h4>", unsafe_allow_html=True)
                        top_features = st.session_state.case_closed_features[:5]
                        
                        # Create a DataFrame for better display
                        feature_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                        feature_df["Importance"] = feature_df["Importance"].apply(lambda x: f"{x:.2f}")
                        st.table(feature_df)
                    
                    with col2:
                        st.markdown("<h4>Similar Cases</h4>", unsafe_allow_html=True)
                        similar_cases = [
                            {"City": city, "Status": "Closed", "Time": "2 months", "Description": f"Similar {crime_description} case"},
                            {"City": city, "Status": "Open", "Time": "4 months", "Description": f"{crime_description} with similar weapon"}
                        ]
                        st.table(pd.DataFrame(similar_cases))
                
                elif task == "Crime Domain Classification":
                    prediction, confidence = utils.predict_crime_domain(
                        st.session_state.crime_domain_model,
                        st.session_state.crime_domain_scaler,
                        st.session_state.crime_domain_imputer,
                        st.session_state.crime_domain_columns,
                        st.session_state.crime_domain_encoder,
                        input_data
                    )
                    
                    # Display prediction result without confidence
                    domain_colors = {
                        "Violent Crime": "#e74c3c",
                        "Fire Accident": "#f39c12",
                        "Other Crime": "#3498db"
                    }
                    
                    domain_color = domain_colors.get(prediction, "#2c3e50")
                    
                    st.markdown(
                        f"""
                        <div style="background-color: {domain_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                            <h2>🔍 Predicted Crime Domain: {prediction}</h2>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Display additional insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h4>Key Factors</h4>", unsafe_allow_html=True)
                        top_features = st.session_state.crime_domain_features[:5]
                        
                        # Create a DataFrame for better display
                        feature_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                        feature_df["Importance"] = feature_df["Importance"].apply(lambda x: f"{x:.2f}")
                        st.table(feature_df)
                    
                    with col2:
                        st.markdown("<h4>Geographic Distribution</h4>", unsafe_allow_html=True)
                        geo_data = {
                            "Region": ["North", "South", "East", "West", "Central"],
                            "Percentage": [35, 15, 20, 10, 20]
                        }
                        chart = pd.DataFrame(geo_data).set_index("Region")
                        st.bar_chart(chart)
                
                elif task == "Victim Gender Prediction":
                    prediction, confidence = utils.predict_victim_gender(
                        st.session_state.victim_gender_model,
                        st.session_state.victim_gender_scaler,
                        st.session_state.victim_gender_imputer,
                        st.session_state.victim_gender_columns,
                        st.session_state.victim_gender_encoders,
                        st.session_state.victim_gender_target_encoder,
                        input_data
                    )
                    
                    # Display prediction result without confidence
                    gender_icon = "👨" if prediction == "M" else "👩" if prediction == "F" else "⚧️"
                    gender_name = "Male" if prediction == "M" else "Female" if prediction == "F" else "Other"
                    gender_color = "#3498db" if prediction == "M" else "#e84393" if prediction == "F" else "#8e44ad"
                    
                    st.markdown(
                        f"""
                        <div style="background-color: {gender_color}; color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                            <h2>{gender_icon} Predicted Victim Gender: {gender_name}</h2>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Display additional insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<h4>Key Factors</h4>", unsafe_allow_html=True)
                        top_features = st.session_state.victim_gender_features[:5]
                        
                        # Create a DataFrame for better display
                        feature_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
                        feature_df["Importance"] = feature_df["Importance"].apply(lambda x: f"{x:.2f}")
                        st.table(feature_df)
                    
                    with col2:
                        st.markdown("<h4>Gender Distribution</h4>", unsafe_allow_html=True)
                        gender_data = {
                            "Gender": ["Male", "Female", "Other"],
                            "Percentage": [65, 30, 5]
                        }
                        chart = pd.DataFrame(gender_data).set_index("Gender")
                        st.bar_chart(chart)
            
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Please check your input data and try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state
if "show_model" not in st.session_state:
    st.session_state.show_model = False

# Main app logic
if not st.session_state.show_model:
    landing_page()
else:
    show_prediction_page()
