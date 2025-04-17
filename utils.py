import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    
    # Convert date/time columns to datetime
    df['Date Reported'] = pd.to_datetime(df['Date Reported'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], format='%d-%m-%Y %H:%M', errors='coerce')
    
    # Extract time-based features
    df['Report_Month'] = df['Date Reported'].dt.month
    df['Report_Hour'] = df['Time of Occurrence'].dt.hour
    
    # Fill missing values for categorical columns
    df['Victim Gender'] = df['Victim Gender'].fillna('Unknown')
    df['Weapon Used'] = df['Weapon Used'].fillna('Unknown')
    
    # Drop unnecessary columns
    columns_to_drop = ['Report Number', 'Date Case Closed', 'Date Reported', 'Date of Occurrence', 'Time of Occurrence']
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
    
    return df

def get_crime_domains(file_path):
    """Get all unique crime domains from the dataset."""
    df = pd.read_csv(file_path)
    return df['Crime Domain'].unique().tolist()

def get_feature_importances(model, feature_columns):
    """Get feature importance rankings from the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_importances = [(feature_columns[i], importances[i]) for i in indices]
    return feature_importances

def train_case_closed_model(file_path):
    """Train model for Case Closed prediction and return model, scaler, and encoders."""
    df = load_and_preprocess_data(file_path)
    
    # Ensure Case Closed has valid values and drop rows with missing target
    df = df.dropna(subset=['Case Closed'])
    df['Case Closed'] = df['Case Closed'].map({'Yes': 1, 'No': 0})
    
    # One-hot encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.drop('Case Closed', errors='ignore')
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('Case Closed', axis=1)
    y = df_encoded['Case Closed']
    
    # Store column names before transformation
    feature_columns = X.columns.tolist()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
    
    # Scale features with feature names
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_columns)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No', 'Yes'])
    
    # Get feature importances
    feature_importances = get_feature_importances(rf, feature_columns)
    
    return rf, scaler, imputer, feature_columns, accuracy, report, feature_importances

def train_crime_domain_model(file_path):
    """Train model for Crime Domain prediction and return model, scaler, and encoders."""
    df = load_and_preprocess_data(file_path)
    
    # Drop rows with missing Crime Domain
    df = df.dropna(subset=['Crime Domain'])
    
    # Convert Crime Domain to categorical codes
    label_encoder = LabelEncoder()
    df['Crime Domain_encoded'] = label_encoder.fit_transform(df['Crime Domain'])
    
    # One-hot encode categorical features (except the target)
    df_without_target = df.drop('Crime Domain_encoded', axis=1)
    df_encoded = pd.get_dummies(df_without_target, drop_first=True)
    
    # Add back the target
    df_encoded['Crime Domain_encoded'] = df['Crime Domain_encoded']
    
    # Separate features and target
    X = df_encoded.drop('Crime Domain_encoded', axis=1)
    y = df_encoded['Crime Domain_encoded']
    
    # Store column names before transformation
    feature_columns = X.columns.tolist()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
    
    # Scale features with feature names
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_columns)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Get feature importances
    feature_importances = get_feature_importances(rf, feature_columns)
    
    return rf, scaler, imputer, feature_columns, label_encoder, accuracy, report, feature_importances

def train_victim_gender_model(file_path):
    """Train model for Victim Gender prediction and return model, scaler, and encoders."""
    df = load_and_preprocess_data(file_path)
    
    # Drop rows with missing Victim Gender
    df = df.dropna(subset=['Victim Gender'])
    
    # Separate features and target
    X = df.drop(columns=['Victim Gender'])
    y = df['Victim Gender']
    
    # Encode categorical features in X
    encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column])
        encoders[column] = encoder
    
    # Store column names before transformation
    feature_columns = X.columns.tolist()
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
    
    # Scale features with feature names
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_columns)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)
    
    # Get feature importances
    feature_importances = get_feature_importances(rf, feature_columns)
    
    return rf, scaler, imputer, feature_columns, encoders, target_encoder, accuracy, report, feature_importances

def predict_case_closed(model, scaler, imputer, feature_columns, input_data):
    """Predict Case Closed for new input data."""
    df_input = pd.DataFrame([input_data])
    
    # One-hot encode categorical features to match training
    categorical_cols = [col for col in df_input.columns if df_input[col].dtype == 'object']
    df_input_encoded = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)
    
    # Align columns with training data
    missing_cols = set(feature_columns) - set(df_input_encoded.columns)
    for col in missing_cols:
        df_input_encoded[col] = 0
    
    # Ensure columns are in the same order as training
    df_input_aligned = df_input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Impute missing values
    df_input_imputed = pd.DataFrame(imputer.transform(df_input_aligned), columns=feature_columns)
    
    # Scale features
    df_input_scaled = pd.DataFrame(scaler.transform(df_input_imputed), columns=feature_columns)
    
    # Predict with probability, keeping feature names
    prediction_proba = model.predict_proba(df_input_scaled)
    prediction_class = model.predict(df_input_scaled)
    print(f"Debug: Prediction probabilities = {prediction_proba[0]}")  # Debug print
    confidence = prediction_proba[0][prediction_class[0]] * 100
    
    return ('Yes' if prediction_class[0] == 1 else 'No'), confidence

def predict_crime_domain(model, scaler, imputer, feature_columns, label_encoder, input_data):
    """Predict Crime Domain for new input data."""
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)
    
    # Align columns with training data
    missing_cols = set(feature_columns) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    
    # Ensure columns are in same order as training
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    
    # Impute missing values
    df_input_imputed = pd.DataFrame(imputer.transform(df_input), columns=feature_columns)
    
    # Scale features
    df_input_scaled = pd.DataFrame(scaler.transform(df_input_imputed), columns=feature_columns)
    
    # Predict with probability, keeping feature names
    prediction_class = model.predict(df_input_scaled)
    prediction_proba = model.predict_proba(df_input_scaled)
    confidence = prediction_proba[0][prediction_class[0]] * 100
    
    return label_encoder.inverse_transform([prediction_class[0]])[0], confidence

def predict_victim_gender(model, scaler, imputer, feature_columns, encoders, target_encoder, input_data):
    """Predict Victim Gender for new input data."""
    df_input = pd.DataFrame([input_data])
    
    # Encode categorical features
    for column, encoder in encoders.items():
        if column in df_input.columns:
            try:
                df_input[column] = encoder.transform(df_input[column])
            except ValueError:
                # Handle unseen categories
                df_input[column] = 0
    
    # Ensure columns match training data
    missing_cols = set(feature_columns) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    
    # Ensure columns are in same order as training
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    
    # Impute missing values
    df_input_imputed = pd.DataFrame(imputer.transform(df_input), columns=feature_columns)
    
    # Scale features
    df_input_scaled = pd.DataFrame(scaler.transform(df_input_imputed), columns=feature_columns)
    
    # Predict with probability, keeping feature names
    prediction_class = model.predict(df_input_scaled)
    prediction_proba = model.predict_proba(df_input_scaled)
    confidence = prediction_proba[0][prediction_class[0]] * 100
    
    return target_encoder.inverse_transform([prediction_class[0]])[0], confidence