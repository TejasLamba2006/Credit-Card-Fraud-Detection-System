"""
Simple, version-agnostic model creation for Streamlit deployment
This creates a basic model without complex preprocessing that can work across sklearn versions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def create_simple_model():
    """
    Create a simple model without ColumnTransformer to avoid version issues
    """
    print("🤖 Creating simple fraud detection model...")

    # Check if we have a dataset
    dataset_file = None
    for filename in ["creditcard_sample.csv", "creditcard.csv"]:
        if os.path.exists(filename):
            dataset_file = filename
            break

    if not dataset_file:
        print("❌ No dataset found.")
        return False

    print(f"📊 Loading dataset: {dataset_file}")
    df = pd.read_csv(dataset_file)

    # Determine target column
    target_col = 'Class' if 'Class' in df.columns else 'isFraud'
    if target_col not in df.columns:
        print("❌ Target column not found.")
        return False

    print(f"🎯 Target column: {target_col}")
    print(f"📈 Dataset shape: {df.shape}")
    print(f"🚨 Fraud cases: {df[target_col].sum():,}")

    # Simple feature preparation (numerical only to avoid version issues)
    numerical_features = []
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        if col in df.columns:
            numerical_features.append(col)

    # Handle categorical 'type' column manually
    if 'type' in df.columns:
        # Simple label encoding for 'type'
        df_model = df.copy()
        le = LabelEncoder()
        df_model['type_encoded'] = le.fit_transform(df_model['type'])
        feature_columns = numerical_features + ['type_encoded']

        # Save the label encoder for later use
        joblib.dump(le, "label_encoder.pkl")
        print("💾 Label encoder saved")
    else:
        feature_columns = numerical_features
        df_model = df.copy()

    if not feature_columns:
        print("❌ No suitable features found")
        return False

    print(f"🔧 Features: {feature_columns}")

    # Create feature matrix and target
    X = df_model[feature_columns].values  # Convert to numpy array
    y = df_model[target_col].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"🔄 Training set: {X_train.shape[0]:,} samples")
    print(f"🧪 Test set: {X_test.shape[0]:,} samples")

    # Simple preprocessing - just scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simple model
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='liblinear'
    )

    print("🚀 Training model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"✅ Training accuracy: {train_score:.3f}")
    print(f"✅ Test accuracy: {test_score:.3f}")

    # Save model components separately (more compatible)
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_columns,
        'has_categorical': 'type' in df.columns
    }

    # Save as a simple dictionary
    joblib.dump(model_data, "simple_fraud_model.pkl")

    # Get file size
    file_size_mb = os.path.getsize("simple_fraud_model.pkl") / 1024 / 1024

    print(f"💾 Model saved: simple_fraud_model.pkl")
    print(f"📦 Model size: {file_size_mb:.2f} MB")

    # Test prediction
    try:
        # Test with first sample
        sample_data = X_test_scaled[:1]
        prediction = model.predict(sample_data)
        probability = model.predict_proba(sample_data)

        print(
            f"🧪 Test prediction: {prediction[0]} (prob: {probability[0][1]:.3f})")
        print("✅ Model is working correctly!")

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

    return True


if __name__ == "__main__":
    print("🤖 Simple Model Generator for Streamlit Deployment")
    print("=" * 55)

    success = create_simple_model()

    if success:
        print("\n🎉 Simple model creation complete!")
        print("\nFiles created:")
        print("- simple_fraud_model.pkl (main model)")
        print("- label_encoder.pkl (if categorical features exist)")
        print("\nNext steps:")
        print("1. Update streamlit_app.py to use simple_fraud_model.pkl")
        print("2. Deploy to Streamlit Cloud")
    else:
        print("\n❌ Model creation failed")
