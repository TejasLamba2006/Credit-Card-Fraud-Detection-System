"""
Generate a basic pre-trained model for deployment
This creates a lightweight model that can be included in the GitHub repository
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os


def create_basic_model():
    """
    Create a basic model for deployment purposes
    """
    print("🤖 Creating basic fraud detection model...")

    # Check if we have a dataset
    dataset_file = None
    for filename in ["creditcard_sample.csv", "creditcard.csv"]:
        if os.path.exists(filename):
            dataset_file = filename
            break

    if not dataset_file:
        print("❌ No dataset found. Please run download.py first.")
        return False

    print(f"📊 Loading dataset: {dataset_file}")
    df = pd.read_csv(dataset_file)

    # Determine target column
    target_col = 'Class' if 'Class' in df.columns else 'isFraud'
    if target_col not in df.columns:
        print("❌ Target column not found. Expected 'Class' or 'isFraud'")
        return False

    print(f"🎯 Target column: {target_col}")
    print(f"📈 Dataset shape: {df.shape}")
    print(f"🚨 Fraud cases: {df[target_col].sum():,}")

    # Prepare features
    feature_columns = []

    # Add categorical features
    if 'type' in df.columns:
        feature_columns.append('type')
        categorical_features = ['type']
    else:
        categorical_features = []

    # Add numerical features
    numerical_features = []
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        if col in df.columns:
            numerical_features.append(col)
            feature_columns.append(col)

    if not numerical_features:
        print("❌ No suitable features found for training")
        return False

    print(f"🔧 Features: {feature_columns}")

    # Create feature matrix and target
    X = df[feature_columns]
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"🔄 Training set: {X_train.shape[0]:,} samples")
    print(f"🧪 Test set: {X_test.shape[0]:,} samples")

    # Create preprocessing pipeline (more compatible approach)
    if categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(drop="first",
                 handle_unknown="ignore"), categorical_features)
            ],
            remainder='passthrough',  # More compatible than 'drop'
            sparse_threshold=0  # Ensure dense output for compatibility
        )
    else:
        preprocessor = StandardScaler()

    # Create and train model pipeline
    if categorical_features:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear'  # More stable for small datasets
            ))
        ])
    else:
        pipeline = Pipeline([
            ("scaler", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear'
            ))
        ])

    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    print(f"✅ Training accuracy: {train_score:.3f}")
    print(f"✅ Test accuracy: {test_score:.3f}")

    # Save model
    model_file = "credit_card_fraud_model.pkl"
    joblib.dump(pipeline, model_file)

    # Get file size
    file_size_mb = os.path.getsize(model_file) / 1024 / 1024

    print(f"💾 Model saved: {model_file}")
    print(f"📦 Model size: {file_size_mb:.2f} MB")

    # Test prediction
    try:
        # Create a sample prediction
        sample_data = X_test.iloc[:1]
        prediction = pipeline.predict(sample_data)
        probability = pipeline.predict_proba(sample_data)

        print(
            f"🧪 Test prediction: {prediction[0]} (prob: {probability[0][1]:.3f})")
        print("✅ Model is working correctly!")

    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

    return True


def check_model_size():
    """
    Check if model is suitable for GitHub
    """
    model_file = "credit_card_fraud_model.pkl"
    if os.path.exists(model_file):
        file_size_mb = os.path.getsize(model_file) / 1024 / 1024
        print(f"\n📦 Model file size: {file_size_mb:.2f} MB")

        if file_size_mb < 25:  # GitHub has 100MB limit, be conservative
            print("✅ Model size is acceptable for GitHub")
            return True
        else:
            print("⚠️  Model might be too large for GitHub")
            print("Consider using Git LFS or external model hosting")
            return False
    else:
        print("❌ Model file not found")
        return False


if __name__ == "__main__":
    print("🤖 Basic Model Generator for Deployment")
    print("=" * 50)

    success = create_basic_model()

    if success:
        check_model_size()
        print("\n🎉 Model creation complete!")
        print("\nNext steps:")
        print("1. Test the model with: streamlit run streamlit_app.py")
        print("2. Add model to git: git add credit_card_fraud_model.pkl")
        print("3. Deploy to Streamlit Cloud")
    else:
        print("\n❌ Model creation failed")
        print("Please check the dataset and try again")
