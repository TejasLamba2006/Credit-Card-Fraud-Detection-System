import os
import pandas as pd
import kagglehub
from pathlib import Path
import shutil


def download_full_dataset():
    """
    Download the full credit card fraud dataset using kagglehub
    """
    print("📥 Downloading full credit card fraud dataset...")
    try:

        path = kagglehub.dataset_download(
            "amanalisiddiqui/fraud-detection-dataset")
        print("Path to dataset files:", path)

        for file in os.listdir(path):
            if file.endswith('.csv'):
                source = os.path.join(path, file)
                destination = "creditcard.csv"
                shutil.copy2(source, destination)
                print(f"✅ Dataset copied to: {destination}")
                return destination

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None


def create_sample_dataset(input_file="creditcard.csv", sample_size=50000):
    """
    Create a smaller sample dataset for GitHub and Streamlit deployment
    """
    sample_path = "creditcard_sample.csv"

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return None

    if os.path.exists(sample_path):
        print(f"✅ Sample dataset already exists: {sample_path}")
        return sample_path

    print(f"📊 Creating sample dataset with {sample_size:,} rows...")

    try:

        df = pd.read_csv(input_file)
        print(f"Original dataset: {len(df):,} rows")

        target_col = 'Class' if 'Class' in df.columns else 'isFraud'

        fraud_df = df[df[target_col] == 1]
        normal_df = df[df[target_col] == 0]

        normal_sample_size = min(sample_size - len(fraud_df), len(normal_df))
        normal_sample = normal_df.sample(n=normal_sample_size, random_state=42)

        sample_df = pd.concat([fraud_df, normal_sample]).sample(
            frac=1, random_state=42).reset_index(drop=True)
        sample_df.to_csv(sample_path, index=False)

        file_size_mb = os.path.getsize(sample_path) / 1024 / 1024

        print(f"✅ Sample dataset created: {sample_path}")
        print(f"   • Total rows: {len(sample_df):,}")
        print(f"   • Fraud cases: {sample_df[target_col].sum():,}")
        print(f"   • Normal cases: {(sample_df[target_col] == 0).sum():,}")
        print(f"   • File size: {file_size_mb:.1f} MB")

        return sample_path

    except Exception as e:
        print(f"❌ Error creating sample: {e}")
        return None


def setup_for_github():
    """
    Setup files for GitHub deployment with size limits
    """
    print("\n🐙 Setting up for GitHub deployment...")

    if os.path.exists("creditcard.csv"):
        file_size_mb = os.path.getsize("creditcard.csv") / 1024 / 1024
        print(f"Full dataset size: {file_size_mb:.1f} MB")

        if file_size_mb > 95:
            print("⚠️  Dataset too large for GitHub (>95MB)")
            print("Creating smaller sample for GitHub...")
            sample_path = create_sample_dataset(sample_size=30000)

            if sample_path:
                print("\n📋 GitHub Deployment Options:")
                print("1. Use the sample dataset (recommended for demo)")
                print("2. Set up Git LFS for the full dataset")
                print("3. Use external hosting for the dataset")

        else:
            print("✅ Dataset size is acceptable for GitHub")
    else:
        print("❌ No dataset found. Run download first.")


if __name__ == "__main__":
    print("🚀 Credit Card Fraud Dataset Setup")
    print("=" * 50)

    dataset_path = download_full_dataset()

    if dataset_path:
        create_sample_dataset()
        setup_for_github()

    print("\n✅ Setup Complete!")
    print("\nNext steps:")
    print("1. Use 'creditcard_sample.csv' for GitHub/Streamlit deployment")
    print("2. Keep 'creditcard.csv' for local development")
    print("3. Update your Streamlit app to use the sample dataset")
