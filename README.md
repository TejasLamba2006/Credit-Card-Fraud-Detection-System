# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive machine learning solution for detecting fraudulent credit card transactions in real-time. This project combines exploratory data analysis, feature engineering, and machine learning to create an accurate fraud detection system with an interactive web interface.

## 🎯 Project Overview

This fraud detection system analyzes transaction patterns, amounts, and account behaviors to identify potentially fraudulent activities. The solution includes:

- **Comprehensive Data Analysis**: In-depth exploration of transaction patterns and fraud indicators
- **Machine Learning Pipeline**: Automated preprocessing and classification system
- **Interactive Web App**: Real-time fraud detection interface built with Streamlit
- **Scalable Architecture**: Production-ready model with proper preprocessing pipelines

## 🚀 Features

### 📊 Data Analysis & Visualization

- Transaction type distribution analysis
- Fraud rate analysis by transaction categories
- Amount-based pattern recognition
- Correlation analysis of key features
- Time-based fraud pattern identification

### 🔍 Machine Learning Model

- **Algorithm**: Logistic Regression with balanced class weights
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical
- **Pipeline**: Integrated preprocessing and modeling workflow
- **Performance**: Optimized for imbalanced fraud detection scenarios

### 🖥️ Interactive Web Application

- **Real-time Predictions**: Instant fraud assessment for new transactions
- **Data Visualizations**: Interactive charts and graphs using Plotly
- **Performance Metrics**: Comprehensive model evaluation dashboard
- **User-friendly Interface**: Intuitive design for non-technical users

## 📁 Project Structure

```
Credit-Card-Fraud-Detection-System/
│
├── analysis_model.ipynb          # Complete data analysis and model training
├── streamlit_app.py              # Interactive web application
├── creditcard.csv                # Dataset (not included in repo)
├── credit_card_fraud_model.pkl   # Trained model (generated after training)
├── download.py                   # Data download utility
│
├── README.md                     # Project documentation (this file)
│
└── requirements.txt              # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/TejasLamba2006/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset

```bash
# Option 1: Run the download script
python download.py

# Option 2: Manual download
# Download the dataset from Kaggle and place creditcard.csv in the project directory
```

### Step 4: Train the Model

```bash
# Open and run the Jupyter notebook
jupyter notebook analysis_model.ipynb

# Run all cells to train and save the model
```

### Step 5: Launch the Web Application

```bash
streamlit run streamlit_app.py
```

## 📊 Dataset Information

The project uses a credit card transactions dataset containing:

- **284,807 transactions** with 31 features
- **492 fraud cases** (0.172% of all transactions)
- **Features include**: Transaction type, amount, account balances, time steps
- **Target variable**: Binary fraud indicator (0=Normal, 1=Fraud)

### Key Features

- `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- `amount`: Transaction amount
- `oldbalanceOrg/newbalanceOrig`: Origin account balance before/after transaction
- `oldbalanceDest/newbalanceDest`: Destination account balance before/after transaction
- `isFraud`: Target variable (1 if fraud, 0 if legitimate)

## 🤖 Model Performance

### Algorithm: Logistic Regression

- **Balanced Class Weights**: Handles imbalanced dataset effectively
- **Preprocessing Pipeline**: Standardized numerical features and encoded categorical variables
- **Cross-validation**: Stratified sampling to maintain class distribution

### Performance Metrics

- **Accuracy**: ~99.9% (typical for imbalanced datasets)
- **Precision**: High precision minimizes false positives
- **Recall**: Optimized to catch actual fraud cases
- **F1-Score**: Balanced performance metric

### Key Insights

1. **TRANSFER** and **CASH_OUT** transactions have highest fraud rates
2. **Account draining** patterns are strong fraud indicators
3. **Amount-based** patterns help distinguish fraudulent behavior
4. **Balance changes** provide crucial fraud detection signals

## 🌐 Web Application Features

### 🏠 Dashboard

- Dataset overview and key statistics
- Transaction type distribution visualizations
- Fraud rate analysis by category
- Amount distribution patterns

### 📊 Data Analysis

- Detailed statistical summaries
- Correlation analysis with interactive heatmaps
- Missing value analysis
- Feature distribution visualizations

### 🔍 Fraud Detection

- **Real-time prediction interface**
- **Input validation and preprocessing**
- **Probability scoring with confidence levels**
- **Risk factor explanations**

### 📈 Model Performance

- Model architecture details
- Performance metrics explanation
- Class distribution analysis
- Limitations and recommendations

## 🚀 Usage Examples

### Real-time Fraud Detection

```python
# Example transaction input
transaction = {
    'type': 'TRANSFER',
    'amount': 5000.00,
    'oldbalanceOrg': 10000.00,
    'newbalanceOrig': 5000.00,
    'oldbalanceDest': 0.00,
    'newbalanceDest': 5000.00
}

# Model prediction
fraud_probability = model.predict_proba(transaction)[0][1]
is_fraud = fraud_probability > 0.5
```

### Batch Processing

```python
# Process multiple transactions
transactions_df = pd.read_csv('new_transactions.csv')
predictions = model.predict(transactions_df)
probabilities = model.predict_proba(transactions_df)
```

## 📈 Future Enhancements

### Technical Improvements

- [ ] **Ensemble Methods**: Random Forest, XGBoost, Neural Networks
- [ ] **Feature Engineering**: Time-based features, user behavior patterns
- [ ] **Real-time Processing**: Apache Kafka integration for streaming data
- [ ] **Model Monitoring**: Performance tracking and drift detection

### Business Enhancements

- [ ] **Threshold Optimization**: Business-specific decision boundaries
- [ ] **Cost-sensitive Learning**: Incorporate financial impact of errors
- [ ] **Explainable AI**: LIME/SHAP for prediction explanations
- [ ] **A/B Testing**: Compare model versions in production

### Integration Features

- [ ] **API Development**: REST API for external system integration
- [ ] **Database Integration**: Real-time data pipeline connections
- [ ] **Alert System**: Automated notifications for high-risk transactions
- [ ] **Compliance Reporting**: Regulatory requirement fulfillment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Developer**: Tejas Lamba
- **Email**: <tejas22feb@gmail.com>
- **LinkedIn**: [Tejas Lamba](https://linkedin.com/in/tejaslamba)
- **Portfolio**: [Tejas Lamba's Portfolio](https://tejaslamba.com)

## 🏆 Acknowledgments

- Dataset provided by Machine Learning Group - ULB
- Inspired by real-world fraud detection challenges
- Built with open-source technologies and frameworks
- Community feedback and contributions

## 📊 Project Stats

![GitHub repo size](https://img.shields.io/github/repo-size/TejasLamba2006/Credit-Card-Fraud-Detection-System)
![GitHub stars](https://img.shields.io/github/stars/TejasLamba2006/Credit-Card-Fraud-Detection-System)
![GitHub forks](https://img.shields.io/github/forks/TejasLamba2006/Credit-Card-Fraud-Detection-System)
![GitHub issues](https://img.shields.io/github/issues/TejasLamba2006/Credit-Card-Fraud-Detection-System)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
