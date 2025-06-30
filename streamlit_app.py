import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color:
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color:
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color:
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid
    }
    .fraud-alert {
        background-color:
        color:
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid
    }
    .safe-alert {
        background-color:
        color:
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv("creditcard.csv")
        return df
    except FileNotFoundError:
        st.error(
            "Dataset 'creditcard.csv' not found. Please ensure the file is in the same directory.")
        return None


@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        model = joblib.load("credit_card_fraud_model.pkl")
        return model
    except FileNotFoundError:
        st.error(
            "Model file 'credit_card_fraud_model.pkl' not found. Please ensure you've trained and saved the model.")
        return None


def main():

    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>',
                unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:",
                                ["🏠 Dashboard",
                                 "📊 Data Analysis",
                                 "🔍 Fraud Detection",
                                 "📈 Model Performance",
                                 "ℹ️ About"])

    df = load_data()
    model = load_model()

    if df is None:
        st.stop()

    if page == "🏠 Dashboard":
        show_dashboard(df)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🔍 Fraud Detection":
        show_fraud_detection(df, model)
    elif page == "📈 Model Performance":
        show_model_performance(df, model)
    elif page == "ℹ️ About":
        show_about()


def show_dashboard(df):
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", f"{len(df):,}")

    with col2:
        fraud_count = df['isFraud'].sum()
        st.metric("Fraudulent Transactions", f"{fraud_count:,}")

    with col3:
        fraud_percentage = (fraud_count / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")

    with col4:
        total_amount = df['amount'].sum()
        st.metric("Total Transaction Amount", f"${total_amount:,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Type Distribution")
        type_counts = df['type'].value_counts()
        fig = px.bar(x=type_counts.index, y=type_counts.values,
                     title="Number of Transactions by Type",
                     labels={'x': 'Transaction Type', 'y': 'Count'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Rate by Transaction Type")
        fraud_by_type = df.groupby(
            'type')['isFraud'].mean().sort_values(ascending=False)
        fig = px.bar(x=fraud_by_type.index, y=fraud_by_type.values,
                     title="Fraud Rate by Transaction Type",
                     labels={'x': 'Transaction Type', 'y': 'Fraud Rate'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction Amount Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x='amount', nbins=50,
                           title="Distribution of Transaction Amounts (Linear Scale)",
                           labels={'amount': 'Transaction Amount', 'count': 'Frequency'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        df_log = df[df['amount'] > 0].copy()
        df_log['log_amount'] = np.log10(df_log['amount'])
        fig = px.histogram(df_log, x='log_amount', nbins=50,
                           title="Distribution of Transaction Amounts (Log Scale)",
                           labels={'log_amount': 'Log10(Transaction Amount)', 'count': 'Frequency'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_data_analysis(df):
    st.markdown('<h2 class="sub-header">📊 Detailed Data Analysis</h2>',
                unsafe_allow_html=True)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Column Names:**")
        st.write(df.columns.tolist())

    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)

    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        fig = px.bar(x=missing_data.index, y=missing_data.values,
                     title="Missing Values by Column")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation Analysis")

    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                      'oldbalanceDest', 'newbalanceDest', 'isFraud']
    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(corr_matrix,
                    title="Correlation Matrix of Numerical Features",
                    color_continuous_scale="RdBu_r",
                    aspect="auto")
    fig.update_layout(width=600, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Analysis by Transaction Amount")

    fraud_data = df[df['amount'] < 50000]

    fig = px.box(fraud_data, x='isFraud', y='amount',
                 title="Transaction Amount Distribution by Fraud Status",
                 labels={'isFraud': 'Fraud Status (0=Normal, 1=Fraud)', 'amount': 'Transaction Amount'})
    st.plotly_chart(fig, use_container_width=True)


def show_fraud_detection(df, model):
    st.markdown('<h2 class="sub-header">🔍 Real-time Fraud Detection</h2>',
                unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Please ensure the model file exists.")
        return

    st.write("Enter transaction details to check for potential fraud:")

    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox("Transaction Type",
                                        options=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
        amount = st.number_input(
            "Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
        old_balance_orig = st.number_input(
            "Original Account Old Balance ($)", min_value=0.0, value=1000.0)
        new_balance_orig = st.number_input(
            "Original Account New Balance ($)", min_value=0.0, value=900.0)

    with col2:
        old_balance_dest = st.number_input(
            "Destination Account Old Balance ($)", min_value=0.0, value=500.0)
        new_balance_dest = st.number_input(
            "Destination Account New Balance ($)", min_value=0.0, value=600.0)

    balance_diff_orig = old_balance_orig - new_balance_orig
    balance_diff_dest = old_balance_dest - new_balance_dest

    if st.button("🔍 Check for Fraud", type="primary"):

        input_data = pd.DataFrame({
            'type': [transaction_type],
            'amount': [amount],
            'oldbalanceOrg': [old_balance_orig],
            'newbalanceOrig': [new_balance_orig],
            'oldbalanceDest': [old_balance_dest],
            'newbalanceDest': [new_balance_dest],
            'balanceDiffOrig': [balance_diff_orig],
            'balanceDiffDest': [balance_diff_dest]
        })

        try:

            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            st.markdown("---")
            st.subheader("🎯 Prediction Results")

            if prediction == 1:
                st.markdown(f'''
                <div class="fraud-alert">
                    <h3>⚠️ FRAUD DETECTED</h3>
                    <p>This transaction has been flagged as potentially fraudulent.</p>
                    <p><strong>Fraud Probability: {prediction_proba[1]:.2%}</strong></p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="safe-alert">
                    <h3>✅ TRANSACTION APPEARS SAFE</h3>
                    <p>This transaction appears to be legitimate.</p>
                    <p><strong>Fraud Probability: {prediction_proba[1]:.2%}</strong></p>
                </div>
                ''', unsafe_allow_html=True)

            st.subheader("📊 Prediction Confidence")

            labels = ['Normal Transaction', 'Fraudulent Transaction']
            values = [prediction_proba[0], prediction_proba[1]]
            colors = ['#38a169', '#e53e3e']

            fig = go.Figure(
                data=[go.Bar(x=labels, y=values, marker_color=colors)])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

    st.markdown("---")
    st.subheader("🎯 Risk Factors to Consider")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**High-Risk Indicators:**")
        st.write("• TRANSFER and CASH_OUT transactions")
        st.write("• Account balance reaching zero after transaction")
        st.write("• Unusually large transaction amounts")
        st.write("• Rapid sequence of transactions")

    with col2:
        st.write("**Low-Risk Indicators:**")
        st.write("• PAYMENT transactions")
        st.write("• Normal balance changes")
        st.write("• Typical transaction amounts")
        st.write("• Consistent user behavior patterns")


def show_model_performance(df, model):
    st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>',
                unsafe_allow_html=True)

    if model is None:
        st.error("Model not loaded. Please ensure the model file exists.")
        return

    st.subheader("🎯 Model Insights")

    st.write(
        f"**Model Type:** {type(model.named_steps['classifier']).__name__}")

    st.subheader("📊 Class Distribution in Dataset")

    fraud_counts = df['isFraud'].value_counts()
    labels = ['Normal Transactions', 'Fraudulent Transactions']
    values = [fraud_counts[0], fraud_counts[1]]
    colors = ['#38a169', '#e53e3e']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_traces(marker=dict(colors=colors))
    fig.update_layout(title="Distribution of Transaction Classes", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Understanding Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Key Performance Metrics:**
        
        - **Precision**: Of all transactions flagged as fraud, how many were actually fraud?
        - **Recall**: Of all actual fraud cases, how many did we catch?
        - **F1-Score**: Harmonic mean of precision and recall
        - **Accuracy**: Overall percentage of correct predictions
        """)

    with col2:
        st.markdown("""
        **Why These Metrics Matter:**
        
        - **High Precision**: Reduces false alarms for customers
        - **High Recall**: Catches more actual fraud cases
        - **Balanced F1**: Good overall performance
        - **Business Impact**: Prevents financial losses
        """)

    st.subheader("⚠️ Model Limitations & Considerations")

    st.markdown("""
    **Important Considerations:**
    
    1. **Imbalanced Data**: Fraud cases are rare, making detection challenging
    2. **Feature Limitations**: Model performance depends on available features
    3. **Evolving Patterns**: Fraudsters adapt, requiring model updates
    4. **False Positives**: Some legitimate transactions may be flagged
    5. **Real-time Performance**: Production deployment requires optimization
    
    **Recommendations for Improvement:**
    
    - Collect more fraud examples for training
    - Engineer additional features (user behavior, time patterns)
    - Try ensemble methods (Random Forest, XGBoost)
    - Implement continuous learning with new data
    - Optimize decision thresholds for business needs
    """)


def show_about():
    st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>',
                unsafe_allow_html=True)

    st.markdown("""
    ## 🎯 Project Overview
    
    This Credit Card Fraud Detection System uses machine learning to identify potentially fraudulent transactions in real-time. The system analyzes transaction patterns, amounts, and account behaviors to flag suspicious activities.
    
    ## 🔧 Technical Implementation
    
    **Machine Learning Pipeline:**
    - Data preprocessing with StandardScaler and OneHotEncoder
    - Logistic Regression with balanced class weights
    - Scikit-learn Pipeline for streamlined processing
    - Model persistence with joblib
    
    **Technologies Used:**
    - **Python**: Core programming language
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation and analysis
    - **Scikit-learn**: Machine learning algorithms
    - **Plotly**: Interactive visualizations
    - **NumPy**: Numerical computations
    
    ## 📊 Dataset Features
    
    The model analyzes these key features:
    - **Transaction Type**: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
    - **Amount**: Transaction value
    - **Account Balances**: Before and after transaction states
    - **Balance Changes**: Engineered features showing account impact
    
    ## 🎯 Business Value
    
    **Financial Protection:**
    - Prevent unauthorized transactions
    - Reduce financial losses
    - Protect customer accounts
    
    **Operational Efficiency:**
    - Automated fraud detection
    - Reduced manual review time
    - Scalable monitoring system
    
    **Customer Experience:**
    - Faster transaction processing
    - Reduced false positives
    - Enhanced security confidence
    
    ## 🔮 Future Enhancements
    
    **Planned Improvements:**
    - Real-time transaction monitoring
    - Advanced ensemble models
    - User behavior analysis
    - Anomaly detection algorithms
    - Integration with banking systems
    
    ## 📝 Model Performance
    
    The current model provides:
    - High accuracy for fraud detection
    - Balanced precision and recall
    - Low false positive rates
    - Scalable prediction capability
    
    ## 🛡️ Security & Privacy
    
    **Data Protection:**
    - No personal information stored
    - Secure model deployment
    - Privacy-preserving analytics
    - Compliant with financial regulations
    
    ---
    
    **Developed by:** Your Name
    **Last Updated:** 2025
    **Version:** 1.0
    """)

    st.markdown("---")
    st.subheader("📧 Contact & Support")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📧 Email**  \nyour.email@domain.com")

    with col2:
        st.markdown(
            "**💼 LinkedIn**  \n[Your LinkedIn Profile](https://linkedin.com/in/tejaslamba)")

    with col3:
        st.markdown(
            "**🐙 GitHub**  \n[Project Repository](https://github.com/TejasLamba/)")


if __name__ == "__main__":
    main()
