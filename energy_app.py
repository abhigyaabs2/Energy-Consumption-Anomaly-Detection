import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import io

# Page configuration
st.set_page_config(
    page_title="Energy Anomaly Detector",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Energy Consumption Anomaly Detector")
st.markdown("""
This application uses **Isolation Forest** algorithm to detect unusual energy consumption patterns.
Upload your data or use sample data to get started!
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Configuration")
contamination = st.sidebar.slider(
    "Contamination Rate (Expected % of anomalies)",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
    help="The proportion of outliers in the dataset"
)

n_estimators = st.sidebar.slider(
    "Number of Trees",
    min_value=50,
    max_value=300,
    value=100,
    step=50,
    help="Number of isolation trees"
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=1000,
    value=42,
    help="For reproducibility"
)

# Function to generate sample data
def generate_sample_data(n_samples=500):
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    data = []
    for i, date in enumerate(dates):
        hour = date.hour
        day_of_week = date.dayofweek
        
        # Base consumption
        consumption = 100
        
        # Hourly pattern (higher during day)
        if 6 <= hour <= 22:
            consumption += 40 + np.random.normal(0, 10)
        else:
            consumption += 10 + np.random.normal(0, 5)
        
        # Weekly pattern (lower on weekends)
        if day_of_week >= 5:
            consumption *= 0.8
        
        # Seasonal pattern
        consumption += 20 * np.sin(2 * np.pi * i / (24 * 30))
        
        # Random noise
        consumption += np.random.normal(0, 5)
        
        # Inject anomalies (about 10%)
        if np.random.random() < 0.10:
            consumption *= np.random.choice([0.3, 2.5])
        
        data.append({
            'timestamp': date,
            'hour': hour,
            'day_of_week': day_of_week,
            'consumption': max(0, consumption),
            'temperature': 20 + 10 * np.sin(2 * np.pi * i / (24 * 365)) + np.random.normal(0, 3)
        })
    
    return pd.DataFrame(data)

# Function to detect anomalies
def detect_anomalies(df, contamination, n_estimators, random_state):
    # Select features for anomaly detection
    features = ['consumption', 'hour', 'day_of_week', 'temperature']
    X = df[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Predict anomalies (-1 for anomalies, 1 for normal)
    df['anomaly'] = iso_forest.fit_predict(X_scaled)
    df['anomaly_score'] = iso_forest.score_samples(X_scaled)
    
    # Convert to binary (1 for anomaly, 0 for normal)
    df['is_anomaly'] = (df['anomaly'] == -1).astype(int)
    
    return df, iso_forest

# Data upload section
st.header("📊 Data Input")
data_option = st.radio(
    "Choose data source:",
    ["Use Sample Data", "Upload CSV File"]
)

df = None

if data_option == "Use Sample Data":
    n_samples = st.slider("Number of samples to generate", 100, 1000, 500, 50)
    if st.button("Generate Sample Data"):
        df = generate_sample_data(n_samples)
        st.success(f"✅ Generated {len(df)} sample records!")

elif data_option == "Upload CSV File":
    st.markdown("""
    **CSV Format Requirements:**
    - Must contain a `consumption` column
    - Optional columns: `timestamp`, `hour`, `day_of_week`, `temperature`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        st.success(f"✅ Uploaded {len(df)} records!")

# Process data if available
if df is not None:
    st.header("🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Display basic statistics in expandable section
    with st.expander("📈 Basic Statistics", expanded=True):
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Mean Consumption", f"{df['consumption'].mean():.2f} kWh")
        with col3:
            st.metric("Median Consumption", f"{df['consumption'].median():.2f} kWh")
        with col4:
            st.metric("Std Deviation", f"{df['consumption'].std():.2f} kWh")
        with col5:
            st.metric("Range", f"{df['consumption'].max() - df['consumption'].min():.2f} kWh")
        
        # Additional statistics
        st.subheader("Distribution Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Minimum", f"{df['consumption'].min():.2f} kWh")
        with col2:
            st.metric("25th Percentile", f"{df['consumption'].quantile(0.25):.2f} kWh")
        with col3:
            st.metric("75th Percentile", f"{df['consumption'].quantile(0.75):.2f} kWh")
        with col4:
            st.metric("Maximum", f"{df['consumption'].max():.2f} kWh")
        
        # Skewness and Kurtosis
        from scipy import stats
        skewness = stats.skew(df['consumption'])
        kurtosis = stats.kurtosis(df['consumption'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skewness", f"{skewness:.3f}", 
                     help="Measure of asymmetry. 0 = symmetric, >0 = right-skewed, <0 = left-skewed")
        with col2:
            st.metric("Kurtosis", f"{kurtosis:.3f}",
                     help="Measure of tail heaviness. 0 = normal, >0 = heavy tails, <0 = light tails")
    
    # Detect anomalies
    if st.button("🚀 Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies..."):
            df, model = detect_anomalies(df, contamination, n_estimators, random_state)
            
            anomalies = df[df['is_anomaly'] == 1]
            normal = df[df['is_anomaly'] == 0]
            
            st.success(f"✅ Detection complete! Found {len(anomalies)} anomalies ({len(anomalies)/len(df)*100:.1f}%)")
            
            # Display anomaly statistics
            st.header("📈 Anomaly Detection Results")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Normal Points", len(normal))
            with col2:
                st.metric("Anomalies Detected", len(anomalies))
            with col3:
                st.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.1f}%")
            with col4:
                avg_anomaly_consumption = anomalies['consumption'].mean() if len(anomalies) > 0 else 0
                st.metric("Avg Anomaly Consumption", f"{avg_anomaly_consumption:.2f} kWh")
            
            # Statistical Comparison: Normal vs Anomalies
            with st.expander("📊 Statistical Comparison: Normal vs Anomalies", expanded=True):
                comparison_data = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                    'Normal': [
                        len(normal),
                        normal['consumption'].mean(),
                        normal['consumption'].median(),
                        normal['consumption'].std(),
                        normal['consumption'].min(),
                        normal['consumption'].max(),
                        normal['consumption'].quantile(0.25),
                        normal['consumption'].quantile(0.75)
                    ],
                    'Anomaly': [
                        len(anomalies),
                        anomalies['consumption'].mean() if len(anomalies) > 0 else 0,
                        anomalies['consumption'].median() if len(anomalies) > 0 else 0,
                        anomalies['consumption'].std() if len(anomalies) > 0 else 0,
                        anomalies['consumption'].min() if len(anomalies) > 0 else 0,
                        anomalies['consumption'].max() if len(anomalies) > 0 else 0,
                        anomalies['consumption'].quantile(0.25) if len(anomalies) > 0 else 0,
                        anomalies['consumption'].quantile(0.75) if len(anomalies) > 0 else 0
                    ]
                })
                
                st.dataframe(comparison_data.style.format({'Normal': '{:.2f}', 'Anomaly': '{:.2f}'}), 
                           use_container_width=True)
            
            # Anomaly Score Statistics
            with st.expander("🎯 Anomaly Score Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Normal Points Scores**")
                    st.write(f"Mean: {normal['anomaly_score'].mean():.4f}")
                    st.write(f"Median: {normal['anomaly_score'].median():.4f}")
                    st.write(f"Std Dev: {normal['anomaly_score'].std():.4f}")
                    st.write(f"Min: {normal['anomaly_score'].min():.4f}")
                    st.write(f"Max: {normal['anomaly_score'].max():.4f}")
                
                with col2:
                    st.write("**Anomaly Points Scores**")
                    if len(anomalies) > 0:
                        st.write(f"Mean: {anomalies['anomaly_score'].mean():.4f}")
                        st.write(f"Median: {anomalies['anomaly_score'].median():.4f}")
                        st.write(f"Std Dev: {anomalies['anomaly_score'].std():.4f}")
                        st.write(f"Min: {anomalies['anomaly_score'].min():.4f}")
                        st.write(f"Max: {anomalies['anomaly_score'].max():.4f}")
                    else:
                        st.write("No anomalies detected")
            
            # Time series plot with anomalies
            st.subheader("📉 Energy Consumption Timeline")
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=normal.index if 'timestamp' not in df.columns else normal['timestamp'],
                y=normal['consumption'],
                mode='markers',
                name='Normal',
                marker=dict(color='lightblue', size=5),
                opacity=0.6
            ))
            
            fig.add_trace(go.Scatter(
                x=anomalies.index if 'timestamp' not in df.columns else anomalies['timestamp'],
                y=anomalies['consumption'],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10, symbol='x'),
            ))
            
            fig.update_layout(
                title="Energy Consumption with Detected Anomalies",
                xaxis_title="Time" if 'timestamp' in df.columns else "Index",
                yaxis_title="Consumption (kWh)",
                hovermode='closest',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot comparison
            st.subheader("📦 Box Plot: Normal vs Anomalies")
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=normal['consumption'], name='Normal', marker_color='lightblue'))
            fig_box.add_trace(go.Box(y=anomalies['consumption'], name='Anomaly', marker_color='red'))
            fig_box.update_layout(
                title="Consumption Distribution Comparison",
                yaxis_title="Consumption (kWh)",
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Anomaly score distribution
            st.subheader("📊 Anomaly Score Distribution")
            fig2 = go.Figure()
            
            fig2.add_trace(go.Histogram(
                x=normal['anomaly_score'],
                name='Normal',
                marker_color='lightblue',
                opacity=0.7,
                nbinsx=50
            ))
            
            fig2.add_trace(go.Histogram(
                x=anomalies['anomaly_score'],
                name='Anomaly',
                marker_color='red',
                opacity=0.7,
                nbinsx=30
            ))
            
            fig2.update_layout(
                title="Distribution of Anomaly Scores",
                xaxis_title="Anomaly Score",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Hourly anomaly distribution
            if 'hour' in df.columns:
                st.subheader("⏰ Anomalies by Hour of Day")
                hourly_anomalies = anomalies.groupby('hour').size().reset_index(name='count')
                hourly_normal = normal.groupby('hour').size().reset_index(name='count')
                
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(x=hourly_normal['hour'], y=hourly_normal['count'], 
                                     name='Normal', marker_color='lightblue'))
                fig3.add_trace(go.Bar(x=hourly_anomalies['hour'], y=hourly_anomalies['count'], 
                                     name='Anomaly', marker_color='red'))
                
                fig3.update_layout(
                    title="Data Distribution by Hour of Day",
                    xaxis_title="Hour of Day",
                    yaxis_title="Count",
                    barmode='stack',
                    height=400
                )
                
                st.plotly_chart(fig3, use_container_width=True)
            
            # Show anomaly details
            st.subheader("🔍 Detected Anomalies Details")
            
            if len(anomalies) > 0:
                # Sort by anomaly score (most anomalous first)
                top_anomalies = anomalies.sort_values('anomaly_score').head(20)
                
                display_cols = ['consumption', 'hour', 'anomaly_score']
                if 'timestamp' in df.columns:
                    display_cols = ['timestamp'] + display_cols
                
                st.dataframe(
                    top_anomalies[display_cols],
                    use_container_width=True
                )
                
                # Anomaly severity classification
                st.subheader("⚠️ Anomaly Severity Classification")
                threshold_severe = anomalies['anomaly_score'].quantile(0.33)
                threshold_moderate = anomalies['anomaly_score'].quantile(0.67)
                
                severe = anomalies[anomalies['anomaly_score'] <= threshold_severe]
                moderate = anomalies[(anomalies['anomaly_score'] > threshold_severe) & 
                                   (anomalies['anomaly_score'] <= threshold_moderate)]
                mild = anomalies[anomalies['anomaly_score'] > threshold_moderate]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔴 Severe", len(severe), 
                             help="Most abnormal consumption patterns")
                with col2:
                    st.metric("🟡 Moderate", len(moderate),
                             help="Moderately unusual patterns")
                with col3:
                    st.metric("🟢 Mild", len(mild),
                             help="Slightly unusual patterns")
            
            # Download results
            st.subheader("💾 Download Results")
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV with Anomaly Labels",
                data=csv,
                file_name="energy_anomalies_detected.csv",
                mime="text/csv"
            )

# Information section
with st.expander("ℹ️ About Isolation Forest"):
    st.markdown("""
    **Isolation Forest** is an unsupervised machine learning algorithm for anomaly detection.
    
    **How it works:**
    - Builds multiple decision trees (isolation trees)
    - Anomalies are isolated quickly (fewer splits needed)
    - Normal points require more splits to be isolated
    - The algorithm assigns an anomaly score to each data point
    
    **Key Parameters:**
    - **Contamination:** Expected proportion of anomalies in the dataset
    - **Number of Trees:** More trees = better accuracy but slower
    - **Random State:** For reproducible results
    
    **Use Cases:**
    - Fraud detection
    - Network intrusion detection
    - Equipment failure prediction
    - Energy consumption monitoring
    """)

st.sidebar.markdown("---")