# ⚡ Energy Consumption Anomaly Detector

A machine learning project that uses **Isolation Forest** algorithm to detect unusual energy consumption patterns.

## 🎯 Features

- **Anomaly Detection**: Identify unusual energy consumption patterns
- **Interactive Visualization**: Real-time charts and graphs using Plotly
- **Data Upload**: Support for custom CSV data
- **Sample Data Generation**: Built-in synthetic data generator
- **Model Tuning**: Adjustable contamination rate and number of estimators
- **Export Results**: Download detected anomalies as CSV

## 📋 Project Structure

```
energy-anomaly-detector/
│
├── streamlit_app.py          # Main Streamlit application
├── notebook.ipynb             # Jupyter notebook for analysis
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # (Optional) Sample data folder
│   └── sample_energy.csv
│
└── models/                    # (Optional) Saved models
    ├── isolation_forest_model.pkl
    └── scaler.pkl
```

## 🚀 Quick Start

### 1. Clone/Create Project

```bash
# Create project directory
mkdir energy-anomaly-detector
cd energy-anomaly-detector
```

### 2. Run Jupyter Notebook (Optional)

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook notebook.ipynb
```

### 3. Run Streamlit App

```bash
streamlit run energy_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Using the Application

### Option 1: Use Sample Data

1. Select "Use Sample Data" in the app
2. Choose the number of samples (100-1000)
3. Click "Generate Sample Data"
4. Click "🚀 Detect Anomalies"

### Option 2: Upload Your Own Data

1. Prepare a CSV file with the following columns:
   - `consumption` (required): Energy consumption values
   - `timestamp` (optional): Date/time of the reading
   - `hour` (optional): Hour of day (0-23)
   - `day_of_week` (optional): Day of week (0-6)
   - `temperature` (optional): Temperature data

2. Example CSV format:
```csv
timestamp,consumption,hour,day_of_week,temperature
2024-01-01 00:00:00,120.5,0,0,22.3
2024-01-01 01:00:00,115.2,1,0,21.8
2024-01-01 02:00:00,110.8,2,0,21.5
```

3. Upload the file in the app
4. Click " Detect Anomalies"

## ⚙️ Configuration Parameters

### Contamination Rate
- **Definition**: Expected proportion of anomalies in the dataset
- **Range**: 0.01 - 0.30 (1% - 30%)
- **Default**: 0.10 (10%)
- **Tip**: If you expect fewer anomalies, decrease this value

### Number of Trees
- **Definition**: Number of isolation trees to build
- **Range**: 50 - 300
- **Default**: 100
- **Tip**: More trees = better accuracy but slower performance

### Random State
- **Definition**: Seed for reproducibility
- **Default**: 42

## 📈 Understanding Results

### Anomaly Score
- **Lower scores** = More likely to be an anomaly
- **Higher scores** = More likely to be normal
- The algorithm automatically determines the threshold based on contamination rate

### Visualizations

1. **Timeline Chart**: Shows all data points with anomalies highlighted
2. **Score Distribution**: Histogram comparing anomaly scores
3. **Hourly Distribution**: Bar chart showing when anomalies occur
4. **Anomaly Table**: Detailed list of detected anomalies

## 🔬 How Isolation Forest Works

Isolation Forest is an unsupervised learning algorithm that:

1. **Builds Random Trees**: Creates multiple decision trees with random splits
2. **Isolates Points**: Anomalies are isolated with fewer splits (shorter paths)
3. **Scores Points**: Calculates anomaly score based on average path length
4. **Detects Outliers**: Points with shorter paths are classified as anomalies

### Why It's Effective

- No need for labeled data
- Works well with high-dimensional data
- Fast and scalable
- Effective for various types of anomalies

```

## 📝 Example Use Cases

1. **Smart Homes**: Detect unusual appliance behavior
2. **Industrial Plants**: Identify equipment malfunctions
3. **Data Centers**: Monitor server power consumption
4. **Smart Grids**: Detect electricity theft or meter errors
5. **Building Management**: Optimize HVAC systems

## 🧪 Testing with Custom Data

Generate sample CSV for testing:

```python
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=100, freq='H')
consumption = 100 + np.random.randn(100) * 10
df = pd.DataFrame({
    'timestamp': dates,
    'consumption': consumption,
    'hour': dates.hour
})

df.to_csv('sample_energy.csv', index=False)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more ML algorithms (LOF, One-Class SVM)
- [ ] Real-time data streaming
- [ ] Email/SMS alerts for anomalies
- [ ] Model comparison dashboard
- [ ] Advanced feature engineering
- [ ] Time series forecasting

## 📚 Resources

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📄 License

MIT License - feel free to use for personal or commercial projects!

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Port already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Data not displaying
- Check CSV format matches requirements
- Ensure `consumption` column exists
- Verify no missing values in key columns

## 📧 Support

For questions or issues, please open an issue on GitHub or contact the maintainer.

---

**Built with ❤️ using Python, scikit-learn, and Streamlit**
