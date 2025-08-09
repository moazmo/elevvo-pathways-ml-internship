# 🛍️ Customer Segmentation Analysis - Mall Dataset

> **Part of Elevvo Pathways ML Internship Portfolio**  
> *Professional customer segmentation analysis using unsupervised machine learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org)

## 📋 Project Overview

This project performs comprehensive customer segmentation analysis on mall customer data using unsupervised machine learning techniques. The goal is to identify distinct customer segments based on their annual income and spending behavior to enable targeted marketing strategies.

## 🎯 Objectives

- ✅ Cluster customers into segments based on income and spending score
- ✅ Perform data scaling and visual exploration of groupings
- ✅ Apply K-Means clustering and determine optimal number of clusters
- ✅ Visualize clusters using 2D plots
- ✅ Compare with alternative clustering algorithms (DBSCAN)
- ✅ Analyze average spending per cluster and provide business insights

## 📊 Dataset

**Source**: Mall Customer Dataset (Kaggle)
**Features**:
- `CustomerID`: Unique identifier
- `Gender`: Male/Female
- `Age`: Customer age
- `Annual Income (k$)`: Income in thousands of dollars
- `Spending Score (1-100)`: Mall-assigned score based on customer behavior

**Size**: 201 customers

## 🛠️ Technologies Used

### Libraries
- **pandas & numpy**: Data manipulation and numerical operations
- **matplotlib & seaborn**: Static visualizations
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning algorithms and preprocessing
- **scipy**: Statistical analysis

### Algorithms
- **K-Means Clustering**: Primary clustering algorithm
- **DBSCAN**: Alternative density-based clustering
- **Elbow Method**: Optimal cluster number determination
- **Silhouette Analysis**: Cluster quality evaluation

## 📁 Project Structure

```
Task_1/
├── Mall_Customers.csv                    # Dataset
├── customer_segmentation_analysis.ipynb  # Main analysis notebook
├── requirements.txt                      # Dependencies
└── README.md                            # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Installation

1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `customer_segmentation_analysis.ipynb`

3. Run all cells to execute the complete analysis

## 📈 Key Features

### 1. Exploratory Data Analysis (EDA)
- Comprehensive data visualization
- Statistical summaries
- Correlation analysis
- Distribution analysis by demographics

### 2. Data Preprocessing
- Feature scaling (StandardScaler vs MinMaxScaler)
- Data quality checks
- Missing value analysis

### 3. Clustering Analysis
- **Elbow Method**: Determine optimal number of clusters
- **Silhouette Analysis**: Evaluate cluster quality
- **K-Means Implementation**: Primary clustering algorithm
- **DBSCAN Comparison**: Alternative clustering approach

### 4. Visualization
- 2D scatter plots of customer segments
- Cluster centers visualization
- Demographic analysis by cluster
- Comparative algorithm visualization

### 5. Business Insights
- Customer segment profiling
- Revenue potential analysis
- Targeted marketing strategies
- Actionable business recommendations

## 🎯 Key Results

### Customer Segments Identified
The analysis typically identifies 5 distinct customer segments:

1. **Budget Conscious**: Low income, low spending
2. **Young Spenders**: Low income, high spending
3. **Conservative High Earners**: High income, low spending
4. **Premium Customers**: High income, high spending
5. **Moderate Customers**: Balanced income and spending

### Business Value
- **Targeted Marketing**: Segment-specific campaigns
- **Resource Allocation**: Optimize customer service and inventory
- **Revenue Optimization**: Focus on high-value segments
- **Customer Retention**: Tailored retention strategies

## 📊 Performance Metrics

- **Silhouette Score**: Measures cluster quality (higher is better)
- **Within-Cluster Sum of Squares (WCSS)**: Measures cluster compactness
- **Cluster Distribution**: Balanced segment sizes
- **Business Relevance**: Actionable customer insights

## 🔄 Future Enhancements

- **Additional Features**: Incorporate purchase history, seasonality
- **Advanced Algorithms**: Try hierarchical clustering, Gaussian Mixture Models
- **Real-time Analysis**: Implement streaming clustering for new customers
- **A/B Testing**: Validate marketing strategies by segment
- **Predictive Modeling**: Predict customer segment migration

## 📝 Usage Examples

### Basic Clustering
```python
# Load and preprocess data
df = pd.read_csv('Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

### Visualization
```python
# Plot clusters
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=clusters)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.show()
```

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Authors

- **Data Science Team** - *Initial work and analysis*

## 🙏 Acknowledgments

- Kaggle for providing the Mall Customer dataset
- Scikit-learn community for excellent machine learning tools
- Matplotlib and Seaborn for visualization capabilities

---

**Note**: This analysis provides a comprehensive foundation for data-driven customer segmentation and targeted business strategies. The insights generated can be directly applied to improve marketing effectiveness and customer satisfaction.
