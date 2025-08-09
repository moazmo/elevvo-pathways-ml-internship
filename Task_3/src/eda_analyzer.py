"""
Exploratory Data Analysis for the loan approval prediction system.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
from pathlib import Path
from loguru import logger
from src.config_loader import config


class EDAAnalyzer:
    """Comprehensive Exploratory Data Analysis for loan approval data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA analyzer.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df.copy()
        self.feature_config = config.get('features')
        self.target_col = self.feature_config['target_column']
        self.categorical_features = self.feature_config['categorical_features']
        self.numerical_features = self.feature_config['numerical_features']
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset."""
        missing_data = {
            'missing_counts': self.df.isnull().sum(),
            'missing_percentages': (self.df.isnull().sum() / len(self.df)) * 100,
            'total_missing': self.df.isnull().sum().sum(),
            'rows_with_missing': self.df.isnull().any(axis=1).sum()
        }
        
        logger.info(f"Total missing values: {missing_data['total_missing']}")
        logger.info(f"Rows with missing values: {missing_data['rows_with_missing']}")
        
        return missing_data
    
    def analyze_target_distribution(self) -> Dict[str, Any]:
        """Analyze target variable distribution and class imbalance."""
        target_counts = self.df[self.target_col].value_counts()
        target_percentages = self.df[self.target_col].value_counts(normalize=True) * 100
        
        imbalance_ratio = target_counts.min() / target_counts.max()
        
        distribution_info = {
            'counts': target_counts.to_dict(),
            'percentages': target_percentages.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio < 0.8
        }
        
        logger.info(f"Target distribution: {distribution_info['counts']}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.3f}")
        
        return distribution_info
    
    def analyze_numerical_features(self) -> Dict[str, Any]:
        """Analyze numerical features distribution and statistics."""
        numerical_analysis = {}
        
        for feature in self.numerical_features:
            if feature in self.df.columns:
                feature_data = self.df[feature].dropna()
                
                analysis = {
                    'basic_stats': feature_data.describe().to_dict(),
                    'skewness': feature_data.skew(),
                    'kurtosis': feature_data.kurtosis(),
                    'outliers_iqr': self._detect_outliers_iqr(feature_data),
                    'zero_values': (feature_data == 0).sum(),
                    'unique_values': feature_data.nunique()
                }
                
                numerical_analysis[feature] = analysis
        
        return numerical_analysis
    
    def analyze_categorical_features(self) -> Dict[str, Any]:
        """Analyze categorical features distribution."""
        categorical_analysis = {}
        
        for feature in self.categorical_features:
            if feature in self.df.columns:
                feature_data = self.df[feature].dropna()
                
                analysis = {
                    'value_counts': feature_data.value_counts().to_dict(),
                    'unique_count': feature_data.nunique(),
                    'mode': feature_data.mode().iloc[0] if len(feature_data.mode()) > 0 else None,
                    'target_relationship': self._analyze_categorical_target_relationship(feature)
                }
                
                categorical_analysis[feature] = analysis
        
        return categorical_analysis
    
    def _detect_outliers_iqr(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(series)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_values': outliers.tolist()[:10]  # First 10 outliers
        }
    
    def _analyze_categorical_target_relationship(self, feature: str) -> Dict[str, Any]:
        """Analyze relationship between categorical feature and target."""
        crosstab = pd.crosstab(self.df[feature], self.df[self.target_col], normalize='index') * 100
        
        return {
            'approval_rates': crosstab.to_dict(),
            'chi_square_test': self._chi_square_test(feature)
        }
    
    def _chi_square_test(self, feature: str) -> Dict[str, float]:
        """Perform chi-square test for categorical feature and target."""
        from scipy.stats import chi2_contingency
        
        try:
            contingency_table = pd.crosstab(self.df[feature], self.df[self.target_col])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            return {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'is_significant': p_value < 0.05
            }
        except Exception as e:
            logger.warning(f"Chi-square test failed for {feature}: {e}")
            return {'error': str(e)}
    
    def generate_correlation_analysis(self) -> Dict[str, Any]:
        """Generate correlation analysis for numerical features."""
        numerical_df = self.df[self.numerical_features].select_dtypes(include=[np.number])
        
        correlation_matrix = numerical_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlation_pairs': high_corr_pairs
        }
    
    def run_comprehensive_eda(self) -> Dict[str, Any]:
        """
        Run comprehensive EDA and return all analysis results.
        
        Returns:
            Dictionary containing all EDA results
        """
        logger.info("Starting comprehensive EDA analysis...")
        
        eda_results = {
            'data_overview': {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'dtypes': self.df.dtypes.to_dict()
            },
            'missing_values': self.analyze_missing_values(),
            'target_distribution': self.analyze_target_distribution(),
            'numerical_analysis': self.analyze_numerical_features(),
            'categorical_analysis': self.analyze_categorical_features(),
            'correlation_analysis': self.generate_correlation_analysis()
        }
        
        logger.info("EDA analysis completed successfully")
        return eda_results
