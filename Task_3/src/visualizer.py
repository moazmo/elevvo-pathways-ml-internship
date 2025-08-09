"""
Visualization utilities for the loan approval prediction system.
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
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
from src.config_loader import config


class LoanDataVisualizer:
    """Creates professional visualizations for loan approval analysis."""
    
    def __init__(self, df: pd.DataFrame, save_plots: bool = True):
        """
        Initialize visualizer.
        
        Args:
            df: DataFrame to visualize
            save_plots: Whether to save plots to results directory
        """
        self.df = df.copy()
        self.save_plots = save_plots
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.feature_config = config.get('features')
        self.target_col = self.feature_config['target_column']
        self.categorical_features = self.feature_config['categorical_features']
        self.numerical_features = self.feature_config['numerical_features']
        
        # Set up plotting parameters
        self.figsize = (12, 8)
        self.color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
    def plot_target_distribution(self) -> None:
        """Plot target variable distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        target_counts = self.df[self.target_col].value_counts()
        ax1.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                colors=self.color_palette[:2])
        ax1.set_title('Loan Approval Distribution', fontsize=14, fontweight='bold')
        
        # Bar plot with percentages
        sns.countplot(data=self.df, x=self.target_col, ax=ax2, palette=self.color_palette[:2])
        ax2.set_title('Loan Approval Counts', fontsize=14, fontweight='bold')
        
        # Add percentage labels on bars
        total = len(self.df)
        for p in ax2.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            ax2.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_missing_values(self, missing_data: Dict[str, Any]) -> None:
        """Plot missing values analysis."""
        missing_counts = missing_data['missing_counts']
        missing_percentages = missing_data['missing_percentages']
        
        # Filter out columns with no missing values
        missing_cols = {k: v for k, v in missing_counts.items() if v > 0}
        
        if not missing_cols:
            logger.info("No missing values found in the dataset")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing value counts
        cols = list(missing_cols.keys())
        counts = list(missing_cols.values())
        
        ax1.barh(cols, counts, color=self.color_palette[0])
        ax1.set_title('Missing Value Counts by Column', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Missing Values')
        
        # Missing value percentages
        percentages = [missing_percentages[col] for col in cols]
        ax2.barh(cols, percentages, color=self.color_palette[1])
        ax2.set_title('Missing Value Percentages by Column', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Percentage of Missing Values')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_numerical_distributions(self) -> None:
        """Plot distributions of numerical features."""
        n_features = len(self.numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(self.numerical_features):
            if feature in self.df.columns:
                ax = axes[i]
                
                # Create histogram with KDE
                self.df[feature].hist(bins=30, alpha=0.7, ax=ax, color=self.color_palette[0])
                ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                
                # Add statistics text
                mean_val = self.df[feature].mean()
                median_val = self.df[feature].median()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.0f}')
                ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.0f}')
                ax.legend()
        
        # Hide empty subplots
        for i in range(len(self.numerical_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_categorical_distributions(self) -> None:
        """Plot distributions of categorical features."""
        n_features = len(self.categorical_features)
        fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 6))
        
        if n_features == 1:
            axes = [axes]
        
        for i, feature in enumerate(self.categorical_features):
            if feature in self.df.columns:
                ax = axes[i]
                
                # Count plot
                sns.countplot(data=self.df, x=feature, ax=ax, palette=self.color_palette)
                ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                total = len(self.df)
                for p in ax.patches:
                    percentage = f'{100 * p.get_height() / total:.1f}%'
                    ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                               ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_target_relationships(self) -> None:
        """Plot relationships between features and target variable."""
        # Numerical features vs target
        n_numerical = len(self.numerical_features)
        if n_numerical > 0:
            n_cols = 3
            n_rows = (n_numerical + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for i, feature in enumerate(self.numerical_features):
                if feature in self.df.columns:
                    ax = axes[i]
                    
                    # Box plot by target
                    sns.boxplot(data=self.df, x=self.target_col, y=feature, ax=ax, palette=self.color_palette[:2])
                    ax.set_title(f'{feature} by Loan Status', fontsize=12, fontweight='bold')
            
            # Hide empty subplots
            for i in range(n_numerical, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.results_dir / 'numerical_target_relationships.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Categorical features vs target
        if self.categorical_features:
            fig, axes = plt.subplots(1, len(self.categorical_features), figsize=(8 * len(self.categorical_features), 6))
            
            if len(self.categorical_features) == 1:
                axes = [axes]
            
            for i, feature in enumerate(self.categorical_features):
                if feature in self.df.columns:
                    ax = axes[i]
                    
                    # Stacked bar plot
                    crosstab = pd.crosstab(self.df[feature], self.df[self.target_col], normalize='index') * 100
                    crosstab.plot(kind='bar', stacked=True, ax=ax, color=self.color_palette[:2])
                    ax.set_title(f'Approval Rate by {feature}', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Percentage')
                    ax.tick_params(axis='x', rotation=45)
                    ax.legend(title='Loan Status')
            
            plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.results_dir / 'categorical_target_relationships.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_correlation_heatmap(self, correlation_matrix: Dict[str, Any]) -> None:
        """Plot correlation heatmap for numerical features."""
        corr_df = pd.DataFrame(correlation_matrix)
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        
        sns.heatmap(corr_df, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.results_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_eda_report(self, eda_results: Dict[str, Any]) -> None:
        """Create comprehensive EDA report with all visualizations."""
        logger.info("Generating comprehensive EDA visualizations...")
        
        # Plot all visualizations
        self.plot_target_distribution()
        self.plot_missing_values(eda_results['missing_values'])
        self.plot_numerical_distributions()
        self.plot_categorical_distributions()
        self.plot_feature_target_relationships()
        self.plot_correlation_heatmap(eda_results['correlation_analysis']['correlation_matrix'])
        
        logger.info("EDA visualization report completed")
