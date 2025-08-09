"""
Visualization utilities for model evaluation and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger


class EvaluationVisualizer:
    """Creates professional visualizations for model evaluation."""
    
    def __init__(self, save_plots: bool = True):
        """
        Initialize evaluation visualizer.
        
        Args:
            save_plots: Whether to save plots to results directory
        """
        self.save_plots = save_plots
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up plotting parameters
        self.figsize = (12, 8)
        self.color_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3A86FF', '#06FFA5']
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette(self.color_palette)
    
    def plot_confusion_matrices(self, evaluation_results: Dict[str, Dict[str, Any]]) -> None:
        """Plot confusion matrices for all models."""
        n_models = len(evaluation_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            ax = axes[i]
            
            cm = np.array(results['confusion_matrix'])
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Rejected', 'Approved'],
                       yticklabels=['Rejected', 'Approved'])
            
            ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, evaluation_results: Dict[str, Dict[str, Any]], 
                       y_test: np.ndarray, models_dict: Dict[str, Any], X_test: np.ndarray) -> None:
        """Plot ROC curves for all models."""
        plt.figure(figsize=self.figsize)
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            if model_name in models_dict:
                model = models_dict[model_name]
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    auc_score = results.get('roc_auc', 0)
                    
                    plt.plot(fpr, tpr, color=self.color_palette[i % len(self.color_palette)],
                            label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(self.results_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, evaluation_results: Dict[str, Dict[str, Any]], 
                                   y_test: np.ndarray, models_dict: Dict[str, Any], X_test: np.ndarray) -> None:
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=self.figsize)
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            if model_name in models_dict:
                model = models_dict[model_name]
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    avg_precision = results.get('average_precision', 0)
                    
                    plt.plot(recall, precision, color=self.color_palette[i % len(self.color_palette)],
                            label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
        
        # Plot baseline
        baseline = np.sum(y_test) / len(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(self.results_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, comparison_analysis: Dict[str, Any]) -> None:
        """Plot model comparison metrics."""
        comparison_df = pd.DataFrame(comparison_analysis['comparison_table'])
        
        # Select key metrics for comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if 'ROC-AUC' in comparison_df.columns:
            available_metrics.append('ROC-AUC')
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Bar plot for each metric
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=self.color_palette[:len(comparison_df)])
            
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, evaluation_results: Dict[str, Dict[str, Any]], top_n: int = 15) -> None:
        """Plot feature importance for models that support it."""
        models_with_importance = {
            name: results for name, results in evaluation_results.items()
            if 'feature_importance' in results or 'feature_coefficients' in results
        }
        
        if not models_with_importance:
            logger.warning("No models with feature importance found")
            return
        
        n_models = len(models_with_importance)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 6 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(models_with_importance.items()):
            ax = axes[i]
            
            # Get importance values
            if 'feature_importance' in results:
                importance_dict = results['feature_importance']
                title_suffix = "Feature Importance"
            else:
                importance_dict = results['feature_coefficients']
                title_suffix = "Feature Coefficients (Absolute Values)"
                # Use absolute values for coefficients
                importance_dict = {k: abs(v) for k, v in importance_dict.items()}
            
            # Get top N features
            top_features = dict(list(importance_dict.items())[:top_n])
            
            # Create horizontal bar plot
            features = list(top_features.keys())
            values = list(top_features.values())
            
            bars = ax.barh(features, values, color=self.color_palette[i % len(self.color_palette)])
            ax.set_title(f'{model_name} - {title_suffix}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Importance Score')
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.annotate(f'{width:.3f}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0),  # 3 points horizontal offset
                           textcoords="offset points",
                           ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_business_impact_analysis(self, evaluation_results: Dict[str, Dict[str, Any]]) -> None:
        """Plot business impact analysis focusing on Type I and Type II errors."""
        business_data = []
        
        for model_name, results in evaluation_results.items():
            business_metrics = results['business_metrics']
            business_data.append({
                'Model': model_name,
                'Type I Error Rate': business_metrics['type_i_error_rate'] * 100,  # Convert to percentage
                'Type II Error Rate': business_metrics['type_ii_error_rate'] * 100,
                'Cost Ratio': business_metrics['cost_ratio'] * 100,
                'Opportunity Loss Ratio': business_metrics['opportunity_loss_ratio'] * 100
            })
        
        business_df = pd.DataFrame(business_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Type I vs Type II Error Rates
        x = np.arange(len(business_df))
        width = 0.35
        
        ax1.bar(x - width/2, business_df['Type I Error Rate'], width, 
               label='Type I Error (Bad Loans Approved)', color=self.color_palette[0])
        ax1.bar(x + width/2, business_df['Type II Error Rate'], width,
               label='Type II Error (Good Loans Rejected)', color=self.color_palette[1])
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Error Rate (%)')
        ax1.set_title('Type I vs Type II Error Rates', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(business_df['Model'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cost vs Opportunity Loss
        ax2.scatter(business_df['Type I Error Rate'], business_df['Type II Error Rate'], 
                   s=100, c=range(len(business_df)), cmap='viridis')
        
        for i, model in enumerate(business_df['Model']):
            ax2.annotate(model, (business_df.iloc[i]['Type I Error Rate'], 
                               business_df.iloc[i]['Type II Error Rate']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Type I Error Rate (%)')
        ax2.set_ylabel('Type II Error Rate (%)')
        ax2.set_title('Error Rate Trade-off Analysis', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Cost and Opportunity Loss Ratios
        ax3.bar(business_df['Model'], business_df['Cost Ratio'], 
               color=self.color_palette[2], alpha=0.7)
        ax3.set_title('Cost Ratio (% of Bad Loans Approved)', fontweight='bold')
        ax3.set_ylabel('Cost Ratio (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(business_df['Model'], business_df['Opportunity Loss Ratio'], 
               color=self.color_palette[3], alpha=0.7)
        ax4.set_title('Opportunity Loss Ratio (% of Good Loans Rejected)', fontweight='bold')
        ax4.set_ylabel('Opportunity Loss Ratio (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'business_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance_radar(self, evaluation_results: Dict[str, Dict[str, Any]]) -> None:
        """Create radar chart comparing model performance across multiple metrics."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Prepare data
        models_data = {}
        for model_name, results in evaluation_results.items():
            models_data[model_name] = [results.get(metric, 0) for metric in metrics]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, (model_name, values) in enumerate(models_data.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=model_name, color=self.color_palette[i % len(self.color_palette)])
            ax.fill(angles, values, alpha=0.1, color=self.color_palette[i % len(self.color_palette)])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Comparison\n(Radar Chart)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        if self.save_plots:
            plt.savefig(self.results_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_evaluation_dashboard(self, evaluation_results: Dict[str, Dict[str, Any]], 
                                  y_test: np.ndarray, models_dict: Dict[str, Any], 
                                  X_test: np.ndarray, comparison_analysis: Dict[str, Any]) -> None:
        """
        Create comprehensive evaluation dashboard with all visualizations.
        
        Args:
            evaluation_results: Model evaluation results
            y_test: Test target values
            models_dict: Dictionary of trained models
            X_test: Test features
            comparison_analysis: Model comparison analysis
        """
        logger.info("Creating comprehensive evaluation dashboard...")
        
        # Generate all visualizations
        self.plot_confusion_matrices(evaluation_results)
        self.plot_roc_curves(evaluation_results, y_test, models_dict, X_test)
        self.plot_precision_recall_curves(evaluation_results, y_test, models_dict, X_test)
        self.plot_model_performance_radar(evaluation_results)
        self.plot_business_impact_analysis(evaluation_results)
        
        # Create summary visualization
        self._plot_evaluation_summary(comparison_analysis)
        
        logger.info("Evaluation dashboard created successfully")
    
    def _plot_evaluation_summary(self, comparison_analysis: Dict[str, Any]) -> None:
        """Create evaluation summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        comparison_df = pd.DataFrame(comparison_analysis['comparison_table'])
        
        # Overall performance comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(available_metrics):
            ax1.bar(x + i * width, comparison_df[metric], width, 
                   label=metric, color=self.color_palette[i])
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Performance Comparison', fontweight='bold')
        ax1.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax1.set_xticklabels(comparison_df['Model'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Best model by metric
        best_models = comparison_analysis['best_models_by_metric']
        metrics_list = list(best_models.keys())
        models_list = [best_models[m]['model'] for m in metrics_list]
        scores_list = [best_models[m]['score'] for m in metrics_list]
        
        bars = ax2.bar(metrics_list, scores_list, color=self.color_palette[:len(metrics_list)])
        ax2.set_title('Best Model by Metric', fontweight='bold')
        ax2.set_ylabel('Best Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add model names on bars
        for i, (bar, model) in enumerate(zip(bars, models_list)):
            height = bar.get_height()
            ax2.annotate(model,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Error analysis
        if 'Type I Error Rate' in comparison_df.columns:
            ax3.bar(comparison_df['Model'], comparison_df['Type I Error Rate'] * 100,
                   color=self.color_palette[0], alpha=0.7, label='Type I Error')
            ax3.bar(comparison_df['Model'], comparison_df['Type II Error Rate'] * 100,
                   bottom=comparison_df['Type I Error Rate'] * 100,
                   color=self.color_palette[1], alpha=0.7, label='Type II Error')
            
            ax3.set_title('Error Analysis by Model', fontweight='bold')
            ax3.set_ylabel('Error Rate (%)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Recommended model highlight
        recommended_model = comparison_analysis['recommended_model']
        recommendation_text = f"Recommended Model: {recommended_model}\n"
        recommendation_text += f"Reason: {comparison_analysis['recommendation_reason']}"
        
        ax4.text(0.5, 0.5, recommendation_text, 
                transform=ax4.transAxes, fontsize=14, fontweight='bold',
                ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor=self.color_palette[0], alpha=0.2))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Model Recommendation', fontweight='bold')
        
        plt.tight_layout()
        if self.save_plots:
            plt.savefig(self.results_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
