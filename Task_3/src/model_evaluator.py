"""
Model evaluation and performance analysis for the loan approval prediction system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score
)
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
from loguru import logger
from src.config_loader import config


class ModelEvaluator:
    """Comprehensive model evaluation and performance analysis."""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize model evaluator.
        
        Args:
            feature_names: List of feature names for interpretability
        """
        self.evaluation_config = config.get_evaluation_config()
        self.feature_names = feature_names
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def evaluate_single_model(self, model: Any, model_name: str, X_test: np.ndarray, 
                             y_test: np.ndarray, y_pred: Optional[np.ndarray] = None,
                             y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model instance
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            y_pred: Predictions (if None, will be generated)
            y_pred_proba: Prediction probabilities (if None, will be generated if possible)
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate all metrics
        evaluation_results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Add AUC metrics if probabilities are available
        if y_pred_proba is not None:
            evaluation_results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            evaluation_results['average_precision'] = average_precision_score(y_test, y_pred_proba)
        
        # Business-focused metrics
        evaluation_results['business_metrics'] = self._calculate_business_metrics(y_test, y_pred)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            evaluation_results['feature_importance'] = self._get_feature_importance(model)
        elif hasattr(model, 'coef_'):
            evaluation_results['feature_coefficients'] = self._get_feature_coefficients(model)
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"  Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"  Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_results['recall']:.4f}")
        logger.info(f"  F1-Score: {evaluation_results['f1_score']:.4f}")
        
        return evaluation_results
    
    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate business-focused metrics for loan approval."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business interpretation:
        # TP: Correctly approved good loans
        # TN: Correctly rejected bad loans  
        # FP: Incorrectly approved bad loans (Type I error - costly)
        # FN: Incorrectly rejected good loans (Type II error - opportunity loss)
        
        total_loans = len(y_true)
        
        business_metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),  # Bad loans approved (high cost)
            'false_negatives': int(fn),  # Good loans rejected (opportunity loss)
            'type_i_error_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False positive rate
            'type_ii_error_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False negative rate
            'cost_ratio': fp / total_loans,  # Proportion of costly errors
            'opportunity_loss_ratio': fn / total_loans  # Proportion of missed opportunities
        }
        
        return business_metrics
    
    def _get_feature_importance(self, model: Any) -> Dict[str, float]:
        """Get feature importance from tree-based models."""
        if not hasattr(model, 'feature_importances_'):
            return {}
        
        importances = model.feature_importances_
        
        if self.feature_names and len(self.feature_names) == len(importances):
            feature_importance = dict(zip(self.feature_names, importances))
        else:
            feature_importance = {f'feature_{i}': imp for i, imp in enumerate(importances)}
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def _get_feature_coefficients(self, model: Any) -> Dict[str, float]:
        """Get feature coefficients from linear models."""
        if not hasattr(model, 'coef_'):
            return {}
        
        coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        
        if self.feature_names and len(self.feature_names) == len(coefficients):
            feature_coef = dict(zip(self.feature_names, coefficients))
        else:
            feature_coef = {f'feature_{i}': coef for i, coef in enumerate(coefficients)}
        
        # Sort by absolute coefficient value
        feature_coef = dict(sorted(feature_coef.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return feature_coef
    
    def compare_models(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and provide recommendations.
        
        Args:
            evaluation_results: Dictionary of evaluation results for each model
            
        Returns:
            Model comparison analysis
        """
        logger.info("Comparing model performances...")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            
            if 'roc_auc' in results:
                row['ROC-AUC'] = results['roc_auc']
            
            # Add business metrics
            business_metrics = results['business_metrics']
            row['Type I Error Rate'] = business_metrics['type_i_error_rate']
            row['Type II Error Rate'] = business_metrics['type_ii_error_rate']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Find best model for each metric
        best_models = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            if metric in comparison_df.columns:
                best_idx = comparison_df[metric].idxmax()
                best_models[metric] = {
                    'model': comparison_df.loc[best_idx, 'Model'],
                    'score': comparison_df.loc[best_idx, metric]
                }
        
        # Overall recommendation based on F1-score (balanced metric for imbalanced data)
        primary_metric = self.evaluation_config['primary_metric']
        if primary_metric == 'f1':
            recommended_model = best_models['F1-Score']['model']
        else:
            recommended_model = best_models.get(primary_metric.upper(), {}).get('model', 'logistic_regression')
        
        comparison_analysis = {
            'comparison_table': comparison_df.to_dict('records'),
            'best_models_by_metric': best_models,
            'recommended_model': recommended_model,
            'recommendation_reason': f"Best {primary_metric.upper()} score for imbalanced classification"
        }
        
        logger.info(f"Model comparison completed. Recommended model: {recommended_model}")
        
        return comparison_analysis
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Dictionary of evaluation results for each model
            
        Returns:
            Complete evaluation report
        """
        logger.info("Generating comprehensive evaluation report...")
        
        # Model comparison
        comparison_analysis = self.compare_models(evaluation_results)
        
        # Summary statistics
        summary_stats = self._generate_summary_statistics(evaluation_results)
        
        # Create final report
        evaluation_report = {
            'model_evaluations': evaluation_results,
            'model_comparison': comparison_analysis,
            'summary_statistics': summary_stats,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save report to JSON
        report_path = self.results_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return evaluation_report
    
    def _generate_summary_statistics(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics across all models."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        summary = {}
        
        for metric in metrics:
            values = [results[metric] for results in evaluation_results.values() if metric in results]
            
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary

    def select_best_model(self, evaluation_results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any]:
        """
        Select the best model based on evaluation results.

        Args:
            evaluation_results: Dictionary of evaluation results for each model

        Returns:
            Tuple of (best_model_name, best_model_instance)
        """
        primary_metric = self.evaluation_config['primary_metric']

        best_score = -np.inf
        best_model_name = None

        # Handle nested structure from generate_evaluation_report
        if 'model_evaluations' in evaluation_results:
            model_results = evaluation_results['model_evaluations']
        else:
            model_results = evaluation_results

        for model_name, results in model_results.items():
            # Convert metric name to match what's in results
            metric_key = primary_metric
            if primary_metric == 'f1':
                metric_key = 'f1_score'

            if metric_key in results:
                score = results[metric_key]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name

        if best_model_name is None:
            raise ValueError("No valid model found for selection")

        logger.info(f"Best model selected: {best_model_name} with {primary_metric}: {best_score:.4f}")

        return best_model_name, best_model_name  # Return model name twice for compatibility
