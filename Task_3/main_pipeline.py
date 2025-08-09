"""
Main execution pipeline for the loan approval prediction system.
This script orchestrates the complete machine learning pipeline from data loading to model deployment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

# Configure logging
logger.add("logs/loan_prediction_{time}.log", rotation="1 day", retention="30 days")

# Import custom modules
from src.data_loader import DataLoader
from src.eda_analyzer import EDAAnalyzer
from src.visualizer import LoanDataVisualizer
from src.data_preprocessor import LoanDataPreprocessor
from src.imbalance_handler import ImbalanceHandler
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.evaluation_visualizer import EvaluationVisualizer
from src.production_pipeline import create_production_pipeline, ModelRegistry
from src.config_loader import config


class LoanApprovalMLPipeline:
    """Complete machine learning pipeline for loan approval prediction."""
    
    def __init__(self):
        """Initialize the ML pipeline."""
        self.results = {}
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        self.trainer = None
        
        logger.info("Loan Approval ML Pipeline initialized")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete machine learning pipeline.
        
        Returns:
            Dictionary with all pipeline results
        """
        logger.info("=" * 80)
        logger.info("STARTING LOAN APPROVAL PREDICTION ML PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Step 1: Data Loading and Validation
            logger.info("Step 1: Data Loading and Validation")
            df, data_summary = self._load_and_validate_data()
            self.results['data_summary'] = data_summary
            
            # Step 2: Exploratory Data Analysis
            logger.info("Step 2: Exploratory Data Analysis")
            eda_results = self._perform_eda(df)
            self.results['eda_results'] = eda_results
            
            # Step 3: Data Preprocessing
            logger.info("Step 3: Data Preprocessing and Feature Engineering")
            X_train, X_val, X_test, y_train, y_val, y_test = self._preprocess_data(df)
            
            # Step 4: Handle Class Imbalance
            logger.info("Step 4: Class Imbalance Analysis and Handling")
            X_train_balanced, y_train_balanced, resampling_method = self._handle_class_imbalance(X_train, y_train)
            self.results['resampling_method'] = resampling_method
            
            # Step 5: Model Training
            logger.info("Step 5: Model Training and Hyperparameter Tuning")
            training_results = self._train_models(X_train_balanced, y_train_balanced, X_val, y_val)
            self.results['training_results'] = training_results
            
            # Step 6: Model Evaluation
            logger.info("Step 6: Model Evaluation and Performance Analysis")
            evaluation_results = self._evaluate_models(X_test, y_test)
            self.results['evaluation_results'] = evaluation_results
            
            # Step 7: Model Selection and Production Setup
            logger.info("Step 7: Model Selection and Production Pipeline Setup")
            production_pipeline = self._setup_production_pipeline()
            self.results['production_pipeline'] = production_pipeline
            
            # Step 8: Generate Final Report
            logger.info("Step 8: Generating Final Report")
            final_report = self._generate_final_report()
            self.results['final_report'] = final_report
            
            logger.info("=" * 80)
            logger.info("LOAN APPROVAL PREDICTION ML PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _load_and_validate_data(self) -> tuple:
        """Load and validate the loan dataset."""
        data_loader = DataLoader()
        df, summary = data_loader.load_and_validate()
        
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Target distribution: {summary['target_distribution']}")
        
        return df, summary
    
    def _perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive exploratory data analysis."""
        # Run EDA analysis
        eda_analyzer = EDAAnalyzer(df)
        eda_results = eda_analyzer.run_comprehensive_eda()
        
        # Create visualizations
        visualizer = LoanDataVisualizer(df)
        visualizer.create_eda_report(eda_results)
        
        logger.info("EDA completed with visualizations saved")
        
        return eda_results
    
    def _preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess data and split into train/val/test sets."""
        preprocessor = LoanDataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process_full_pipeline(df)
        
        # Store preprocessor for later use
        self.preprocessor = preprocessor
        
        logger.info("Data preprocessing completed")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """Analyze and handle class imbalance."""
        imbalance_handler = ImbalanceHandler()
        
        # Analyze class distribution
        distribution_info = imbalance_handler.analyze_class_distribution(y_train)
        self.results['class_distribution'] = distribution_info
        
        # Apply best resampling method
        X_train_balanced, y_train_balanced, method = imbalance_handler.apply_best_resampling(X_train, y_train)
        
        logger.info(f"Class imbalance handled using: {method}")
        
        return X_train_balanced, y_train_balanced, method
    
    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train multiple models with hyperparameter tuning."""
        self.trainer = ModelTrainer()

        # Train all models
        training_results = self.trainer.train_all_models(X_train, y_train, X_val, y_val)

        # Store models for evaluation
        self.models = self.trainer.models
        
        logger.info(f"Trained {len(training_results)} models successfully")
        
        return training_results
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate all trained models comprehensively."""
        evaluator = ModelEvaluator(feature_names=self.preprocessor.feature_names)
        
        # Evaluate each model
        evaluation_results = {}
        for model_name, model in self.models.items():
            results = evaluator.evaluate_single_model(model, model_name, X_test, y_test)
            evaluation_results[model_name] = results
        
        # Generate comprehensive evaluation report
        evaluation_report = evaluator.generate_evaluation_report(evaluation_results)
        
        # Create evaluation visualizations
        viz = EvaluationVisualizer()
        viz.create_evaluation_dashboard(
            evaluation_results, y_test, self.models, X_test, 
            evaluation_report['model_comparison']
        )
        
        # Select best model from the comparison analysis
        comparison_analysis = evaluation_report['model_comparison']
        self.best_model_name = comparison_analysis['recommended_model']
        self.best_model = self.models[self.best_model_name]

        logger.info(f"Best model selected: {self.best_model_name}")
        logger.info(f"Best model type: {type(self.best_model)}")
        
        logger.info(f"Model evaluation completed. Best model: {self.best_model_name}")
        
        return evaluation_report
    
    def _setup_production_pipeline(self) -> Dict[str, Any]:
        """Set up production-ready pipeline with the best model."""
        if self.best_model is None or self.best_model_name is None:
            raise ValueError("No best model selected")

        # Ensure self.trainer is initialized
        if self.trainer is None:
            self.trainer = ModelTrainer()

        # Get performance metrics for the best model
        best_model_results = self.results['evaluation_results']['model_evaluations'][self.best_model_name]
        performance_metrics = {
            'accuracy': best_model_results['accuracy'],
            'precision': best_model_results['precision'],
            'recall': best_model_results['recall'],
            'f1_score': best_model_results['f1_score'],
            'roc_auc': best_model_results.get('roc_auc', 0)
        }
        
        # Save the best model using the existing trainer instance
        model_path = self.trainer.save_model(
            self.best_model, 
            self.best_model_name, 
            {
                'performance_metrics': performance_metrics,
                'resampling_method': self.results['resampling_method'],
                'feature_names': self.preprocessor.feature_names
            }
        )
        
        # Create production pipeline
        production_pipeline = create_production_pipeline(
            self.best_model_name, 
            model_path, 
            performance_metrics
        )
        
        logger.info("Production pipeline setup completed")
        
        return {
            'model_path': model_path,
            'performance_metrics': performance_metrics,
            'pipeline_status': 'ready'
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        report = {
            'pipeline_execution_summary': {
                'execution_timestamp': datetime.now().isoformat(),
                'dataset_size': self.results['data_summary']['shape'],
                'target_distribution': self.results['data_summary']['target_distribution'],
                'resampling_method_used': self.results['resampling_method'],
                'models_trained': list(self.models.keys()),
                'best_model': self.best_model_name
            },
            'model_performance_summary': {},
            'business_recommendations': self._generate_business_recommendations(),
            'technical_recommendations': self._generate_technical_recommendations()
        }
        
        # Add model performance summary
        if 'evaluation_results' in self.results:
            model_comparison = self.results['evaluation_results']['model_comparison']
            report['model_performance_summary'] = {
                'recommended_model': model_comparison['recommended_model'],
                'recommendation_reason': model_comparison['recommendation_reason'],
                'best_models_by_metric': model_comparison['best_models_by_metric']
            }
        
        # Save report
        report_path = Path("results/final_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Final report saved to {report_path}")
        
        return report
    
    def _generate_business_recommendations(self) -> List[str]:
        """Generate business-focused recommendations."""
        recommendations = [
            "Implement the trained model as part of the loan approval workflow",
            "Focus on precision to minimize bad loan approvals and reduce financial risk",
            "Regularly retrain the model with new data to maintain performance",
            "Consider manual review for borderline cases (confidence < 0.7)",
            "Monitor model performance in production and set up alerts for degradation"
        ]
        
        # Add specific recommendations based on results
        if self.best_model_name:
            if 'logistic' in self.best_model_name.lower():
                recommendations.append("The logistic regression model provides good interpretability for regulatory compliance")
            elif 'tree' in self.best_model_name.lower():
                recommendations.append("The decision tree model offers clear decision rules that can be explained to stakeholders")
        
        return recommendations
    
    def _generate_technical_recommendations(self) -> List[str]:
        """Generate technical recommendations."""
        recommendations = [
            "Deploy the model using the provided FastAPI service for scalable predictions",
            "Implement model monitoring to track prediction drift and performance degradation",
            "Set up automated retraining pipeline with new data",
            "Use A/B testing to validate model improvements before full deployment",
            "Implement proper logging and error handling for production use"
        ]
        
        # Add resampling-specific recommendations
        if self.results.get('resampling_method') == 'smote':
            recommendations.append("SMOTE was used to handle class imbalance - monitor for overfitting in production")
        
        return recommendations


def main():
    """Main execution function."""
    try:
        # Initialize and run pipeline
        pipeline = LoanApprovalMLPipeline()
        results = pipeline.run_complete_pipeline()
        
        # Print summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        final_report = results['final_report']
        summary = final_report['pipeline_execution_summary']
        
        print(f"Dataset Size: {summary['dataset_size']}")
        print(f"Best Model: {summary['best_model']}")
        print(f"Resampling Method: {summary['resampling_method_used']}")
        
        if 'model_performance_summary' in final_report:
            perf_summary = final_report['model_performance_summary']
            print(f"Recommended Model: {perf_summary['recommended_model']}")
            print(f"Reason: {perf_summary['recommendation_reason']}")
        
        print("\nBusiness Recommendations:")
        for i, rec in enumerate(final_report['business_recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\nTechnical Recommendations:")
        for i, rec in enumerate(final_report['technical_recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 80)
        print("Pipeline completed successfully!")
        print("Check the 'results' folder for detailed analysis and visualizations.")
        print("Use 'python src/api_server.py' to start the prediction API service.")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
