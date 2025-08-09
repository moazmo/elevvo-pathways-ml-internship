"""
Class imbalance handling for the loan approval prediction system.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from typing import Tuple, Dict, Any, List
from loguru import logger
from src.config_loader import config


class ImbalanceHandler:
    """Handles class imbalance using various resampling techniques."""
    
    def __init__(self):
        """Initialize imbalance handler with configuration."""
        self.imbalance_config = config.get('imbalance')
        self.data_config = config.get_data_config()
        
    def analyze_class_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze class distribution and imbalance metrics.
        
        Args:
            y: Target variable array
            
        Returns:
            Dictionary with distribution analysis
        """
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Calculate imbalance metrics
        majority_class = max(class_counts.values())
        minority_class = min(class_counts.values())
        imbalance_ratio = minority_class / majority_class
        
        distribution_info = {
            'class_counts': dict(class_counts),
            'class_percentages': {k: (v/total_samples)*100 for k, v in class_counts.items()},
            'total_samples': total_samples,
            'majority_class_count': majority_class,
            'minority_class_count': minority_class,
            'imbalance_ratio': imbalance_ratio,
            'is_severely_imbalanced': imbalance_ratio < 0.5,
            'is_moderately_imbalanced': 0.5 <= imbalance_ratio < 0.8
        }
        
        logger.info(f"Class distribution analysis:")
        logger.info(f"  Class counts: {distribution_info['class_counts']}")
        logger.info(f"  Imbalance ratio: {imbalance_ratio:.3f}")
        logger.info(f"  Severely imbalanced: {distribution_info['is_severely_imbalanced']}")
        
        return distribution_info
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique).
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Resampled X and y arrays
        """
        smote = SMOTE(
            sampling_strategy=self.imbalance_config['smote_sampling_strategy'],
            k_neighbors=self.imbalance_config['smote_k_neighbors'],
            random_state=self.data_config['random_state']
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"SMOTE applied:")
        logger.info(f"  Original shape: {X.shape}")
        logger.info(f"  Resampled shape: {X_resampled.shape}")
        logger.info(f"  New class distribution: {Counter(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def apply_random_oversampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random oversampling."""
        ros = RandomOverSampler(
            sampling_strategy='auto',
            random_state=self.data_config['random_state']
        )
        
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        logger.info(f"Random oversampling applied:")
        logger.info(f"  Original shape: {X.shape}")
        logger.info(f"  Resampled shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def apply_adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply ADASYN (Adaptive Synthetic Sampling)."""
        try:
            adasyn = ADASYN(
                sampling_strategy='auto',
                random_state=self.data_config['random_state']
            )
            
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
            logger.info(f"ADASYN applied:")
            logger.info(f"  Original shape: {X.shape}")
            logger.info(f"  Resampled shape: {X_resampled.shape}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.warning(f"ADASYN failed: {e}. Falling back to SMOTE.")
            return self.apply_smote(X, y)
    
    def apply_smote_tomek(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE + Tomek links (combination of over and under sampling)."""
        smote_tomek = SMOTETomek(
            sampling_strategy='auto',
            random_state=self.data_config['random_state']
        )
        
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        logger.info(f"SMOTE + Tomek applied:")
        logger.info(f"  Original shape: {X.shape}")
        logger.info(f"  Resampled shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def compare_resampling_methods(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Compare different resampling methods.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Dictionary with results from different resampling methods
        """
        methods = {
            'original': (X, y),
            'smote': self.apply_smote(X, y),
            'random_oversampling': self.apply_random_oversampling(X, y),
            'adasyn': self.apply_adasyn(X, y),
            'smote_tomek': self.apply_smote_tomek(X, y)
        }
        
        comparison_results = {}
        
        for method_name, (X_res, y_res) in methods.items():
            distribution = Counter(y_res)
            comparison_results[method_name] = {
                'shape': X_res.shape,
                'class_distribution': dict(distribution),
                'imbalance_ratio': min(distribution.values()) / max(distribution.values())
            }
        
        logger.info("Resampling methods comparison completed")
        return comparison_results
    
    def get_recommended_method(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Get recommended resampling method based on data characteristics.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Recommended method name
        """
        distribution_info = self.analyze_class_distribution(y)
        
        # Decision logic for method selection
        if distribution_info['is_severely_imbalanced']:
            if X.shape[0] < 1000:  # Small dataset
                return 'smote'
            else:  # Larger dataset
                return 'smote_tomek'
        elif distribution_info['is_moderately_imbalanced']:
            return 'smote'
        else:
            return 'original'  # No resampling needed
    
    def apply_best_resampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Apply the best resampling method based on data characteristics.
        
        Args:
            X: Feature array
            y: Target array
            
        Returns:
            Resampled X, y arrays and method name used
        """
        if not self.imbalance_config['apply_smote']:
            logger.info("SMOTE disabled in configuration. Using original data.")
            return X, y, 'original'
        
        recommended_method = self.get_recommended_method(X, y)
        
        method_map = {
            'original': lambda x, y: (x, y),
            'smote': self.apply_smote,
            'random_oversampling': self.apply_random_oversampling,
            'adasyn': self.apply_adasyn,
            'smote_tomek': self.apply_smote_tomek
        }
        
        X_resampled, y_resampled = method_map[recommended_method](X, y)
        
        logger.info(f"Applied resampling method: {recommended_method}")
        return X_resampled, y_resampled, recommended_method
