# src/models/customer_ltv_prediction.py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class CustomerLTVPredictor:
    """Predict customer lifetime value."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.feature_importance = None

    def prepare_features(self,
                         customers_df: pd.DataFrame,
                         transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for LTV prediction."""
        # Calculate current LTV
        current_ltv = transactions_df.groupby('customer_id')['total_amount'].sum()

        # Merge with customer features
        features = customers_df.merge(
            current_ltv.reset_index(),
            on='customer_id',
            how='left'
        )

        # Create customer behavior features
        customer_behavior = transactions_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'items_count': 'mean',
            'is_online': 'mean',
            'total_amount': ['mean', 'std']
        }).reset_index()

        customer_behavior.columns = [
            'customer_id',
            'transaction_count',
            'avg_items',
            'online_ratio',
            'avg_transaction_amount',
            'transaction_amount_std'
        ]

        # Merge all features
        features = features.merge(customer_behavior, on='customer_id', how='left')

        # Calculate customer age
        features['account_age_days'] = (
                pd.Timestamp('2024-12-11') -
                pd.to_datetime(features['registration_date'])
        ).dt.days

        # Select features for prediction
        feature_cols = [
            'age', 'transaction_count', 'avg_items', 'online_ratio',
            'avg_transaction_amount', 'transaction_amount_std', 'account_age_days'
        ]

        # Fill missing values
        features = features.fillna(0)

        return features[feature_cols], features['total_amount']

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the LTV prediction model."""
        logger.info("Training customer LTV prediction model")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = {
            'train': {
                'mae': mean_absolute_error(y_train, train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'r2': r2_score(y_train, train_pred)
            },
            'test': {
                'mae': mean_absolute_error(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'r2': r2_score(y_test, test_pred)
            }
        }

        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict customer LTV."""
        return self.model.predict(X)