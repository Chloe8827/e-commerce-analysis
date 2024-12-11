# src/models/sales_prediction.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class SalesPredictor:
    """Predict future sales using Random Forest."""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.feature_importance = None

    def prepare_features(self,
                         transactions_df: pd.DataFrame,
                         target_col: str = 'total_amount') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for sales prediction."""
        # Create time-based features
        df = transactions_df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # Aggregate daily sales
        daily_sales = df.groupby('date')[target_col].sum().reset_index()

        # Create lagged features
        for lag in [1, 7, 14, 30]:
            daily_sales[f'lag_{lag}'] = daily_sales[target_col].shift(lag)

        # Create rolling means
        for window in [7, 14, 30]:
            daily_sales[f'rolling_mean_{window}'] = (
                daily_sales[target_col].rolling(window=window).mean()
            )

        # Add calendar features
        daily_sales['dayofweek'] = daily_sales['date'].dt.dayofweek
        daily_sales['month'] = daily_sales['date'].dt.month
        daily_sales['day'] = daily_sales['date'].dt.day

        # Drop missing values
        daily_sales = daily_sales.dropna()

        # Separate features and target
        features = daily_sales.drop([target_col, 'date'], axis=1)
        target = daily_sales[target_col]

        return features, target

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the sales prediction model."""
        logger.info("Training sales prediction model")

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
        """Make sales predictions."""
        return self.model.predict(X)