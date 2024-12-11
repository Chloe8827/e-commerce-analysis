# src/features/build_features.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for e-commerce data."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def create_customer_features(self,
                                 customers_df: pd.DataFrame,
                                 transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from customer and transaction data."""
        logger.info("Creating customer features")

        # Basic customer features
        features = customers_df.copy()

        # Encode categorical variables
        categorical_cols = ['gender', 'membership_type']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            features[col] = self.label_encoders[col].fit_transform(features[col])

        # Calculate customer lifetime value
        customer_transactions = transactions_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'total_amount': 'sum',
            'items_count': 'sum',
            'is_online': 'mean'
        }).reset_index()

        customer_transactions.columns = [
            'customer_id',
            'transaction_count',
            'total_spend',
            'total_items',
            'online_purchase_ratio'
        ]

        # Merge with customer features
        features = features.merge(customer_transactions, on='customer_id', how='left')

        # Calculate average transaction value
        features['avg_transaction_value'] = (
                features['total_spend'] / features['transaction_count']
        )

        # Calculate customer age
        features['account_age_days'] = (
                pd.Timestamp('2024-12-11') -
                pd.to_datetime(features['registration_date'])
        ).dt.days

        # Calculate purchase frequency (transactions per month)
        features['purchase_frequency'] = (
                features['transaction_count'] / (features['account_age_days'] / 30)
        )

        # Fill missing values
        features = features.fillna(0)

        return features

    def create_transaction_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from transaction data."""
        logger.info("Creating transaction features")

        features = transactions_df.copy()

        # Temporal features
        features['hour'] = pd.to_datetime(features['date']).dt.hour
        features['day_of_week'] = pd.to_datetime(features['date']).dt.dayofweek
        features['month'] = pd.to_datetime(features['date']).dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

        # Encode categorical variables
        categorical_cols = ['product_category', 'payment_method']
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            features[col] = self.label_encoders[col].fit_transform(features[col])

        # Calculate average item price
        features['avg_item_price'] = features['amount'] / features['items_count']

        return features

    def scale_features(self, features: pd.DataFrame,
                       exclude_cols: List[str]) -> pd.DataFrame:
        """Scale numerical features."""
        # Identify numerical columns to scale
        numeric_cols = features.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Scale features
        features[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])

        return features