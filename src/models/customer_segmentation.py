# src/models/customer_segmentation.py

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """Customer segmentation using K-means clustering."""

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()

    def segment_customers(self,
                          features: pd.DataFrame,
                          feature_cols: list) -> Tuple[pd.DataFrame, Dict]:
        """Perform customer segmentation."""
        logger.info("Performing customer segmentation")

        # Scale features
        X = self.scaler.fit_transform(features[feature_cols])

        # Perform clustering
        clusters = self.kmeans.fit_predict(X)
        features['customer_segment'] = clusters

        # Calculate segment profiles
        segment_profiles = self._create_segment_profiles(features)

        return features, segment_profiles

    def _create_segment_profiles(self, features: pd.DataFrame) -> Dict:
        """Create profiles for each customer segment."""
        profiles = {}

        for segment in range(self.n_clusters):
            segment_data = features[features['customer_segment'] == segment]

            profiles[f"Segment_{segment}"] = {
                'size': len(segment_data),
                'size_percentage': len(segment_data) / len(features) * 100,
                'avg_age': segment_data['age'].mean(),
                'avg_total_spend': segment_data['total_spend'].mean(),
                'avg_transaction_count': segment_data['transaction_count'].mean(),
                'avg_transaction_value': segment_data['avg_transaction_value'].mean(),
                'online_purchase_ratio': segment_data['online_purchase_ratio'].mean()
            }

        return profiles