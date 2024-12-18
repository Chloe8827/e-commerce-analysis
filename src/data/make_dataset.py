# src/main.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

from features.build_features import FeatureEngineer
from models.customer_segmentation import CustomerSegmentation
from models.sales_prediction import SalesPredictor
from models.customer_ltv_prediction import CustomerLTVPredictor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ECommerceAnalyzer:
    """Main class for e-commerce data analysis."""

    def __init__(self, data_path: str = 'src/data/data/'):
        self.data_path = Path(data_path)
        self.results_path = Path('results')
        self.results_path.mkdir(exist_ok=True)

        # 初始化组件
        self.feature_engineer = FeatureEngineer()
        self.customer_segmentation = CustomerSegmentation(n_clusters=4)
        self.sales_predictor = SalesPredictor()
        self.ltv_predictor = CustomerLTVPredictor()

        # 存储结果
        self.results = {}

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载已处理的数据."""
        logger.info("Loading preprocessed data")

        # 直接从processed文件夹加载数据
        self.customers_df = pd.read_csv(self.data_path / 'customers_processed.csv')
        self.transactions_df = pd.read_csv(self.data_path / 'transactions_processed.csv')

        logger.info(f"Loaded {len(self.customers_df)} customers and {len(self.transactions_df)} transactions")

        # 转换日期列
        self.customers_df['registration_date'] = pd.to_datetime(self.customers_df['registration_date'])
        self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])

        # 创建特征
        self.customer_features = self.feature_engineer.create_customer_features(
            self.customers_df,
            self.transactions_df
        )
        self.transaction_features = self.feature_engineer.create_transaction_features(
            self.transactions_df
        )

        return self.customers_df, self.transactions_df

    def perform_customer_segmentation(self) -> Dict:
        """执行客户分群分析."""
        logger.info("Performing customer segmentation")

        # 选择用于分群的特征
        segmentation_features = [
            'age', 'transaction_count', 'total_spend',
            'avg_transaction_value', 'online_purchase_ratio'
        ]

        # 执行分群
        segmented_customers, segment_profiles = self.customer_segmentation.segment_customers(
            self.customer_features,
            segmentation_features
        )

        # 保存分群结果到原始客户数据
        self.customers_df['segment'] = segmented_customers['customer_segment']

        # 保存结果
        self.results['customer_segmentation'] = {
            'profiles': segment_profiles,
            'feature_importance': segmentation_features
        }

        # 可视化分群结果
        self._plot_segment_profiles(segment_profiles)

        return segment_profiles

    def predict_sales(self) -> Dict:
        """预测销售趋势."""
        logger.info("Predicting sales")

        # 准备特征
        X, y = self.sales_predictor.prepare_features(self.transactions_df)

        # 训练模型并获取指标
        metrics = self.sales_predictor.train(X, y)

        # 生成预测
        future_predictions = self.sales_predictor.predict(X.tail(30))

        # 保存结果
        self.results['sales_prediction'] = {
            'metrics': metrics,
            'feature_importance': self.sales_predictor.feature_importance.to_dict(),
            'last_30_days_prediction': future_predictions.tolist()
        }

        # 可视化结果
        self._plot_sales_prediction(y, future_predictions)

        return metrics

    def predict_customer_ltv(self) -> Dict:
        """预测客户终身价值."""
        logger.info("Predicting customer LTV")

        # 准备特征
        X, y = self.ltv_predictor.prepare_features(
            self.customers_df,
            self.transactions_df
        )

        # 训练模型并获取指标
        metrics = self.ltv_predictor.train(X, y)

        # 生成预测
        ltv_predictions = self.ltv_predictor.predict(X)

        # 将预测结果添加到客户数据中
        self.customers_df['predicted_ltv'] = ltv_predictions

        # 保存结果
        self.results['ltv_prediction'] = {
            'metrics': metrics,
            'feature_importance': self.ltv_predictor.feature_importance.to_dict(),
            'predictions_summary': {
                'mean': float(np.mean(ltv_predictions)),
                'median': float(np.median(ltv_predictions)),
                'std': float(np.std(ltv_predictions))
            }
        }

        # 可视化结果
        self._plot_ltv_distribution(ltv_predictions)

        return metrics

    def save_results(self):
        """保存分析结果."""
        logger.info("Saving analysis results")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 创建results目录（如果不存在）
        self.results_path.mkdir(parents=True, exist_ok=True)

        # 保存JSON结果
        with open(self.results_path / f'analysis_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=4)

        # 保存更新后的客户数据（包含分群和LTV预测）
        self.customers_df.to_csv(
            self.results_path / f'customers_with_predictions_{timestamp}.csv',
            index=False
        )

        logger.info(f"Results saved to {self.results_path}")

    # [其他方法保持不变...]


def main():
    """主函数."""
    try:
        # 初始化分析器
        analyzer = ECommerceAnalyzer()

        # 加载数据
        customers_df, transactions_df = analyzer.load_and_prepare_data()
        logger.info(f"Data loaded successfully: {len(customers_df)} customers, {len(transactions_df)} transactions")

        # 执行客户分群
        segment_profiles = analyzer.perform_customer_segmentation()
        logger.info("Customer segmentation completed")

        # 预测销售
        sales_metrics = analyzer.predict_sales()
        logger.info(f"Sales prediction completed. Test R2 score: {sales_metrics['test']['r2']:.3f}")

        # 预测客户终身价值
        ltv_metrics = analyzer.predict_customer_ltv()
        logger.info(f"LTV prediction completed. Test R2 score: {ltv_metrics['test']['r2']:.3f}")

        # 保存所有结果
        analyzer.save_results()
        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()