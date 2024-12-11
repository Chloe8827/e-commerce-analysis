# src/data/preprocess.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self):
        self.preprocessing_stats = {}

    def preprocess_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理客户数据"""
        logger.info("开始处理客户数据")

        df = df.copy()

        # 转换日期格式
        df['registration_date'] = pd.to_datetime(df['registration_date'])

        # 创建年龄组
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 50, 100],
            labels=['18-25', '26-35', '36-50', '50+']
        )

        # 计算注册年份
        df['registration_year'] = df['registration_date'].dt.year

        logger.info("客户数据处理完成")
        return df

    def preprocess_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理交易数据"""
        logger.info("开始处理交易数据")

        df = df.copy()

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])

        # 计算总金额
        df['total_amount'] = df['amount'] * df['items_count']

        # 提取时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        # 处理缺失的支付方式
        df['payment_method'].fillna('Unknown', inplace=True)

        logger.info("交易数据处理完成")
        return df


def main(input_filepath: str = 'data/raw/', output_filepath: str = 'data/processed/'):
    """主函数：读取原始数据，预处理，并保存处理后的数据"""
    logger.info('开始数据预处理流程')

    # 读取原始数据
    input_path = Path(input_filepath)
    customers_df = pd.read_csv(input_path / 'customers.csv')
    transactions_df = pd.read_csv(input_path / 'transactions.csv')

    # 预处理数据
    preprocessor = DataPreprocessor()
    customers_processed = preprocessor.preprocess_customers(customers_df)
    transactions_processed = preprocessor.preprocess_transactions(transactions_df)

    # 创建输出目录并保存处理后的数据
    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)

    customers_processed.to_csv(output_path / 'customers_processed.csv', index=False)
    transactions_processed.to_csv(output_path / 'transactions_processed.csv', index=False)

    logger.info('数据预处理完成并保存')
    return customers_processed, transactions_processed


if __name__ == '__main__':
    main()