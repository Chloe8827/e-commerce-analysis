import pandas as pd

# 读取原始数据
customers_raw = pd.read_csv('../../data/raw/customers.csv')
transactions_raw = pd.read_csv('../../data/raw/transactions.csv')

# 读取处理后的数据
customers_processed = pd.read_csv('../../data/processed/customers_processed.csv')
transactions_processed = pd.read_csv('../../data/processed/transactions_processed.csv')

# 查看数据
print("原始客户数据：")
print(customers_raw.head())
print("\n处理后的客户数据：")
print(customers_processed.head())