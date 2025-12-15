import pandas as pd
from sklearn.model_selection import train_test_split


def data_preprocessing(path):
    # 1.获取数据源
    data = pd.read_csv(path)
    # 2.去重
    data.dropna(inplace=True)
    return data


if __name__ == '__main__':
    data = data_preprocessing("../data/garbage_sorting.csv")
    print(data.type.value_counts())