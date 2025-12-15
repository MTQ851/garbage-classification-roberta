import pandas as pd
import os


def check_data():
    # 路径配置
    data_path = "../data/garbage_sorting.csv"

    print(f"正在检查数据: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='gbk')

    # 1. 基础信息
    print(f"数据总行数: {len(df)}")

    # 2. 检查是否有冲突标签
    # 逻辑：如果同一个 garbage_name 对应了多个不同的 type，就是冲突
    conflicts = df.groupby('garbage_name')['type'].nunique()
    conflicts = conflicts[conflicts > 1]

    if not conflicts.empty:
        print("\n" + "!" * 50)
        print(f"[严重警告] 发现 {len(conflicts)} 个垃圾存在标签冲突！")
        print("模型无法学习冲突的数据，请在CSV中手动修正以下条目：")
        print("!" * 50)

        for name in conflicts.index:
            entries = df[df['garbage_name'] == name]
            print(f"\n冲突垃圾名: 【{name}】")
            print(entries[['id', 'garbage_name', 'type']].to_string(index=False))

        print("\n" + "!" * 50)
        print("请修正 CSV 文件后再运行训练脚本！")
        exit(1)
    else:
        print("\n[√] 数据质量检查通过：没有发现标签冲突。")
        print("[√] 可以开始训练。")


if __name__ == '__main__':
    check_data()