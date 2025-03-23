import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==== 配置部分 ====
CSV_PATH = "data_factory/raw/DanDLN_Data.csv"
TIMESTAMP_COL = "ts"
PLOT_VARIABLES = ["voltage", "temperature", "current"]  # 可以修改

# ==== 1. 加载数据 ====
df = pd.read_csv(CSV_PATH)
df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
df = df.sort_values(TIMESTAMP_COL)
df = df.set_index(TIMESTAMP_COL)

print("\n📌 数据中的列（特征）有：")
print(list(df.columns))

print("\n🔍 基本信息:")
print(df.info())

print("\n📊 描述性统计:")
print(df.describe())

print("\n❓ 缺失值统计:")
print(df.isnull().sum())

# ==== 2. 检查时间间隔 ====
df["time_diff"] = df.index.to_series().diff().dt.total_seconds()
intervals = df["time_diff"].dropna().unique()
print(f"\n⏱ 不同时间间隔（秒）: {sorted(intervals)}")

if len(intervals) > 1:
    print("⚠️ 时间间隔不一致，可能需要重采样（resample）")
else:
    print("✅ 时间间隔一致，可直接送入 TimeGPT")

# ==== 3. 缺失值可视化 ====
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu")
plt.title("🧊 缺失值可视化")
plt.tight_layout()
plt.savefig("missing_values.png")
plt.close()

# ==== 4. 各变量趋势图 ====
os.makedirs("plots", exist_ok=True)

for col in PLOT_VARIABLES:
    if col in df.columns:
        plt.figure(figsize=(12, 4))
        df[col].plot()
        plt.title(f"📈 {col} 时间趋势")
        plt.xlabel("Time")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f"plots/{col}_trend.png")
        plt.close()
        print(f"✅ 图已保存: plots/{col}_trend.png")
    else:
        print(f"⚠️ 没找到列: {col}")

print("\n✅ 探查完成，已输出统计信息与趋势图！")
