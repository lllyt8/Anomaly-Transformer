import pandas as pd
import matplotlib.pyplot as plt
from timegpt import TimeGPT
import datetime

# ====== 配置区 ======
CSV_PATH = "your_battery_data.csv"  # 你自己的电池数据文件
TARGET_COL = "voltage"              # 要分析的主序列
EXOG_COLS = ["temperature", "current"]  # 外生变量列
API_KEY = "nixak-qaTEyzVzAXINDnW5dIkHG3bzsYfdmnRWlcNaav8wRfp9F9xhzbtP3i8hltsRRVMh5MQYOKgSmPHXH3q3"    # 你的 TimeGPT API Key
BATTERY_ID = "Battery-B08"          # 电池标识（用于报告）

# ====== 1. 加载数据 ======
df = pd.read_csv(CSV_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values("timestamp")

# TimeGPT 要求 index 是 timestamp
df = df.set_index("timestamp").asfreq("5min")  # 这里假设是 5 分钟间隔
df = df.interpolate()

# ====== 2. 调用 TimeGPT 检测异常 ======
model = TimeGPT(api_key=API_KEY)

df_reset = df.reset_index()
anomalies = model.detect_anomalies(
    df=df_reset[["timestamp", TARGET_COL] + EXOG_COLS],
    exogenous_columns=EXOG_COLS
)

# ====== 3. 可视化输出 ======
def plot_anomalies(anomalies_df):
    plt.figure(figsize=(12, 5))
    plt.plot(anomalies_df["timestamp"], anomalies_df["value"], label="Voltage")
    plt.scatter(
        anomalies_df["timestamp"][anomalies_df["anomaly"]],
        anomalies_df["value"][anomalies_df["anomaly"]],
        color="red", label="Anomaly", zorder=10
    )
    plt.title(f"Anomaly Detection for {BATTERY_ID}")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{BATTERY_ID}_anomaly_plot.png")
    plt.close()

plot_anomalies(anomalies)

# ====== 4. 生成 Markdown 报告 ======
def generate_markdown_report(anomalies_df):
    detected = anomalies_df[anomalies_df["anomaly"]]
    report = f"# 🔧 Battery Anomaly Report for {BATTERY_ID}\n\n"
    report += f"⏰ Generated at: {datetime.datetime.now()}\n\n"
    report += f"**Total Anomalies Detected:** {len(detected)}\n\n"
    report += "## 🧨 Anomaly Timestamps:\n"
    for row in detected.itertuples():
        report += f"- {row.timestamp}: {row.value:.2f}V\n"
    report += "\n![Anomaly Plot](./" + f"{BATTERY_ID}_anomaly_plot.png" + ")"
    with open(f"{BATTERY_ID}_report.md", "w") as f:
        f.write(report)

generate_markdown_report(anomalies)
print(f"✅ Done! Report + Plot saved for {BATTERY_ID}")
