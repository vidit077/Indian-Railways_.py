
"""
INDIAN RAILWAYS DELAY & ROUTE BOTTLENECK ANALYZER
Libraries used: pandas, numpy, matplotlib only.

Run: python main.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "train_routes_delays_Sep2024.csv"
OUTPUT_DIR = "outputs"
DELAY_LIMIT = 15
MIN_RECORDS = 20
SHOW_PLOTS = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=130)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


print("=" * 60)
print("INDIAN RAILWAYS DELAY & ROUTE BOTTLENECK ANALYZER")
print("=" * 60)

if not os.path.exists(CSV_FILE):
    print(f"'{CSV_FILE}' not found -> creating SAMPLE data (synthetic).")
    print("For real results, download the Kaggle file and put it next to main.py.\n")
    from generate_sample_data import make_sample
    make_sample(CSV_FILE)
    USING_SAMPLE = True
else:
    USING_SAMPLE = False

df = pd.read_csv(CSV_FILE)
print("Raw records:", len(df))
print(df.head(), "\n")

df.columns = df.columns.str.strip().str.lower()
df["train"] = df["train"].astype(str).str.strip()
df["station"] = df["station"].astype(str).str.strip()
df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

before = len(df)
df = df.dropna(subset=["train", "station", "date", "arr_delay"])
df = df.drop_duplicates()
df = df[df["arr_delay"] >= 0]
print(f"Cleaning: {before} -> {len(df)} records ({before - len(df)} removed)\n")

df["delayed"] = df["arr_delay"] > DELAY_LIMIT
df["weekday"] = df["date"].dt.day_name()
df["category"] = np.select(
    [df["arr_delay"] <= DELAY_LIMIT, df["arr_delay"] <= 60, df["arr_delay"] <= 120],
    ["On Time", "Moderate", "High"], default="Severe")
CAT_ORDER = ["On Time", "Moderate", "High", "Severe"]

avg_delay = df["arr_delay"].mean()
median_delay = df["arr_delay"].median()
p90_delay = np.percentile(df["arr_delay"], 90)
delayed_pct = df["delayed"].mean() * 100

print("OVERALL SUMMARY")
print(f"  Records            : {len(df)}")
print(f"  Trains / Stations  : {df['train'].nunique()} / {df['station'].nunique()}")
print(f"  Date range         : {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Average delay      : {avg_delay:.1f} min")
print(f"  Median delay       : {median_delay:.1f} min")
print(f"  90th percentile    : {p90_delay:.1f} min")
print(f"  Max delay          : {df['arr_delay'].max():.0f} min")
print(f"  Delayed (> {DELAY_LIMIT} min)  : {delayed_pct:.1f} %\n")

train_stats = (df.groupby("train")
                 .agg(records=("arr_delay", "size"),
                      avg_delay=("arr_delay", "mean"),
                      delayed_pct=("delayed", lambda s: s.mean() * 100))
                 .query("records >= @MIN_RECORDS")
                 .sort_values("avg_delay", ascending=False)
                 .round(1))
print("TOP 10 MOST DELAYED TRAINS")
print(train_stats.head(10).to_string(), "\n")

station_stats = (df.groupby("station")
                   .agg(records=("arr_delay", "size"),
                        avg_delay=("arr_delay", "mean"),
                        delayed_pct=("delayed", lambda s: s.mean() * 100),
                        trains=("train", "nunique"))
                   .query("records >= @MIN_RECORDS")
                   .round(1))

station_stats["hotspot_score"] = (
    50 * station_stats["avg_delay"] / station_stats["avg_delay"].max()
    + 50 * station_stats["delayed_pct"] / station_stats["delayed_pct"].max()).round(1)
station_stats = station_stats.sort_values("hotspot_score", ascending=False)
print("TOP 10 DELAY HOTSPOT STATIONS")
print(station_stats.head(10).to_string(), "\n")

daily = df.groupby("date")["arr_delay"].mean()
daily_roll = daily.rolling(7, min_periods=1).mean()

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_avg = df.groupby("weekday")["arr_delay"].mean().reindex(day_order)

worst_day = daily.idxmax().date()
print(f"WORST DAY: {worst_day} (avg delay {daily.max():.1f} min)")
print(f"WORST WEEKDAY: {weekday_avg.idxmax()} ({weekday_avg.max():.1f} min)\n")

top_trains = train_stats.head(8).index
top_stations = station_stats.head(8).index
heat = (df[df["train"].isin(top_trains) & df["station"].isin(top_stations)]
        .pivot_table(index="train", columns="station", values="arr_delay", aggfunc="mean"))

BLUE, RED, GREY = "#2563eb", "#dc2626", "#6b7280"
cat_colors = {"On Time": "#16a34a", "Moderate": "#eab308", "High": "#f97316", "Severe": "#dc2626"}

plt.figure(figsize=(8, 4.5))
plt.hist(df["arr_delay"].clip(upper=180), bins=36, color=BLUE, edgecolor="white")
plt.axvline(DELAY_LIMIT, color=RED, ls="--", label=f"{DELAY_LIMIT} min limit")
plt.axvline(avg_delay, color="black", ls=":", label=f"Mean {avg_delay:.0f} min")
plt.title("Arrival Delay Distribution (clipped at 180 min)")
plt.xlabel("Delay (minutes)")
plt.ylabel("Records")
plt.legend()
save("1_delay_distribution.png")

t = train_stats.head(10).iloc[::-1]
plt.figure(figsize=(9, 5))
plt.barh(t.index, t["avg_delay"], color=BLUE)
plt.title("Top 10 Most Delayed Trains (average delay)")
plt.xlabel("Average delay (minutes)")
save("2_top_delayed_trains.png")

s = station_stats.head(10).iloc[::-1]
plt.figure(figsize=(9, 5))
plt.barh(s.index, s["hotspot_score"], color=RED)
plt.title("Top 10 Delay Hotspot Stations (hotspot score)")
plt.xlabel("Hotspot score (0-100)")
save("3_hotspot_stations.png")

plt.figure(figsize=(9, 4.5))
plt.plot(daily.index, daily.values, color=GREY, alpha=0.6, marker="o", ms=3, label="Daily average")
plt.plot(daily_roll.index, daily_roll.values, color=BLUE, lw=2.5, label="7-day average")
plt.title("Daily Average Arrival Delay")
plt.xlabel("Date")
plt.ylabel("Minutes")
plt.legend()
plt.xticks(rotation=30)
save("4_daily_trend.png")

counts = df["category"].value_counts().reindex(CAT_ORDER).fillna(0)
plt.figure(figsize=(7, 4.5))
bars = plt.bar(counts.index, counts.values, color=[cat_colors[c] for c in counts.index])
for b, v in zip(bars, counts.values):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v / counts.sum() * 100:.1f}%",
             ha="center", va="bottom")
plt.title("Delay Categories")
plt.ylabel("Records")
save("5_delay_categories.png")

plt.figure(figsize=(8, 4.5))
plt.bar([d[:3] for d in day_order], weekday_avg.values, color=BLUE)
plt.title("Average Delay by Weekday")
plt.ylabel("Minutes")
save("6_weekday_pattern.png")

if not heat.empty:
    plt.figure(figsize=(10, 5.5))
    plt.imshow(heat.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(label="Average delay (min)")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)
    plt.grid(False)
    plt.title("Where Do Trains Lose Time? (Train x Hotspot Station)")
    save("7_train_station_heatmap.png")

train_stats.to_csv(os.path.join(OUTPUT_DIR, "train_summary.csv"))
station_stats.to_csv(os.path.join(OUTPUT_DIR, "station_summary.csv"))

top_station = station_stats.index[0]
top_train = train_stats.index[0]
severe_pct = (df["category"] == "Severe").mean() * 100

print("=" * 60)
print("FINAL INSIGHTS")
print("=" * 60)

skew = ("a few very long delays pull the average up."
        if avg_delay > median_delay * 1.15 else "delays are fairly evenly spread.")

print(f"1. Average arrival delay is {avg_delay:.1f} min vs median {median_delay:.1f} min -> {skew}")
print(f"2. {delayed_pct:.1f}% of arrivals are later than {DELAY_LIMIT} min; "
      f"{severe_pct:.1f}% are Severe (> 120 min).")
print(f"3. Most delayed train: {top_train} ({train_stats.iloc[0]['avg_delay']} min on average).")
print(f"4. Top hotspot station: {top_station} "
      f"(avg {station_stats.iloc[0]['avg_delay']} min, "
      f"{station_stats.iloc[0]['delayed_pct']}% delayed).")
print(f"5. Worst day: {worst_day}; worst weekday: {weekday_avg.idxmax()}.")

print("\nNOTE: A high-delay station is only a possible hotspot. This dataset alone")
print("does not prove a physical bottleneck (delay may be carried from earlier stops).")

if USING_SAMPLE:
    print("\n*** These results use SYNTHETIC SAMPLE DATA, not real Indian Railways data. ***")

print(f"\nCharts and CSV summaries saved in the '{OUTPUT_DIR}' folder. Analysis completed!")
