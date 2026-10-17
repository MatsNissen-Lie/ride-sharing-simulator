import pandas as pd

# Simulate the user's data
# "2025-01-03T16:11:30-05:00" -> This would be interpreted as 21:11 UTC if not handled.
data = {"timestamp": ["2025-01-03 16:11:30-05:00"]}
df = pd.DataFrame(data)

# Logic from generate_plots.py
df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Aware Timestamp: {df['timestamp'].iloc[0]}")

if df["timestamp"].dt.tz is not None:
    df["timestamp"] = df["timestamp"].apply(lambda x: x.replace(tzinfo=None))

print(f"Naive Timestamp: {df['timestamp'].iloc[0]}")

# Verification
# We want 16:11, regardless of the -05:00 offset in the string meaning it is actually 21:11 UTC.
# The user wants "local time" i.e. what is written in the string.
assert df["timestamp"].iloc[0].hour == 16
assert df["timestamp"].iloc[0].minute == 11
assert df["timestamp"].iloc[0].tzinfo is None
print("\nVerification Passed: 16:11 preserved and tzinfo removed!")
