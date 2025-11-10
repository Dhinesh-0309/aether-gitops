# controller/train_synthetic.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

np.random.seed(42)
N = 8000

# Features we want the model to learn from
# (cpu%, mem%, load1, disk busy %, disk read/write Bps, net rx/tx Bps, fs usage %)
cpu = np.clip(np.random.normal(40, 20, N), 0, 100)
mem = np.clip(np.random.normal(55, 20, N), 0, 100)
load1 = np.clip(np.random.normal(1.0, 1.0, N), 0, None)
disk_io_busy = np.clip(np.random.beta(2, 10, N) * 100, 0, 100)
disk_r = np.abs(np.random.normal(2e6, 2e6, N))
disk_w = np.abs(np.random.normal(2e6, 2e6, N))
net_rx = np.abs(np.random.normal(1e6, 1e6, N))
net_tx = np.abs(np.random.normal(1e6, 1e6, N))
fs_usage = np.clip(np.random.normal(60, 20, N), 0, 100)

# Pseudo “expert” rule to create labels: 1=cloud, 0=edge
label = (
    (cpu > 70) |
    (mem > 80) |
    (disk_io_busy > 70) |
    (fs_usage > 90) |
    ((cpu > 55) & (mem > 70) & (load1 > 2.0)) |
    ((net_rx + net_tx) > 6e6)
).astype(int)

df = pd.DataFrame({
    "cpu": cpu,
    "mem": mem,
    "load1": load1,
    "disk_io_busy": disk_io_busy,
    "disk_r": disk_r,
    "disk_w": disk_w,
    "net_rx": net_rx,
    "net_tx": net_tx,
    "fs_usage": fs_usage,
    "target": label
})

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Feature importances:")
for n, v in zip(X.columns, clf.feature_importances_):
    print(f"  {n}: {v:.3f}")
print("\nReport:\n", classification_report(y_test, y_pred, digits=3))

# Save model bundle (model + feature order)
dump({"model": clf, "features": list(X.columns)}, "model.pkl")
print("\nSaved model.pkl with features:", list(X.columns))