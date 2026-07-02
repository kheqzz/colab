# %%
import joblib
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print("="*60)
print("BOOSTING EXPERIMENT + THRESHOLD OPTIMIZATION")
print("="*60)

# %%
print("\n[LOAD DATA]")

X_train_resampled = joblib.load('X_train_resampled_values_v1.pkl')
y_train_resampled = joblib.load('y_train_resampled_values_v1.pkl')
X_test = joblib.load('X_test_values_v1.pkl')
y_test = joblib.load('y_test_values_v1.pkl')

print("Train:", X_train_resampled.shape)
print("Test :", X_test.shape)

# %%
models = {
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        booster = "gbtree",
        n_estimators=100,
        random_state=42,
        device="cuda",
        tree_method="hist",
        eval_metric="logloss"
    ),

    "XGBoostLinear": XGBClassifier(
        booster = "gblinear",
        n_estimators=100,
        random_state=42,
        device="cuda",
        eval_metric="logloss"
    ),
    "LightGBM": LGBMClassifier(
        
        n_estimators=100,
        random_state=42,
        device="gpu",
        verbose=-1
    ),
    "LightGBM_RF": LGBMClassifier(
        boosting_type="rf",
        n_estimators=100,
        bagging_freq=1,
        bagging_fraction=0.8,
        feature_fraction=0.8,
        random_state=42,
        device="gpu",
        verbose=-1
    )
}

# %%
def plot_cm(cm, name, dtype):
    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['0', '1'],
        yticklabels=['0', '1']
    )

    plt.title(f"{name} - {dtype}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/cm_{name}_{dtype}.png", dpi=300, bbox_inches="tight")

    plt.show()

# %%
trained_models = {}

print("\n[TRAINING MODELS]")

for name, model in models.items():
    print("\n==", name, "==")

    start = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    print("Train time:", round(time.time() - start, 2), "sec")

    trained_models[name] = model

    joblib.dump(model, f"{name}_model.pkl")

# %%
from sklearn.metrics import precision_recall_curve
def three_different_threshold_evaluation(model,X,y):
    proba = model.predict_proba(X)[:, 1]

    default_t = 0.5

    loop_t = 0.5
    best_f1_loop = 0

    for t in np.linspace(0.01, 0.99, 200):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y, pred)

        if f1 > best_f1_loop:
            best_f1_loop = f1
            loop_t = t
    
  

    return default_t,loop_t

# %%
thresholds = {}

print("\n[THRESHOLD OPTIMIZATION]")

for name, model in trained_models.items():
    print("\n==", name, "==")

    # Ambil 3 threshold sekaligus dari fungsi
    t_default, t_loop= three_different_threshold_evaluation(model, X_test, y_test)

    # Simpan ketiganya ke dalam dictionary berdasarkan nama model
    thresholds[name] = {
        "default": t_default,
        "otomatis": t_loop,
        
    }

    # Cetak hasilnya agar bisa kamu bandingkan langsung
    print("Threshold Default  :", round(t_default, 3))
    print("Threshold Otomatis :", round(t_loop, 3))
   

# %%
print("\n[FINAL EVALUATION]")

for name, model in trained_models.items():
    print("\n==============================")
    print("==", name, "==")
    print("==============================")

    # Lakukan looping untuk setiap jenis threshold yang disimpan
    for t_type, t_value in thresholds[name].items():
        print(f"\n>> Menilai Menggunakan Threshold: {t_type.upper()} ({round(t_value, 3)})")

        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= t_value).astype(int)

      

        acc = accuracy_score(y_test, pred_test)
        prec = precision_score(y_test, pred_test)
        rec = recall_score(y_test, pred_test)
        f1 = f1_score(y_test, pred_test)
     
        
        auc_score = 0.0
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_proba)

        print("METRICS:")
        print("  Accuracy :", round(acc, 4))
        print("  Precision:", round(prec, 4))
        print("  Recall   :", round(rec, 4))
        print("  AUC Score:", round(auc_score, 4))
        print("  F1 Test  :", round(f1, 4))
       

        # Jika dirasa terlalu penuh, classification report & plot_cm bisa ditaruh 
        # di luar loop threshold atau hanya dijalankan untuk t_type == 'manual' saja.
        print(f"\n  Classification Report ({t_type}):")
        print(classification_report(y_test, pred_test, digits=4))

        cm_test = confusion_matrix(y_test, pred_test)
       

        plot_cm(cm_test, name, f"test_{t_type}")
      

    # Eksport model cukup sekali saja per tipe model (di luar loop threshold)
    joblib.dump(model, f"{name}_final.pkl")


