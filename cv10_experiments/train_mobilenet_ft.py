import os
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

IMG_SIZE = (416, 416)
BATCH_SIZE = 16
EPOCHS = 20
EPOCHS_FROZEN = 5
N_SPLITS = 10
RESULTS_DIR = "results_mobilenet_ft"
SEED = 42
FINE_TUNE_AT = 100
LR_FROZEN = 1e-3
LR_FINE = 1e-5

os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = "data-ar"
CLASS_DIRS = ["arabica", "robusta"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(sorted(CLASS_DIRS))}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(len(CLASS_TO_IDX))]
print(f"Classes: {CLASS_NAMES}")

filepaths, labels = [], []
for cls in CLASS_DIRS:
    paths = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(DATA_DIR, cls, pat)))
    filepaths.extend(paths)
    labels.extend([CLASS_TO_IDX[cls]] * len(paths))
filepaths = np.array(filepaths)
labels = np.array(labels)
print(f"Number of samples: {len(filepaths)} (arabica={np.sum(labels==CLASS_TO_IDX['arabica'])}, robusta={np.sum(labels==CLASS_TO_IDX['robusta'])})")

AUTOTUNE = tf.data.AUTOTUNE

def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method='bilinear')
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, tf.cast(label, tf.float32)

augmenter = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

def make_ds(paths, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def build_model():
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model, base_model

def unfreeze_for_finetune(model, base_model, fine_tune_at=FINE_TUNE_AT, keep_bn_frozen=True):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    if keep_bn_frozen:
        for layer in base_model.layers[fine_tune_at:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

early_stopping_p1 = EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True)
early_stopping_p2 = EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)

def plot_training(history, filename_prefix):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    auc_tr = history.history.get('auc', [])
    auc_val = history.history.get('val_auc', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1); plt.plot(epochs_range, acc, label='Train Acc'); plt.plot(epochs_range, val_acc, label='Val Acc'); plt.legend(); plt.title('Accuracy')
    plt.subplot(1, 3, 2); plt.plot(epochs_range, loss, label='Train Loss'); plt.plot(epochs_range, val_loss, label='Val Loss'); plt.legend(); plt.title('Loss')
    if auc_tr and auc_val and len(auc_tr) == len(auc_val):
        plt.subplot(1, 3, 3); plt.plot(epochs_range, auc_tr, label='Train AUC'); plt.plot(epochs_range, auc_val, label='Val AUC'); plt.legend(); plt.title('AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename_prefix}_training_plot.png"))
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, filename_prefix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename_prefix}_confusion_matrix.png"))
    plt.close()

def plot_roc(y_true, y_pred_proba, filename_prefix):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC'); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename_prefix}_roc.png"))
    plt.close()
    return roc_auc

cv_metrics = []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), start=1):
    prefix = f"{fold_idx}"  # 1..10
    print(f"\n===== Fold {fold_idx}/{N_SPLITS} =====")
    x_train, x_val = filepaths[train_idx], filepaths[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    train_ds = make_ds(x_train, y_train, training=True)
    val_ds = make_ds(x_val, y_val, training=False)

    model, base_model = build_model()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=[early_stopping_p1],
        verbose=1
    )
    plot_training(history_p1, filename_prefix=f"{prefix}_phase1")
    pd.DataFrame(history_p1.history).to_csv(os.path.join(RESULTS_DIR, f"{prefix}_history_phase1.csv"), index=False)

    unfreeze_for_finetune(model, base_model, fine_tune_at=FINE_TUNE_AT, keep_bn_frozen=True)
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping_p2],
        verbose=1
    )
    plot_training(history_p2, filename_prefix=f"{prefix}_phase2")
    pd.DataFrame(history_p2.history).to_csv(os.path.join(RESULTS_DIR, f"{prefix}_history_phase2.csv"), index=False)

    y_true, y_prob = [], []
    for imgs, lbls in val_ds:
        probs = model.predict(imgs, verbose=0).flatten()
        y_prob.extend(probs.tolist())
        y_true.extend(lbls.numpy().tolist())
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    roc_auc_val = plot_roc(y_true, y_prob, filename_prefix=prefix)

    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, filename_prefix=prefix)
    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open(os.path.join(RESULTS_DIR, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(
            report_txt + f"\nAUC: {roc_auc_val:.4f}\n"
            f"ACC: {acc:.4f}  PREC: {prec:.4f}  REC: {rec:.4f}  F1: {f1:.4f}\n"
            f"phase1_best_epoch(val_auc): {int(np.argmax(history_p1.history.get('val_auc', [0])) + 1)}\n"
            f"phase2_best_epoch(val_auc): {int(np.argmax(history_p2.history.get('val_auc', [0])) + 1)}\n"
        )

    # Meta per fold
    with open(os.path.join(RESULTS_DIR, f"{prefix}_meta.json"), "w") as f:
        json.dump({
            "fold": fold_idx,
            "samples_train": int(len(x_train)),
            "samples_val": int(len(x_val)),
            "fine_tune_at": FINE_TUNE_AT,
            "lr_frozen": LR_FROZEN,
            "lr_fine": LR_FINE,
            "acc": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc_val),
            "phase1_best_epoch": int(np.argmax(history_p1.history.get('val_auc', [0])) + 1),
            "phase2_best_epoch": int(np.argmax(history_p2.history.get('val_auc', [0])) + 1)
        }, f, indent=2)

    cv_metrics.append({
        "fold": fold_idx,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc_val
    })

cv_df = pd.DataFrame(cv_metrics).sort_values("fold")
cv_df.to_csv(os.path.join(RESULTS_DIR, "cv_metrics_per_fold.csv"), index=False)

summary = {
    "acc_mean": cv_df["acc"].mean(),
    "acc_std": cv_df["acc"].std(ddof=0),
    "precision_mean": cv_df["precision"].mean(),
    "precision_std": cv_df["precision"].std(ddof=0),
    "recall_mean": cv_df["recall"].mean(),
    "recall_std": cv_df["recall"].std(ddof=0),
    "f1_mean": cv_df["f1"].mean(),
    "f1_std": cv_df["f1"].std(ddof=0),
    "roc_auc_mean": cv_df["roc_auc"].mean(),
    "roc_auc_std": cv_df["roc_auc"].std(ddof=0),
    "created_at": datetime.now().isoformat()
}
pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, "cv_metrics_summary.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "README_results.txt"), "w") as f:
    f.write(
        "Results of MobileNetV2 (ImageNet fine-tune) 10-fold CV on the data-ar dataset (arabica vs robusta)\n"
        f"Classes: {CLASS_NAMES}\n"
        "Per fold: {1..10}_history_phase1.csv, {1..10}_training_plot_phase1.png, "
        "{1..10}_history_phase2.csv, {1..10}_training_plot_phase2.png, "
        "{1..10}_confusion_matrix.png, {1..10}_roc.png, {1..10}_classification_report.txt, {1..10}_meta.json\n"
        "Aggregate: cv_metrics_per_fold.csv, cv_metrics_summary.csv\n"
    )

print("✅ Completed 10-fold fine-tuning of MobileNetV2 on ImageNet. Results in folder:", RESULTS_DIR)
