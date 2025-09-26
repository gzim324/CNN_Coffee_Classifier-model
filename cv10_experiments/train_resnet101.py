import csv
import glob
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (416, 416)
BATCH_SIZE = 16
EPOCHS = 20
RESULTS_DIR = "results_resnet101"
DATA_DIR = "data-ar"
N_SPLITS = 10
SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_filepaths_and_labels(data_dir):
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not classes:
        raise RuntimeError(f"No directories class in {data_dir}")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    filepaths, labels = [], []
    for c in classes:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for p in glob.glob(os.path.join(data_dir, c, ext)):
                filepaths.append(p)
                labels.append(class_to_idx[c])
    if not filepaths:
        raise RuntimeError(f"Not found pictures in {data_dir}/<class>/*.(jpg|png)")
    return np.array(filepaths), np.array(labels), classes


filepaths, labels, class_names = load_filepaths_and_labels(DATA_DIR)
num_classes = len(class_names)
print(f"Classes: {class_names} | number of samples: {len(filepaths)}")

AUTOTUNE = tf.data.AUTOTUNE


def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method='bilinear')
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


augmenter = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augmenter")


def make_dataset(paths, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, y.astype(np.float32)))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, t: (decode_image(p), t), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, t: (augmenter(x, training=True), t), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def build_model():
    base_model = ResNet101(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None
    )
    base_model.trainable = True

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max',
    restore_best_weights=True
)


def plot_training(history, prefix):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    auc_tr = history.history.get('auc', [])
    auc_val = history.history.get('val_auc', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1);
    plt.plot(epochs_range, acc, label='Training Accuracy');
    plt.plot(epochs_range, val_acc, label='Validation Accuracy');
    plt.legend();
    plt.title('Accuracy')
    plt.subplot(1, 3, 2);
    plt.plot(epochs_range, loss, label='Training Loss');
    plt.plot(epochs_range, val_loss, label='Validation Loss');
    plt.legend();
    plt.title('Loss')
    if auc_tr and auc_val and len(auc_tr) == len(epochs_range):
        plt.subplot(1, 3, 3);
        plt.plot(epochs_range, auc_tr, label='Training AUC');
        plt.plot(epochs_range, auc_val, label='Validation AUC');
        plt.legend();
        plt.title('AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_training_plot.png"))
    plt.close()


def save_history_csv(history, prefix):
    keys = list(history.history.keys())
    rows = zip(*[history.history[k] for k in keys])
    out_path = os.path.join(RESULTS_DIR, f"{prefix}_history.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i, row in enumerate(rows):
            writer.writerow([i + 1] + list(row))


def save_confusion_matrix(y_true, y_pred, prefix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label');
    plt.ylabel('True Label');
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_confusion_matrix.png"))
    plt.close()


def plot_roc(y_true, y_pred_proba, prefix):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('ROC');
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{prefix}_roc.png"))
    plt.close()
    return roc_auc


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

cv_all_csv = os.path.join(RESULTS_DIR, "cv_metrics_per_fold.csv")
with open(cv_all_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "val_accuracy", "precision", "recall", "f1", "roc_auc", "best_epoch", "train_time_s"])

cv_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels), start=1):
    print(f"\n===== Fold {fold_idx}/{N_SPLITS} =====")
    prefix = f"{fold_idx}"

    x_train, x_val = filepaths[train_idx], filepaths[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    train_ds = make_dataset(x_train, y_train, training=True)
    val_ds = make_dataset(x_val, y_val, training=False)

    model = build_model()

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )
    train_time = time.time() - t0

    plot_training(history, prefix)
    save_history_csv(history, prefix)

    y_prob = model.predict(val_ds, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = y_val

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_val = plot_roc(y_true, y_prob, prefix)

    save_confusion_matrix(y_true, y_pred, prefix)
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(RESULTS_DIR, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(report + f"\nAUC: {auc_val:.4f}\n"
                         f"ACC: {acc:.4f}  PREC: {prec:.4f}  REC: {rec:.4f}  F1: {f1:.4f}\n")

    best_epoch = int(np.argmax(history.history.get("val_auc", [0])) + 1)

    with open(cv_all_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([fold_idx, f"{acc:.6f}", f"{prec:.6f}", f"{rec:.6f}",
                         f"{f1:.6f}", f"{auc_val:.6f}", best_epoch, f"{train_time:.3f}"])

    cv_metrics.append([acc, prec, rec, f1, auc_val])

cv_arr = np.array(cv_metrics)
means = cv_arr.mean(axis=0)
stds = cv_arr.std(axis=0, ddof=0)
summary_csv = os.path.join(RESULTS_DIR, "cv_metrics_summary.csv")
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "mean", "std"])
    for name, m, s in zip(["val_accuracy", "precision", "recall", "f1", "roc_auc"], means, stds):
        writer.writerow([name, f"{m:.6f}", f"{s:.6f}"])

with open(os.path.join(RESULTS_DIR, "README_results.txt"), "w") as f:
    f.write(
        "Results of ResNet101 10-fold CV on the data-ar dataset (arabica vs robusta)\n"
        f"Classes: {class_names}\n"
        "Per fold: {1..10}_training_plot.png, {1..10}_history.csv, "
        "{1..10}_confusion_matrix.png, {1..10}_roc.png, {1..10}_classification_report.txt\n"
        "Aggregate: cv_metrics_per_fold.csv, cv_metrics_summary.csv\n"
        f"Created: {datetime.now().isoformat()}\n"
    )

print(f"\n✅ 10-fold cross-validation completed. Results: {RESULTS_DIR}")
