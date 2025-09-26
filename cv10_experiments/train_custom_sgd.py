import os
import sys
import time
import glob
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from contextlib import redirect_stdout
from tabulate import tabulate
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical

class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        try:
            if callable(lr):
                lr = lr(self.model.optimizer.iterations).numpy()
            else:
                lr = float(tf.keras.backend.get_value(lr))
        except Exception:
            lr = float(getattr(self.model.optimizer, "lr", 0.0))
        print(f"Epoch {epoch + 1}: Learning Rate = {lr}")

ts = time.time()

RESULTS_DIR = "results_custom_sgd"
os.makedirs(RESULTS_DIR, exist_ok=True)

params = {
    "optimizer": "SGD",  # adam, adadelta, SGD
    "dropout": 0.3,
    "kernel_size": 3,
    "img_height": 416,
    "img_width": 416,
    "loss": "categorical_crossentropy",
    "epochs": 50,
    "batch_size": 32,
    "activation": "relu",
    "name": int(ts)
}

DATA_DIR = 'data-ar'
IMG_HEIGHT = int(params["img_height"])
IMG_WIDTH = int(params["img_width"])
BATCH_SIZE = int(params["batch_size"])
EPOCHS = int(params["epochs"])
ACTIVATION = params["activation"]
OPT_NAME = params["optimizer"]
DROPOUT = float(params["dropout"])
KERNEL_SIZE = (int(params["kernel_size"]), int(params["kernel_size"]))
LOSS_FN = params["loss"]
N_SPLITS = 10
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

def list_files_and_labels(data_dir):
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        raise RuntimeError(f"No catalogs in {data_dir}")

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    filepaths, labels = [], []

    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    for cname in class_names:
        for ext in exts:
            for p in glob.glob(os.path.join(data_dir, cname, ext)):
                filepaths.append(p)
                labels.append(class_to_idx[cname])
    if not filepaths:
        raise RuntimeError(f"Not found pictures in {data_dir}/<class>/*.jpg")

    return np.array(filepaths), np.array(labels), class_names

filepaths, labels, CLASS_NAMES = list_files_and_labels(DATA_DIR)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes: {CLASS_NAMES} | number of samples: {len(filepaths)}")

AUTOTUNE = tf.data.AUTOTUNE

def decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH), method='bilinear')
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img

def paths_to_dataset(paths, y_indices, training=False):
    y_onehot = tf.one_hot(y_indices, depth=NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, y_onehot))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (decode_image(p), y), num_parallel_calls=AUTOTUNE)
    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def build_model():
    model = Sequential(name="custom_cnn")
    model.add(Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, activation=ACTIVATION))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, activation=ACTIVATION))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(DROPOUT))

    model.add(Conv2D(filters=256, kernel_size=KERNEL_SIZE, activation=ACTIVATION))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(DROPOUT))

    model.add(Flatten())
    model.add(Dense(256, activation=ACTIVATION))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

    model.compile(optimizer=optimizer, loss=LOSS_FN, metrics=['accuracy'])
    return model

def get_current_lr(model):
    lr = model.optimizer.learning_rate
    try:
        if callable(lr):
            return float(lr(model.optimizer.iterations).numpy())
        return float(tf.keras.backend.get_value(lr))
    except Exception:
        return float(getattr(model.optimizer, "lr", 0.0))

def calculate_metrics(eval_dataset, model, num_classes):
    y_true = np.concatenate([y.numpy() for _, y in eval_dataset], axis=0)
    y_pred_proba = model.predict(eval_dataset, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)

    y_true_categorical = to_categorical(y_true_idx, num_classes=num_classes)
    roc_auc = roc_auc_score(y_true_categorical, y_pred_proba, multi_class='ovr', average='macro')
    precision = precision_score(y_true_idx, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true_idx, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true_idx, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true_idx, y_pred)
    return roc_auc, precision, recall, f1, cm, float((y_true_idx == y_pred).mean())

def display_results(history, cm, current_lr, test_accuracy, precision, recall, f1, roc_auc,
                    total_time, optimizer, batch_size, dropout, kernel_size,
                    img_height, img_width, name_prefix):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    actual_epochs = len(history.epoch)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(actual_epochs), acc, label='Training Accuracy')
    plt.plot(range(actual_epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right'); plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(actual_epochs), loss, label='Training Loss')
    plt.plot(range(actual_epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right'); plt.title('Training and Validation Loss')

    acc_loss_path = os.path.join(RESULTS_DIR, f"{name_prefix}_acc-loss.png")
    plt.tight_layout(); plt.savefig(acc_loss_path); plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    matrix_path = os.path.join(RESULTS_DIR, f"{name_prefix}_matrix.png")
    plt.tight_layout(); plt.savefig(matrix_path); plt.close()

    params_table = [
        ["Epochs", actual_epochs],
        ["Batch Size", batch_size],
        ["Optimizer", optimizer],
        ["Dropout", dropout],
        ["Kernel Size", kernel_size],
        ["Image Size", f"{img_height}x{img_width}"],
        ["Learning Rate", f"{current_lr:.6f}"],
        ["Val Accuracy", f"{test_accuracy:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"],
        ["ROC AUC Score", f"{roc_auc:.4f}"],
        ["Training Time (s)", f"{round(total_time, 1)}"],
    ]
    results_table = tabulate(params_table, headers=["Parameter", "Value"], tablefmt="pretty")

    summary_path = os.path.join(RESULTS_DIR, f"{name_prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PARAMS AND RESULTS\n")
        f.write(results_table)

    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    csv_headers = [
        "fold", "img_size", "epochs", "optimizer", "dropout", "kernel_size", "learning_rate",
        "val_accuracy", "precision", "recall", "f1", "roc_auc", "total_time(s)"
    ]

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow([
            name_prefix, f"{img_height}x{img_width}", actual_epochs, optimizer, dropout, kernel_size,
            f"{current_lr:.6f}", f"{test_accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}",
            f"{f1:.4f}", f"{roc_auc:.4f}", round(total_time, 1)
        ])

    print("Done - Check results.csv")


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

all_metrics_csv = os.path.join(RESULTS_DIR, "cv_metrics_all.csv")
with open(all_metrics_csv, "w", newline="") as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["fold", "val_accuracy", "precision", "recall", "f1", "roc_auc", "epochs_run", "train_time_s"])

cv_summary = []

fold_no = 0
for train_idx, val_idx in skf.split(filepaths, labels):
    fold_no += 1
    print(f"\n===== Fold {fold_no}/{N_SPLITS} =====")

    x_train, x_val = filepaths[train_idx], filepaths[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    train_ds = paths_to_dataset(x_train, y_train, training=True)
    val_ds = paths_to_dataset(x_val, y_val, training=False)

    model = build_model()
    EarlyStop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
    lr_logger = LearningRateLogger()

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        verbose=1,
        validation_data=val_ds,
        callbacks=[EarlyStop, lr_logger]
    )
    total_time = time.time() - start_time

    roc_auc, precision, recall, f1, cm, val_accuracy = calculate_metrics(val_ds, model, num_classes=NUM_CLASSES)

    current_lr = get_current_lr(model)
    prefix = f"{fold_no}"  # 1..10
    display_results(
        history=history,
        cm=cm,
        current_lr=current_lr,
        test_accuracy=val_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        total_time=total_time,
        optimizer=OPT_NAME,
        batch_size=BATCH_SIZE,
        dropout=DROPOUT,
        kernel_size=KERNEL_SIZE,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        name_prefix=prefix
    )

    with open(all_metrics_csv, "a", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([prefix, f"{val_accuracy:.6f}", f"{precision:.6f}", f"{recall:.6f}",
                         f"{f1:.6f}", f"{roc_auc:.6f}", len(history.epoch), round(total_time, 3)])

    cv_summary.append([val_accuracy, precision, recall, f1, roc_auc])

cv_array = np.array(cv_summary)  # shape: (folds, 5)
means = cv_array.mean(axis=0)
stds = cv_array.std(axis=0, ddof=0)

summary_csv = os.path.join(RESULTS_DIR, "cv_metrics_summary.csv")
with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(["metric", "mean", "std"])
    for name, m, s in zip(["val_accuracy", "precision", "recall", "f1", "roc_auc"], means, stds):
        writer.writerow([name, f"{m:.6f}", f"{s:.6f}"])

summary_path = os.path.join(RESULTS_DIR, f"{params['name']}-summary.txt")
with open(summary_path, "a") as f:
    f.write("\n\nMODEL SUMMARY (last fold)\n")
    with redirect_stdout(f):
        model.summary()

print(f"\n✅ 10-fold cross-validation completed. Results saved in: {RESULTS_DIR}")
print(" - Per-fold files: {1..10}_acc-loss.png, {1..10}_matrix.png, {1..10}_summary.txt")
print(" - Aggregate files: results.csv, cv_metrics_all.csv, cv_metrics_summary.csv")
