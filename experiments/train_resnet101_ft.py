import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = (416, 416)
BATCH_SIZE = 16
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNE = 10
RESULTS_DIR = "results_resnet101-ft"

train_dir = "data/train"
val_dir = "data/valid"
test_dir = "data/test"

os.makedirs(RESULTS_DIR, exist_ok=True)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode='binary'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='binary'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode='binary'
)

class_names = train_ds.class_names
print(f"Klasy: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

base_model = ResNet101(input_shape=IMG_SIZE + (3,),
                       include_top=False,
                       weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max',
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_INITIAL,
    callbacks=[early_stopping]
)

base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[early_stopping]
)


def plot_training(history, filename_prefix):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    auc_train = history.history.get('auc', [])
    auc_val = history.history.get('val_auc', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    if auc_train and auc_val and len(auc_train) == len(epochs_range):
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, auc_train, label='Training AUC')
        plt.plot(epochs_range, auc_val, label='Validation AUC')
        plt.legend()
        plt.title('AUC')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename_prefix}_training_plot.png"))
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def plot_roc(y_true, y_pred_proba, filename):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()

    return roc_auc


plot_training(history, filename_prefix="initial")
plot_training(history_fine, filename_prefix="fine_tuning")

pd.DataFrame(history.history).to_csv(os.path.join(RESULTS_DIR, "training_history_initial.csv"), index=False)
pd.DataFrame(history_fine.history).to_csv(os.path.join(RESULTS_DIR, "training_history_fine_tuning.csv"), index=False)

y_test = []
y_test_pred = []
y_test_pred_proba = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_test_pred_proba.extend(preds.flatten())
    y_test_pred.extend((preds.flatten() > 0.5).astype(int))
    y_test.extend(labels.numpy())

save_confusion_matrix(y_test, y_test_pred, class_names, filename="confusion_matrix_test.png")
roc_auc_test = plot_roc(y_test, y_test_pred_proba, filename="roc_curve_test.png")

report_test = classification_report(y_test, y_test_pred, target_names=class_names)
report_test_full = report_test + f"\n\nAUC-ROC (test): {roc_auc_test:.4f}\n"

with open(os.path.join(RESULTS_DIR, "classification_report_test.txt"), "w") as f:
    f.write(report_test_full)

print("✅ All results saved in the folder:", RESULTS_DIR)
