import csv
import os
import sys
import time
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay
from tabulate import tabulate
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, LeakyReLU, PReLU, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_cimport csv
import os
import sys
import time
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay
from tabulate import tabulate
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, LeakyReLU, PReLU, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical


class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.model.optimizer.iterations).numpy()
        else:
            lr = lr.numpy()
        print(f"Epoch {epoch + 1}: Learning Rate = {lr}")


ts = time.time()

os.makedirs("results", exist_ok=True)


def calculate_metrics(test_dataset, model, num_classes):
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred_proba = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_true, axis=1)

    y_true_categorical = to_categorical(y_true, num_classes=num_classes)

    roc_auc = roc_auc_score(y_true_categorical, y_pred_proba, multi_class='ovr', average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    return roc_auc, precision, recall, f1, cm


def display_results(history, test_accuracy, precision, recall, f1, roc_auc, total_time, optimizer, batch_size, dropout,
                    kernel_size, img_height, img_width, name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    actual_epochs = len(history.epoch)
    current_lr = model.optimizer.learning_rate.numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(actual_epochs), acc, label='Training Accuracy')
    plt.plot(range(actual_epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(actual_epochs), loss, label='Training Loss')
    plt.plot(range(actual_epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    acc_loss_path = os.path.join("results", f"{name}-acc-loss.png")
    plt.savefig(acc_loss_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    matrix_path = os.path.join("results", f"{name}-matrix.png")
    plt.savefig(matrix_path)
    plt.close()

    print("Done - Check results.csv")
    params_table = [
        ["Epochs", actual_epochs],
        ["Batch Size", batch_size],
        ["Optimizer", optimizer],
        ["Dropout", dropout],
        ["Kernel Size", kernel_size],
        ["Image Size", f"{img_height}x{img_width}"],
        ["Learning Rate", f"{current_lr:.6f}"],
        ["Test Accuracy", f"{test_accuracy:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"],
        ["ROC AUC Score", f"{roc_auc:.4f}"],
        ["Training Time (s)", f"{round(total_time, 1)}"],
    ]
    results = tabulate(params_table, headers=["Parameter", "Value"], tablefmt="pretty")

    summary_path = os.path.join("results", f"{name}-summary.txt")
    with open(summary_path, "w") as f:
        f.write("PARAMS AND RESULTS\n")
        f.write(results)

    csv_path = os.path.join("results", "results.csv")
    csv_headers = [
        "files_name", "img_size", "epochs", "optimizer", "dropout", "kernel_size", "learning_rate",
        "test_accuracy", "precision", "recall", "f1", "roc_auc", "total_time(s)"
    ]
    row_data = [
        name, f"{img_height}x{img_width}", actual_epochs, optimizer, dropout, kernel_size, f"{current_lr:.6f}",
        test_accuracy, precision, recall, f"{f1:.4f}", f"{roc_auc:.4f}", round(total_time, 1)
    ]

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow(row_data)


# Settings

params = {
    "optimizer": "adam",  # adam, adadelta, SGD
    "dropout": 0.3,
    "kernel_size": 2,
    "img_height": 416,
    "img_width": 416,
    "loss": "categorical_crossentropy",
    "epochs": 50,
    "batch_size": 32,
    "activation": "relu",
    "name": int(ts)
}

if __name__ == "__main__":
    arguments = sys.argv[1:]

    for arg in arguments:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key in params:
                params[key] = value
                sufix_value = value
            else:
                print(f"Unknown parameter: {key}")
        else:
            print(f"Arg {arg} has wrong syntax.")

data_directory = 'data'

img_height = int(params["img_height"])
img_width = int(params["img_height"])
batch_size = int(params["batch_size"])
epochs = int(params["epochs"])
activation = params["activation"]
optimizer = params["optimizer"]
dropout = float(params["dropout"])
kernel_size = (int(params["kernel_size"]), int(params["kernel_size"]))
loss = params["loss"]

# Load data
train_dataset = image_dataset_from_directory(
    f'{data_directory}/train',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

validation_dataset = image_dataset_from_directory(
    f'{data_directory}/valid',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    f'{data_directory}/test',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

num_classes = len(train_dataset.class_names)

model = Sequential()
model.add(Input(shape=(img_height, img_width, 3)))

model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=kernel_size, activation=activation))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=256, kernel_size=kernel_size, activation=activation))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation=activation))

model.add(Dense(num_classes, activation='softmax'))

if params["optimizer"] == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
elif params["optimizer"] == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
elif params["optimizer"] == "adadelta":
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=3,
                          verbose=1)

lr_logger = LearningRateLogger()

# Training
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=epochs,
    verbose=1,
    batch_size=batch_size,
    validation_data=validation_dataset,
    callbacks=[EarlyStop, lr_logger]
)

total_time = time.time() - start_time

test_loss, test_accuracy = model.evaluate(test_dataset)

roc_auc, precision, recall, f1, cm = calculate_metrics(
    test_dataset,
    model,
    num_classes=num_classes
)

display_results(
    history=history,
    test_accuracy=test_accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    roc_auc=roc_auc,
    total_time=total_time,
    optimizer=params["optimizer"],
    batch_size=batch_size,
    dropout=dropout,
    kernel_size=kernel_size,
    img_height=img_height,
    img_width=img_width,
    name=params["name"]
)

summary_path = os.path.join("results", f"{params['name']}-summary.txt")
with open(summary_path, "a") as f:
    f.write("\n\nMODEL SUMMARY\n")
    with redirect_stdout(f):
        model.summary()
ategorical


class LearningRateLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if callable(lr):
            lr = lr(self.model.optimizer.iterations).numpy()
        else:
            lr = lr.numpy()
        print(f"Epoch {epoch + 1}: Learning Rate = {lr}")


ts = time.time()

os.makedirs("results", exist_ok=True)


def calculate_metrics(test_dataset, model, num_classes):
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred_proba = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_true, axis=1)

    y_true_categorical = to_categorical(y_true, num_classes=num_classes)

    roc_auc = roc_auc_score(y_true_categorical, y_pred_proba, multi_class='ovr', average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    return roc_auc, precision, recall, f1, cm


def display_results(history, test_accuracy, precision, recall, f1, roc_auc, total_time, optimizer, batch_size, dropout,
                    kernel_size, img_height, img_width, name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    actual_epochs = len(history.epoch)
    current_lr = model.optimizer.learning_rate.numpy()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(actual_epochs), acc, label='Training Accuracy')
    plt.plot(range(actual_epochs), val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(actual_epochs), loss, label='Training Loss')
    plt.plot(range(actual_epochs), val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    acc_loss_path = os.path.join("results", f"{name}-acc-loss.png")
    plt.savefig(acc_loss_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    matrix_path = os.path.join("results", f"{name}-matrix.png")
    plt.savefig(matrix_path)
    plt.close()

    print("Done - Check results.csv")
    params_table = [
        ["Epochs", actual_epochs],
        ["Batch Size", batch_size],
        ["Optimizer", optimizer],
        ["Dropout", dropout],
        ["Kernel Size", kernel_size],
        ["Image Size", f"{img_height}x{img_width}"],
        ["Learning Rate", f"{current_lr:.6f}"],
        ["Test Accuracy", f"{test_accuracy:.4f}"],
        ["Precision", f"{precision:.4f}"],
        ["Recall", f"{recall:.4f}"],
        ["F1 Score", f"{f1:.4f}"],
        ["ROC AUC Score", f"{roc_auc:.4f}"],
        ["Training Time (s)", f"{round(total_time, 1)}"],
    ]
    results = tabulate(params_table, headers=["Parameter", "Value"], tablefmt="pretty")

    summary_path = os.path.join("results", f"{name}-summary.txt")
    with open(summary_path, "w") as f:
        f.write("PARAMS AND RESULTS\n")
        f.write(results)

    csv_path = os.path.join("results", "results.csv")
    csv_headers = [
        "files_name", "img_size", "epochs", "optimizer", "dropout", "kernel_size", "learning_rate",
        "test_accuracy", "precision", "recall", "f1", "roc_auc", "total_time(s)"
    ]
    row_data = [
        name, f"{img_height}x{img_width}", actual_epochs, optimizer, dropout, kernel_size, f"{current_lr:.6f}",
        test_accuracy, precision, recall, f"{f1:.4f}", f"{roc_auc:.4f}", round(total_time, 1)
    ]

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if not file_exists:
            writer.writerow(csv_headers)
        writer.writerow(row_data)


# Settings

params = {
    "optimizer": "adam",  # adam, adadelta, SGD
    "dropout": 0.3,
    "kernel_size": 2,
    "img_height": 416,
    "img_width": 416,
    "loss": "categorical_crossentropy",
    "epochs": 50,
    "batch_size": 32,
    "activation": "relu",
    "name": int(ts)
}

if __name__ == "__main__":
    arguments = sys.argv[1:]

    for arg in arguments:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key in params:
                params[key] = value
                sufix_value = value
            else:
                print(f"Unknown parameter: {key}")
        else:
            print(f"Arg {arg} has wrong syntax.")

data_directory = 'data'

img_height = int(params["img_height"])
img_width = int(params["img_height"])
batch_size = int(params["batch_size"])
epochs = int(params["epochs"])
activation = params["activation"]
optimizer = params["optimizer"]
dropout = float(params["dropout"])
kernel_size = (int(params["kernel_size"]), int(params["kernel_size"]))
loss = params["loss"]

# Load data
train_dataset = image_dataset_from_directory(
    f'{data_directory}/train',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

validation_dataset = image_dataset_from_directory(
    f'{data_directory}/valid',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    f'{data_directory}/test',
    label_mode='categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

num_classes = len(train_dataset.class_names)

model = Sequential()
model.add(Input(shape=(img_height, img_width, 3)))

model.add(Conv2D(filters=64, kernel_size=kernel_size, activation=activation))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=128, kernel_size=kernel_size, activation=activation))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=256, kernel_size=kernel_size, activation=activation))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation=activation))

model.add(Dense(num_classes, activation='softmax'))

if params["optimizer"] == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
elif params["optimizer"] == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
elif params["optimizer"] == "adadelta":
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-4)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=3,
                          verbose=1)

lr_logger = LearningRateLogger()

# Training
start_time = time.time()

history = model.fit(
    train_dataset,
    epochs=epochs,
    verbose=1,
    batch_size=batch_size,
    validation_data=validation_dataset,
    callbacks=[EarlyStop, lr_logger]
)

total_time = time.time() - start_time

test_loss, test_accuracy = model.evaluate(test_dataset)

roc_auc, precision, recall, f1, cm = calculate_metrics(
    test_dataset,
    model,
    num_classes=num_classes
)

display_results(
    history=history,
    test_accuracy=test_accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    roc_auc=roc_auc,
    total_time=total_time,
    optimizer=params["optimizer"],
    batch_size=batch_size,
    dropout=dropout,
    kernel_size=kernel_size,
    img_height=img_height,
    img_width=img_width,
    name=params["name"]
)

summary_path = os.path.join("results", f"{params['name']}-summary.txt")
with open(summary_path, "a") as f:
    f.write("\n\nMODEL SUMMARY\n")
    with redirect_stdout(f):
        model.summary()
