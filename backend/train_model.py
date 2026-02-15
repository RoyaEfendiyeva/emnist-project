import os
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

train_path = os.path.join(DATASET_DIR, "emnist-byclass-train.csv")
test_path = os.path.join(DATASET_DIR, "emnist-byclass-test.csv")

print("TRAIN EXISTS:", os.path.exists(train_path))
print("TEST EXISTS:", os.path.exists(test_path))


train = pd.read_csv(train_path, nrows=30000)  
test  = pd.read_csv(test_path, nrows=5000)     

x_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values
x_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values


x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)


def fix_orientation(img):

    img = np.rot90(img, k=1)
    img = np.fliplr(img)
    return img

x_train = np.array([fix_orientation(im) for im in x_train])
x_test = np.array([fix_orientation(im) for im in x_test])


x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]


model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5), 
    layers.Dense(62, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")


model_path = os.path.join(MODEL_DIR, "emnist_cnn.h5")
model.save(model_path)

print(f" Model saved to: {model_path}")

