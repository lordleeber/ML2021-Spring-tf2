import numpy as np
import csv
import os

import tensorflow as tf


train_path = 'covid.train.csv'  # path to training data
test_path = 'covid.test.csv'   # path to testing data


#train_data = np.genfromtxt(train_path, delimiter=",")
with open(train_path, 'r') as fp:
    train_data = list(csv.reader(fp))
    train_data = np.array(train_data[1:])[:, 1:].astype(float)
train_data[:, 40:] = (train_data[:, 40:] - train_data[:, 40:].mean(axis=0, keepdims=True)) / train_data[:, 40:].std(axis=0, keepdims=True)

dev_data = train_data[:270, ...]
train_data = train_data[270:, ...]

#test_data = np.genfromtxt(test_path, delimiter=",")
with open(test_path, 'r') as fp:
    test_data = list(csv.reader(fp))
    test_data = np.array(test_data[1:])[:, 1:].astype(float)
test_data[:, 40:] = (test_data[:, 40:] - test_data[:, 40:].mean(axis=0, keepdims=True)) / test_data[:, 40:].std(axis=0, keepdims=True)

config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape=(93,), activation='relu'))
model.add(tf.keras.layers.Dense(32, input_shape=(64,), activation='relu'))
model.add(tf.keras.layers.Dense(1, input_shape=(32,), activation=None))

model.summary()

model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        loss=tf.keras.losses.mean_squared_error,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )


model.fit(train_data[:, :93], train_data[:, 93], epochs=config['n_epochs'], batch_size=config['batch_size'], validation_split=0.2)

















































