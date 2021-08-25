import tensorflow as tf
import numpy as np
import pandas as pd
import os
from time import time


dir_path = ''
num_classes = 2
n_neurons = 20
n_hidden_layers = 4

data_train = pd.read_csv(os.path.join('datasets', dir_path, 'train.csv')).to_numpy()
data_test = pd.read_csv(os.path.join('datasets', dir_path, 'test.csv')).to_numpy()

x_train, y_train = data_train[:, :-1], data_train[:, -1]
x_test, y_test = data_test[:, :-1], data_test[:, -1]

y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[x_train.shape[1]]),
])

for _ in range(n_hidden_layers):
    model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))

model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'], )

model_path = os.path.join('datasets', dir_path, f'model_{n_hidden_layers}layers_{dir_path}.h5')

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
ck = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

start = time()
model.fit(x_train, y_train_ohe, batch_size=4, epochs=100, validation_data=(x_test, y_test_ohe), verbose=2, callbacks=[ck, es])
print(f'Tempo de Treinamento: {time()-start}')

model = tf.keras.models.load_model(model_path)
print('Resultado Treinamento')
model.evaluate(x_train, y_train_ohe, verbose=2)

print('Resultado Teste')
model.evaluate(x_test, y_test_ohe, verbose=2)
