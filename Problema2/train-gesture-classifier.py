import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from pathlib import Path
from keras.layers import Input, Dropout
from keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# Configurar para que TensorFlow utilice la GPU por defecto
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configurar para que TensorFlow asigne memoria dinámicamente
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Especificar la GPU por defecto
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Manejar error
        print(e)

x_data = np.load('./rps_dataset.npy')
y_data = np.load('./rps_labels.npy')


x_temp, x_test, y_temp, y_test = train_test_split(x_data, y_data, test_size=0.2,random_state=123,stratify=y_data)

x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=321, stratify=y_temp)

y_train_cat = to_categorical(y_train, num_classes=3)
y_val_cat = to_categorical(y_val, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)



def build_model(input_shape):
    i = Input(input_shape, dtype=tf.float32)

    x = Dense(32)(i)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    
    x = Dense(3)(x)
    output = Activation('softmax')(x)

    return tf.keras.Model(inputs=i, outputs=output)


print("Building model")
model = build_model(input_shape=(x_train.shape[1],))

model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=['accuracy'])

early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="min", restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=8, verbose=1, min_delta=1e-4, mode="min"
)

history = model.fit(
    x_train,
    y_train_cat,
    validation_data=(x_val, y_val_cat),
    epochs=100,
    batch_size=32,
    callbacks=[reduce_lr, early_stopping],
)

model.save('rps_model.h5')