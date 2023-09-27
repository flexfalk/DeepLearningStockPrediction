import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def get_sequence_model(dropout, MAX_SEQ_LENGTH, NUM_FEATURES, num_rnn_layers=2, num_rnn_units=32):



  features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
  x = keras.layers.Dense(32, activation="relu")(features_input)

  for i in range(num_rnn_layers - 1):
    x = keras.layers.GRU(num_rnn_units, return_sequences=True)(x)
    x = keras.layers.Dropout(dropout)(x)

  x = keras.layers.GRU(num_rnn_units)(x)
  x = keras.layers.Dropout(dropout)(x)

  output = keras.layers.Dense(1, activation="sigmoid")(x)
  rnn_model = keras.Model(features_input, output)

  rnn_model.compile(loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"])

  return rnn_model


def run_experiment(train_X, train_Y, EPOCHS, windowsize, NUM_FEATURES, dropout=0.4, early_stopping_patience=10, VALIDATION=0.25):

    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True,
    #                              save_best_only=True, verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    seq_model = get_sequence_model(MAX_SEQ_LENGTH=windowsize, dropout = dropout, NUM_FEATURES=NUM_FEATURES)
    history = seq_model.fit(train_X, train_Y,
        validation_split=VALIDATION,
        epochs=EPOCHS,
        callbacks=[early_stopping])

    return history, seq_model