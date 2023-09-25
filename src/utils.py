import pandas as pd
import numpy as np
from typing import Tuple
import keras
from sklearn.preprocessing import StandardScaler

def get_csv_path():

    return r"C:\Users\sofu0\PycharmProjects\Taiwan\DeepLearningStockPrediction\data\SIEMEND_stock_price.csv"


def create_window(data : pd.DataFrame, start: int, windowsize : int, labelling : str = "avg") -> Tuple: 
    """this is a function that creates a single event given data, start point and windowsize_

    Args:
        data (pd.DataFrame): _description_
        start (int): _description_
        windowsize (int): _description_

    Returns:
        Tuple: returns the tuple of window, next_dat_price a.k.a label and next day date, if we want to keep track of that
    """    

    data = data.to_numpy()

    window = data[start:start + windowsize, 1:]
    
    this_day_price = data[start + windowsize, 4]
    next_day_price = data[start + windowsize + 1, 4]
        
    # next_day_date = data[start + windowsize + 1, 1]

    next_seven_days_price = np.mean(data[start + windowsize + 1 : start + windowsize + 8, 4])

    if labelling == "next":
        label = this_day_price < next_day_price
    
    if labelling == "avg":
        #not implemented
        label = this_day_price < next_seven_days_price
        
    #scale window
    window_scaler = StandardScaler()  # Create a scaler for the window
    window = window_scaler.fit_transform(window)  # Scale the window

    return (window, label)


def create_events(data : pd.DataFrame, windowsize = 7, step_size = 2, labelling='avg') -> Tuple:
    """this is a function that creates all the events

    Args:
        data (pd.DataFrame, windowsize, optional): _description_. Defaults to 7, step_size = 4)->Tuple(.

    Returns:
        tuple: a tuple with events, labels and dates 
    """



    events = np.array([]).reshape((0, windowsize, 6))
    # labels = np.array([]).reshape((0, ))
    labels = []
    # prices = []
    # dates = []
    n_days = len(data)

    last_day = windowsize
    first_day = 0

    while last_day < n_days - windowsize:


        event,  label = create_window(data, first_day, windowsize, labelling=labelling)


        events = np.concatenate((events, event.reshape(1, windowsize, 6)), axis=0)

        labels.append(label)


        first_day += step_size
        last_day += step_size


    return (events,np.array(labels))


def preprocessing_pipeline(windowsize, stepsize, labelling):
   
    data = pd.read_csv(get_csv_path())

    print(data.columns)

    # print("here")
    all_events, labels = create_events(data, windowsize, step_size=stepsize, labelling=labelling)

    # if labelling == 'avg':
    #     labels = labels.astype(np.float32)

    split = 0.75

    # scaler = StandardScaler()

    train_X = all_events[0: int(len(all_events)*split)].astype(np.float32)
    # train_X = scaler.fit_transform(train_X.reshape(-1, 6)).reshape(train_X.shape)
    train_Y = labels[0: int(len(all_events)*split)].astype(np.int32)

    
    test_X = all_events[int(len(all_events)*split) :].astype(np.float32)
    # test_X = scaler.fit_transform(test_X.reshape(-1, 6)).reshape(test_X.shape)
    test_Y = labels[int(len(all_events)*split):].astype(np.int32)

    return train_X, train_Y, test_X, test_Y

# @title ##Load Utility for our seqeuence model
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

from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
def main():
    # print("yoo")
    epochs = 10
    dropout = 0.4
    windowsize = 90
    stepsize = 3
    labelling = 'avg'
    n_features = 1

    #make preprocessing 
    train_X, train_Y, test_X, test_Y = preprocessing_pipeline(windowsize=windowsize, stepsize = stepsize, labelling = labelling)

    
    # print(test_Y.shape)
    # print(test_X.shape)
    # print(train_Y.shape)
    # print(train_X.shape)

    print(np.bincount(train_Y))

    print(np.bincount(test_Y))

    print(train_X[:, :, 3:4].shape)


    #train model
    history, sequence_model = run_experiment(train_X=train_X[:, :, 3:4], train_Y=train_Y, EPOCHS=epochs, NUM_FEATURES=n_features, dropout=dropout, windowsize=windowsize, early_stopping_patience=10, VALIDATION=0.25)

    #predict
    probabilities = sequence_model.predict(test_X[:, :, 3:4])

    threshold = 0.5
    pred_label = (probabilities >= threshold).astype(int).flatten().tolist()


    # pred_label = np.argmax(probabilities, axis=1)

    for j in range(len(pred_label)):
        print(pred_label[j], test_Y[j])



    print("F1 : ", f1_score(test_Y, pred_label, average='macro'))
    print("Accuracy : ", accuracy_score(test_Y, pred_label))

    #examples
    plt.plot([i for i in range(len(test_X[0]))],test_X[0][:, 3])
    plt.savefig('figures/example_stock_plot.png', format='png')

    plt.figure()
    plt.title("54%% accuracy")
    plt.plot([i for i in range(len(history.history["loss"]))], history.history["loss"], label='loss')
    plt.plot([i for i in range(len(history.history["loss"]))], history.history["val_loss"], label='val_loss')
    plt.legend()
    plt.savefig('figures/loss.png', format='png')



if __name__ == "__main__":
    main()

