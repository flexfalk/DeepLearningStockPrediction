import pandas as pd
import numpy as np
from typing import Tuple
import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt


def get_csv_path() -> str:

    """This function gives me the csv path of the stock i want to analyze

    Returns:
        str: the path where the stock price csv is
    """    

    return r"C:\Users\sofu0\PycharmProjects\Taiwan\DeepLearningStockPrediction\data\SIEMEND_stock_price.csv"

    

def create_window(data : pd.DataFrame, start: int, windowsize : int, labelling : str) -> Tuple: 
    """this is a 0function that creates a single event given data, start point and windowsize_

    Args:
        data (pd.DataFrame): _description_
        start (int): _description_
        windowsize (int): _description_

    Returns:
        Tuple: returns the tuple of window, next_dat_price a.k.a label and next day date, if we want to keep track of that
    """    

    data = data.to_numpy()

    window = data[start:start + windowsize, :]
    
    this_day_price = data[start + windowsize, 0]
    next_day_price = data[start + windowsize + 1, 0]
        
    # next_day_date = data[start + windowsize + 1, 1]

    next_seven_days_price = np.mean(data[start + windowsize + 1 : start + windowsize + 8, 0])

    if labelling == "next":
        label = this_day_price < next_day_price
    
    if labelling == "avg":
        #not implemented
        label = this_day_price < next_seven_days_price

    #create derivative here inhere

    price = window[:, 0].reshape(-1, 1)

    # Compute the derivative
    derivative = np.gradient(price, axis=0).reshape(-1, 1)
    # print(derivative.shape)
    # print(price.shape)

    window = np.concatenate((price, derivative), axis=1)

        
    #scale window. This could be moved to another place in the code
    window_scaler = StandardScaler()  # Create a scaler for the window
    window = window_scaler.fit_transform(window)  # Scale the window

    return (window, label)


def create_events(data : pd.DataFrame, windowsize, step_size, labelling) -> Tuple:
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


        event1,  label1 = create_window(data, first_day, windowsize, labelling=labelling)

        event2,  label2 = create_window(data, first_day, int(windowsize/2), labelling=labelling)        
        event2 = np.repeat(event2, 2, axis=0)

        event3,  label3 = create_window(data, first_day, int(windowsize/4), labelling=labelling)
        event3 = np.repeat(event3, 4, axis=0)
        # print(event1.shape)

        all_events = np.concatenate((event1, event2, event3), axis=1)
        # print(all_events.shape)

        events = np.concatenate((events, all_events.reshape(1, windowsize, 6)), axis=0)

        # events2 = np.concatenate((events, event2.reshape(1, windowsize, 2)), axis=0)
        # events3 = np.concatenate((events, event1.reshape(1, windowsize, 2)), axis=0)
        # print(events.shape)

        labels.append(label1)
        first_day += step_size
        last_day += step_size

    return (events,np.array(labels))


def preprocessing_pipeline(windowsize, stepsize, labelling):
   
    data = pd.read_csv(get_csv_path())

    print(data.columns)

    #remove features
    data = data[["Close", "Volume"]]

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


from gru_keras import run_experiment, get_sequence_model
import random

def main():
    # print("yoo")
    epochs = 20
    dropout = 0.2
    windowsize = 80
    stepsize = 3
    labelling = 'avg'
    n_features=6
    threshold = 0.5

    #make preprocessing 
    train_X, train_Y, test_X, test_Y = preprocessing_pipeline(windowsize=windowsize, stepsize = stepsize, labelling = labelling)

    # #okay lets try to make the derivative as well
    # print(train_X.shape)

    # #keras training model
    # history, sequence_model = run_experiment(train_X=train_X[:, :, :], train_Y=train_Y, EPOCHS=epochs, 
    #                                          NUM_FEATURES=n_features, dropout=dropout, windowsize=windowsize, early_stopping_patience=10, VALIDATION=0.25)
    # #predict
    # probabilities = sequence_model.predict(test_X[:, :, :])
    # pred_label = (probabilities >= threshold).astype(int).flatten().tolist()

    # print("F1 : ", f1_score(test_Y, pred_label, average='macro'))
    # print("Accuracy : ", accuracy_score(test_Y, pred_label))

    # plt.scatter(train_X[0, :, 0], train_X[0, :, 1])
    # plt.savefig('figures/scatter.png', format='png')
    #examples

    #genenrates som pictures
    random_ingegers = [random.randint(0, 90) for _ in range(3)]
    print(train_X.shape)
    print(train_Y.shape)
    for ri in random_ingegers:
        print(train_X[ri,:, 0].shape)
        plt.figure()
        plt.plot( [i for i in range(80)], train_X[ri, :, 0], label=train_Y[ri])
        plt.legend()
        plt.savefig('figures/figure' + str(ri) + '.png', format='png')




    # plt.plot([i for i in range(len(test_X[0]))],test_X[0][:, 3])
    # plt.savefig('figures/example_stock_plot.png', format='png')

    # plt.figure()
    # plt.title("54%% accuracy")
    # plt.plot([i for i in range(len(history.history["loss"]))], history.history["loss"], label='loss')
    # plt.plot([i for i in range(len(history.history["loss"]))], history.history["val_loss"], label='val_loss')
    # plt.legend()
    # plt.savefig('figures/loss.png', format='png')
    
    #pytorch training
    # model = StockPriceGRU(n_features, hidden_size, output_size)

    # criterion = nn.BCEWithLogitLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Training loop
    # num_epochs = 100
    # for epoch in range(num_epochs):
    #     model.train()
    #     total_loss = 0.0
        
    #     for batch_inputs, batch_labels in train_dataloader:  # Iterate over training data
    #         optimizer.zero_grad()
    #         outputs = model(batch_inputs)
    #         loss = criterion(outputs, batch_labels.float())  # Assuming batch_labels are FloatTensors
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
        
    #     # Calculate average training loss for the epoch
    #     average_loss = total_loss / len(train_dataloader)
        
    #     # Validation loop (evaluate model on the validation set)
    #     model.eval()
    #     with torch.no_grad():
    #         # Perform validation and calculate validation metrics
            
    #     # Print training and validation metrics for monitoring
    #     print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {average_loss:.4f} - Validation Metrics: ...")

    #     # Test the trained model on the test set and calculate test metrics
    #     model.eval()
    #     with torch.no_grad():
    #         # Perform testing and calculate test metrics





        




if __name__ == "__main__":
    main()

