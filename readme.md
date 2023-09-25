This is a repository where i try to play around with making a GRU that can do forecasting on a stock

## Plan

Im gonna pick a stock that is good for swing trading 

Siemens Ltd

The plan:

The plan is to see this problem as a similar problem to action recognition on videos. We use a sequence of videos to predict what is going on. I imagine it would be good to have multiple stream, 1 that has the price of the last weak, 1 of the last month, and 1 of the last year.
Theese periods will be our "sliding window" in which we will slide through the graph to make predictions. Of course it is important to not have some from the future in the training. So we cant just shuffle the data. 

We have to basically split it into stags of sliding windows with a label, that is so to say either the price we will predict, or like a label. I think it would be best with an actual price. 

So we make 3 different models i think. 
make an algorithm that goes through all the data from like 2015 to 2021 for training and then 2021- present for testing. We create "events", 
so 
a sliding window and then with a label on is one event. We can start with this proto type that only knows the last 7 days f.eks. and then we will see how it preforms. 

## Evaluation

How do we evaluate it?

Well, if we are actually trying to predict the price we will use some kind of MSE . But i think it could be more fun to actually make it to "predict if it will go up or go down" and then see the accutacy of that. 

Lest actually make so that it if the avg stock price from next week is over og under

THen we can afterwards try to model it, by constructing a trading bot that will buy if the predicted price is X over the current price, and sell if the predicted price is Y under the current price. We can make different scenarios.

## Data gathering. 

We need to just download closing price of the stock for the last 10 years, and put it under the data folder. It cant actually be that much data, so it should be easy. 

## Preprocessing 

I dont think there should be other preprocessing than just the split into events, and then the creation of labels of "up or down". So in a way we actuall need to look at a sliding window of 8 days, where the 8th - 7th day determined the label of the 1 - 7 days.
Be carefull here to not let it know the day it should predict on. 
I think we can shuffle in the evens as long as the periods of test and train are seperately, and specifically that the train should be older that the test data. 

## Model creation. 
We will just use a standard GRU with some dropout layers in Pytorch. If i recall correctly i dont even think we need to use GPU for this, since GRU is very lightweight, and i cant imagine the data to be big in any way. We can find inspiration in the BadmintonProject. 



## Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')