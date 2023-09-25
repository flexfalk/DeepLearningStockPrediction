# import modules
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
  
# initialize parameters
start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 1, 1)
  
# get the data
data = yf.download('SIEMENS.NS', start = start_date,
                   end = end_date)

# print(data)
data.to_csv("data/SIEMEND_stock_price.csv")