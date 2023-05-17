import pandas as pd
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('TkAgg')
from IPython.display import display
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import data_preprocessing

class Portfolio:

    benchmark = "BTC"

    def __init__(self, portfolio_token, frequency):
        
        self.data_range = ["2022-03", "2023-03"]

        self.train_sample_r2 = 0
        self.test_sample_r2 = 0

        self.tokens = portfolio_token  # list
        self.frequency = frequency  # string
        self.cointergration = 0
        self.weight = []  
        self.weighted_price_train = []
        self.weighted_price_test = []

        self.spread_train = []
        self.spread_test = []
        
        self.train_x = []
        self.train_x_open = []
        self.train_y = []
        self.train_y_open = []
        self.test_x = []
        self.test_x_open = []
        self.test_y = []
        self.test_y_open = []
        #self.predict_ = []
        self.model_ = LinearRegression(fit_intercept=False)
        # function
        data_preprocessing.Preprocess(portfolio_token, frequency)
        self.fetch_data(2, 1, "2022-03")
        #self.forward_fit() 
    
    def fetch_data(self, train_window, test_window, start_date):
        time_string_start_train = datetime.strptime(start_date + "-01 08:00:00", "%Y-%m-%d %H:%M:%S")
        time_span_train = relativedelta(months= train_window)
        time_string_end_train = time_string_start_train + time_span_train
        time_span_test = relativedelta(months= test_window)
        time_string_start_test = time_string_end_train 
        time_string_end_test = time_string_start_test + time_span_test

        index_1 = int(datetime.timestamp(time_string_start_train))
        index_2 = int(datetime.timestamp(time_string_end_train))
        index_3 = int(datetime.timestamp(time_string_end_test))
        print(index_1, index_2, index_3)
        train_range = []
        test_range = []
        for i in range(index_1, index_2, 3600):
            train_range.append( i * 1000.0)
        for i in range(index_2, index_3, 3600):
            test_range.append( i * 1000.0)
        #print(train_range)
        self.train_x = pd.read_csv('data/clean_data/' + self.frequency  + '-CP_data.csv',dtype=np.float64,index_col='TimeStamp').loc[train_range[0]:train_range[-1], :]
        self.test_x = pd.read_csv('data/clean_data/'  + self.frequency  + '-CP_data.csv', dtype=np.float64,index_col='TimeStamp').loc[test_range[0]:test_range[-1], :]
        self.train_y = pd.read_csv('data/clean_data/BTC-1h-data.csv', dtype=np.float64,index_col='TimeStamp').loc[train_range[0]:train_range[-1], "Consecutive Payoff"]
        self.test_y = pd.read_csv('data/clean_data/BTC-1h-data.csv', dtype=np.float64,index_col='TimeStamp').loc[test_range[0]:test_range[-1], "Consecutive Payoff"]
        
        self.train_x_open = pd.read_csv('data/clean_data/' + self.frequency  + '-open_data.csv',dtype=np.float64,index_col='TimeStamp').loc[train_range[0]:train_range[-1], :]
        self.test_x_open = pd.read_csv('data/clean_data/'  + self.frequency  + '-open_data.csv', dtype=np.float64,index_col='TimeStamp').loc[test_range[0]:test_range[-1], :]
        self.train_y_open = pd.read_csv('data/clean_data/BTC-1h-data.csv', dtype=np.float64,index_col='TimeStamp').loc[train_range[0]:train_range[-1], "Open"]
        self.test_y_open = pd.read_csv('data/clean_data/BTC-1h-data.csv', dtype=np.float64,index_col='TimeStamp').loc[test_range[0]:test_range[-1], "Open"]
        
        # time leangth
        self.time_train = train_range
        self.time_test =  test_range

    def fetch_single_token(self, token_name, frequncy ,data_type):
        target_df = pd.read_csv('data/clean_data/' + token_name + '-' + frequncy + '-' + data_type +'_data.csv', dtype=np.float64,index_col='TimeStamp')
        return target_df           
    
    def forward_fit(self, train_window, test_window, start_date):
        self.fetch_data(train_window, test_window, start_date)
        self.model_.fit(self.train_x, self.train_y)
        self.weight = self.model_.coef_

    def predict_(self):
        self.predict_ = self.model_.predict(self.test_x)

    def cointergrqted_coef(self):
        # fitting cointergrated coeff
        # N0 : coef is exisdted
        # N1 : doesn't existed
        model = LinearRegression(fit_intercept= False)
        model.fit(np.reshape(self.weighted_price_train, (-1, 1)), self.train_y_open)
        self.cointergartion = model.coef_
        print(self.cointergartion)

    def ADF(self):
        return(adfuller(self.spread_train))

    def portfolio_pricing(self):
        #calculate the Weighted price
        
        for i in self.time_train:
            # training data
            self.weighted_price_train.append(np.sum(self.train_x_open.loc[i, :] * self.weight))
            # testing data
        
        for i in self.time_test:    
            self.weighted_price_test.append(np.sum(self.test_x_open.loc[i, :] * self.weight))

    def calcu_spread(self):
        self.spread_train = self.train_y_open - (self.weighted_price_train * self.cointergartion)
        self.spread_test = self.test_y_open - (self.weighted_price_test * self.cointergartion)

    def spread_plot(self):
        plt.plot(self.time_test, self.spread_test)
        plt.show()
        
        
if __name__ == "__main__":
    demo = Portfolio(['ETH', 'DOGE' ,'SOL' ,'DOT' ,'LINK' ,'XLM' ,'EOS' ,'XMR', 'TRX'], '1h')
    
    demo.forward_fit(1, 1, "2022-03")
    print(demo.weight)
    demo.portfolio_pricing()
    demo.cointergrqted_coef()
    demo.calcu_spread()
    print(demo.spread_train)
    demo.spread_plot()
    print(demo.ADF())
        
