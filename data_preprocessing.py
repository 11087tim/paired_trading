import pandas as pd
import numpy as np
import math
from IPython.display import display

#portfoilo = ['ETH', 'ADA', 'DOGE' ,'SOL' ,'DOT' ,'LINK' ,'XLM' ,'EOS' ,'XMR', 'TRX']

# for portfolio train and portfolio test and preprocess all the given token at specific time zone
def Preprocess(token_names, time_range):

   preprocessing('BTC', time_range)
   data_cp =  {'TimeStamp' : pd.read_csv('data/clean_data/BTC-1h-data.csv', index_col=0, dtype=np.float64).index}
   data_open =  {'TimeStamp' : pd.read_csv('data/clean_data/BTC-1h-data.csv', index_col=0, dtype=np.float64).index}
  
   data_cp = pd.DataFrame(data_cp)
   data_open = pd.DataFrame(data_open)
   

   for token in token_names:
        dataPath = 'data/clean_data/' + token  + '-' + time_range + '-data.csv'
        preprocessing(token, time_range)
        data_cp = pd.concat([data_cp, pd.read_csv(dataPath)['Consecutive Payoff']], axis=1)
        data_open = pd.concat([data_open, pd.read_csv(dataPath)['Open']], axis=1)
        

   #index
   data_cp.index = data_cp['TimeStamp'] 
   data_open.index = data_open['TimeStamp'] 

   # drop dummy 
   data_cp= data_cp.drop(columns='TimeStamp')
   data_open = data_open.drop(columns='TimeStamp') 

   # columns names
   data_cp.columns = token_names
   data_open.columns = token_names
   
   
   #store
   data_cp.to_csv('data/clean_data/' + time_range + '-CP_data.csv', encoding='utf-8')
   data_open.to_csv('data/clean_data/' + time_range + '-open_data.csv', encoding='utf-8')
   

   
# preprocessing for each token at specific time zone
def preprocessing(token_name, time_range = '1d'):
    data_df = {'TimeStamp' : [0], 'Open': [0], 'Close' :[0] }
    data_df = pd.DataFrame(data=data_df)
    

    #test_df = {'TimeStamp' : [0] , 'Open': [0], 'Close' : [0]}
    #test_df = pd.DataFrame(data=test_df)
    
    for i in range(3, 16):
        year = '2022'
        month = i
        if month > 12:
            month = month - 12
            year = '2023'

        if month < 10:
            month = '0'+ str(month)
        #data/spot/monthly/klines/ETHUSDT/1d/2022-03-01_2023-03-01/ETHUSDT-1d-2022-03.csv
        dataPath = 'data/spot/monthly/klines/'+ \
        token_name\
        + 'USDT/'\
        + time_range\
        + '/2022-03-01_2023-03-01/'\
        + token_name\
        + 'USDT-'\
        + time_range\
        +'-'\
        + year\
        + '-'\
        + str(month)\
        +'.csv'

        #print(dataPath)
        try:  
            token_curmon_df = pd.read_csv(dataPath,header=None, dtype=np.float64)
        except:
            continue    
        token_curmon_df.columns = np.arange(0, 12, 1, dtype=int).astype(str)
        token_curmon_df = token_curmon_df.drop([ '2', '3', '5', '6', '7', '8', '9', '10', '11'], axis=1)
        token_curmon_df.columns = ['TimeStamp', 'Open', 'Close']

        data_df = pd.concat([data_df, token_curmon_df], axis=0)    

    
   
    data_df = data_df.iloc[1:, :]    
    #test_df = test_df.drop([0])
    data_df.index = data_df['TimeStamp']
    #test_df.index = test_df['TimeStamp']
    #print(train_df.index)
    data_df = data_df.drop(columns=['TimeStamp'])
    #test_df = test_df.drop(columns=['TimeStamp'])

    data_df['Consecutive Payoff'] =   pd.to_numeric(data_df['Close'])  / pd.to_numeric(data_df['Open'])
    #test_df['Consecutive Payoff'] =    pd.to_numeric(test_df['Close'])  / pd.to_numeric(test_df['Open'])
    
    data_df.to_csv('data/clean_data/' + token_name + '-' + time_range + '-data.csv', encoding='utf-8')
    #test_df.to_csv( 'data/clean_data/' + token_name + '-' + time_range + '-test_data.csv', encoding='utf-8')



   

   