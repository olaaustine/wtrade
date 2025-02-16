import pandas as pd 

class LoadTrainingData():

    def load_training_data(self,filename):
        read_data = pd.read_csv(filename,  sep='\t')

        read_data[['Open', 'High', 'Low', 'Close', 'Volume']] = read_data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

        X = read_data[['Open', 'High', 'Low', 'Volume']]
        Y = read_data['Close']

        return X, Y
    

    def load_average_data(self, filename):
        read_data = pd.read_csv(filename, sep='\t')
        
        read_data[['Open', 'High', 'Low', 'Close', 'Volume']] = read_data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

        open_mean = read_data['Open'].mean(skipna=True)
        high_mean = read_data['High'].mean(skipna=True)
        low_mean = read_data['Low'].mean(skipna=True)
        vol_mean = read_data['Volume'].mean(skipna=True)
        close_mean = read_data['Close'].mean(skipna=True)

        X = pd.DataFrame([[open_mean, high_mean, low_mean, vol_mean]], 
                     columns=['Open_Mean', 'High_Mean', 'Low_Mean', 'Volume_Mean'])
        Y = close_mean

        return X, Y
    
    def load_daily_return(self, filename):
        read_data = pd.read_csv(filename, sep="\t")

        read_data[['Open', 'High', 'Low', 'Close', 'Volume']] = read_data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

        X = read_data[['Open', 'Close']].dropna()
        Y = (read_data['Close'] - read_data['Open']) / read_data['Open'].dropna()


        return X, Y 
    
    def load_seven_day_average(self, filename):
        read_data = pd.read_csv(filename, sep="\t")

        read_data[['Open', 'High', 'Low', 'Close', 'Volume']] = read_data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')
        X = read_data['Open'].rolling(window=7).mean().dropna()
        Y = read_data['Close'].rolling(window=7).mean().dropna()

        return X, Y 
    

    def load_thirty_day_average(self, filename):
        read_data = pd.read_csv(filename, sep="\t")

        read_data[['Open', 'High', 'Low', 'Close', 'Volume']] = read_data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

        X = read_data['Open'].rolling(window=30).mean().dropna()
        Y = read_data['Close'].rolling(window=30).mean().dropna()


        return X, Y




