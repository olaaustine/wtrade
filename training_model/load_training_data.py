import pandas as pd 

class LoadTrainingData():

    def load_training_data(self,filename):
        read_data = pd.read_csv(filename,  sep='\t')

        X = read_data[['Open', 'High', 'Low', 'Volume']]
        Y = read_data['Close']

        return X, Y
    

    def load_average_data(self, filename):
        read_data = pd.read_csv(filename, sep='\t')

        open_mean = read_data['Open'].mean()
        high_mean = read_data['High'].mean()
        low_mean = read_data['Low'].mean()
        vol_mean = read_data['Volume'].mean()
        close_mean = read_data['Close'].mean()

        X = [open_mean, high_mean, low_mean, vol_mean]
        Y = close_mean

        return X, Y
    
    def load_daily_return(self, filename):
        read_data = pd.read_csv(filename, sep="\t")

        X = read_data[['Open', 'Close']]
        Y = (read_data['Close'] - read_data['Open']) / read_data['Open']


        return X, Y 
    
    def load_seven_day_average(self, filename):
        read_data = pd.read_csv(filename, sep="\t")

        X = read_data['Close'].rolling(window=7).mean()
        Y = read_data['Open'].rolling(window=7).mean()

        return X, Y 
    

    def load_thirty_day_average(self, filename):
        read_data = pd.read_csv(file=filename, sep="\t")

        X = read_data['Open'].rolling(window=30).mean()
        Y = read_data['Close'].rolling(window=30).mean()


        return X, Y




