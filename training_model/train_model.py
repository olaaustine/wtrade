import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from load_training_data import LoadTrainingData


data_loader = LoadTrainingData()

file_path = os.getenv("AGENT_TRADE_PATH", "./data/")
file_paths = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith(".csv")]

class TrainTheModel():
    
    def train_model(self, X, Y):

        params = {
            'objective': 'reg:squarederror',  # Regression task, giving it a regression task because regression is better to train the stock model
            'eval_metric': 'rmse',  # Root Mean Squared Error, because it is a regression task 
            'eta': 0.3,  # Step size shrinkage, used in update to prevent overfitting, also known as learning rate
            'max_depth': 6,  # Maximum depth of trees, 6 is the default
            'subsample': 0.5,  # Fraction of samples used per tree randomly sample half of the training data prior to growing trees. and this will prevent overfitting
            'colsample_bytree': 0.8,  # Fraction of features used per tree  is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
        }
        
        # Split data into training (80%) and testing (20%) sets
        #training size is 0.8
        # test size is 0.2
        # random state to keep only a certain selection continuously
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train, label=Y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=Y_test, enable_categorical=True)

        # Train the model
        # Number of boosting rounds
        model = xgb.train(params, dtrain, num_boost_round=100)

        Y_train_pred = model.predict(dtrain)  # Predictions on training set
        Y_test_pred = model.predict(dtest)  # Predictions on test set

        # Evaluate using RMSE
        # measures how far the predicted values are from the actual values
        #lower RSME means better performance
        train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

        # Evaluate using R² Score
        # measures how well the model explains variance in the target variable 
        # 1 is perfect 
        # 0 performs as well as predicting the mean
        train_r2 = r2_score(Y_train, Y_train_pred)
        test_r2 = r2_score(Y_test, Y_test_pred)

        print(f"Training RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        print(f"Training R² Score: {train_r2:.4f}, Test R² Score: {test_r2:.4f}")

        return model, Y_test_pred
    
    def concat_all_files_together(self, file_path, load_data):
        X_all = []
        Y_all = []
        for file in file_path:
            X, Y = data_loader.load_training_data(file)  # Replace with actual file path
            # X_average, Y_average = data_loader.load_average_data(file)
            # X_daily_return, Y_daily_return = data_loader.load_daily_return(file)
            # X_seven_return, Y_seven_return = data_loader.load_seven_day_average(file)
            # X_thirty_return, Y_thirty_return = data_loader.load_thirty_day_average(file)

            #            # Merge all features into a single DataFrame
            # X = pd.concat([X, X_average, X_daily_return, X_seven_return, X_thirty_return], axis=1)

            # Y = pd.concat([pd.Series(Y_average).repeat(len(Y_daily_return)).reset_index(drop=True), 
            #    Y_daily_return.reset_index(drop=True), 
            #    Y_seven_return.reset_index(drop=True), 
            #    Y_thirty_return.reset_index(drop=True)], axis=1)


            X.dropna(inplace=True) # remove any rows where one of the column is Nan
            # X_average.dropna(inplace=True)
            # X_daily_return.dropna(inplace=True)
            # X_seven_return.dropna(inplace=True)
            # X_thirty_return.dropna(inplace=True)
            Y.dropna(inplace=True) # remove any rows where one of the column is Nan
            # Y_average.dropna(inplace=True)
            # Y_daily_return.dropna(inplace=True)
            # Y_seven_return.dropna(inplace=True)
            # Y_thirty_return.dropna(inplace=True)

            X_all.append(X)
            Y_all.append(Y)

            # Convert lists to DataFrame
        X_combined = pd.concat(X_all, axis=0, ignore_index=True)
        Y_combined = pd.concat(Y_all, axis=0, ignore_index=True)

        return X_combined, Y_combined
    



if __name__ == "__main__":
    # Initialize the class
    trainer = TrainTheModel()

    # Combine all data
    X, Y = trainer.concat_all_files_together(file_paths, data_loader)

    trainer.train_model(X, Y)