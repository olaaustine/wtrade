import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

        # Make predictions
        Y_pred = model.predict(dtest)

        return model, Y_pred # we return model for future use
    
    def concat_all_files_together(self, file_path, load_data):
        X_all = []
        Y_all = []
        for file in file_path:
            X, Y = data_loader.load_training_data(file)  # Replace with actual file path

            # Handle missing values

            X = X.apply(pd.to_numeric, errors="coerce") # this will turn my df to numeric and makes sure empty column is turned to Nan
            Y = pd.to_numeric(Y, errors="coerce") # this will turn my series to numeric and makes sure empty column is turned to Nan

            X.dropna(inplace=True) # remove any rows where one of the column is Nan
            Y.dropna(inplace=True) # remove any rows where one of the column is Nan

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

    print(trainer.train_model(X, Y))