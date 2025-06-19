import matplotlib.pyplot as plt
import keras
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

#------------------------------------------------------------------------------
# Helper Functions
# These are functions used several times, but are not the main functions to run linear regression or DNN models
#------------------------------------------------------------------------------

# This function adds a datetime callback before each epoch
class TimeCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        epoch_begin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{time} ---".format(time=epoch_begin),end=(""))


def plot_model_loss(epochs, train_loss, val_loss, title="model loss"):
    # This function creates the train & validation loss comparison plots and saves them to a file
    # Input:    epochs - (int) number of epochs to run
    #           train_loss - (series) contains the training loss over the # of epochs
    #           val_loss - (series) contains the validation loss over the # of epochs
    #           title - (string) this is used to label the plot and create the filname
    # Output: None
    plt.plot(range(0,epochs),train_loss, label="training set")
    plt.plot(range(0,epochs),val_loss, label="validation set")
    
    plt.xlabel("epoch")
    plt.ylabel("mean squared error loss")
    plt.title(title)
    plt.legend()

    filename = title.replace(" ", "_").replace(",","").replace("(","").replace(":","").replace(")","").replace("0.","")
    plt.savefig("./plots/"+filename)
    

def split_data(per_train, df, k=1):
    # This function splits the input data into train, validation, and test based on the training percentage entered
    # Input:    per_train - (float) % of data to use as training
    #           df - (DataFrame) the input data
    # Output:   df_train - (DataFrame) training data based on the % specified
    #           df_test - (DataFrame) testing data, half of the data remaining after training data selected
    #           df_val - (DataFrame) validation data, half of the data remaining after training data selected
    
    ind_split = int(round(per_train*len(df),0))
    ind_split2 = int(round((1-per_train)/2*len(df),0)+ind_split)

    df_train = df[:ind_split]
    df_test = df[ind_split:ind_split2]
    df_val = df[ind_split2:]

    return df_train, df_test, df_val

