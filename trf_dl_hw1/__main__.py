import logging
from argparse import ArgumentParser
import pandas as pd

from preprocess import pre_process
from utils import split_data
from linear_regression import run_lr_model
from DNN import run_DNN16_model, run_DNN30_8_model, run_DNN30_16_8_model, run_DNN30_16_8_4_model
import warnings

warnings.filterwarnings('ignore')

model_help = """The model to run (input number of corresponding model):   1:linear regression,  2:DNN-16,  3:DNN-30-8,  4:DNN-30-16-8,  5:DNN-30-16-8-4"""

def parse_args():
    arg_parser = ArgumentParser(
        prog="Cancer_data_Loss_comparison", usage="Runs different linear regression & DNN models on the cancer data (cancer_reg.csv) to compare loss. The comparison plots are saved to png files."
    )
    arg_parser.add_argument(
        "-m", "--model", type=int, choices=[1,2,3,4,5], dest="model", help=model_help)
    arg_parser.add_argument(
        "-l", "--learn_rate", type=float, default=0.01, dest="lr", help="The learning rate. Default = 0.01")
    arg_parser.add_argument(
        "-e", "--epochs", type=int, default=11, dest="epochs", help="Number of epochs to run. Default = 11")

    return arg_parser.parse_args()


if __name__ == "__main__":
    input_args = parse_args()
    try:
        df = pd.read_csv("cancer_reg.csv")

        # Pre-process & Split Data into training, test, & validation 
        df_norm = pre_process(df)
        df_train, df_test, df_val = split_data(.6, df_norm)

        # Split data into x and y 
        target = 'TARGET_deathRate'
        data_dic = {
            "Y_train": df_train[target],
            "Y_test": df_test[target],
            "Y_val": df_val[target],

            "X_train": df_train.drop(columns=[target]),
            "X_test": df_test.drop(columns=[target]),
            "X_val": df_val.drop(columns=[target]),
        }
        
        # Run model using the data that was processed
        if input_args.model == 1:
            run_lr_model(data_dic,input_args.lr)

        elif input_args.model == 2:
            r2, mse = run_DNN16_model(
                data_dic, 
                learning_rate=input_args.lr, 
                epochs=input_args.epochs
                )
            print(r2)
            
        elif input_args.model == 3:
            r2, mse = run_DNN30_8_model(
                data_dic, 
                learning_rate=input_args.lr, 
                epochs=input_args.epochs
                )
            
        elif input_args.model == 4:
            r2, mse = run_DNN30_16_8_model(
                data_dic, 
                learning_rate=input_args.lr, 
                epochs=input_args.epochs
                )
            
        elif input_args.model == 5:
            r2, mse = run_DNN30_16_8_4_model(
                data_dic, 
                learning_rate=input_args.lr, 
                epochs=input_args.epochs
                )
    
    except(FileNotFoundError):
        print("""FileNotFoundError - Double check the following:
              - file exists
              - file name includes .csv extension""")
