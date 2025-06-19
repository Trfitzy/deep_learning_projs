import logging
from argparse import ArgumentParser
import keras
import numpy as np

from utils import load_data, build_u_model, fit_n_assess_model, plot_images
import warnings

warnings.filterwarnings('ignore')

model_help = """Runs model comparisons for Retina Image Segmentation"""

def parse_args():
    arg_parser = ArgumentParser(
        prog="Retina_Image_Segmentation", usage="Trains and tests 3 models for retina image segmentation"
    )
    arg_parser.add_argument(
        "action", type=str, choices=["train", "test"], help=model_help)
    arg_parser.add_argument(
        "data_path", type=str)
    
    
    return arg_parser.parse_args()


if __name__ == "__main__":
    input_args = parse_args()
    #"C:/Users/trfit/Documents/learning/UC/Deep_Learning/DL_HW3/Data"

    # Pull in data
    #try:
    X_train, y_train = load_data(input_args.data_path + "/Data/train")
    X_test, y_test = load_data(input_args.data_path + "/Data/test")

    data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_test': X_test,
    'y_test': y_test
    }

    filter_num = 16
    model_info = {
            'epochs': 50,
            'learning_rate': 0.01,
            'loss':  "binary_crossentropy", #'MeanSquaredError', #keras.losses.SparseCategoricalCrossentropy(), # multiclass crossentropy
            "activation": 'relu',
            "final_activation": 'softmax',
            'optimizer':'adam',
            'name': 'u_net',
            'metrics':["accuracy",keras.metrics.BinaryIoU(target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None),]
        }

    for num_layers in [2, 3, 4, 5]:
        model_info['name'] = str(num_layers)+" layers u_net"

        # Build Model
        model = build_u_model(num_layers, filter_num)
        model.summary()

        weights_filepath = input_args.data_path + "/DL_HW3_Fitzgerald/weights/U-Net_" + str(num_layers) + "_layers.weights.h5"

        # Check train or test
        if input_args.action == "train":
            
            # Train Model
            hist = fit_n_assess_model(model, model_info, data, input_args.data_path + "/DL_HW3_Fitzgerald/plots/")

            # Save Model weights
            model.save_weights(weights_filepath, overwrite=True)
            print("Weights saved to " + weights_filepath)

        if input_args.action == "test":

            # Load saved weights

            print("Loading model weights from "+weights_filepath)
            #saved_model = keras.models.load_model(weights_filepath)
            model.load_weights(weights_filepath)

            # Run prediction
            y_pred = model.predict(X_test, batch_size=None, steps=None, callbacks=None)    

            # Plot results for image 10
            plot_images(X_test[10],y_test[10],[y_pred[10]],input_args.data_path + "/DL_HW3_Fitzgerald/plots/",[str(num_layers) + "_layers"])

        else:
            print("No action entered")
    
    #except FileNotFoundError:
    #    print("The file path " + input_args.data_path + " does not contain train & test folders or those folders are missing files.")

    

    



    