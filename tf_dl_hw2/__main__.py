import logging
from argparse import ArgumentParser
import keras
import numpy as np

from utils import conv_block, ident_block, fit_n_assess_model, test_model, all_vis_output
import warnings

warnings.filterwarnings('ignore')

model_help = """The model to run for MNIST data (input number of corresponding model):   1:DNN,  2:CNN,  3:VGG,  4:ResNet"""

def parse_args():
    arg_parser = ArgumentParser(
        prog="MNIST", usage="Runs differnt models to classify MNIST data."
    )
    arg_parser.add_argument(
        "-m", "--model", type=int, choices=[1,2,3,4], dest="model", help=model_help)
    arg_parser.add_argument(
        "-o", "--optemizer", type=str, default='adagrad', choices=['adam','SGD','adagrad'], dest="opt", help="The optimizer to use. Default is Adagrad")
    arg_parser.add_argument(
        "-l", "--learn_rate", type=float, default=0.01, dest="lr", help="The learning rate. Default = 0.01")
    
    return arg_parser.parse_args()


if __name__ == "__main__":
    input_args = parse_args()
    #try:

    path = "/content/sample_data/mnist_train_small.csv"
    # Pull in data
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    data_dic = {
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,
        "X_test": X_test,
    }
    model_info = {
        'epochs': 5,
        'learning_rate': input_args.lr,
        'loss':  keras.losses.SparseCategoricalCrossentropy(), # multiclass crossentropy
        "activation": 'relu',
        "final_activation": 'softmax',
    }

    if input_args.opt.lower() == "adam":
        model_info["optimizer"] = keras.optimizers.Adam(learning_rate=model_info['learning_rate'])
    elif input_args.opt.lower() == "sgd":
        model_info["optimizer"] = keras.optimizers.SGD(learning_rate=model_info['learning_rate'])
    else:
        model_info["optimizer"] = keras.optimizers.Adagrad(learning_rate=model_info['learning_rate'])
    
    # Run model using the data that was processed
    if input_args.model == 1:
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation=model_info['activation']),
                keras.layers.Dense(256, activation=model_info['activation']),
                keras.layers.Dense(128, activation=model_info['activation']),
                keras.layers.Dense(64, activation=model_info['activation']),
                keras.layers.Dense(10, activation=model_info['final_activation']),
            ],
            name = 'DNN-MNIST'
            )

    elif input_args.model == 2:
        model = keras.Sequential(
            [
                # Block 1
                keras.Input(shape=(28, 28, 1)),
                keras.layers.Conv2D(16, kernel_size=(3, 3), activation=model_info['activation']),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                
                # Block 2
                keras.layers.Conv2D(32, kernel_size=(3, 3), activation=model_info['activation']),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),

                keras.layers.Flatten(),
                keras.layers.Dense(10, activation=model_info['final_activation']),
            ],
            name = 'CNN-MNIST'
        )
        
    elif input_args.model == 3:
        model = keras.Sequential(
            [
                # Block 1
                keras.Input(shape=(28, 28, 1)),
                keras.layers.Conv2D(8, kernel_size=(3, 3), padding="same", activation=model_info['activation']),
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                # Block 2
                keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation=model_info['activation']),
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                
                # Block 3
                keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=model_info['activation']),
                keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation=model_info['activation']),
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                # Block 4
                keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=model_info['activation']),
                keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation=model_info['activation']),
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                keras.layers.Flatten(),
                keras.layers.Dense(512, activation=model_info['activation']),
                keras.layers.Dense(512, activation=model_info['activation']),
                keras.layers.Dense(100, activation=model_info['activation']),
                keras.layers.Dense(10, activation=model_info['final_activation']),
            ],
            name = 'VGG-MNIST'
        )
        
        
    elif input_args.model == 4:
        
        resnet_input = keras.Input(shape=(28, 28, 1))#, name="img")
        conv1 = keras.layers.Conv2D(8, kernel_size=(7, 7), strides=(2, 2), padding="same", activation=model_info['activation'])(resnet_input)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

        conv2_1 = ident_block(8, conv1, model_info)
        conv2_2 = ident_block(8, conv2_1, model_info)

        conv3_1 = conv_block(16, conv2_2, model_info)
        conv3_2 = ident_block(16, conv3_1, model_info)

        conv4_1 = conv_block(32, conv3_2, model_info)

        conv5_1 = conv_block(64, conv4_1, model_info)

        model_output = keras.layers.GlobalMaxPooling2D()(conv5_1)
        model_output = keras.layers.Dense(10, activation=model_info['final_activation'])(model_output)

        model = keras.Model(resnet_input, model_output, name="ResNet-MNIST")

    hist1 = fit_n_assess_model(model, model_info, data_dic)
    test_model(model, X_test, y_test)
    model.save("cnn_model.keras")
    if input_args.model != 1:
        all_vis_output(model, X_test[np.random.randint(1,1000)], model.name, 3)
            
    
    #except(FileNotFoundError):
    #    print("""FileNotFoundError - Double check the following:
    #          - file exists
    #          - file name includes .csv extension""")
