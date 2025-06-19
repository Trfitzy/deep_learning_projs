from sklearn.metrics import mean_squared_error, r2_score
import keras
from keras import layers

from utils import plot_model_loss, TimeCallback

#-----------------------------------------------------------------------------
# Main DNN function.
#-----------------------------------------------------------------------------

def test_model(model, data_dic):
    # Use testing data to make a prediction on the model
    Y_pred = model.predict(data_dic['X_test'])

    # The predicted data is used to get a R2 score and MSE
    r2 = r2_score(data_dic['Y_test'],Y_pred)
    mse = mean_squared_error(data_dic['Y_test'],Y_pred)
    print("R2 score : %.4f" % r2 )
    print("Mean squared error: %.4f" % mse)
    return r2, mse
    

def fit_eval_model(model, data_dic, name, learning_rate=0.01, epochs=1):
    """
    This function takes in model information and compiles, fits, and evaluates the model. This is used below for each individual model
    Input:  model 
            data_dic - (dic of DataFrames) contains all of the data to run the model
            name - (string) the name of the model. Used in creating the plot
            learning_rate - (float) learning rate of the model. Can be entered when running the code or defaults to 0.01
            epochs - (int) number of epochs for the model. Can be entered when running the code or defaults to 11.

    Output: r2_score_val - (float) the r2 score using the test data
            mse - (float) the mean squared error using the test data
    """
    # Compile model with MSR as the loss function and SGD as the optimizer
    model.compile(
        loss=keras.losses.MeanSquaredError(name="mean_squared_error", dtype=None),
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        #metrics=['mean_squared_error']
    )

    # Fit the model with the training data from the data dictionary.
    print("\nStart fitting {} model with {} epochs ".format(name, epochs))
    hist1 = model.fit(x=data_dic['X_train'],
                    y=data_dic['Y_train'],
                    epochs=epochs,
                    callbacks=[TimeCallback()],
                    validation_data=(data_dic['X_val'],data_dic['Y_val']))
    
    # Pull training and validation loss from history output object
    train_loss = hist1.history['loss']
    val_loss = hist1.history['val_loss']
    
    
    # Evaluate the model with the testing data from the data dictionary
    test_loss = model.evaluate(data_dic['X_test'],data_dic['Y_test'])
    print("Test loss: ",round(test_loss,4))

    # Run test_model
    r2, mse = test_model(model, data_dic)

    # Plots the loss and saves it to a file with the model information
    plot_model_loss(
        epochs, 
        train_loss, 
        val_loss, 
        title=name+" loss (epochs: "+str(epochs)+", learning rate: "+str(learning_rate)+")"
        )
    filepath = "./weights/"+name+".weights.h5"
    model.save_weights(filepath, overwrite=True)
    #print(model.get_weights())
    
    return r2, mse

#--------------------------------------------------------------------------
# Individual DNN functions
#--------------------------------------------------------------------------

def run_DNN16_model(data_dic, learning_rate=0.01, epochs=10):
    """ 
    This function runs DNN-16. A model with 1 hidden layer with 16 nodes.
    Input:  data_dic - (dic of DataFrames) contains all of the data to run the model
            name - (string) the name of the model. Used in creating the plot
            learning_rate - (float) learning rate of the model. Can be entered when running the code or defaults to 0.01
            epochs - (int) number of epochs for the model. Can be entered when running the code or defaults to 11.
    Output: r2_score_val - (float) the r2 score using the test data
            mse - (float) the mean squared error using the test data
    """
    # Prepare the input
    inputs = keras.Input(shape=(len(data_dic['X_train'].columns),))

    # Create the layers
    x = layers.Dense(16)(inputs)
    
    # Create the model with the layers
    model = keras.Model(inputs=inputs, outputs=layers.Dense(1)(x), name='DNN-16')

    # Run the function to compile, fit, evaluate, and predict the model
    r2, mse = fit_eval_model(model, data_dic, 'DNN-16', learning_rate, epochs=epochs)
    return r2, mse 

#--------------------------------------------------------------------------    

def run_DNN30_8_model(data_dic, learning_rate=0.01, epochs=10):
    """ 
    This function runs DNN-30-8. A model with 2 hidden layers with 30 and 8 nodes, respectively.
    Input:  data_dic - (dic of DataFrames) contains all of the data to run the model
            name - (string) the name of the model. Used in creating the plot
            learning_rate - (float) learning rate of the model. Can be entered when running the code or defaults to 0.01
            epochs - (int) number of epochs for the model. Can be entered when running the code or defaults to 11.
    Output: r2_score_val - (float) the r2 score using the test data
            mse - (float) the mean squared error using the test data
    """
    # Prepare the input
    inputs = keras.Input(shape=(len(data_dic['X_train'].columns),))
    
    # Create the layers
    x1 = layers.Dense(30)(inputs)
    x2 = layers.Dense(8)(x1)

    # Create the model with the layers
    model = keras.Model(inputs=inputs, outputs=layers.Dense(1)(x2), name='DNN-30-8')

    # Run the function to compile, fit, evaluate, and predict the model
    r2, mse = fit_eval_model(model, data_dic, 'DNN-30-8', learning_rate, epochs=epochs)
    return r2, mse

#--------------------------------------------------------------------------    

def run_DNN30_16_8_model(data_dic, learning_rate=0.01, epochs=10):
    """ 
    This function runs DNN-30-16-8. A model with 3 hidden layers with 30, 16, and 8 nodes, respectively.
    Input:  data_dic - (dic of DataFrames) contains all of the data to run the model
            name - (string) the name of the model. Used in creating the plot
            learning_rate - (float) learning rate of the model. Can be entered when running the code or defaults to 0.01
            epochs - (int) number of epochs for the model. Can be entered when running the code or defaults to 11.
    Output: r2_score_val - (float) the r2 score using the test data
            mse - (float) the mean squared error using the test data
    """
    # Prepare the input
    name = 'DNN-30-16-8'
    inputs = keras.Input(shape=(len(data_dic['X_train'].columns),))
    
    # Create the layers
    x1 = layers.Dense(30)(inputs)
    x2 = layers.Dense(16)(x1)
    x3 = layers.Dense(8)(x2)

    # Create the model with the layers
    model = keras.Model(inputs=inputs, outputs=layers.Dense(1)(x3), name='DNN-30-16-8')

    # Run the function to compile, fit, evaluate, and predict the model
    r2, mse = fit_eval_model(model, data_dic, name, learning_rate, epochs=epochs)
    return r2, mse 

#--------------------------------------------------------------------------    

def run_DNN30_16_8_4_model(data_dic, learning_rate=0.01, epochs=10):
    """ 
    This function runs DNN-30-16-8-4. A model with 2 hidden layers with 30, 16, 8, and 4  nodes, respectively.
    Input:  data_dic - (dic of DataFrames) contains all of the data to run the model
            name - (string) the name of the model. Used in creating the plot
            learning_rate - (float) learning rate of the model. Can be entered when running the code or defaults to 0.01
            epochs - (int) number of epochs for the model. Can be entered when running the code or defaults to 11.
    Output: r2_score_val - (float) the r2 score using the test data
            mse - (float) the mean squared error using the test data
    """
    # Prepare the input
    name = 'DNN-30-16-8-4'
    inputs = keras.Input(shape=(len(data_dic['X_train'].columns),))

    # Create the layers
    x1 = layers.Dense(30)(inputs)
    x2 = layers.Dense(16)(x1)
    x3 = layers.Dense(8)(x2)
    x4 = layers.Dense(4)(x3)

    # Create the model with the layers
    model = keras.Model(inputs=inputs, outputs=layers.Dense(1)(x4), name='DNN-30-16-8-4')

    # Run the function to compile, fit, evaluate, and predict the model
    r2, mse = fit_eval_model(model, data_dic, name, learning_rate, epochs=epochs)
    return r2, mse 
