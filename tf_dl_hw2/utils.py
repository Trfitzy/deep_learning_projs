import matplotlib.pyplot as plt
import keras
from datetime import datetime
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay, f1_score, roc_auc_score, roc_curve, RocCurveDisplay
from collections import Counter

#------------------------------------------------------------------------------
# Model Helper Functions
# These are the main functions to compile, fit, and evaluate the models
#------------------------------------------------------------------------------
# This function adds a datetime callback before each epoch

class TimeCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        epoch_begin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{time} ---".format(time=epoch_begin),end=(""))

#----------------------------------------------------------------------------------------------------------------
# Resnet convolution block
def conv_block(filter, input_data, model_info):
  conv = keras.layers.Conv2D(filter, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=model_info['activation'])(input_data)
  conv = keras.layers.BatchNormalization()(conv)
  conv = keras.layers.Conv2D(filter, kernel_size=(3, 3), padding="same")(conv)

  input_data = keras.layers.Conv2D(filter, kernel_size=(3, 3), strides=(2, 2), padding="same", activation=model_info['activation'])(input_data)
  conv = keras.layers.Add()([input_data, conv])
  output_data = keras.activations.relu(conv)
  return output_data

# Resnet identity block
def ident_block(filter, input_data, model_info):
  conv = keras.layers.Conv2D(filter, kernel_size=(3, 3), padding="same", activation=model_info['activation'])(input_data)
  conv = keras.layers.BatchNormalization()(conv)
  conv = keras.layers.Conv2D(filter, kernel_size=(3, 3), padding="same")(conv)

  conv = keras.layers.Add()([input_data, conv])
  output_data = keras.activations.relu(conv)
  return output_data

def fit_n_assess_model(nmodel, model_info, data):
    print(nmodel.summary())

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    start = datetime.now()

    # Compile model with MSR as the loss function and SGD as the optimizer
    nmodel.compile(loss=model_info['loss'], optimizer=model_info['optimizer'], metrics=["accuracy"])

    print("Fit model")
    # Fit the model with the training data from the data dictionary.
    nhist1 = nmodel.fit(x = X_train, y = y_train, epochs=model_info['epochs'],
                        callbacks=[TimeCallback()],
                        validation_split=0.2,)

    stop = datetime.now()
    model_info['tdelta'] = str(stop-start)
    print("Runtime: ", str(stop-start))

    print("Evaluate model")
    model_info['test_accuracy'] = str(round(nmodel.evaluate(X_test, y_test, verbose=True)[1],4))
    print("Test accuracy: ",model_info['test_accuracy'])

    
    plot_model_loss(model_info, nhist1.history, title=nmodel.name)
    print("Loss and accuracy plots saved.")
    return nhist1

def test_model(nmodel, X_test, y_test):
  # This function gets a prediction for the test data and then finds and prints 
  #   the following:
  #     - f1_score
  #     - auc
  #     - roc curve
  #     - precision & recall plots
  # Input:  nmodel - (keras model) trained model
  #         X_test - (array) contains input values for testing the model
  #         y_test - (array) contains true answers for test values
  # Output: None

  # Make predictions on X test data
  y_pred = nmodel.predict(X_test, batch_size=None, steps=None, callbacks=None)

  # Calculate f1 score & auc
  f1 = f1_score(y_test, np.argmax(y_pred, axis=1), average='micro')
  auc = np.round(roc_auc_score(y_test, y_pred, multi_class='ovr'), 3)

  print("f1_score for our sample data is {}".format(f1))
  print("Auc for our sample data is {}".format(auc))

  # Modify multi-class y values to a binary class for each multi-class (One Hot Encoding)
  y_onehot_test = LabelBinarizer().fit_transform(y_test)

  # Get plots for the rest of the metrics
  roc_curve(y_onehot_test, y_pred, nmodel.name)
  print("Roc curve plot saved")
  recall_prec_plots(y_onehot_test, y_pred, nmodel.name)
  print("Precision and recall curve plots saved")

#---------------------------------------------------------------------------------------
# Plotting functions
# TODO: need to save to a file instead of showing plots
#------------------------------------------------------------------------------------

def roc_curve(y_onehot_test, y_score, title):
    # This function creates the roc curve
    # Input:  y_onehot_test - (array) contains true answers for test values, one hot encoded
    #         y_score - (array) contains predicted probabilities for answers to test values
    # Output: None
    fig, ax = plt.subplots(figsize=(6, 6))

    for class_id in range(10):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {class_id}",
            ax=ax,
            plot_chance_level=(class_id == 9),
        )

    _ = ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="MNIST classification\nto One-vs-Rest multiclass",
    )

    filename = title+"_roc_curve.png"
    plt.savefig("./plots/"+filename)
    plt.clf()

def recall_prec_plots(y_test, y_score, title):
    # This function creates the recall-precision curves
    # Input:  y_onehot_test - (array) contains true answers for test values, one hot encoded
    #         y_score - (array) contains predicted probabilities for answers to test values
    # Output: None

    # For each class
    precision = dict()
    recall = dict()
    threshold = dict()
    average_precision = dict()

    for i in range(10):
        precision[i], recall[i], threshold[i] = precision_recall_curve(y_test[:,i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:,i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], threshold['micro'] = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
        prevalence_pos_label=Counter(y_test.ravel())[1] / y_test.size,
    )
    display.plot(plot_chance_level=True)
    
    filename = title+"_avg_precision.png"
    plt.savefig("./plots/"+filename)
    plt.clf()

    _ = display.ax_.set_title("Micro-averaged over all classes")

    _, ax = plt.subplots(figsize=(7, 8))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

    for i in range(10):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}")

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    filename = title+"_precision_recall_curve"
    plt.savefig("./plots/"+filename)
    plt.clf()
    #plt.show()
    #return precision, recall, threshold, average_precision

def plot_model_loss(model_info, hist, title="model loss"):
    # This function creates the train & validation loss comparison plots and saves them to a file
    # 
    # Output: None

    train_loss = hist['loss']
    val_loss = hist['val_loss']
    train_acc = hist['accuracy']
    val_acc = hist['val_accuracy']

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2,1,1)
    
    ax1.plot(range(0,model_info['epochs']),train_loss, label="training set")
    ax1.plot(range(0,model_info['epochs']),val_loss, label="validation set")
    ax1.set_ylabel(model_info['loss'].name)
    ax1.legend(loc="upper right")
    
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(range(0,model_info['epochs']), train_acc, label="training set")
    ax2.plot(range(0,model_info['epochs']), val_acc, label="validation set")
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc="upper right")
    ax2.set_ylabel('Epoch')

    plot_title = title+" \n("
    cnt = 0
    for key, value in model_info.items():
      if key != 'loss':
        if key == "optimizer":
          if type(value) != str:
            value = value.name
        plot_title += key + ": "+ str(value) + ", "
        cnt += 1
        if cnt == 4:
          plot_title += "\n"
          cnt = 0
    plot_title += ")"

    fig.suptitle(plot_title, y=1.02, fontsize=12)
    plt.tight_layout()
    #plt.show()

    filename = title+"_loss_accuracy.png"
    fig.savefig("./plots/"+filename, bbox_inches='tight')
    plt.clf()

def vis_output(inputs, outputs, image, title):
    # redefine mode to output right after the first hidden layer
    nLayer = keras.Model(inputs = inputs, outputs = outputs)
    #nLayer.summary()

    # get feature map for 1st layer
    feature_maps = nLayer.predict(image)

    out_size = feature_maps.shape[-1]
    if round(out_size/4) != (out_size/4):
        print("Size not div by 4: ",out_size)
    elif len(feature_maps.shape) < 4:
        print("Feature map too small for images: ",feature_maps.shape)
    else:

        # plot maps
        for n in range(out_size):
            ax = plt.subplot(int(out_size/4), 4, n+1)
            plt.imshow(feature_maps[0, :, :, n], cmap='gray')
        #plt.show()

def all_vis_output(nmodel, image, title, n_layers='all'):

    if n_layers == "all":
       n_layers = len(nmodel.layers)
    
    filename = title+"_image.png"
    plt.imsave("./plots/"+filename, image, cmap='gray')
    #plt.show()
    plt.clf()

    for n in range(n_layers):
        vis_output(nmodel.inputs, nmodel.layers[n].output, np.expand_dims(image, axis=0), title)