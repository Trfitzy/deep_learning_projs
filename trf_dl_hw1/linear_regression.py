from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#----------------------------------------------------------------------------
# Linear Regression
#----------------------------------------------------------------------------
def run_lr_model(data_dic, lr):

    sgd_model = linear_model.SGDRegressor(loss='squared_error', learning_rate='constant', eta0=lr)

    sgd_model.fit(data_dic['X_train'], data_dic['Y_train'])

    Y_pred = sgd_model.predict(data_dic['X_test'])

    # The coefficients
    print("Coefficients: \n", sgd_model.coef_)

    # The predicted data is used to get a R2 score and MSE
    r2 = r2_score(data_dic['Y_test'],Y_pred)
    mse = mean_squared_error(data_dic['Y_test'],Y_pred)
    print("R2 score : %.4f" % r2 )
    print("Mean squared error: %.4f" % mse)
