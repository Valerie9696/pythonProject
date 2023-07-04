from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def plot_prediction(predictions, truth):
    values_0 = list((predictions['targets']['t+0']).items())
    values_1 = list((predictions['targets']['t+1']).items())
    values_2 = list((predictions['targets']['t+2']).items())
    values_3 = list((predictions['targets']['t+3']).items())
    values_4 = list((predictions['targets']['t+4']).items())
    all = [[i[1], j[1], k[1], l[1], m[1]] for i, j, k, l, m in zip(values_0, values_1, values_2, values_3, values_4)]
    timestep = 0
    for row in all:
        xaxis = [timestep + x for x in range(0, len(row))]
        yaxis = row
        plt.plot(xaxis, yaxis, color='red')
        timestep += 1
    plt.plot(truth, color='cyan')



"""
Mean Squared Error (MSE): 
MSE calculates the average of the squared differences between the predicted and actual values.
It is a popular metric for measuring the overall accuracy of forecasts.

Root Mean Squared Error (RMSE):
RMSE is the square root of the MSE.
It provides a measure of the average magnitude of the prediction errors in the original scale of the data.

Mean Absolute Error (MAE): 
MAE calculates the average of the absolute differences between the predicted and actual values.
It provides a measure of the average magnitude of the errors, irrespective of their direction.

Mean Absolute Percentage Error (MAPE):
MAPE calculates the average of the percentage differences between the predicted and actual values.
It is useful for evaluating the accuracy of forecasts relative to the magnitude of the actual values.

Symmetric Mean Absolute Percentage Error (SMAPE):
SMAPE is a variation of MAPE that addresses some of its limitations.
It calculates the average of the percentage differences between the predicted and actual values,
taking into account the average of the absolute values of the predicted and actual values.
3.737212826880342 13.966759713398957 788.8864244086889
RMSE:  1.0317886215839918
MSE:  1.746246607100744
MAPE:  3.138148764024348
"""

def metrics(predictions, truth):
    values_0 = list((predictions['targets']['t+0']).items())
    values_1 = list((predictions['targets']['t+1']).items())
    values_2 = list((predictions['targets']['t+2']).items())
    values_3 = list((predictions['targets']['t+3']).items())
    values_4 = list((predictions['targets']['t+4']).items())
    all = [[i[1], j[1], k[1], l[1], m[1]] for i, j, k, l, m in zip(values_0, values_1, values_2, values_3, values_4)]
    timestep = 0
    mse_errors = []
    rmse_errors = []
    mape_errors = []
    for y_pred in all:
        if timestep+len(y_pred)>=len(truth):
            break
        y_true = truth[timestep:timestep+len(y_pred)]
        try:
            mse_errors.append(mean_squared_error(y_true, y_pred, squared=True))
            rmse_errors.append(mean_squared_error(y_true, y_pred, squared=False))
            mape_errors.append(mean_absolute_percentage_error(y_true, y_pred))
        except:
            print('NAN')
            print(y_true)
            print(y_pred)
        timestep += 1
    print(max(rmse_errors), max(mse_errors), max(mape_errors))
    plt.title('MSE')
    plt.plot(mse_errors)
    plt.show()
    plt.title('RMSE')
    plt.plot(rmse_errors)
    plt.show()
    rmse = sum(rmse_errors)/len(rmse_errors)
    mse = sum(mse_errors)/len(mse_errors)
    mape = sum(mape_errors)/len(mape_errors)
    #rsme = sum(rmse_errors)/len(rmse_errors)
    print('RMSE: ', rmse)
    print('MSE: ', mse)
    print('MAPE: ', mape)