from utilities import *
from forecast_model import ForecastModel

# Params and flag definition ############################################################################

model_type = 'Residual'
predict_param = 'TOA'
hyper_parameter_tuning_flag = False
temporal_res = 'year'

def main(experiment):

    ################## Load the validation data (from 2013 to 2023) #################################
    data_processor = DataProcessor(predict_param = predict_param, future_data = False, remove_small_samples = True)
    df_train = data_processor.load_data(file_path = 'df_merged') 
    df_val = data_processor.load_data(file_path = f'df_merged_val{experiment}') 

    X_train, _, _ = data_processor.data_filtering_and_feature_engineering(df_train)
    X_val, y_val, _ = data_processor.data_filtering_and_feature_engineering(df_val)
    X_val = X_val[X_train.columns]

    ##################### PLOT AND COMPARE THE MEASURED AND FORECASTED TARGET VARIABLE ################

    # LOAD THE MODELS
    forecast_model = ForecastModel(model_type = model_type, predict_param = predict_param)
    forecast_model.load()

    #Forecast the target variable
    y_pred = forecast_model.model.predict(X_val)

    #######################COMPUTE AND PLOT SEN'S SLOPE TREND####################################

    fontsize = 20
    plt.figure(figsize=(10, 10))
    plt.scatter(y_pred,y_val.values.ravel(),s = 20,color='k')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_val.values.ravel())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)
    plt.style.use('classic')
    plt.rcParams['font.family'] = 'sans-serif'
    if predict_param == 'TOA':
        plt.xlim([-8,4])
        plt.ylim([-8,4])
        plt.plot([-8,4],[-8,4],color='k',linestyle='--')
        lrline = np.array([slope*-8 + intercept,slope*4+intercept])
        lr = plt.plot(np.array([-8,4]),lrline,color='r')
        plt.text(1.0,-0.5,'R2=' + str(round(np.corrcoef(y_pred,y_val.values.ravel())[0,1]**2,2)), fontsize=fontsize)
        plt.text(1.0,-1,'RMSE=' + str(round(np.sqrt(np.sum((y_pred-y_val.values.ravel())**2)/len(y_pred)),2)), fontsize=fontsize)
        plt.xlabel('Predicted TOA CRF [W/m$^2$]', fontsize=fontsize)
        plt.ylabel('Measured TOA CRF [W/m$^2$]', fontsize=fontsize)
        plt.legend(lr,['$y=' + str(round(slope,2)) + '\cdot x+' + str(round(intercept,2)) + '$'],loc='upper right', prop={'size': fontsize})
    elif predict_param == 'SFC':
        plt.xlim([-8,2])
        plt.ylim([-8,2])
        plt.plot([-8,2],[-8,2],color='k',linestyle='--')
        lrline = np.array([slope*-8 + intercept,slope*2+intercept])
        lr = plt.plot(np.array([-8,2]),lrline,color='r')
        plt.text(-0.5,-0.5,'R2=' + str(round(np.corrcoef(y_pred,y_val.values.ravel())[0,1]**2,2)), fontsize=fontsize)
        plt.text(-0.5,-1,'RMSE=' + str(round(np.sqrt(np.sum((y_pred-y_val.values.ravel())**2)/len(y_pred)),2)), fontsize=fontsize)
        plt.xlabel('Predicted SFC CRF [W/m$^2$]', fontsize=fontsize)
        plt.ylabel('Measured SFC CRF [W/m$^2$]', fontsize=fontsize)
        plt.legend(lr,['$y=' + str(round(slope,2)) + '\cdot x+' + str(round(intercept,2)) + '$'],loc='upper right', prop={'size': fontsize})
    elif predict_param == 'DIF':
        plt.xlim([0,8])
        plt.ylim([0,8])
        plt.plot([0,8],[0,8],color='r',linestyle='--')
        plt.text(0.5,0.5,'R2=' + str(round(np.corrcoef(y_pred,y_val.values.ravel())[0,1]**2,2)))
        plt.text(0.5,1,'RMSE=' + str(round(np.sqrt(np.sum((y_pred-y_val.values.ravel())**2)/len(y_pred)),2)))
        plt.xlabel('Predicted SFC CRF [W/m$^2$]')
        plt.ylabel('Measured SFC CRF [W/m$^2$]')
        
    plt.grid('on')
    plt.savefig(f'../03Figures/{predict_param}{experiment}scatter.png',dpi=300)

if __name__ == "__main__":
    for experiment in experiments:
        main(experiment)