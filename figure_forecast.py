from utilities import *
from forecast_model import ForecastModel

# Params and flag definition ############################################################################

model_type = 'Residual'
predict_param = 'TOA'
hyper_parameter_tuning_flag = False
temporal_res = 'year'

def main():

    ################## Load the measured/historical data (from 2003 to 2023) #################################
    data_processor = DataProcessor(predict_param = predict_param, future_data = False, remove_small_samples = False)
    df_train = data_processor.load_data(file_path = 'df_merged') 
    df_val = data_processor.load_data(file_path = f'df_merged_val{experiments[0]}') 
    df_measured = pd.concat([df_train,df_val])

    X_measured, y_measured, t_measured = data_processor.data_filtering_and_feature_engineering(df_measured)
    
    plt.figure(figsize=(10, 6))

    # Plot the Sen's slope line of the measured data
    #Compute the measured data Sen's slope
    trend_slope, trend_intercept = data_processor.sens_slope(t_measured, y_measured, temporal_res)
    print(f"Sen's slope measured: {trend_slope}")

    # Plotting the measured data
    plt.plot(t_measured, y_measured, color=[0.5,0.5,0.5])
    start_year = t_measured.iloc[0]
    end_year = t_measured.iloc[-1]
    plt.plot([start_year, end_year], [trend_intercept + trend_slope * (start_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12),
                                    trend_intercept + trend_slope * (end_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12)], 
            label=f'Measured Sen\'s Slope: {np.round(trend_slope,10)} W/m$^2$/{temporal_res}', color='k', linestyle='--', linewidth=2)

    color = [173/255, 0, 31/255]
    for experiment in experiments:

        # Load the future CMIP data (from 2013 to 2049) and forecast the target variable #################################
        data_processor = DataProcessor(predict_param = predict_param, future_data = True, remove_small_samples = False)
        df_future = data_processor.load_data(file_path = f'df_merged_val{experiment}_future') 

        X_future, _, t_future  = data_processor.data_filtering_and_feature_engineering(df_future)
        X_future = X_future[X_measured.columns]

        # LOAD THE MODELS
        forecast_model = ForecastModel(model_type = model_type, predict_param = predict_param)
        forecast_model.load()

        #Forecast the target variable
        y_future = forecast_model.model.predict(X_future)

        #######################COMPUTE AND PLOT SEN'S SLOPE TREND####################################

        #Compute the forecasted data Sen's slope
        trend_slope, trend_intercept = data_processor.sens_slope(t_future, y_future, temporal_res)
        print(f"Sen's slope forecasted: {trend_slope}")

        # Plotting the forecasted data
        plt.plot(t_future, y_future, color=color)

        # Plot the Sen's slope line of the forecasted data
        start_year = t_future.iloc[0]
        end_year = t_future.iloc[-1]
        plt.plot([start_year, end_year], [trend_intercept + trend_slope * (start_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12),
                                        trend_intercept + trend_slope * (end_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12)], 
                label=f'Forecasted Sen\'s Slope {experiment}: {np.round(trend_slope,10)} W/m$^2$/{temporal_res}', color=color , linestyle='--', linewidth=2)

        color = [el + 0.1 for el in color]

    plt.grid('on')
    plt.title('Trend of Time Series with Sen\'s Slope')
    plt.xlabel('Year')
    plt.ylabel(f'{predict_param} yearly mean [W/$m^2$]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../03Figures/Figure_{predict_param}_sensslope.png',dpi=300)

if __name__ == "__main__":
    main()