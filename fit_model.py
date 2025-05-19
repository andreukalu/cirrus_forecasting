from utilities import *   
from forecast_model import ForecastModel

model_type = 'Residual_GB'
predict_param = 'TOA'
hyper_parameter_tuning_flag = True

def main():

    # LOAD AND PRE-PROCESS THE DATASET
    data_processor = DataProcessor(predict_param = predict_param)

    df_train = data_processor.load_data(file_path = 'df_merged')

    X_train, y_train, _ = data_processor.data_filtering_and_feature_engineering(df_train)

    # FIND/FIT THE ML MODEL
    model = ForecastModel(predict_param = predict_param,model_type = model_type,\
                        hyper_parameter_tuning = hyper_parameter_tuning_flag,\
                        loss='combined_sens')

    model.fit(X_train, y_train)
    
    # SAVE THE ML MODEL
    model.save()

    for experiment in experiments:

        df_val = data_processor.load_data(file_path = f'df_merged_val{experiment}')

        X_val, y_val, _ = data_processor.data_filtering_and_feature_engineering(df_val)

        X_val = X_val[X_train.columns]

        # VALIDATE THE ML MODEL
        model.validate(X_val, y_val)

if __name__ == "__main__":
    main()