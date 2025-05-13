from utilities import *   

def main():

    model_type = 'Residual'
    predict_param = 'TOA'
    hyper_parameter_tuning_flag = True

    # LOAD AND PRE-PROCESS THE DATASET
    data_processor = DataProcessor(predict_param = predict_param)

    df_train = data_processor.load_data(file_path = 'df_merged')
    df_val = data_processor.load_data(file_path = 'df_merged_val')

    X_train, y_train = data_processor.data_filtering_and_feature_engineering(df_train)
    X_val, y_val = data_processor.data_filtering_and_feature_engineering(df_val)

    # FIND/FIT/VALIDATE THE ML MODEL
    model = ForecastModel(predict_param = predict_param, model_type = model_type, hyper_parameter_tuning = hyper_parameter_tuning_flag)

    model.fit(X_train, y_train)

    model.validate(X_val, y_val)

    model.save()

if __name__ == "__main__":
    main()