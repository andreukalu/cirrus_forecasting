from utilities import *   
from find_model import FindModel  

def main():

    model_type = 'Residual'
    predict_param = 'TOA'
    hyper_parameter_tuning_flag = True

    find_model = FindModel(model_type = model_type, predict_param = predict_param, hyper_parameter_tuning = hyper_parameter_tuning_flag)

    df_train, df_val = find_model.load_data(train_dataframe_filename = 'df_merged', val_dataframe_filename = 'df_merged_val')

    X_train, X_val, y_train, y_val = find_model.data_filtering_and_feature_engineering(df_train, df_val)

    find_model.instantiate_model()

    if hyper_parameter_tuning_flag == True:
        find_model.define_hyper_parameter_tuning()

    find_model.model_fit(X_train, y_train)

    find_model.model_validate(X_val, y_val)

    find_model.save_model()

if __name__ == "__main__":
    main()