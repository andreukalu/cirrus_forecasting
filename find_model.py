#IMPORTS DEFINED IN UTILITIES
from utilities import *

class FindModel:
#### CLASS INITIALIZATION PARAMETERS ###############################################################################################
    def __init__(self, model_type, predict_param, pca_flag = False, weights_flag = False, hyper_parameter_tuning = False, shift_flag = False, cols = []):

        self.model_type = model_type                    # Model archietcture to be fitted
        self.predict_param = predict_param              # Parameter to be forecasted. 
                                                        # 'TOA': Top of the atmosphere
                                                        # 'SFC': Surface
                                                        # 'DIF': Difference between TOA and SFC

        self.cols = cols                                # Specific columns of the dataset to be used. If [], use the whole dataset

        self.datapath = '../02Data'                     # Path of the data to be used for train and validation

        self.pca_flag = pca_flag                        # Flag for principal component analysis (PCA)
        self.pca_params = 30                            # How many parameters to use in the PCA
        self.weights_flag = weights_flag                # Flag for using training weights
        self.find_model = hyper_parameter_tuning        # Flag for hyper-parameter tuning
        self.shift_flag = shift_flag                    # Flag for using moving window feature engineering

        self.model = None                               # ML model to be fitted
        self.hyper_parameter_search_model = None        # Hyper-parameter tuning object

        self.model_name = predict_param + 'model.pkl'   # Filename of the saved model

##### CLASS METHODS ###############################################################################################################

    # LOAD THE DATA 
    def load_data(self, train_dataframe_filename = 'df_merged', val_dataframe_filename = 'df_merged_val'):
        # Load the historical data (from 2003 to 2013) for training
        path = os.path.join(self.datapath,'df_merged')
        df_train = pd.read_pickle(path).dropna(axis=1)

        # Load the historical data (from 2013 to 2023) for validation
        path = os.path.join(self.datapath,'df_merged_val')
        df_val = pd.read_pickle(path).dropna(axis=1)

        return df_train, df_val
    
    # Data filtering and Feature Engineering on the train and validation sets
    def data_filtering_and_feature_engineering(self, df_train, df_val):
        # TRAIN SET:
        df_train = df_train[df_train['COUNT']>100]
        df_train.loc[:,'year'] = df_train.apply(lambda x: x['time'].year + x['time'].month/12, axis=1)
        df_train.loc[:,'month_sin'] = df_train.apply(lambda x: np.sin(x['time'].month*2*np.pi/12), axis=1)
        df_train.loc[:,'month_cos'] = df_train.apply(lambda x: np.cos(x['time'].month*2*np.pi/12), axis=1)

        # VALIDATION SET:
        df_val = df_val[df_val['COUNT']>100]
        df_val.loc[:,'year'] = df_val.apply(lambda x: x['time'].year, axis=1)
        df_val.loc[:,'month_sin'] = df_val.apply(lambda x: np.sin(x['time'].month*2*np.pi/12), axis=1)
        df_val.loc[:,'month_cos'] = df_val.apply(lambda x: np.cos(x['time'].month*2*np.pi/12), axis=1)

        # If past data is to be used, apply window feature engineering
        if self.shift_flag == True:
        # TRAINING SET:
            shifted = df_train.iloc[:, 8:].diff().rolling(window=12).mean()
            shifted.columns = [f"{col}_next" for col in shifted.columns]
            df_train = pd.concat([df_train,shifted],axis=1).dropna()
            shifted = df_train.iloc[:, 8:].rolling(window=12).mean()
            shifted.columns = [f"{col}_next" for col in shifted.columns]
            df_train = pd.concat([df_train,shifted],axis=1).dropna()
            
        # VALIDATION SET:
            shifted = df_val.iloc[:, 8:].diff().rolling(window=12).mean()
            shifted.columns = [f"{col}_next" for col in shifted.columns]
            df_val = pd.concat([df_val,shifted],axis=1).dropna()
            shifted = df_val.iloc[:, 8:].rolling(window=12).mean()
            shifted.columns = [f"{col}_next" for col in shifted.columns]
            df_val = pd.concat([df_val,shifted],axis=1).dropna()

        # Get the target measurement parameter
        # Top of the Atmosphere
        if self.predict_param == 'TOA':  
            y_train = df_train[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)
            y_val = df_val[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)

        # Surface
        elif self.predict_param == 'SFC':
            y_train = df_train[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
            y_val = df_val[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)

        # Difference between TOA and SFC
        elif self.predict_param == 'DIF':
            y_train = df_train[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)-df_train[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
            y_val = df_val[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)-df_val[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
            self.cols = ['rh12', 'rh13', 'rh14', 'ps', 't4', 't9', 't12', 't19', 'albedo']

        # Remove the non-used and target columns
        if len(self.cols)>0:
            X_train = df_train[self.cols]
            X_val = df_val[self.cols]
        else:
            X_train = df_train.iloc[:,8:]
            X_val = df_val.iloc[:,8:]

        # Apply PCA if required
        if self.pca_flag == True:
        # Standarize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Apply PCA
            pca = PCA(n_components='mle')
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            
            with open('pca.pkl', 'wb') as f:
                pickle.dump(pca, f)
                print('SAVED PCA SUCCESSFULLY')

        # Return the data
        return X_train, X_val, y_train, y_val 

    # DEFINE THE FORECASTING MODELS
    def instantiate_model(self): 
        if self.model_type == 'LinearRegression':
            self.model = LinearRegression()
        elif self.model_type == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=333,bootstrap=True,max_depth=100,max_features=1.0,min_samples_leaf=1,min_samples_split=2)
        elif self.model_type == 'GradientBoosting':
            self.model = GradientBoostingRegressor(learning_rate=0.1,loss='huber',max_depth=100,max_features=0.72,min_samples_leaf=1,min_samples_split=12,n_estimators=100)
        elif self.model_type == 'KNN':
            self.model = KNeighborsRegressor(n_neighbors=10)
        elif self.model_type == 'Lasso':
            self.model = Lasso(alpha=0.1)
        elif self.model_type == 'Ridge':
            ridge = Ridge(alpha=1.0)
            self.model = Pipeline([('scaler',StandardScaler()),('ridge',ridge)])
        elif self.model_type == 'Stacked':
            model = LinearRegression()
            rf = RandomForestRegressor(n_estimators=333,bootstrap=True,max_depth=100,max_features=1.0,min_samples_leaf=1,min_samples_split=2)
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

            # Stacking
            self.model = StackingRegressor(
                estimators=[('ridge', ridge),('rf',rf)], 
                final_estimator=model
            )
        elif self.model_type == 'Residual':
            base = Ridge(alpha=1.0)
            residual = RandomForestRegressor()
            residual = GradientBoostingRegressor()
            
            self.model = Pipeline([('scaler',StandardScaler()),('residual_model',ResidualCorrector(base_model=base, residual_model=residual))])

    # HYPERPARAMETER TUNING ON THE DEFINED MODELS
    def define_hyper_parameter_tuning(self):
        if self.model_type == 'RandomForest':
            param_grid = {
                'n_estimators': np.arange(10, 50, 10).tolist(),
                'max_depth': np.arange(30, 40, 5).tolist(),
                'min_samples_split': np.arange(2, 20, 5).tolist(),
                'min_samples_leaf': np.arange(3, 4, 1).tolist(),
                'bootstrap': [True],
                'max_features' : np.arange(0.2, 1.1, 0.1).tolist()
            }
        elif self.model_type == 'GradientBoosting':
            param_grid = {
                'loss': ['huber','squared_error'],
                'learning_rate' : [0.2,0.1,0.05],
                'n_estimators': np.arange(50, 200, 50).tolist(),
                'max_depth': np.arange(5, 50, 10).tolist(),
                'min_samples_split': np.arange(2, 50, 10).tolist(),
                'min_samples_leaf': np.arange(1, 2, 1).tolist(),
                'max_features' : np.arange(0.4, 1.1, 0.2).tolist()
            }
        elif self.model_type == 'KNN':
            param_grid = {
                'n_neighbors': np.arange(5, 100, 10).tolist(),
                'weights' : ['uniform','distance'],
                'algorithm': ['auto','ball_tree'],
                'leaf_size': np.arange(1, 5, 1).tolist()
            }
        elif self.model_type == 'Ridge':
            param_grid = {
                'ridge__alpha': [0.01, 0.1, 0.5, 1, 2, 4, 10,15,20,30,40],
            }
        elif self.model_type == 'Residual':
            param_grid = {
                'residual_model__residual_model__n_estimators': [2,5,10,20,50,100,200],
                'residual_model__residual_model__loss': ['squared_error'],
                'residual_model__residual_model__max_depth': [4,10,20,30,50],
                'residual_model__residual_model__min_samples_split': [2,10,20,30,40],
                'residual_model__residual_model__learning_rate': [0.1,0.05,0.025],
                'residual_model__base_model__alpha': [0.1,1]
            }
        
        cv = TimeSeriesSplit(n_splits=2)
        self.hyper_parameter_search_model = GridSearchCV(self.model, param_grid, cv=cv, verbose = 3, scoring='neg_mean_squared_error', n_jobs=-1)

    # MODEL FITTING OR HYPERPARAMETER TUNING
    def model_fit(self, X_train, y_train, weights= None):
        print(self.find_model)
        # Fit the model
        if self.find_model == False:
            if (self.model_type == 'GradientBoosting') | (self.model_type == 'RandomForest'): 
                self.model = self.model.fit(X_train, y_train, sample_weight = weights)
            else:
                self.model = self.model.fit(X_train, y_train)
        else:
            self.hyper_parameter_search_model = self.hyper_parameter_search_model.fit(X_train, y_train)
        # Print the best parameters and best score
            print("Best parameters:", self.hyper_parameter_search_model.best_params_)
            print("Best cross-validation accuracy:", self.hyper_parameter_search_model.best_score_)
            
        # Store the best-found model
            self.model = self.hyper_parameter_search_model.best_estimator_

    # VALIDATE THE MODEL WITH THE VAL DATASET
    def model_validate(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        y_pred = np.array(y_pred)
        y_val = np.array(y_val)

        print('R2=' + str(round(np.corrcoef(y_pred,y_val)[0,1]**2,2)))
        print('RMSE=' + str(np.sqrt(np.sum((y_pred-y_val)**2)/len(y_pred))))

        return y_pred

    # SAVE THE FITTED MODEL AS PICKLE
    def save_model(self):

        with open(self.model_name,'wb') as f:
            pickle.dump(self.model,f)
            print('model saved')
    
    # SAVE THE FITTED MODEL AS PICKLE
    def load_model(self):

        with open(self.model_name, 'rb') as f:
            self.model = pickle.load(f)
            print('Model loaded')

    # GET THE FEATURE IMPORTANCES
    def get_feature_importances(self):

        if self.model_type == 'RandomForest':
            # Compute importances
            importances = self.model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in reg.estimators_], axis=0)
            print(X_train.columns[importances>0.025])
            
            # Rename columns
            X_train.columns = [self.rename_col(col) for col in X_train.columns]
            X_train = X_train.rename(columns={'PS':'Surface Pressure'})
            # Create a Series and sort it
            forest_importances = pd.Series(importances, index=X_train.columns)
            forest_importances = forest_importances.sort_values(ascending=False)*100

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.grid('on')
            forest_importances.plot.bar(ax=ax, color='skyblue', edgecolor='black')
            ax.set_title("Feature importances")
            ax.set_ylabel("Importance [%]")
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            plt.savefig(f'../03Figures/{self.predict_param}RFparameters.png',dpi=300)

        elif self.model_type == 'Ridge':
            # Compute importances
            ridge = self.model.named_steps['ridge']
            abs_coefs = np.abs(ridge.coef_)
            importances = 100 * abs_coefs / abs_coefs.sum()
            print(X_train.columns[importances>0.025])
            
            # Rename columns
            X_train.columns = [self.rename_col(col) for col in X_train.columns]
            X_train = X_train.rename(columns={'PS':'Surface Pressure'})
            # Create a Series and sort it
            forest_importances = pd.Series(importances, index=X_train.columns)
            forest_importances = forest_importances.sort_values(ascending=False)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.grid('on')
            forest_importances.plot.bar(ax=ax, color='skyblue', edgecolor='black')
            ax.set_title("Feature importances")
            ax.set_ylabel("Importance [%]")
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            plt.savefig(f'../03Figures/{self.predict_param}Ridgeparameters.png',dpi=300)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.grid('on')
            forest_importances[:20].plot.bar(ax=ax, color='skyblue', edgecolor='black')
            ax.set_title("20 most important features")
            ax.set_ylabel("Importance [%]")
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            plt.savefig(f'../03Figures/{self.predict_param}Ridgeparameters20.png',dpi=300)
    
    # RENAME THE DATASET COLUMNS FOR PLOTTING
    def rename_col(self, col):
        
        level = [100000., 92500., 85000., 70000., 60000., 50000., 40000., 30000., 25000., 20000., 
                15000., 10000., 7000., 5000., 3000., 2000., 1000., 500., 100.]
        level = [l / 100 for l in level]  # divide each value by 100

        var_map = {
            'hur': 'RH',         # Relative Humidity
            'ta': 'T',           # Air Temperature
            'ps': 'PS',          # Surface Air Pressure
            'snw': 'SNW',        # Snow Depth
            'clt': 'CLT',        # Total Cloud Cover
            'pr': 'PR',          # Precipitation
            'rsus': 'RSUS',      # Surface Upwelling SW Radiation
            'rsds': 'RSDS',      # Surface Downwelling SW Radiation
            'rlut': 'OLR',       # TOA Outgoing Longwave Radiation
            'rsdt': 'RSDT',      # TOA Incoming Shortwave Radiation
            'rtdiff': 'TOA CRF', # Computed variable (RSDT - RSUS - OLR)
            'year': 'Year',
            'month_sin': 'Month (sin)',
            'month_cos': 'Month (cos)'
        }

        match = re.match(r'([a-zA-Z]+)(\d+)', col)
        if match:
            prefix, idx = match.groups()
            idx = int(idx) - 1
            name = var_map.get(prefix.lower(), prefix.upper())
            
            if 0 <= idx < len(level):
                return f"{name} @ {level[idx]} hPa"
            else:
                return f"{name} #{idx+1}"
        else:
            return var_map.get(col.lower(), col.upper())
