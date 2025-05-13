from utilities import *

################### Params and flag definition ############################################################################

pca_flag = False
pca_params = 30
weights_flag = False
find_model = True
shift_flag = False
model = 'Residual'

predict_param = 'TOA'

cols = []

datapath = '../02Data'

# Load the historical data (from 2003 to 2013) for training
path = os.path.join(datapath,'df_merged')
df_train = pd.read_pickle(path).dropna(axis=1)

# Load the historical data (from 2013 to 2023) for validation
path = os.path.join(datapath,'df_merged_val')
df_val = pd.read_pickle(path).dropna(axis=1)

# Data filtering and Feature Engineering on the train and validation sets
# TRAIN SET:
df_train = df_train[df_train['COUNT']>100]
df_train['year'] = df_train.apply(lambda x: x['time'].year + x['time'].month/12, axis=1)
df_train['month_sin'] = df_train.apply(lambda x: np.sin(x['time'].month*2*np.pi/12), axis=1)
df_train['month_cos'] = df_train.apply(lambda x: np.cos(x['time'].month*2*np.pi/12), axis=1)

# VALIDATION SET:
df_val = df_val[df_val['COUNT']>100]
df_val['year'] = df_val.apply(lambda x: x['time'].year, axis=1)
df_val['month_sin'] = df_val.apply(lambda x: np.sin(x['time'].month*2*np.pi/12), axis=1)
df_val['month_cos'] = df_val.apply(lambda x: np.cos(x['time'].month*2*np.pi/12), axis=1)

# If past data is to be used, apply window feature engineering
if shift_flag == True:
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
if predict_param == 'TOA':
    ytrain = df_train[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)
    yval = df_val[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)
elif predict_param == 'SFC':
    ytrain = df_train[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
    yval = df_val[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
elif predict_param == 'DIF':
    ytrain = df_train[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)-df_train[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
    yval = df_val[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)-df_val[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
    cols = ['rh12', 'rh13', 'rh14', 'ps', 't4', 't9', 't12', 't19', 'albedo']

# Remove the non-used and target columns
if len(cols)>0:
    Xtrain = df_train[cols]
    Xval = df_val[cols]
else:
    Xtrain = df_train.iloc[:,8:]
    Xval = df_val.iloc[:,8:]

# Get the training weights if required
if weights_flag == True:
    weights = df_train['COUNT']/df_train['COUNT'].max()
else:
    weights = None

# Apply PCA if required
if pca_flag == True:
# Standarize the data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xval = scaler.transform(Xval)

# Apply PCA
    pca = PCA(n_components='mle')
    Xtrain = pca.fit_transform(Xtrain)
    Xval = pca.transform(Xval)
    
    with open('pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
        print('SAVED PCA SUCCESSFULLY')

######################## DEFINE THE FORECASTING MODELS ##########################
if model == 'LinearRegression':
    reg = LinearRegression()
elif model == 'RandomForest':
    reg = RandomForestRegressor(n_estimators=333,bootstrap=True,max_depth=100,max_features=1.0,min_samples_leaf=1,min_samples_split=2)
elif model == 'GradientBoosting':
    reg = GradientBoostingRegressor(learning_rate=0.1,loss='huber',max_depth=100,max_features=0.72,min_samples_leaf=1,min_samples_split=12,n_estimators=100)
elif model == 'KNN':
    reg = KNeighborsRegressor(n_neighbors=10)
elif model == 'Lasso':
    reg = Lasso(alpha=0.1)
elif model == 'Ridge':
    ridge = Ridge(alpha=1.0)
    reg = Pipeline([('scaler',StandardScaler()),('ridge',ridge)])
elif model == 'Stacked':
    reg = LinearRegression()
    rf = RandomForestRegressor(n_estimators=333,bootstrap=True,max_depth=100,max_features=1.0,min_samples_leaf=1,min_samples_split=2)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    # Stacking
    reg = StackingRegressor(
        estimators=[('ridge', ridge),('rf',rf)], 
        final_estimator=reg
    )
elif model == 'Residual':
    base = Ridge(alpha=1.0)
    residual = RandomForestRegressor()
    residual = GradientBoostingRegressor()
    
    reg = Pipeline([('scaler',StandardScaler()),('residual_model',ResidualCorrector(base_model=base, residual_model=residual))])

########################### HYPERPARAMETER TUNING ON THE DEFINED MODELS #####################################################   
if find_model==True:
    if model == 'RandomForest': #TOA: {'bootstrap': True, 'max_depth': 21, 'max_features': 0.5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 33}
        param_grid = {
            'n_estimators': np.arange(10, 50, 10).tolist(),
            'max_depth': np.arange(30, 40, 5).tolist(),
            'min_samples_split': np.arange(2, 20, 5).tolist(),
            'min_samples_leaf': np.arange(3, 4, 1).tolist(),
            'bootstrap': [True],
            'max_features' : np.arange(0.2, 1.1, 0.1).tolist()
        }

        # Set up RandomizedSearchCV
        reg = GridSearchCV(
            reg, param_grid=param_grid, cv=10, verbose = 3, scoring='neg_mean_squared_error', n_jobs=-1
        )
    elif model == 'GradientBoosting': #TOA: {'learning_rate': 0.1, 'loss': 'huber', 'max_depth': 30, 'max_features': 0.72, 'min_samples_leaf': 1, 'min_samples_split': 12, 'n_estimators': 40}
                                        # SFC: {'learning_rate': 0.2, 'loss': 'huber', 'max_depth': 20, 'max_features': 0.1, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 10}
        param_grid = {
            'loss': ['huber','squared_error'],
            'learning_rate' : [0.2,0.1,0.05],
            'n_estimators': np.arange(50, 200, 50).tolist(),
            'max_depth': np.arange(5, 50, 10).tolist(),
            'min_samples_split': np.arange(2, 50, 10).tolist(),
            'min_samples_leaf': np.arange(1, 2, 1).tolist(),
            'max_features' : np.arange(0.4, 1.1, 0.2).tolist()
        }

        # Set up RandomizedSearchCV
        reg = GridSearchCV(
            reg, param_grid=param_grid, cv=10, verbose = 3, scoring='neg_mean_squared_error', n_jobs=-1
        )
    elif model == 'KNN': #{'algorithm': 'ball_tree', 'leaf_size': 5, 'n_neighbors': 45, 'weights': 'distance'}
        param_grid = {
            'n_neighbors': np.arange(5, 100, 10).tolist(),
            'weights' : ['uniform','distance'],
            'algorithm': ['auto','ball_tree'],
            'leaf_size': np.arange(1, 5, 1).tolist()
        }

        # Set up RandomizedSearchCV
        reg = GridSearchCV(
            reg, param_grid=param_grid, cv=10, verbose = 3, scoring='r2', n_jobs=-1
        )
    elif model == 'Ridge': #{'algorithm': 'ball_tree', 'leaf_size': 5, 'n_neighbors': 45, 'weights': 'distance'}
        param_grid = {
            'ridge__alpha': [0.01, 0.1, 0.5, 1, 2, 4, 10,15,20,30,40],  # Regularization strength
        }

        # Set up RandomizedSearchCV
        cv = TimeSeriesSplit(n_splits=2)  # each split adds more data
        reg = GridSearchCV(
            reg, param_grid=param_grid, cv=cv, verbose = 3, scoring='max_error', n_jobs=-1
        )
    elif model == 'Residual':
        param_grid = {
            'residual_model__residual_model__n_estimators': [2,5,10,20,50,100,200],
            'residual_model__residual_model__loss': ['squared_error'],
            'residual_model__residual_model__max_depth': [4,10,20,30,50],
            'residual_model__residual_model__min_samples_split': [2,10,20,30,40],
            'residual_model__residual_model__learning_rate': [0.1,0.05,0.025],
            'residual_model__base_model__alpha': [0.1,1]
        }
        cv = TimeSeriesSplit(n_splits=2)  # each split adds more data
        reg = GridSearchCV(reg, param_grid, cv=cv, verbose = 3, scoring='neg_mean_squared_error', n_jobs=-1)


if (model == 'GradientBoosting') | (model == 'RandomForest'): 
    reg = reg.fit(Xtrain, ytrain, sample_weight = weights)
    y_pred = reg.predict(Xval)
    y_pred = np.array(y_pred)
    yval = np.array(yval)
else:
    reg = reg.fit(Xtrain, ytrain)
    y_pred = reg.predict(Xval)
    y_pred = np.array(y_pred)
    yval = np.array(yval)

if find_model==True:
    # Print the best parameters and best score
    print("Best parameters:", reg.best_params_)
    print("Best cross-validation accuracy:", reg.best_score_)
    reg = reg.best_estimator_

yval = np.array(yval)
y_pred = np.array(y_pred)
print('R2=' + str(round(np.corrcoef(y_pred,yval)[0,1]**2,2)))
print('RMSE=' + str(np.sqrt(np.sum((y_pred-yval)**2)/len(y_pred))))


# SAVE THE FITTED MODEL
if predict_param == 'TOA':
    model_name = 'TOAmodel.pkl'
elif predict_param == 'SFC':
    model_name = 'SFCmodel.pkl'
elif predict_param == 'DIF':
    model_name = 'DIFmodel.pkl'

with open(model_name,'wb') as f:
    pickle.dump(reg,f)
    print('model saved')

###################### GET THE FEATURE IMPORTANCES ###################################

level = [100000., 92500., 85000., 70000., 60000., 50000., 40000., 30000., 25000., 20000., 
         15000., 10000., 7000., 5000., 3000., 2000., 1000., 500., 100.]
level = [l / 100 for l in level]  # divide each value by 100

# Function to rename based on pattern
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

def rename_col(col):
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

if model == 'RandomForest':
    # Compute importances
    importances = reg.feature_importances_
    std = np.std([tree.feature_importances_ for tree in reg.estimators_], axis=0)
    print(Xtrain.columns[importances>0.025])
    
    # Rename columns
    Xtrain.columns = [rename_col(col) for col in Xtrain.columns]
    Xtrain = Xtrain.rename(columns={'PS':'Surface Pressure'})
    # Create a Series and sort it
    forest_importances = pd.Series(importances, index=Xtrain.columns)
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
    plt.savefig(f'../03Figures/{predict_param}RFparameters.png',dpi=300)

elif model == 'Ridge':
    # Compute importances
    ridge = reg.named_steps['ridge']
    abs_coefs = np.abs(ridge.coef_)
    importances = 100 * abs_coefs / abs_coefs.sum()
    print(Xtrain.columns[importances>0.025])
    
    # Rename columns
    Xtrain.columns = [rename_col(col) for col in Xtrain.columns]
    Xtrain = Xtrain.rename(columns={'PS':'Surface Pressure'})
    # Create a Series and sort it
    forest_importances = pd.Series(importances, index=Xtrain.columns)
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
    plt.savefig(f'../03Figures/{predict_param}Ridgeparameters.png',dpi=300)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid('on')
    forest_importances[:20].plot.bar(ax=ax, color='skyblue', edgecolor='black')
    ax.set_title("20 most important features")
    ax.set_ylabel("Importance [%]")
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'../03Figures/{predict_param}Ridgeparameters20.png',dpi=300)
    