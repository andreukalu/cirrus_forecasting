from utilities import *

################### Params and flag definition ############################################################################

predict_param = 'TOA'
temporal_res = 'year'
shift_flag = False
pca_flag = False

cols = []

################## Load the future CMIP data (from 2013 to 2049) and forecast the target variable #################################

#Load the prediction model pre-trained
if predict_param == 'TOA':
    model_name = 'TOAmodel.pkl'
elif predict_param == 'SFC':
    model_name = 'SFCmodel.pkl'
    cols = []
elif predict_param == 'DIF':
    model_name = 'DIFmodel.pkl'
    cols = []

with open(model_name, 'rb') as f:
    reg = pickle.load(f)

#Define the data folder
datapath = '../02Data'

#Load the CMIP6 dataset, from 2013 to 2049
path = os.path.join(datapath,'df_merged_future_future')
df_future = pd.read_pickle(path).dropna(axis=1)    

#Prepare and filter the data to adequate to feature engineering unsed in the training
df_future['year'] = df_future.apply(lambda x: x['time'].year + x['time'].month/12, axis=1)
df_future['month_sin'] = df_future.apply(lambda x: np.sin(x['time'].month*2*np.pi/12), axis=1)
df_future['month_cos'] = df_future.apply(lambda x: np.cos(x['time'].month*2*np.pi/12), axis=1)

if shift_flag == True:
    t_future = df_future.iloc[12:-11]['time']
    shifted = df_future.diff().rolling(window=12).mean()
    shifted.columns = [f"{col}_next" for col in shifted.columns]
    df_future = pd.concat([df_future,shifted],axis=1).dropna()
    shifted = df_future.rolling(window=12).mean()
    shifted.columns = [f"{col}_next" for col in shifted.columns]
    df_future = pd.concat([df_future,shifted],axis=1).dropna()
else:
    t_future = df_future['time']
    df_future = df_future.drop(columns=['time'])

X_future = df_future.copy()

if len(cols)>0:
    X_future = X_future[cols]

#Apply PCA if required
if pca_flag == True:
    with open('pca.pkl','rb') as f:
        pca = pickle.load(f)
        
    scaler = StandardScaler()
    X_future = scaler.fit_transform(X_future)

    X_future = pca.fit_transform(X_future)

#Forecast the target variable
y_future = reg.predict(X_future)

################## Load the measured/historical data (from 2003 to 2023) #################################

#Load the training and validation dataset, CMIP and MPL from 2003 to 2023 and concatenate them:
path = os.path.join(datapath,'df_merged')
df_train = pd.read_pickle(path).dropna(axis=1)
path = os.path.join(datapath,'df_merged_val')
df_val = pd.read_pickle(path).dropna(axis=1)
df_measured = pd.concat([df_train,df_val])

#Get the target measurement parameter
if predict_param == 'TOA':
    y_measured = df_measured[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)
elif predict_param == 'SFC':
    y_measured = df_measured[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
elif predict_param == 'DIF':
    y_measured = df_measured[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)-df_measured[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)

#Get the time vector
t_measured = df_measured['time']

##################### PLOT AND COMPARE THE MEASURED AND FORECASTED TARGET VARIABLE ################

#Plot the forecasted and measured variables
p2 = plt.plot(t_measured,y_measured,color='#A2142F',label='Measured')
p1 = plt.plot(t_future,y_future,color='k',label='Predicted')
plt.grid('on')
plt.ylabel(f'TOA {predict_param} [$W/m^2$]')
plt.legend()
plt.savefig(f'../03Figures/{predict_param}temporal.png',dpi=300)
plt.show()

#######################COMPUTE AND PLOT SEN'S SLOPE TREND####################################

plt.figure(figsize=(10, 6))

#Compute the measured data Sen's slope
trend_slope, trend_intercept = sens_slope(t_measured, y_measured, temporal_res)
print(f"Sen's slope measured: {trend_slope}")

# Plotting the measured data
plt.plot(t_measured, y_measured, color=[0.5,0.5,0.5])

# Plot the Sen's slope line of the measured data
start_year = t_measured.iloc[0]
end_year = t_measured.iloc[-1]
plt.plot([start_year, end_year], [trend_intercept + trend_slope * (start_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12),
                                 trend_intercept + trend_slope * (end_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12)], 
         label=f'Measured Sen\'s Slope: {np.round(trend_slope,10)} W/m$^2$/{temporal_res}', color='k', linestyle='--', linewidth=2)

#Compute the forecasted data Sen's slope
trend_slope, trend_intercept = sens_slope(t_future, y_future, temporal_res)
print(f"Sen's slope forecasted: {trend_slope}")

# Plotting the forecasted data
plt.plot(t_future, y_future, color='#d62041')

# Plot the Sen's slope line of the forecasted data
start_year = t_future.iloc[0]
end_year = t_future.iloc[-1]
plt.plot([start_year, end_year], [trend_intercept + trend_slope * (start_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12),
                                 trend_intercept + trend_slope * (end_year - start_year).total_seconds() / (60 * 60 * 24 * 30 * 12)], 
         label=f'Forecasted Sen\'s Slope: {np.round(trend_slope,10)} W/m$^2$/{temporal_res}', color='#ad001f', linestyle='--', linewidth=2)

plt.grid('on')
plt.title('Trend of Time Series with Sen\'s Slope')
plt.xlabel('Year')
plt.ylabel(f'TOA {predict_param} yearly mean [W/$m^2$]')
plt.legend()
plt.tight_layout()
plt.savefig(f'../03Figures/{predict_param}sensslope.png',dpi=300)