from utilities import *

class DataProcessor:
    def __init__(self,predict_param = 'TOA', cols = [], pca_flag = False, pca_params = 30, shift_flag = False, future_data = False, remove_small_samples = True):
        
        self.datapath = '../02Data'                     # Path of the data to be used for train and validation
        self.predict_param = predict_param              # Parameter to be forecasted. 
                                                        # 'TOA': Top of the atmosphere
                                                        # 'SFC': Surface
                                                        # 'DIF': Difference between TOA and SFC

        self.remove_small_samples = remove_small_samples# Flag to remove months with less than 100 MPL records
        self.future_data = future_data                  # Flag to specify if there is only CMIP6 data
        self.cols = cols                                # Specific columns of the dataset to be used. If [], use the whole dataset

        self.pca_flag = pca_flag                        # Flag for principal component analysis (PCA)
        self.pca_params = pca_params                    # How many parameters to use in the PCA
        self.shift_flag = shift_flag                    # Flag for using moving window feature engineering

    # LOAD THE DATA 
    def load_data(self, file_path = 'df_merged'):

        # Load data
        path = os.path.join(self.datapath,file_path)
        df = pd.read_pickle(path).dropna(axis=1)

        print(f'{file_path} Dataset loaded')

        return df
    
    # Data filtering and Feature Engineering on the train and validation sets
    def data_filtering_and_feature_engineering(self, df):
        
        if self.future_data == False and self.remove_small_samples == True:
            df = df[df['COUNT']>100]
        df.loc[:,'year'] = df.apply(lambda x: x['time'].year + x['time'].month/12, axis=1)
        df.loc[:,'month_sin'] = df.apply(lambda x: np.sin(x['time'].month*2*np.pi/12), axis=1)
        df.loc[:,'month_cos'] = df.apply(lambda x: np.cos(x['time'].month*2*np.pi/12), axis=1)

        # If past data is to be used, apply window feature engineering
        if self.shift_flag == True:
        # TRAINING SET:
            shifted = df.iloc[:, 8:].diff().rolling(window=12).mean()
            shifted.columns = [f"{col}_next" for col in shifted.columns]
            df = pd.concat([df,shifted],axis=1).dropna()
            shifted = df.iloc[:, 8:].rolling(window=12).mean()
            shifted.columns = [f"{col}_next" for col in shifted.columns]
            df = pd.concat([df,shifted],axis=1).dropna()

        # Get the target measurement parameter
        if self.future_data == False:
            # Top of the Atmosphere
            if self.predict_param == 'TOA':  
                y = df[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)
            # Surface
            elif self.predict_param == 'SFC':
                y = df[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
            # Difference between TOA and SFC
            elif self.predict_param == 'DIF':
                y = df[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)-df[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)
                self.cols = ['rh12', 'rh13', 'rh14', 'ps', 't4', 't9', 't12', 't19', 'albedo']
        else:
            y = None

        # Remove the non-used and target columns
        if len(self.cols)>0:
            X = df[self.cols]
        else:
            if self.future_data == True:
                X = df.drop(columns=['time'])
            else:
                X = df.iloc[:,8:]

        # Apply PCA if required
        if self.pca_flag == True:
        # Standarize the data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Apply PCA
            pca = PCA(n_components='mle')
            X = pca.fit_transform(X)
            
            with open('pca.pkl', 'wb') as f:
                pickle.dump(pca, f)
                print('SAVED PCA SUCCESSFULLY')
        
        time = df['time']

        print('Datasets processed')
        # Return the data
        return X, y, time
    
    # Function to calculate Sen's slope
    def sens_slope(self, x, y, time_target='year'):
        
        n = len(x)
        slopes = []
        
        times_dict = {'day': 60 * 60 * 24,'month': 60 * 60 * 24 * 30.5,'year': 60 * 60 * 24 * 365.25}
        
        time_factor = times_dict[time_target]

        # Convert datetime objects to numeric values (e.g., days since the first date)
        x_numeric = (x - x.min()).dt.total_seconds()  # Convert to the number of seconds since the first date
        
        # Ensure that x and y are numpy arrays to allow proper indexing
        x = np.array(x_numeric)  # x should be a numeric array (days)
        y = np.array(y)  # y should be a numeric array (predictions)

        for i in range(n):
            for j in range(i + 1, n):
                # Calculate slope between pairs of points
                slope = (y[j] - y[i]) / (x[j] - x[i])  # x_numeric is now numeric values (days)
                slopes.append(slope)

        median_slope = np.median(slopes) # slope in units per second

        # Compute intercepts using the median slope
        intercepts = y - median_slope * x
        median_intercept = np.median(intercepts)
        
        #Convert slope and itnercept to target time unit
        median_slope *= time_factor
        
        return median_slope, median_intercept