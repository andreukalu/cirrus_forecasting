from utilities import *

predict_param = 'TOA'
shift_flag = False

cols = []
if predict_param == 'TOA':
    model_name = 'TOAmodel.pkl'
    # cols = ['rh10', 'rh12', 'rh15', 'rh18', 'ps', 't10', 't12', 't19', 'month']
elif predict_param == 'SFC':
    model_name = 'SFCmodel.pkl'
    cols = []
elif predict_param == 'DIF':
    model_name = 'DIFmodel.pkl'
    cols = ['rh9', 'rh12', 'rh13', 'rh14', 'ps', 't2', 't4', 't5', 't9', 't12',
       't14', 't15', 't18', 't19', 'rsds', 'albedo', 'month']

#Load the prediction model
with open(model_name, 'rb') as f:
    reg = pickle.load(f)

datapath = '../02Data'
path = os.path.join(datapath,'df_merged_val')
df = pd.read_pickle(path).dropna(axis=1).dropna()
if shift_flag == True:
    shifted = df.iloc[:, 8:].diff().rolling(window=12).mean()
    shifted.columns = [f"{col}_next" for col in shifted.columns]
    df = pd.concat([df,shifted],axis=1).dropna()
    shifted = df.iloc[:, 8:].rolling(window=12).mean()
    shifted.columns = [f"{col}_next" for col in shifted.columns]
    df = pd.concat([df,shifted],axis=1).dropna()
df = df[df['COUNT']>100]


t = df['time']
X = df
X['year'] = df.apply(lambda x: x['time'].year + x['time'].month/12, axis=1)
X['month_sin'] = df.apply(lambda x: np.sin(x['time'].month*2*np.pi/12), axis=1)
X['month_cos'] = df.apply(lambda x: np.cos(x['time'].month*2*np.pi/12), axis=1)

if len(cols)>0:
    X = X[cols]
else:
    X = X.iloc[:,8:]

path = os.path.join(datapath,'df_merged_val')
df = pd.read_pickle(path).dropna(axis=1)

if shift_flag == True:
    shifted = df.iloc[:, 8:].diff().rolling(window=12).mean()
    shifted.columns = [f"{col}_next" for col in shifted.columns]
    df = pd.concat([df,shifted],axis=1).dropna()
    shifted = df.iloc[:, 8:].rolling(window=12).mean()
    shifted.columns = [f"{col}_next" for col in shifted.columns]
    df = pd.concat([df,shifted],axis=1).dropna()
df = df[df['COUNT']>100]

if predict_param == 'TOA':
    yval = df[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)
elif predict_param == 'SFC':
    yval = df[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)#,'SFC30sr8CASR'
elif predict_param == 'DIF':
    yval = df[['TOA20sr8CASR','TOA30sr8CASR']].mean(axis=1)-df[['SFC20sr8CASR','SFC30sr8CASR']].mean(axis=1)#,'SFC30sr8CASR'
ttrain = df['time']
y_pred = reg.predict(X)

fontsize = 20
plt.figure(figsize=(10, 10))
plt.scatter(y_pred,yval.values.ravel(),s = 20,color='k')
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, yval.values.ravel())
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
    plt.text(1.0,-0.5,'R2=' + str(round(np.corrcoef(y_pred,yval.values.ravel())[0,1]**2,2)), fontsize=fontsize)
    plt.text(1.0,-1,'RMSE=' + str(round(np.sqrt(np.sum((y_pred-yval.values.ravel())**2)/len(y_pred)),2)), fontsize=fontsize)
    plt.xlabel('Predicted TOA CRF [W/m$^2$]', fontsize=fontsize)
    plt.ylabel('Measured TOA CRF [W/m$^2$]', fontsize=fontsize)
    plt.legend(lr,['$y=' + str(round(slope,2)) + '\cdot x+' + str(round(intercept,2)) + '$'],loc='upper right', prop={'size': fontsize})
elif predict_param == 'SFC':
    plt.xlim([-8,2])
    plt.ylim([-8,2])
    plt.plot([-8,2],[-8,2],color='k',linestyle='--')
    lrline = np.array([slope*-8 + intercept,slope*2+intercept])
    lr = plt.plot(np.array([-8,2]),lrline,color='r')
    plt.text(-0.5,-0.5,'R2=' + str(round(np.corrcoef(y_pred,yval.values.ravel())[0,1]**2,2)), fontsize=fontsize)
    plt.text(-0.5,-1,'RMSE=' + str(round(np.sqrt(np.sum((y_pred-yval.values.ravel())**2)/len(y_pred)),2)), fontsize=fontsize)
    plt.xlabel('Predicted SFC CRF [W/m$^2$]', fontsize=fontsize)
    plt.ylabel('Measured SFC CRF [W/m$^2$]', fontsize=fontsize)
    plt.legend(lr,['$y=' + str(round(slope,2)) + '\cdot x+' + str(round(intercept,2)) + '$'],loc='upper right', prop={'size': fontsize})
elif predict_param == 'DIF':
    plt.xlim([0,8])
    plt.ylim([0,8])
    plt.plot([0,8],[0,8],color='r',linestyle='--')
    plt.text(0.5,0.5,'R2=' + str(round(np.corrcoef(y_pred,yval.values.ravel())[0,1]**2,2)))
    plt.text(0.5,1,'RMSE=' + str(round(np.sqrt(np.sum((y_pred-yval.values.ravel())**2)/len(y_pred)),2)))
    plt.xlabel('Predicted SFC CRF [W/m$^2$]')
    plt.ylabel('Measured SFC CRF [W/m$^2$]')
    
plt.grid('on')
plt.savefig(f'../03Figures/{predict_param}scatter.png',dpi=300)