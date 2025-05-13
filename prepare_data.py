from utilities import *

#PREPARE DATA FOR THE PAST (TRAIN)
datapath = '../02Data'
path = os.path.join(datapath,'MPL_last.txt')
df = pd.read_csv(path,sep='\t')
df['time'] = pd.to_datetime(dict(year=df['YEAR'], month=df['MONTH'], day=1))

df = df.drop(columns=['MONTH','YEAR','ALB','CTT','CBT','CTH','CBH','SZA'])

path = os.path.join(datapath,'rh.nc')
nc_data = Dataset(path)

rh = nc_data.variables['hur'][:]
plev = nc_data.variables['plev'][:]
time = nc_data.variables['time'][:]

rh = np.squeeze(rh)
columns = [f'rh{i+1}' for i in range(rh.shape[1])]

df_rh = pd.DataFrame(rh,columns=columns)
df_rh['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_rh['time'].to_list()]
df_rh['time'] = t
df_rh['time'] = df_rh['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'temp.nc')
nc_data = Dataset(path)

ta = nc_data.variables['ta'][:]
plev = nc_data.variables['plev'][:]
time = nc_data.variables['time'][:]

ta = np.squeeze(ta)
columns = [f't{i+1}' for i in range(ta.shape[1])]

df_t = pd.DataFrame(ta,columns=columns)
df_t['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_t['time'].to_list()]
df_t['time'] = t
df_t['time'] = df_t['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'sap.nc')
nc_data = Dataset(path)
# print("Dimensions:")
# for dim_name, dim in nc_data.dimensions.items():
#     print(f"{dim_name}: {len(dim)}")

# print("Variables:")
# for var_name in nc_data.variables.keys():
#     print(var_name)

ps = nc_data.variables['ps'][:]
time = nc_data.variables['time'][:]

df_p = pd.DataFrame(np.squeeze(ps),columns=['ps'])
df_p['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_p['time'].to_list()]
df_p['time'] = t
df_p['time'] = df_p['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'ssa.nc')
nc_data = Dataset(path)

snw = nc_data.variables['snw'][:]
time = nc_data.variables['time'][:]

df_snw = pd.DataFrame(np.squeeze(snw[:,1,:]),columns=['snw'])
df_snw['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_snw['time'].to_list()]
df_snw['time'] = t
df_snw['time'] = df_p['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'tcc.nc')
nc_data = Dataset(path)

ttc = nc_data.variables['clt'][:]
time = nc_data.variables['time'][:]

df_ttc = pd.DataFrame(np.squeeze(ttc[:,1,:]),columns=['ttc'])
df_ttc['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_ttc['time'].to_list()]
df_ttc['time'] = t
df_ttc['time'] = df_ttc['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'precip.nc')
nc_data = Dataset(path)

precip = nc_data.variables['pr'][:]
time = nc_data.variables['time'][:]

df_precip = pd.DataFrame(np.squeeze(ttc[:,1,:]),columns=['pr'])
df_precip['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_precip['time'].to_list()]
df_precip['time'] = t
df_precip['time'] = df_precip['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

df_out = df.merge(df_rh,on=['time']).merge(df_p,on=['time']).merge(df_t,on=['time']).merge(df_snw,on=['time']).merge(df_ttc,on=['time']).merge(df_precip,on=['time'])

path = os.path.join(datapath,'df_merged')
df_out.to_pickle(path)

#PREPARE DATA FOR THE FUTURE (TRAIN)
datapath = '../02Data'
path = os.path.join(datapath,'MPL_last.txt')
df = pd.read_csv(path,sep='\t')
df['time'] = pd.to_datetime(dict(year=df['YEAR'], month=df['MONTH'], day=1))

df = df.drop(columns=['MONTH','YEAR','ALB','CTT','CBT','CTH','CBH','SZA','SFC20sr8CASR','SFC30sr8CASR'])

path = os.path.join(datapath,'rh_future.nc')
nc_data = Dataset(path)

rh = nc_data.variables['hur'][:]
plev = nc_data.variables['plev'][:]
time = nc_data.variables['time'][:]

rh = np.squeeze(rh)
columns = [f'rh{i+1}' for i in range(rh.shape[1])]

df_rh = pd.DataFrame(rh,columns=columns)
df_rh['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_rh['time'].to_list()]
df_rh['time'] = t
df_rh['time'] = df_rh['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'temp_future.nc')
nc_data = Dataset(path)

ta = nc_data.variables['ta'][:]
plev = nc_data.variables['plev'][:]
time = nc_data.variables['time'][:]

ta = np.squeeze(ta)
columns = [f't{i+1}' for i in range(ta.shape[1])]

df_t = pd.DataFrame(ta,columns=columns)
df_t['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_t['time'].to_list()]
df_t['time'] = t
df_t['time'] = df_t['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'sap_future (2).nc')
nc_data = Dataset(path)
# print("Dimensions:")
# for dim_name, dim in nc_data.dimensions.items():
#     print(f"{dim_name}: {len(dim)}")

# print("Variables:")
# for var_name in nc_data.variables.keys():
#     print(var_name)

ps = nc_data.variables['ps'][:]
time = nc_data.variables['time'][:]

df_p = pd.DataFrame(np.squeeze(ps),columns=['ps'])
df_p['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_p['time'].to_list()]
df_p['time'] = t
df_p['time'] = df_p['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'ssa_future.nc')
nc_data = Dataset(path)

snw = nc_data.variables['snw'][:]
time = nc_data.variables['time'][:]

df_snw = pd.DataFrame(np.squeeze(snw[:]),columns=['snw'])
df_snw['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_snw['time'].to_list()]
df_snw['time'] = t
df_snw['time'] = df_p['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'ttc_future.nc')
nc_data = Dataset(path)

ttc = nc_data.variables['clt'][:]
time = nc_data.variables['time'][:]

df_ttc = pd.DataFrame(np.squeeze(ttc[:]),columns=['ttc'])
df_ttc['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_ttc['time'].to_list()]
df_ttc['time'] = t
df_ttc['time'] = df_ttc['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

path = os.path.join(datapath,'prec_future.nc')
nc_data = Dataset(path)

precip = nc_data.variables['pr'][:]
time = nc_data.variables['time'][:]

df_precip = pd.DataFrame(np.squeeze(ttc[:]),columns=['pr'])
df_precip['time'] = time
t = [timedelta(days=time)+datetime(1,1,1,0,0,0) for time in df_precip['time'].to_list()]
df_precip['time'] = t
df_precip['time'] = df_precip['time'].apply(lambda x: x.replace(day=1).replace(hour=0))

df_out = df.merge(df_rh,on=['time']).merge(df_p,on=['time']).merge(df_t,on=['time']).merge(df_snw,on=['time']).merge(df_ttc,on=['time']).merge(df_precip,on=['time'])

path = os.path.join(datapath,'df_merged_val')
df_out.to_pickle(path)