from utilities import *

datapath = '../02Data'

# path = os.path.join(datapath,'df_merged_val')
# df_val = pd.read_pickle(path).dropna(axis=1)
path = os.path.join(datapath,'df_merged')
df = pd.read_pickle(path).dropna(axis=1)
path = os.path.join(datapath,'df_merged_future_future')
df_future = pd.read_pickle(path).dropna(axis=1)

# df = pd.concat([df_train,df_val])
print(df.columns)
cols = ['rh10', 'rh12', 'rh15', 'rh18', 'ps', 't10', 't12', 't19', 'month', 'albedo']
for col in cols:
    t = df['time']
    variable = df[col]
    df_aux = pd.DataFrame(data={'time':t,'pred':variable})
    df_aux['year'] = df_aux['time'].dt.year
    # Aggregate data by year (e.g., using mean of monthly values)
    yearly_data = df_aux.groupby('year')['pred'].mean()
    years_filtered = yearly_data.index[1:-1]  # Exclude first and last year
    yearly_data = yearly_data.loc[years_filtered]

    # Applying Sen's slope to the yearly data
    years = yearly_data.index.values
    values = yearly_data.values

    trend_slope = sens_slope(years, values)
    print(f"Sen's slope: {trend_slope}")

    # Plotting the data and the trend line (using Sen's slope)
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_data.index, yearly_data, label='Yearly Mean', marker='o')

    # Plot the Sen's slope line
    start_year = yearly_data.index[0]
    end_year = yearly_data.index[-1]
    plt.plot([start_year, end_year], [yearly_data[start_year+1] + trend_slope * (start_year - start_year),
                                    yearly_data[start_year+1] + trend_slope * (end_year - start_year)], 
            label=f'Sen\'s Slope: {np.round(trend_slope,4)} W/m$^2$/year', color='red', linestyle='--')
    
    t = df_future['time']
    variable = df_future[col]
    df_aux = pd.DataFrame(data={'time':t,'pred':variable})
    df_aux['year'] = df_aux['time'].dt.year
    # Aggregate data by year (e.g., using mean of monthly values)
    yearly_data = df_aux.groupby('year')['pred'].mean()
    years_filtered = yearly_data.index[1:-1]  # Exclude first and last year
    yearly_data = yearly_data.loc[years_filtered]

    # Applying Sen's slope to the yearly data
    years = yearly_data.index.values
    values = yearly_data.values

    trend_slope = sens_slope(years, values)
    print(f"Sen's slope: {trend_slope}")

    # Plotting the data and the trend line (using Sen's slope)
    plt.plot(yearly_data.index, yearly_data, label='Yearly Mean', marker='o')

    # Plot the Sen's slope line
    start_year = yearly_data.index[0]
    end_year = yearly_data.index[-1]
    plt.plot([start_year, end_year], [yearly_data[start_year+1] + trend_slope * (start_year - start_year),
                                    yearly_data[start_year+1] + trend_slope * (end_year - start_year)], 
            label=f'Sen\'s Slope: {np.round(trend_slope,4)} W/m$^2$/year', color='red', linestyle='--')
    plt.grid('on')

    plt.title('Yearly Trend of Time Series with Sen\'s Slope')
    plt.xlabel('Year')
    plt.ylabel(col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../03Figures/{col}.png',dpi=300)