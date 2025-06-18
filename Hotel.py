import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")


df = pd.read_excel("Consumption Dataset.xlsx")
df['Date Time Served'] = pd.to_datetime(df['Date Time Served'])
df['Week'] = df['Date Time Served'].dt.to_period('W').apply(lambda r: r.start_time)


bar_brand_combos = df.groupby(['Bar Name', 'Brand Name']).size().reset_index()[['Bar Name', 'Brand Name']]


forecast_results = []
Z = 1.65  

for _, row in bar_brand_combos.iterrows():
    bar = row['Bar Name']
    brand = row['Brand Name']

  
    ts = (
        df[(df['Bar Name'] == bar) & (df['Brand Name'] == brand)]
        .groupby('Week')['Consumed (ml)']
        .sum()
        .sort_index()
    )

   
    if (ts > 0).sum() < 5:
        continue


    ts = ts.asfreq('W').replace(0, np.nan).interpolate(method='linear').fillna(method='bfill')

    try:
        
        model = ARIMA(ts, order=(1, 0, 0))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=1).summary_frame()

        forecast_mean = forecast['mean'].iloc[0]
        last_8_std = ts[-8:].std()
        safety_stock = Z * last_8_std
        par_level = round(forecast_mean + safety_stock)

        forecast_results.append({
            'Bar Name': bar,
            'Brand Name': brand,
            'Forecast Week': ts.index[-1] + pd.Timedelta(weeks=1),
            'Forecasted Demand': round(forecast_mean, 2),
            'Safety Stock': round(safety_stock, 2),
            'Par Level (Recommended)': par_level
        })

    except Exception as e:
        continue

forecast_df = pd.DataFrame(forecast_results)
forecast_df.to_excel("Weekly_Inventory_Forecast.xlsx", index=False)
print("Forecast completed and saved to Weekly_Inventory_Forecast.xlsx")
