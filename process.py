import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

df = pd.read_csv('Math.csv')
df = df[df['Level'] == 'HL']

df = df.sort_values('Session')

sessions = df['Session'].tolist()
all_sessions = sessions + ['M25'] 


grades = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7']
for grade in grades:
    df[grade + ' Low'] = df[grade].str.split('-').str[0].astype(int)

plt.figure(figsize=(10,6))

for grade in grades:
    y_actual = df[grade + ' Low'].values

   
    model = ExponentialSmoothing(y_actual, trend='add', seasonal=None, initialization_method="estimated")
    model_fit = model.fit()

    n_forecast = len(all_sessions) - len(y_actual)
    y_pred = model_fit.forecast(steps=n_forecast)
    y_full = np.concatenate([y_actual, y_pred])

    plt.plot(sessions, y_actual, marker='o', label=f'{grade} Actual')

    plt.plot(all_sessions, y_full, linestyle='--', label=f'{grade} ExpSmooth')

plt.xlabel('Session')
plt.ylabel('Lower Boundary')
plt.title('IB Math AA HL Grade Boundaries and Exponential Smoothing (to M25)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
