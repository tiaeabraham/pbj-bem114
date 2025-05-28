# NatGas#


# Import necessary libraries
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from keras.optimizers import Adam  # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import load_model  # type: ignore
import numpy as np
import pandas as pd
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense, Dropout  # type: ignore
import seaborn as sns

# Boolean to decide whether to train the model
train = False

# Filepath for storing the model
model_filepath = 'model/best_gasoline_model.h5'

# Load the dataset
data = pd.read_csv('model/combined_gasoline_data.csv')
# data.head())

# Separate dates for future plotting
train_dates = pd.to_datetime(data['Date'], format='%m/%d/%y')
# print(data['Date'].tail(15))  # Print last few dates

# data.isnull().values.any() # Check for NaN values
data_interpolated = data.interpolate()  # Interpolate to fill NaN values
# data_interpolated.isnull().values.any() # Check for NaN values after interpolation
data = data_interpolated  # Replace the original DataFrame with the interpolated data

# Select variables used for training (everything except the first column, which is Date)
cols = list(data)[1:]
# print(cols)

# Create new dataframe with only training data
df_for_training = data[cols].astype(float)

# LSTM models use sigmoid and tang which are sensitive to magnitude - we need to normalize the data
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# Reshape input data inot n_samples x timesteps x n_features as required for LSTM network

# Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1  # number of days we want to look into the future based on the past days
n_past = 14  # number of past days we want to use to predict the future

# Reformat input data into shape (n_samples x timesteps x n_features)
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(
        df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

# Use the Spread column as a feature along with other features
# Use Date_ordinal and other columns as features, including Spread
features = data.drop(columns=['Date']).values
target = data['Spread'].values.reshape(-1, 1)  # 'Spread' is still the target

trainX, trainY = np.array(trainX), np.array(trainY)

print(f'trainX shape == {trainX.shape}')
print(f'trainY shape == {trainY.shape}')


# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target)

'''
trainX shape == (1019, 14, 275)
trainY shape == (1019, 1)

1019 becuase we are looking back 14 days (1033 - 14 = 1019)
We cannot look back 14 days until we get to the 15th day

trainY has a shape of (1019, 1). This model only predicts a single value,
but it needs all feature variables to make the prediction of the single variable.
This is why we can only predict a single day after our training, the day after where
our data ends. To predict more days in the future, we need all the feature variables
which we do not yet have We need to predict all variables if we want to do that.
'''

# Define the autoenoder model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(
    trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=5, batch_size=16,
                    validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

# Predicting...
# Libraries that will help us extract only business days in the US.
# Otherwise our dates would be wrong when we look back (or forward).
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
# Remember that we can only predict one day in future as our model needs 5 variables
# as inputs for prediction. We only have all 5 variables until the last day in our dataset.
n_past = 16
n_days_for_prediction = 15  # let us predict past 15 days

predict_period_dates = pd.date_range(
    list(train_dates)[-n_past], periods=n_days_for_prediction, freq=us_bd).tolist()
print(predict_period_dates)

# Make prediction
# shape = (n, 1) where n is the n_days_for_prediction
prediction = model.predict(trainX[-n_days_for_prediction:])

# Perform inverse transformation to rescale back to original range
# Since we used 5 variables for transform, the inverse expects same dimensions
# Therefore, let us copy our values 5 times and discard them after inverse transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]

# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame(
    {'Date': np.array(forecast_dates), 'Spread': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])


original = data[['Date', 'Spread']]
# original['Date'] = pd.to_datetime(original['Date'])
original.loc[:, 'Date'] = pd.to_datetime(original['Date'], format='%m/%d/%y')


# original = original.loc[original['Date'] >= '2020-5-1'] # limits the dates for visualization purposes

# sns.lineplot(original['Date'], original['Spread'])
# sns.lineplot(df_forecast['Date'], df_forecast['Spread'])
# Plotting the original data
'''sns.lineplot(x='Date', y='Spread', data=original)

# Plotting the forecast data
sns.lineplot(x='Date', y='Spread', data=df_forecast)

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Spread')
plt.title('Original vs Forecast Spread')
plt.legend()

# Show the plot
plt.show()
'''

# Assuming you have the necessary data for portfolio and benchmark returns
# Portfolio returns would typically be calculated based on your model's predictions vs actuals
# Assuming 'Actual' and 'Predicted' columns are present in your data
data['Return'] = data['Spread'].pct_change()  # Daily returns for the Spread

# Load the benchmark returns (for example, S&P 500 returns) if not already in your data
# Assuming benchmark returns are stored in a column named 'Benchmark_Return' in your data
# and it's aligned with the dates in your 'Date' column

# Calculate daily returns
# Replace 'Benchmark_Return' with your actual benchmark column name
data['Benchmark_Return'] = data['Benchmark_Return'].pct_change()

# Drop NaN values that result from pct_change
data = data.dropna()

# Calculate average daily return for the portfolio
average_daily_return = data['Return'].mean()
print(f"Average Daily Portfolio Return: {average_daily_return:.4f}")

# Calculate beta (Covariance of portfolio and benchmark returns / Variance of benchmark returns)
cov_matrix = np.cov(data['Return'], data['Benchmark_Return'])
beta = cov_matrix[0, 1] / cov_matrix[1, 1]
print(f"Portfolio Beta: {beta:.4f}")
