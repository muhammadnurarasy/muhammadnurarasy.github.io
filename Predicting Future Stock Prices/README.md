## Predicting Future Stock Prices
Project Link: https://colab.research.google.com/drive/1vrGnOg-i3oTrGee_njP0I8XpndJ96Ely?usp=sharing
### Loading the Model and Fetching Data

To predict stock prices for a different ticker or future dates, load the trained model and fetch the new data:

```python
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('stock_price_lstm_model.h5')

# Function to fetch and preprocess data for a new ticker
def get_stock_data(ticker, start_date='2015-01-01'):
    data = yf.download(ticker, start=start_date)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return data, scaled_data, scaler

# Fetch data for the new ticker
new_ticker = 'GOOGL'
data, scaled_data, scaler = get_stock_data(new_ticker)
```

Making Predictions
Use the trained model to make future predictions:

```python
def create_sequences(data, sequence_length=60):
    x_data = []
    for i in range(sequence_length, len(data)):
        x_data.append(data[i-sequence_length:i, 0])
    x_data = np.array(x_data)
    return x_data

def predict_future_prices(model, initial_data, num_predictions, scaler, sequence_length=60):
    predictions = []
    current_data = initial_data[-sequence_length:]
    
    for _ in range(num_predictions):
        current_data = np.reshape(current_data, (1, sequence_length, 1))
        predicted_price = model.predict(current_data)
        predictions.append(predicted_price[0, 0])
        current_data = np.append(current_data[0, 1:], predicted_price[0, 0])
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Create sequences and predict future prices
x_data = create_sequences(scaled_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

# Predict prices for the next 365 days (approx. 1 year)
num_predictions = 365
future_predictions = predict_future_prices(model, scaled_data, num_predictions, scaler)
```

Visualizing Predictions


```python
import matplotlib.pyplot as plt

# Generate dates for the predicted prices
last_date = data.index[-1]
prediction_dates = pd.date_range(last_date, periods=num_predictions+1, closed='right')

# Create a DataFrame for plotting
future_df = pd.DataFrame(data=future_predictions, index=prediction_dates, columns=['Predicted Close'])

plt.figure(figsize=(14, 8))
plt.plot(data['Close'], label='Historical Prices')
plt.plot(future_df['Predicted Close'], label='Predicted Prices', linestyle='--')
plt.title(f'{new_ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```
