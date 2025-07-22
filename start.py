import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock


class BitcoinPricePredictor:
    def __init__(self, lookback_window=288, prediction_horizon=289):
        """
        Bitcoin Price Prediction Model for 24-hour forecasting
        
        Args:
            lookback_window (int): Number of 5-min periods to look back (288 = 24 hours)
            prediction_horizon (int): Number of 5-min periods to predict (289 = 24 hours + 1)
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.scaler = None
        self.price_scaler = None
        self.model = None
        self.feature_columns = ['macd', 'stoch_rsi', 'inter_slope', 'dpo', 'coppock']
        
    def extract_features(self, price_data):
        """Extract technical indicators from price data"""
        print("Extracting technical indicators...")
        
        # Calculate technical indicators
        macd = Macd(price_data, 6, 12, 3).values
        stoch_rsi = StochRsi(price_data, period=14).hist_values
        dpo = Dpo(price_data, period=4).values
        cop = Coppock(price_data, wma_pd=10, roc_long=6, roc_short=3).values
        inter_slope = PolyInter(price_data, progress_bar=True).values
        
        # Align all indicators (remove NaN values from beginning)
        min_length = min(len(macd), len(stoch_rsi), len(dpo), len(cop), len(inter_slope))
        start_idx = len(price_data) - min_length
        
        features = np.array([
            macd[-min_length:],
            stoch_rsi[-min_length:],
            inter_slope[-min_length:],
            dpo[-min_length:],
            cop[-min_length:]
        ]).T
        
        # Corresponding price data
        aligned_prices = price_data[start_idx:]
        
        return features, aligned_prices
    
    def create_sequences(self, features, prices):
        """Create input sequences and target sequences for training"""
        X, y = [], []
        
        # Create sequences where X is features + price, y is future prices
        for i in range(len(features) - self.lookback_window - self.prediction_horizon + 1):
            # Input: lookback_window of features + current price
            feature_seq = features[i:(i + self.lookback_window)]
            price_seq = prices[i:(i + self.lookback_window)].reshape(-1, 1)
            
            # Combine features and prices
            input_seq = np.concatenate([feature_seq, price_seq], axis=1)
            X.append(input_seq)
            
            # Target: next prediction_horizon prices
            target_seq = prices[(i + self.lookback_window):(i + self.lookback_window + self.prediction_horizon)]
            y.append(target_seq)
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, price_data, train_split=0.8):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Extract features
        features, aligned_prices = self.extract_features(price_data)
        
        # Create sequences
        X, y = self.create_sequences(features, aligned_prices)
        
        # Split temporally (preserve time order)
        split_point = int(len(X) * train_split)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Scale features (fit on training data only)
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        X_train = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        X_test = X_test_scaled.reshape(len(X_test), n_timesteps, n_features)
        
        # Scale prices separately (for better reconstruction)
        self.price_scaler = MinMaxScaler()
        y_train_scaled = self.price_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_test_scaled = self.price_scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        
        print(f"Training sequences: {len(X_train)}")
        print(f"Testing sequences: {len(X_test)}")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        
        return X_train, X_test, y_train_scaled, y_test_scaled, y_test
    
    def build_model(self, input_shape):
        """Build the LSTM model for multi-step price prediction"""
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers for multi-step prediction
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            
            # Output layer - predict 289 future prices
            Dense(self.prediction_horizon, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train(self, price_data, epochs=100, batch_size=16, validation_split=0.2):
        """Train the model"""
        print("Starting model training...")
        
        # Prepare data
        X_train, X_test, y_train_scaled, y_test_scaled, y_test_original = self.prepare_data(price_data)
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        print("Model architecture:")
        self.model.summary()
        
        # Train model
        history = self.model.fit(
            X_train, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test_scaled),
            shuffle=False,  # Important: don't shuffle time series data
            verbose=1
        )
        
        # Evaluate model
        test_predictions_scaled = self.model.predict(X_test)
        test_predictions = self.price_scaler.inverse_transform(
            test_predictions_scaled.reshape(-1, 1)
        ).reshape(test_predictions_scaled.shape)
        
        # Calculate metrics
        mae = np.mean(np.abs(test_predictions - y_test_original))
        mape = np.mean(np.abs((y_test_original - test_predictions) / y_test_original)) * 100
        
        print(f"\nModel Performance:")
        print(f"MAE: ${mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return history, test_predictions, y_test_original
    
    def predict_next_24h(self, recent_data):
        """
        Predict next 24 hours (289 5-minute intervals) of Bitcoin prices
        
        Args:
            recent_data: Array of recent price data (at least lookback_window length)
            
        Returns:
            Array of 289 predicted prices
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if len(recent_data) < self.lookback_window + 50:  # Need extra for technical indicators
            raise ValueError(f"Need at least {self.lookback_window + 50} recent price points")
        
        # Use the most recent data
        recent_prices = recent_data[-self.lookback_window - 50:]
        
        # Extract features
        features, aligned_prices = self.extract_features(recent_prices)
        
        # Use the last lookback_window for prediction
        if len(features) < self.lookback_window:
            raise ValueError("Not enough data after feature extraction")
        
        last_features = features[-self.lookback_window:]
        last_prices = aligned_prices[-self.lookback_window:]
        
        # Combine features and prices
        input_seq = np.concatenate([
            last_features, 
            last_prices.reshape(-1, 1)
        ], axis=1)
        
        # Scale the input
        input_scaled = self.scaler.transform(input_seq.reshape(-1, input_seq.shape[1]))
        input_scaled = input_scaled.reshape(1, self.lookback_window, -1)
        
        # Make prediction
        prediction_scaled = self.model.predict(input_scaled, verbose=0)
        
        # Inverse scale to get actual prices
        prediction = self.price_scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten()
        
        return prediction
    
    def monte_carlo_prediction(self, recent_data, n_simulations=100, noise_std=0.01):
        """
        Generate Monte Carlo predictions by adding noise to inputs
        
        Args:
            recent_data: Recent price data
            n_simulations: Number of Monte Carlo simulations
            noise_std: Standard deviation of noise to add
            
        Returns:
            Array of shape (n_simulations, 289) with price predictions
        """
        print(f"Running {n_simulations} Monte Carlo simulations...")
        
        base_prediction = self.predict_next_24h(recent_data)
        simulations = []
        
        for i in range(n_simulations):
            if i % 100 == 0:
                print(f"Simulation {i}/{n_simulations}")
            
            # Add noise to recent data
            noisy_data = recent_data + np.random.normal(0, noise_std * np.std(recent_data), len(recent_data))
            
            try:
                prediction = self.predict_next_24h(noisy_data)
                simulations.append(prediction)
            except:
                # If prediction fails, use base prediction with noise
                noisy_prediction = base_prediction + np.random.normal(0, noise_std * np.std(base_prediction), len(base_prediction))
                simulations.append(noisy_prediction)
        
        simulations = np.array(simulations)
        
        # Calculate statistics
        mean_prediction = np.mean(simulations, axis=0)
        std_prediction = np.std(simulations, axis=0)
        percentile_5 = np.percentile(simulations, 5, axis=0)
        percentile_95 = np.percentile(simulations, 95, axis=0)
        
        results = {
            'mean': mean_prediction,
            'std': std_prediction,
            'lower_bound': percentile_5,
            'upper_bound': percentile_95,
            'all_simulations': simulations
        }
        
        return results
    
    def save_model(self, filepath='models/bitcoin_predictor.h5'):
        """Save the trained model and scalers"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        self.model.save(filepath)
        joblib.dump(self.scaler, 'models/feature_scaler.pkl')
        joblib.dump(self.price_scaler, 'models/price_scaler.pkl')
        
        # Save model parameters
        params = {
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'feature_columns': self.feature_columns
        }
        with open('models/model_params.json', 'w') as f:
            json.dump(params, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/bitcoin_predictor.h5'):
        """Load a trained model and scalers"""
        self.model = load_model(filepath)
        self.scaler = joblib.load('models/feature_scaler.pkl')
        self.price_scaler = joblib.load('models/price_scaler.pkl')
        
        # Load model parameters
        with open('models/model_params.json', 'r') as f:
            params = json.load(f)
        
        self.lookback_window = params['lookback_window']
        self.prediction_horizon = params['prediction_horizon']
        self.feature_columns = params['feature_columns']
        
        print(f"Model loaded from {filepath}")
    
    def plot_predictions(self, recent_data, monte_carlo_results, save_path=None):
        """Plot the predictions with confidence intervals"""
        # Create time axis (5-minute intervals)
        time_axis = np.arange(0, self.prediction_horizon * 5, 5)  # Minutes
        hours = time_axis / 60  # Convert to hours
        
        # Get recent prices for context
        recent_prices = recent_data[-50:]  # Last 50 points for context
        recent_time = np.arange(-len(recent_prices) * 5, 0, 5) / 60  # Hours before prediction
        
        fig = go.Figure()
        
        # Plot recent prices
        fig.add_trace(go.Scatter(
            x=recent_time,
            y=recent_prices,
            mode='lines',
            name='Recent Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Plot mean prediction
        fig.add_trace(go.Scatter(
            x=hours,
            y=monte_carlo_results['mean'],
            mode='lines',
            name='Mean Prediction',
            line=dict(color='red', width=2)
        ))
        
        # Plot confidence interval
        fig.add_trace(go.Scatter(
            x=hours,
            y=monte_carlo_results['upper_bound'],
            mode='lines',
            name='95th Percentile',
            line=dict(color='rgba(255,0,0,0.3)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=monte_carlo_results['lower_bound'],
            mode='lines',
            name='5th Percentile',
            line=dict(color='rgba(255,0,0,0.3)'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            showlegend=True
        ))
        
        fig.update_layout(
            title='Bitcoin Price Prediction - Next 24 Hours',
            xaxis_title='Time (Hours)',
            yaxis_title='Bitcoin Price ($)',
            hovermode='x unified'
        )
        
        if save_path:
            py.plot(fig, filename=save_path)
        else:
            fig.show()


# Usage Example
if __name__ == '__main__':
    # Initialize predictor
    predictor = BitcoinPricePredictor(
        lookback_window=288,  # 24 hours of 5-min data
        prediction_horizon=289  # Next 24 hours + 1
    )
    
    # Load your Bitcoin data
    with open('historical_data/hist_data_pyth.json') as f:
        data = json.load(f)
    
    # Assuming your data is in data['prices'] or similar
    bitcoin_prices = np.array(data['close'])  # Adjust this to your data structure
    
    # Train the model
    history, test_predictions, y_test = predictor.train(
        bitcoin_prices, 
        epochs=100, 
        batch_size=16
    )
    
    # Save the trained model
    predictor.save_model()
    
    # Make predictions for next 24 hours
    recent_data = bitcoin_prices[-500:]  # Last 500 data points
    
    # Single prediction
    next_24h_prices = predictor.predict_next_24h(recent_data)
    print(f"Predicted prices for next 24h: {len(next_24h_prices)} points")
    
    # Monte Carlo predictions
    mc_results = predictor.monte_carlo_prediction(
        recent_data, 
        n_simulations=100,
        noise_std=0.01
    )
    
    # Plot results
    predictor.plot_predictions(recent_data, mc_results, 'bitcoin_prediction.html')
    
    print(f"Mean predicted price in 24h: ${mc_results['mean'][-1]:.2f}")
    print(f"95% confidence interval: ${mc_results['lower_bound'][-1]:.2f} - ${mc_results['upper_bound'][-1]:.2f}")