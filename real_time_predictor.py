#!/usr/bin/env python3
"""
Bitcoin Real-Time Predictor
Lightweight real-time updates with 24-hour predictions every 5 minutes
"""

import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

from keras.models import load_model
from keras.optimizers import Adam
import joblib
import requests
from collections import deque

from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock


class RealTimeBitcoinPredictor:
    def __init__(self, base_model_path='models/full_dataset_model.h5'):
        """
        Real-time Bitcoin predictor with lightweight updates
        """
        self.lookback_window = 288  # 24 hours
        self.prediction_horizon = 289  # Next 24 hours + 1
        
        # Load base model and scalers
        self.load_base_model(base_model_path)
        
        # Real-time data buffer (store recent 1000 data points)
        self.price_buffer = deque(maxlen=1000)
        self.timestamp_buffer = deque(maxlen=1000)
        self.feature_buffer = deque(maxlen=500)
        
        # Prediction storage
        self.current_prediction = None
        self.prediction_timestamp = None
        self.prediction_confidence = None
        
        # Update tracking
        self.last_update = None
        self.update_count = 0
        self.running = False
        
    def load_base_model(self, model_path):
        """Load the pre-trained base model"""
        try:
            print("Loading base model...")
            self.model = load_model(model_path)
            self.feature_scaler = joblib.load('models/feature_scaler_full.pkl')
            self.price_scaler = joblib.load('models/price_scaler_full.pkl')
            
            with open('models/full_model_config.json', 'r') as f:
                self.config = json.load(f)
            
            print("Base model loaded successfully!")
            print(f"Model config: {self.config}")
            
        except Exception as e:
            print(f"Error loading base model: {e}")
            raise
    
    def fetch_latest_bitcoin_price(self):
        """Fetch latest Bitcoin price from API"""
        try:
            # Using CoinGecko API (free, no API key required)
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=1)
            trading_pair = 'Crypto.BTC/USD'
            url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
            response = requests.get(url, params={
                "symbol": trading_pair,
                "resolution": '5',
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp())
            })
            
            price = float(response.json()['c'][-1])
            timestamp = int(response.json()['t'][-1])
            
            return price, timestamp
            
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None, None
    
    def fetch_bitcoin_ohlcv(self, limit=100):
        """Fetch recent OHLCV data for better price history"""
        try:
            # Using CoinGecko API for historical data
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=10)
            trading_pair = 'Crypto.BTC/USD'
            url = f"https://benchmarks.pyth.network/v1/shims/tradingview/history"
            response = requests.get(url, params={
                "symbol": trading_pair,
                "resolution": '5',
                "from": int(start_dt.timestamp()),
                "to": int(end_dt.timestamp())
            })
            data = response.json()
            
            # Extract closing prices and timestamps
            prices = np.array(data['c'][:-limit], dtype=np.float32)
            timestamps = np.array(data['t'][:-limit], dtype=np.int64)
            
            return prices, timestamps
            
        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            return [], []
    
    def initialize_price_buffer(self):
        """Initialize the price buffer with recent data"""
        print("Initializing price buffer with recent Bitcoin data...")
        
        # Fetch recent OHLCV data
        prices, timestamps = self.fetch_bitcoin_ohlcv(limit=500)
        
        if len(prices) > 0:
            self.price_buffer.extend(prices)
            self.timestamp_buffer.extend(timestamps)
            print(f"Initialized with {len(prices)} historical price points")
        else:
            print("Warning: Could not fetch historical data, starting with empty buffer")
    
    def calculate_features_fast(self, prices):
        """Fast feature calculation for real-time updates"""
        try:
            if len(prices) < 50:  # Need minimum data for indicators
                return None
            
            # Convert to numpy array
            price_array = np.array(prices)
            
            # Calculate technical indicators (optimized for speed)
            macd = Macd(price_array, 6, 12, 3).values
            stoch_rsi = StochRsi(price_array, period=14).hist_values
            dpo = Dpo(price_array, period=4).values
            cop = Coppock(price_array, wma_pd=10, roc_long=6, roc_short=3).values
            inter_slope = PolyInter(price_array, progress_bar=False).values
            
            # Find minimum length
            min_len = min(len(macd), len(stoch_rsi), len(dpo), len(cop), len(inter_slope))
            
            if min_len < self.lookback_window:
                return None
            
            # Take the last values
            features = np.column_stack([
                macd[-min_len:],
                stoch_rsi[-min_len:],
                inter_slope[-min_len:],
                dpo[-min_len:],
                cop[-min_len:]
            ])
            
            # Add price-based features
            aligned_prices = price_array[-min_len:]
            returns = np.diff(aligned_prices) / aligned_prices[:-1]
            volatility = pd.Series(returns).rolling(window=20).std().fillna(0).values
            
            price_features = np.column_stack([
                aligned_prices[1:],
                returns,
                volatility[1:]
            ])
            
            # Combine features
            combined_features = np.column_stack([
                features[1:],
                price_features
            ])
            
            return combined_features, aligned_prices[1:]
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
    
    def predict_24h(self):
        """Generate 24-hour price prediction"""
        try:
            if len(self.price_buffer) < 300:  # Need sufficient data
                print("Insufficient data for prediction")
                return None
            
            # Calculate features
            result = self.calculate_features_fast(list(self.price_buffer))
            if result is None:
                print("Could not calculate features")
                return None
            
            features, prices = result
            
            if len(features) < self.lookback_window:
                print("Not enough feature data")
                return None
            
            # Prepare input sequence
            input_features = features[-self.lookback_window:]
            input_prices = prices[-self.lookback_window:]
            
            # Combine features and prices
            input_sequence = np.concatenate([
                input_features,
                input_prices.reshape(-1, 1)
            ], axis=1)
            
            # Scale input
            input_scaled = self.feature_scaler.transform(
                input_sequence.reshape(-1, input_sequence.shape[1])
            ).reshape(1, self.lookback_window, -1)
            
            # Make prediction
            prediction_scaled = self.model.predict(input_scaled, verbose=0)
            
            # Inverse scale
            prediction = self.price_scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).flatten()
            
            # Calculate confidence (based on recent price volatility)
            recent_volatility = np.std(prices[-50:])
            confidence = max(0.1, 1.0 - (recent_volatility / np.mean(prices[-50:])))
            
            return {
                'prices': prediction,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'current_price': prices[-1],
                'lookback_start': prices[-self.lookback_window],
                'volatility': recent_volatility
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def monte_carlo_prediction(self, n_simulations=500, noise_factor=0.005):
        """Fast Monte Carlo simulation for real-time use"""
        print("Running Monte Carlo simulation...")
        
        base_pred = self.predict_24h()
        if base_pred is None:
            return None
        
        simulations = []
        base_prices = base_pred['prices']
        
        # Generate simulations quickly
        for _ in range(n_simulations):
            # Add noise to recent prices
            noisy_buffer = list(self.price_buffer)
            noise = np.random.normal(0, noise_factor * np.std(noisy_buffer[-100:]), len(noisy_buffer))
            noisy_buffer = np.array(noisy_buffer) + noise
            
            # Quick prediction with noisy data
            try:
                # Update buffer temporarily
                old_buffer = self.price_buffer.copy()
                self.price_buffer.clear()
                self.price_buffer.extend(noisy_buffer)
                
                pred = self.predict_24h()
                if pred is not None:
                    simulations.append(pred['prices'])
                else:
                    # Fallback: add noise to base prediction
                    noise_pred = base_prices + np.random.normal(0, noise_factor * np.std(base_prices), len(base_prices))
                    simulations.append(noise_pred)
                
                # Restore original buffer
                self.price_buffer = old_buffer
                
            except:
                # Fallback if anything fails
                noise_pred = base_prices + np.random.normal(0, noise_factor * np.std(base_prices), len(base_prices))
                simulations.append(noise_pred)
        
        # Calculate statistics
        simulations = np.array(simulations)
        
        result = {
            'mean': np.mean(simulations, axis=0),
            'std': np.std(simulations, axis=0),
            'lower_5': np.percentile(simulations, 5, axis=0),
            'upper_95': np.percentile(simulations, 95, axis=0),
            'lower_25': np.percentile(simulations, 25, axis=0),
            'upper_75': np.percentile(simulations, 75, axis=0),
            'base_prediction': base_pred,
            'n_simulations': len(simulations),
            'timestamp': datetime.now()
        }
        
        return result
    
    def update_model_lightweight(self, learning_rate=0.0001):
        """Lightweight model update with recent data"""
        try:
            if len(self.price_buffer) < 500:
                return False
            
            # Prepare recent data for update
            result = self.calculate_features_fast(list(self.price_buffer))
            if result is None:
                return False
            
            features, prices = result
            
            # Create a few recent sequences for fine-tuning
            if len(features) < self.lookback_window + self.prediction_horizon:
                return False
            
            # Take only the most recent sequences (lightweight update)
            n_recent = min(10, len(features) - self.lookback_window - self.prediction_horizon + 1)
            
            X_update, y_update = [], []
            for i in range(-n_recent, 0):
                idx = len(features) + i
                
                input_seq = np.concatenate([
                    features[idx:idx + self.lookback_window],
                    prices[idx:idx + self.lookback_window].reshape(-1, 1)
                ], axis=1)
                
                target_seq = prices[idx + self.lookback_window:idx + self.lookback_window + self.prediction_horizon]
                
                if len(target_seq) == self.prediction_horizon:
                    X_update.append(input_seq)
                    y_update.append(target_seq)
            
            if not X_update:
                return False
            
            X_update = np.array(X_update)
            y_update = np.array(y_update)
            
            # Scale data
            X_update_scaled = self.feature_scaler.transform(
                X_update.reshape(-1, X_update.shape[2])
            ).reshape(X_update.shape)
            
            y_update_scaled = self.price_scaler.transform(
                y_update.reshape(-1, 1)
            ).reshape(y_update.shape)
            
            # Compile model with lower learning rate for fine-tuning
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='huber',
                metrics=['mae']
            )
            
            # Quick update (few epochs)
            self.model.fit(
                X_update_scaled, y_update_scaled,
                epochs=3,
                batch_size=len(X_update),
                verbose=0,
                shuffle=False
            )
            
            return True
            
        except Exception as e:
            print(f"Error in lightweight update: {e}")
            return False
    
    def continuous_prediction_loop(self, update_interval=300, mc_simulations=500):
        """Main loop for continuous predictions every 5 minutes"""
        print("Starting continuous prediction loop...")
        print(f"Predictions every {update_interval} seconds ({update_interval/60:.1f} minutes)")
        
        self.running = True
        
        while self.running:
            try:
                start_time = time.time()
                print(f"\n{'='*60}")
                print(f"UPDATE #{self.update_count + 1} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                # Fetch latest price
                latest_price, timestamp = self.fetch_latest_bitcoin_price()
                if latest_price:
                    self.price_buffer.append(latest_price)
                    self.timestamp_buffer.append(timestamp)
                    print(f"Current Bitcoin Price: ${latest_price:,.2f}")
                else:
                    print("Warning: Could not fetch latest price")
                
                # Generate prediction
                print("Generating 24-hour prediction...")
                mc_result = self.monte_carlo_prediction(n_simulations=mc_simulations)
                
                if mc_result:
                    self.current_prediction = mc_result
                    self.prediction_timestamp = datetime.now()
                    
                    # Display prediction summary
                    current_price = mc_result['base_prediction']['current_price']
                    mean_24h = mc_result['mean'][-1]
                    lower_95 = mc_result['lower_5'][-1]
                    upper_95 = mc_result['upper_95'][-1]
                    
                    print(f"\n📊 PREDICTION SUMMARY:")
                    print(f"Current Price: ${current_price:,.2f}")
                    print(f"24h Prediction: ${mean_24h:,.2f}")
                    print(f"Change: ${mean_24h - current_price:+,.2f} ({((mean_24h/current_price - 1) * 100):+.2f}%)")
                    print(f"95% Confidence Interval: ${lower_95:,.2f} - ${upper_95:,.2f}")
                    print(f"Confidence Level: {mc_result['base_prediction']['confidence']:.2%}")
                    
                    # Save prediction
                    self.save_prediction(mc_result)
                    
                    # Lightweight model update every 10 predictions
                    if self.update_count % 10 == 0 and self.update_count > 0:
                        print("\n🔄 Performing lightweight model update...")
                        if self.update_model_lightweight():
                            print("✅ Model updated successfully")
                        else:
                            print("⚠️ Model update failed")
                
                else:
                    print("❌ Prediction failed")
                
                self.update_count += 1
                self.last_update = datetime.now()
                
                # Calculate next update time
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                next_update = datetime.now() + timedelta(seconds=sleep_time)
                
                print(f"\n⏱️ Update completed in {elapsed:.1f}s")
                print(f"Next update at: {next_update.strftime('%H:%M:%S')}")
                print(f"Sleeping for {sleep_time:.1f}s...")
                
                # Sleep until next update
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("\n⏹️ Stopping continuous prediction loop...")
                break
            except Exception as e:
                print(f"❌ Error in prediction loop: {e}")
                print("⏳ Waiting 30 seconds before retry...")
                time.sleep(30)
        
        self.running = False
        print("🛑 Continuous prediction loop stopped")
    
    def save_prediction(self, prediction_data):
        """Save prediction to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'predictions/prediction_{timestamp}.json'
            
            # Create directory if it doesn't exist
            import os
            os.makedirs('predictions', exist_ok=True)
            
            # Prepare data for JSON serialization
            save_data = {
                'timestamp': timestamp,
                'current_price': float(prediction_data['base_prediction']['current_price']),
                'predictions_24h': prediction_data['mean'].tolist(),
                'confidence_intervals': {
                    'lower_5': prediction_data['lower_5'].tolist(),
                    'upper_95': prediction_data['upper_95'].tolist(),
                    'lower_25': prediction_data['lower_25'].tolist(),
                    'upper_75': prediction_data['upper_75'].tolist()
                },
                'confidence': float(prediction_data['base_prediction']['confidence']),
                'volatility': float(prediction_data['base_prediction']['volatility']),
                'n_simulations': prediction_data['n_simulations'],
                'update_number': self.update_count
            }
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            # Also save latest prediction as 'latest.json'
            with open('predictions/latest.json', 'w') as f:
                json.dump(save_data, f, indent=2)
            
        except Exception as e:
            print(f"Error saving prediction: {e}")
    
    def get_latest_prediction(self):
        """Get the most recent prediction"""
        return self.current_prediction
    
    def start_background_predictions(self, update_interval=300):
        """Start predictions in background thread"""
        if self.running:
            print("Prediction loop already running!")
            return
        
        # Initialize data buffer
        self.initialize_price_buffer()
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(
            target=self.continuous_prediction_loop,
            args=(update_interval,),
            daemon=True
        )
        self.prediction_thread.start()
        print("Background prediction started!")
    
    def stop_predictions(self):
        """Stop the prediction loop"""
        self.running = False
        print("Stopping predictions...")
    
    def status(self):
        """Get current status"""
        status = {
            'running': self.running,
            'buffer_size': len(self.price_buffer),
            'last_update': self.last_update,
            'update_count': self.update_count,
            'current_price': list(self.price_buffer)[-1] if self.price_buffer else None,
            'prediction_available': self.current_prediction is not None
        }
        return status


def main():
    """Main function for standalone execution"""
    print("🚀 Bitcoin Real-Time Predictor Starting...")
    
    try:
        # Initialize predictor
        predictor = RealTimeBitcoinPredictor()
        
        print("\n📋 Predictor Status:")
        print(f"Model: Loaded successfully")
        print(f"Lookback Window: {predictor.lookback_window} periods (24 hours)")
        print(f"Prediction Horizon: {predictor.prediction_horizon} periods (24 hours)")
        
        # Initialize and start
        predictor.initialize_price_buffer()
        
        print(f"\n🔄 Starting continuous predictions every 5 minutes...")
        print("Press Ctrl+C to stop")
        
        # Start continuous loop
        predictor.continuous_prediction_loop(
            update_interval=300,  # 5 minutes
            mc_simulations=100    # 500 Monte Carlo simulations
        )
        
    except FileNotFoundError:
        print("❌ Base model not found!")
        print("Please run the base model training first:")
        print("python base_model_trainer.py")
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")


# API endpoint class for integration
class PredictionAPI:
    """Simple API wrapper for integration with other systems"""
    
    def __init__(self):
        self.predictor = RealTimeBitcoinPredictor()
        self.predictor.initialize_price_buffer()
    
    def get_prediction(self, monte_carlo=True, simulations=500):
        """Get current 24-hour prediction"""
        if monte_carlo:
            return self.predictor.monte_carlo_prediction(n_simulations=simulations)
        else:
            return self.predictor.predict_24h()
    
    def get_status(self):
        """Get predictor status"""
        return self.predictor.status()
    
    def update_with_price(self, price, timestamp=None):
        """Manually add a price point"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.predictor.price_buffer.append(float(price))
        self.predictor.timestamp_buffer.append(timestamp)


if __name__ == "__main__":
    main()