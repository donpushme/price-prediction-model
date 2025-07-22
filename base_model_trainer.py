#!/usr/bin/env python3
"""
Dynamic Memory Management Bitcoin Predictor
Handles ALL data with intelligent memory management
"""

import numpy as np
import pandas as pd
import json
import os
import gc
import psutil
import h5py
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock


class MemoryMonitor:
    """Monitor and manage memory usage dynamically"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    
    @staticmethod
    def get_available_memory():
        """Get available memory in GB"""
        return psutil.virtual_memory().available / 1024 / 1024 / 1024
    
    @staticmethod
    def force_garbage_collection():
        """Aggressive garbage collection"""
        gc.collect()
        gc.collect()
        gc.collect()
    
    @staticmethod
    def calculate_optimal_batch_size(available_memory_gb, sequence_size_mb):
        """Calculate optimal batch size based on available memory"""
        # Reserve 2GB for model and other operations
        usable_memory_gb = max(1, available_memory_gb - 2)
        usable_memory_mb = usable_memory_gb * 1024
        
        # Use 70% of usable memory for batches
        batch_memory_mb = usable_memory_mb * 0.7
        optimal_batch_size = max(4, int(batch_memory_mb / sequence_size_mb))
        
        return min(optimal_batch_size, 128)  # Cap at 128


class HDF5DataManager:
    """Manage data storage and retrieval using HDF5 for memory efficiency"""
    
    def __init__(self, data_path='data/training_data.h5'):
        self.data_path = data_path
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
    def save_features(self, features, prices, chunk_size=10000):
        """Save features to HDF5 with compression"""
        print("Saving features to HDF5...")
        
        with h5py.File(self.data_path, 'w') as f:
            # Save with compression
            f.create_dataset('features', data=features, 
                           compression='gzip', compression_opts=9,
                           chunks=True, shuffle=True)
            f.create_dataset('prices', data=prices,
                           compression='gzip', compression_opts=9,
                           chunks=True, shuffle=True)
            
            # Save metadata
            f.attrs['n_samples'] = len(features)
            f.attrs['n_features'] = features.shape[1]
            f.attrs['created'] = datetime.now().isoformat()
        
        print(f"Features saved to {self.data_path}")
        
        # Clear from memory
        del features, prices
        MemoryMonitor.force_garbage_collection()
    
    def load_chunk(self, start_idx, end_idx):
        """Load a chunk of data from HDF5"""
        with h5py.File(self.data_path, 'r') as f:
            features_chunk = f['features'][start_idx:end_idx]
            prices_chunk = f['prices'][start_idx:end_idx]
        
        return features_chunk, prices_chunk
    
    def get_data_info(self):
        """Get information about stored data"""
        with h5py.File(self.data_path, 'r') as f:
            return {
                'n_samples': f.attrs['n_samples'],
                'n_features': f.attrs['n_features'],
                'features_shape': f['features'].shape,
                'prices_shape': f['prices'].shape
            }


class ChunkedDataGenerator(Sequence):
    """Memory-efficient data generator that loads chunks from HDF5"""
    
    def __init__(self, data_manager, lookback_window, prediction_horizon,
                 start_idx=0, end_idx=None, batch_size=32,
                 feature_scaler=None, price_scaler=None, shuffle=False):
        
        self.data_manager = data_manager
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.feature_scaler = feature_scaler
        self.price_scaler = price_scaler
        self.shuffle = shuffle
        
        # Get data info
        data_info = data_manager.get_data_info()
        self.total_samples = data_info['n_samples']
        
        # Set range
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else self.total_samples
        
        # Calculate valid sequences
        self.total_sequences = max(0, self.end_idx - self.start_idx - lookback_window - prediction_horizon + 1)
        self.sequence_indices = np.arange(self.total_sequences)
        
        if shuffle:
            np.random.shuffle(self.sequence_indices)
        
        # Memory management
        self.chunk_cache = {}
        self.max_cache_size = 3  # Keep max 3 chunks in memory
        
    def __len__(self):
        return max(1, int(np.ceil(self.total_sequences / self.batch_size)))
    
    def _get_chunk(self, chunk_start, chunk_size=10000):
        """Get chunk with caching"""
        chunk_key = (chunk_start, chunk_size)
        
        if chunk_key not in self.chunk_cache:
            # Remove oldest cache entries if cache is full
            if len(self.chunk_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.chunk_cache))
                del self.chunk_cache[oldest_key]
            
            # Load new chunk
            chunk_end = min(chunk_start + chunk_size, self.total_samples)
            features_chunk, prices_chunk = self.data_manager.load_chunk(chunk_start, chunk_end)
            self.chunk_cache[chunk_key] = (features_chunk, prices_chunk)
        
        return self.chunk_cache[chunk_key]
    
    def __getitem__(self, idx):
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.total_sequences)
        batch_sequence_indices = self.sequence_indices[start_idx:end_idx]
        
        batch_X, batch_y = [], []
        
        for seq_idx in batch_sequence_indices:
            actual_idx = self.start_idx + seq_idx
            
            # Determine which chunk(s) we need
            sequence_start = actual_idx
            sequence_end = actual_idx + self.lookback_window + self.prediction_horizon
            
            # Load appropriate chunk(s)
            chunk_start = (sequence_start // 10000) * 10000
            features_chunk, prices_chunk = self._get_chunk(chunk_start)
            
            # Calculate relative indices within chunk
            rel_start = sequence_start - chunk_start
            rel_end = rel_start + self.lookback_window + self.prediction_horizon
            
            # Handle case where sequence spans multiple chunks
            if rel_end > len(features_chunk):
                # Load next chunk and concatenate
                next_chunk_start = chunk_start + 10000
                if next_chunk_start < self.total_samples:
                    next_features, next_prices = self._get_chunk(next_chunk_start)
                    
                    features_data = np.concatenate([features_chunk[rel_start:], next_features])
                    prices_data = np.concatenate([prices_chunk[rel_start:], next_prices])
                else:
                    continue  # Skip this sequence
            else:
                features_data = features_chunk[rel_start:rel_end]
                prices_data = prices_chunk[rel_start:rel_end]
            
            # Create sequence
            if len(features_data) >= self.lookback_window + self.prediction_horizon:
                # Input sequence
                feature_seq = features_data[:self.lookback_window]
                input_seq = feature_seq  # Price already included in features
                
                # Target sequence
                target_seq = prices_data[self.lookback_window:self.lookback_window + self.prediction_horizon]
                
                if len(target_seq) == self.prediction_horizon:
                    batch_X.append(input_seq)
                    batch_y.append(target_seq)
        
        if not batch_X:
            # Return dummy batch if no valid sequences
            dummy_input_shape = (self.lookback_window, 8)  # Assuming 8 features
            dummy_X = np.zeros((1, *dummy_input_shape))
            dummy_y = np.zeros((1, self.prediction_horizon))
            return dummy_X, dummy_y
        
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        
        # Apply scaling
        if self.feature_scaler is not None:
            original_shape = batch_X.shape
            batch_X_reshaped = batch_X.reshape(-1, batch_X.shape[-1])
            print("Transforming batch with shape:", batch_X_reshaped.shape)
            print("Scaler expects features:", self.feature_scaler.n_features_in_)
            if batch_X_reshaped.shape[1] != self.feature_scaler.n_features_in_:
                raise ValueError(f"Feature count mismatch: batch has {batch_X_reshaped.shape[1]} features, but scaler expects {self.feature_scaler.n_features_in_}. Check your feature extraction pipeline for consistency.")
            batch_X_scaled = self.feature_scaler.transform(batch_X_reshaped)
            batch_X = batch_X_scaled.reshape(original_shape)
        
        if self.price_scaler is not None:
            batch_y = self.price_scaler.transform(batch_y.reshape(-1, 1)).reshape(batch_y.shape)
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sequence_indices)
        
        # Clear cache periodically to free memory
        if len(self.chunk_cache) > 2:
            self.chunk_cache.clear()
            MemoryMonitor.force_garbage_collection()


class DynamicMemoryBitcoinPredictor:
    """Bitcoin predictor with dynamic memory management for full dataset"""
    
    def __init__(self, lookback_window=288, prediction_horizon=289):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.feature_scaler = None
        self.price_scaler = None
        self.model = None
        self.data_manager = HDF5DataManager()
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        print(f"Initial memory usage: {MemoryMonitor.get_memory_usage():.2f} GB")
    
    def extract_and_save_features(self, price_data, chunk_size=50000):
        """Extract features in chunks and save to HDF5"""
        print("=" * 60)
        print("EXTRACTING FEATURES WITH DYNAMIC MEMORY MANAGEMENT")
        print("=" * 60)
        
        total_length = len(price_data)
        print(f"Processing {total_length:,} data points in chunks of {chunk_size:,}")
        
        all_features = []
        all_prices = []
        
        # Process in overlapping chunks to maintain continuity
        overlap = max(100, self.lookback_window)  # Overlap to maintain context
        
        for chunk_start in range(0, total_length, chunk_size - overlap):
            chunk_end = min(chunk_start + chunk_size, total_length)
            
            print(f"\nProcessing chunk {chunk_start:,} to {chunk_end:,}")
            print(f"Memory usage: {MemoryMonitor.get_memory_usage():.2f} GB")
            
            # Extract chunk
            price_chunk = price_data[chunk_start:chunk_end]
            
            try:
                # Extract features for this chunk
                chunk_features, chunk_prices = self._extract_chunk_features(price_chunk)
                
                # Remove overlap from previous chunks (except first chunk)
                if chunk_start > 0 and len(chunk_features) > overlap:
                    chunk_features = chunk_features[overlap:]
                    chunk_prices = chunk_prices[overlap:]
                
                all_features.append(chunk_features)
                all_prices.append(chunk_prices)
                
                print(f"Chunk processed: {chunk_features.shape}")
                
                # Force garbage collection
                MemoryMonitor.force_garbage_collection()
                
            except Exception as e:
                print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                continue
        
        # Combine all chunks
        print("\nCombining all chunks...")
        combined_features = np.vstack(all_features)
        combined_prices = np.concatenate(all_prices)
        
        print(f"Final combined shape: {combined_features.shape}")
        print(f"Memory usage before save: {MemoryMonitor.get_memory_usage():.2f} GB")
        
        # Save to HDF5
        self.data_manager.save_features(combined_features, combined_prices)
        
        # Clear memory
        del all_features, all_prices, combined_features, combined_prices
        MemoryMonitor.force_garbage_collection()
        
        print(f"Memory usage after save: {MemoryMonitor.get_memory_usage():.2f} GB")
        return True
    
    def _extract_chunk_features(self, price_data):
        """Extract features from a single chunk"""
        try:
            # Calculate technical indicators
            macd = Macd(price_data, 6, 12, 3).values
            stoch_rsi = StochRsi(price_data, period=14).hist_values
            dpo = Dpo(price_data, period=4).values
            cop = Coppock(price_data, wma_pd=10, roc_long=6, roc_short=3).values
            inter_slope = PolyInter(price_data, progress_bar=False).values
            
            # Apply consistent truncation
            features_dict = {
                'macd': macd[30:-1] if len(macd) > 31 else macd[30:] if len(macd) > 30 else macd,
                'stoch_rsi': stoch_rsi[30:-1] if len(stoch_rsi) > 31 else stoch_rsi[30:] if len(stoch_rsi) > 30 else stoch_rsi,
                'inter_slope': inter_slope[30:-1] if len(inter_slope) > 31 else inter_slope[30:] if len(inter_slope) > 30 else inter_slope,
                'dpo': dpo[30:-1] if len(dpo) > 31 else dpo[30:] if len(dpo) > 30 else dpo,
                'cop': cop[30:-1] if len(cop) > 31 else cop[30:] if len(cop) > 30 else cop
            }
            
            # Find minimum length and align
            min_len = min(len(feature) for feature in features_dict.values() if len(feature) > 0)
            
            if min_len == 0:
                raise ValueError("All features have zero length")
            
            aligned_features = []
            for feature in features_dict.values():
                if len(feature) >= min_len:
                    aligned_features.append(feature[-min_len:])
                else:
                    aligned_features.append(feature)
            
            # Stack technical features
            X = np.array(aligned_features).T
            
            # Calculate price features
            price_start_idx = max(0, len(price_data) - min_len - 1)
            aligned_prices = price_data[price_start_idx:price_start_idx + min_len + 1]
            
            if len(aligned_prices) > 1:
                returns = np.diff(aligned_prices) / (aligned_prices[:-1] + 1e-8)  # Avoid division by zero
                
                if len(returns) >= 20:
                    volatility = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0).values
                else:
                    volatility = np.full_like(returns, np.std(returns) if len(returns) > 1 else 0.0)
                
                current_prices = aligned_prices[1:]
                
                # Final alignment
                final_len = min(len(X), len(current_prices), len(returns), len(volatility))
                X = X[-final_len:]
                current_prices = current_prices[-final_len:]
                returns = returns[-final_len:]
                volatility = volatility[-final_len:]
                
                # Combine features
                price_features = np.column_stack([current_prices, returns, volatility])
                combined_features = np.column_stack([X, price_features])
                
                return combined_features, current_prices
            else:
                raise ValueError("Not enough price data in chunk")
                
        except Exception as e:
            print(f"Error in chunk feature extraction: {e}")
            raise
    
    def prepare_scalers(self, sample_fraction=0.01):
        """Prepare scalers using a sample of the data"""
        print("Preparing scalers...")
        
        data_info = self.data_manager.get_data_info()
        total_samples = data_info['n_samples']
        
        # Sample data for scaler fitting
        sample_size = max(1000, int(total_samples * sample_fraction))
        sample_indices = np.random.choice(total_samples, size=min(sample_size, total_samples), replace=False)
        
        # Load samples in small chunks
        sample_features = []
        sample_prices = []
        
        for i in range(0, len(sample_indices), 1000):
            batch_indices = sample_indices[i:i+1000]
            for idx in batch_indices:
                features_chunk, prices_chunk = self.data_manager.load_chunk(idx, idx+1)
                sample_features.extend(features_chunk)
                sample_prices.extend(prices_chunk)
        
        # Convert to arrays
        sample_features = np.array(sample_features)
        sample_prices = np.array(sample_prices)
        
        print(f"Using {len(sample_features)} samples for scaler fitting")
        
        # Fit scalers
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(sample_features)
        
        self.price_scaler = MinMaxScaler()
        self.price_scaler.fit(sample_prices.reshape(-1, 1))
        
        # Clean up
        del sample_features, sample_prices
        MemoryMonitor.force_garbage_collection()
    
    def create_dynamic_generators(self, validation_split=0.1, test_split=0.1):
        """Create data generators with dynamic batch sizing"""
        print("Creating dynamic data generators...")
        
        data_info = self.data_manager.get_data_info()
        total_samples = data_info['n_samples']
        
        # Calculate splits
        train_size = int(total_samples * (1 - validation_split - test_split))
        val_size = int(total_samples * validation_split)
        
        print(f"Total samples: {total_samples:,}")
        print(f"Train samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Test samples: {total_samples - train_size - val_size:,}")
        
        # Calculate optimal batch size
        available_memory = MemoryMonitor.get_available_memory()
        sequence_size_mb = (self.lookback_window * 8 * 4) / (1024 * 1024)  # Rough estimate
        optimal_batch_size = MemoryMonitor.calculate_optimal_batch_size(available_memory, sequence_size_mb)
        
        print(f"Available memory: {available_memory:.2f} GB")
        print(f"Calculated optimal batch size: {optimal_batch_size}")
        
        # Create generators
        train_gen = ChunkedDataGenerator(
            self.data_manager, self.lookback_window, self.prediction_horizon,
            start_idx=0, end_idx=train_size, batch_size=optimal_batch_size,
            feature_scaler=self.feature_scaler, price_scaler=self.price_scaler, shuffle=True
        )
        
        val_gen = ChunkedDataGenerator(
            self.data_manager, self.lookback_window, self.prediction_horizon,
            start_idx=train_size, end_idx=train_size + val_size, batch_size=optimal_batch_size,
            feature_scaler=self.feature_scaler, price_scaler=self.price_scaler, shuffle=False
        )
        
        return train_gen, val_gen, optimal_batch_size
    
    def build_adaptive_model(self, input_shape):
        """Build model that adapts to available memory"""
        print("Building adaptive LSTM model...")
        
        available_memory = MemoryMonitor.get_available_memory()
        
        # Adjust model size based on available memory
        if available_memory > 8:  # High memory
            lstm_units = [256, 128, 64]
            dense_units = [128, 64]
        elif available_memory > 4:  # Medium memory
            lstm_units = [128, 64, 32]
            dense_units = [64, 32]
        else:  # Low memory
            lstm_units = [64, 32]
            dense_units = [32]
        
        print(f"Using model configuration for {available_memory:.1f}GB memory:")
        print(f"LSTM units: {lstm_units}")
        print(f"Dense units: {dense_units}")
        
        # Build model
        model = Sequential()
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(0.3))
        
        # Dense layers
        for units in dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(self.prediction_horizon, activation='linear'))
        
        # Compile
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        
        return model
    
    def train_full_dataset(self, price_data, epochs=100):
        """Train on the full dataset with dynamic memory management"""
        print("=" * 60)
        print("TRAINING ON FULL DATASET WITH DYNAMIC MEMORY MANAGEMENT")
        print("=" * 60)
        
        # Step 1: Extract and save features
        if not os.path.exists(self.data_manager.data_path):
            self.extract_and_save_features(price_data)
        else:
            print("Using existing feature data...")
        
        # Step 2: Prepare scalers
        self.prepare_scalers()
        
        # Step 3: Create generators
        train_gen, val_gen, batch_size = self.create_dynamic_generators()
        
        # Step 4: Build model
        sample_batch_x, _ = train_gen[0]
        input_shape = sample_batch_x.shape[1:]
        self.model = self.build_adaptive_model(input_shape)
        
        print("Model Architecture:")
        self.model.summary()
        
        # Step 5: Set up callbacks
        callbacks = [
            ModelCheckpoint(
                'models/full_dataset_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Step 6: Train
        print(f"Training with batch size: {batch_size}")
        print(f"Steps per epoch: {len(train_gen)}")
        
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def save_full_model(self):
        """Save the complete model"""
        print("Saving full model...")
        
        self.model.save('models/full_dataset_model.h5')
        joblib.dump(self.feature_scaler, 'models/feature_scaler_full.pkl')
        joblib.dump(self.price_scaler, 'models/price_scaler_full.pkl')
        
        config = {
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'model_type': 'full_dataset_lstm',
            'training_time': datetime.now().isoformat(),
            'data_path': self.data_manager.data_path
        }
        
        with open('models/full_model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("Full model saved successfully!")


def main():
    """Train on full dataset with dynamic memory management"""
    print("Dynamic Memory Bitcoin Predictor - FULL DATASET")
    print("=" * 60)
    
    # Monitor initial system state
    print(f"Available memory: {MemoryMonitor.get_available_memory():.2f} GB")
    print(f"Initial process memory: {MemoryMonitor.get_memory_usage():.2f} GB")
    
    # Initialize predictor
    predictor = DynamicMemoryBitcoinPredictor(
        lookback_window=288,  # Keep original parameters
        prediction_horizon=289
    )
    
    try:
        # Load ALL data
        print("Loading Bitcoin price data...")
        with open('historical_data/hist_data_pyth.json', 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            bitcoin_prices = np.array(data.get('close', list(data.values())[0]))
        else:
            bitcoin_prices = np.array(data)
        
        print(f"Loaded {len(bitcoin_prices):,} price points")
        print(f"Price range: ${bitcoin_prices.min():.2f} - ${bitcoin_prices.max():.2f}")
        print(f"Data covers approximately {len(bitcoin_prices) * 5 / (60 * 24):.1f} days")
        
        # Train on full dataset
        history = predictor.train_full_dataset(bitcoin_prices, epochs=100)
        
        # Save everything
        predictor.save_full_model()
        
        print("\n" + "=" * 60)
        print("FULL DATASET TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final memory usage: {MemoryMonitor.get_memory_usage():.2f} GB")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()