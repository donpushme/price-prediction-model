import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
from keras.utils import to_categorical
import json
import os

from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock


def extract_data(data):
    # obtain labels
    labels = Genlabels(data, window=25, polyorder=3).labels

    # obtain features
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(data, progress_bar=True).values

    # truncate bad values and shift label
    X = np.array([macd[30:-1], 
                  stoch_rsi[30:-1], 
                  inter_slope[30:-1],
                  dpo[30:-1], 
                  cop[30:-1]])

    X = np.transpose(X)
    labels = labels[31:]

    return X, labels

def adjust_data_temporal(X, y, split=0.8):
    """
    Split time series data while preserving temporal order.
    Handle class imbalance through class weights instead of sampling.
    """
    # Calculate split point based on time (not random sampling)
    split_point = int(len(X) * split)
    
    # Split data temporally - earlier data for training, later for testing
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    
    # Calculate class weights to handle imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training class distribution: {np.bincount(y_train.astype(int))}")
    print(f"Testing class distribution: {np.bincount(y_test.astype(int))}")
    print(f"Class weights: {class_weight_dict}")
    
    return X_train, X_test, y_train, y_test, class_weight_dict

def adjust_data_balanced_temporal(X, y, split=0.8, balance_method='weights'):
    """
    Alternative approach: Use sliding window sampling while preserving order.
    This keeps more data but ensures balanced training in each epoch.
    """
    if balance_method == 'weights':
        return adjust_data_temporal(X, y, split)
    
    elif balance_method == 'sequential_sampling':
        # Find indices of each class
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        
        # Split each class temporally
        split_0 = int(len(idx_0) * split)
        split_1 = int(len(idx_1) * split)
        
        train_idx_0 = idx_0[:split_0]
        test_idx_0 = idx_0[split_0:]
        train_idx_1 = idx_1[:split_1]
        test_idx_1 = idx_1[split_1:]
        
        # Balance the training set by taking equal amounts from each class
        min_train_samples = min(len(train_idx_0), len(train_idx_1))
        
        # Take the most recent samples of the minority class to maintain temporal relevance
        if len(train_idx_0) > len(train_idx_1):
            # More 0s than 1s, take the last min_train_samples of 0s
            train_idx_0_balanced = train_idx_0[-min_train_samples:]
            train_idx_1_balanced = train_idx_1
        else:
            # More 1s than 0s, take the last min_train_samples of 1s  
            train_idx_0_balanced = train_idx_0
            train_idx_1_balanced = train_idx_1[-min_train_samples:]
        
        # Combine indices while preserving temporal order
        train_indices = np.sort(np.concatenate([train_idx_0_balanced, train_idx_1_balanced]))
        test_indices = np.sort(np.concatenate([test_idx_0, test_idx_1]))
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        print(f"Training samples: {len(X_train)} (balanced)")
        print(f"Testing samples: {len(X_test)}")
        print(f"Training class distribution: {np.bincount(y_train.astype(int))}")
        print(f"Testing class distribution: {np.bincount(y_test.astype(int))}")
        
        return X_train, X_test, y_train, y_test, None

# Updated model training function

def shape_data(X, y, timesteps=10):
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if not os.path.exists('models'):
        os.mkdir('models')

    joblib.dump(scaler, 'models/scaler.dump')

    # reshape data with timesteps
    reshaped = []
    for i in range(timesteps, X.shape[0] + 1):
        reshaped.append(X[i - timesteps:i])
    
    # account for data lost in reshaping
    X = np.array(reshaped)
    y = y[timesteps - 1:]

    return X, y

def build_and_train_model(X_train, X_test, y_train, y_test, class_weights=None):
    """
    Build and train model with proper handling of class imbalance
    """
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.utils import to_categorical
    
    # Convert to categorical for softmax
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    
    # Build model
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Train with class weights if provided
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=8,
        shuffle=True,  # Shuffle batches between epochs, not the sequence order
        validation_data=(X_test, y_test_cat),
        class_weight=class_weights,  # Handle imbalance here
        verbose=1
    )
    
    return model, history

if __name__ == '__main__':
    with open('historical_data/hist_data_pyth.json') as f:
        data = json.load(f)

    # load and reshape data
    X, y = extract_data(np.array(data['close']))
    X, y = shape_data(X, y, timesteps=100)

    # ensure equal number of labels, shuffle, and split
    X_train, X_test, y_train, y_test, class_weights = adjust_data_temporal(X, y, split=0.8)
    model, history = build_and_train_model(X_train, X_test, y_train, y_test, class_weights)
    
    # build and train model
    model.save('models/lstm_model.h5')
