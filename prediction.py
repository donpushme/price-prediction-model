import numpy as np
import joblib
from keras.models import load_model
from keras.utils import to_categorical
import json
from sklearn.preprocessing import StandardScaler

# Import your technical analysis modules (same as training)
from technical_analysis.generate_labels import Genlabels
from technical_analysis.macd import Macd
from technical_analysis.rsi import StochRsi
from technical_analysis.poly_interpolation import PolyInter
from technical_analysis.dpo import Dpo
from technical_analysis.coppock import Coppock

def load_trained_model():
    """Load the saved model and scaler"""
    model = load_model('models/lstm_model.h5')
    scaler = joblib.load('models/scaler.dump')
    return model, scaler

def preprocess_new_data(data, scaler, timesteps=5):
    """
    Preprocess new data in the same way as training data
    
    Args:
        data: Raw price data (close prices)
        scaler: Fitted scaler from training
        timesteps: Same timesteps used in training (default 5)
    
    Returns:
        X: Preprocessed data ready for prediction
    """
    
    # Extract features (same as training)
    macd = Macd(data, 6, 12, 3).values
    stoch_rsi = StochRsi(data, period=14).hist_values
    dpo = Dpo(data, period=4).values
    cop = Coppock(data, wma_pd=10, roc_long=6, roc_short=3).values
    inter_slope = PolyInter(data, progress_bar=False).values
    
    # Truncate bad values (same as training)
    X = np.array([macd[30:-1], 
                  stoch_rsi[30:-1], 
                  inter_slope[30:-1],
                  dpo[30:-1], 
                  cop[30:-1]])
    
    X = np.transpose(X)
    
    # Scale data using the fitted scaler
    X_scaled = scaler.transform(X)
    
    # Reshape with timesteps
    reshaped = []
    for i in range(timesteps, X_scaled.shape[0] + 1):
        reshaped.append(X_scaled[i - timesteps:i])
    
    X_final = np.array(reshaped)
    
    if X.shape[0] == 0:
        raise ValueError("No samples to predict on. Check your data extraction and feature engineering steps.")
    
    return X_final

def make_predictions(model, X):
    """
    Make predictions using the loaded model
    
    Args:
        model: Loaded Keras model
        X: Preprocessed input data
    
    Returns:
        predictions: Raw prediction probabilities
        predicted_classes: Predicted class labels
    """
    
    # Get prediction probabilities
    predictions = model.predict(X)
    
    # Get predicted classes (0 or 1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predictions, predicted_classes

def predict_single_sample(model, scaler, new_data, timesteps=5):
    """
    Predict for a single new data sample
    
    Args:
        model: Loaded model
        scaler: Fitted scaler
        new_data: New price data
        timesteps: Timesteps used in training
    
    Returns:
        prediction_prob: Probability for each class
        predicted_class: Predicted class (0 or 1)
        confidence: Confidence score
    """
    
    # Preprocess the new data
    X = preprocess_new_data(new_data, scaler, timesteps)
    
    # Take the last sample for prediction
    X_sample = X[-1:]  # Get last timestep sequence
    
    # Make prediction
    prediction_prob = model.predict(X_sample)[0]
    predicted_class = np.argmax(prediction_prob)
    confidence = np.max(prediction_prob)
    
    return prediction_prob, predicted_class, confidence

# ==================== USAGE EXAMPLES ====================

def main():
    print("=== Loading Trained Model ===")
    model, scaler = load_trained_model()
    
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Example 1: Predict on new data
    print("\n=== Example 1: Predict on New Data ===")
    
    # Load new data (replace with your actual new data)
    try:
        with open('historical_data/new_data.json') as f:
            new_data = json.load(f)
        
        new_close_prices = np.array(new_data['close'])
        print(new_close_prices)
        
        # Preprocess and predict
        X_new = preprocess_new_data(new_close_prices, scaler)
        predictions, predicted_classes = make_predictions(model, X_new)
        
        print(f"Number of predictions: {len(predictions)}")
        print(f"Sample predictions (first 5):")
        for i in range(min(5, len(predictions))):
            prob_class_0 = predictions[i][0]
            prob_class_1 = predictions[i][1]
            predicted = predicted_classes[i]
            print(f"Sample {i}: Class 0: {prob_class_0:.3f}, Class 1: {prob_class_1:.3f}, Predicted: {predicted}")
            
    except FileNotFoundError:
        print("New data file not found. Using synthetic data for demonstration.")
        
        # Generate synthetic data for demonstration
        synthetic_data = np.random.randn(200) * 10 + 100  # Random walk around 100
        X_synthetic = preprocess_new_data(synthetic_data, scaler)
        predictions, predicted_classes = make_predictions(model, X_synthetic)
        
        print(f"Synthetic data predictions (first 5):")
        for i in range(min(5, len(predictions))):
            prob_class_0 = predictions[i][0]
            prob_class_1 = predictions[i][1]
            predicted = predicted_classes[i]
            print(f"Sample {i}: Class 0: {prob_class_0:.3f}, Class 1: {prob_class_1:.3f}, Predicted: {predicted}")
    
    # Example 2: Single prediction with confidence
    print("\n=== Example 2: Single Prediction with Confidence ===")
    
    # Generate some sample data
    sample_data = np.random.randn(100) * 5 + 50
    
    try:
        prediction_prob, predicted_class, confidence = predict_single_sample(
            model, scaler, sample_data
        )
        
        print(f"Prediction probabilities: {prediction_prob}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.3f}")
        
        if predicted_class == 0:
            print("Prediction: Bearish signal (Class 0)")
        else:
            print("Prediction: Bullish signal (Class 1)")
            
    except Exception as e:
        print(f"Error in single prediction: {e}")
    
    # Example 3: Batch prediction with interpretation
    print("\n=== Example 3: Batch Prediction Analysis ===")
    
    try:
        # Generate batch data
        batch_data = np.random.randn(150) * 8 + 75
        X_batch = preprocess_new_data(batch_data, scaler)
        predictions, predicted_classes = make_predictions(model, X_batch)
        
        # Analyze results
        bullish_signals = np.sum(predicted_classes == 1)
        bearish_signals = np.sum(predicted_classes == 0)
        total_signals = len(predicted_classes)
        
        print(f"Total predictions: {total_signals}")
        print(f"Bullish signals (Class 1): {bullish_signals} ({bullish_signals/total_signals*100:.1f}%)")
        print(f"Bearish signals (Class 0): {bearish_signals} ({bearish_signals/total_signals*100:.1f}%)")
        
        # High confidence predictions
        confidences = np.max(predictions, axis=1)
        high_confidence_idx = confidences > 0.8
        
        print(f"High confidence predictions (>80%): {np.sum(high_confidence_idx)}")
        
    except Exception as e:
        print(f"Error in batch prediction: {e}")

def real_time_prediction_example():
    """
    Example of how to use the model for real-time predictions
    """
    print("\n=== Real-time Prediction Example ===")
    
    model, scaler = load_trained_model()
    
    # Simulate real-time data (replace with actual data stream)
    def get_latest_price_data():
        """Simulate getting latest price data"""
        # In reality, this would fetch from an API or database
        return np.random.randn(100) * 10 + 100
    
    # Get latest data
    latest_data = get_latest_price_data()
    
    try:
        # Make prediction
        prediction_prob, predicted_class, confidence = predict_single_sample(
            model, scaler, latest_data
        )
        
        # Interpret results
        signal_type = "BULLISH" if predicted_class == 1 else "BEARISH"
        
        print(f"Latest Signal: {signal_type}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Raw probabilities: Bearish={prediction_prob[0]:.3f}, Bullish={prediction_prob[1]:.3f}")
        
        # Trading decision logic
        if confidence > 0.75:
            if predicted_class == 1:
                print("📈 Strong BUY signal detected!")
            else:
                print("📉 Strong SELL signal detected!")
        elif confidence > 0.6:
            if predicted_class == 1:
                print("📊 Weak BUY signal detected")
            else:
                print("📊 Weak SELL signal detected")
        else:
            print("⚠️ Uncertain signal - consider waiting")
            
    except Exception as e:
        print(f"Error in real-time prediction: {e}")

if __name__ == '__main__':
    main()
    real_time_prediction_example()