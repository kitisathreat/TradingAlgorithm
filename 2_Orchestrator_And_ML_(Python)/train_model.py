import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sentiment_analyzer import SentimentAnalyzer

def train_and_save_model():
    """
    Loads historical data, preprocesses it for a neural network,
    builds and trains the network, evaluates it, and saves the final assets.
    """
    print("--- Starting Neural Network Training Phase ---")

    # --- Step 1: Load Data ---
    training_data_file = 'interactive_training_app/backend/investor_decisions_with_vision.csv'
    if not os.path.exists(training_data_file):
        print(f"FATAL: Training data '{training_data_file}' not found. Please run an interactive session first.")
        return

    df = pd.read_csv(training_data_file)
    # Drop rows with missing data for simplicity
    df.dropna(inplace=True)
    if df.empty:
        print("Aborting training: No data available after cleaning.")
        return

    # --- Step 2: Feature Engineering ---
    print("Performing feature engineering...")
    analyzer = SentimentAnalyzer()
    df['News_Sentiment'] = df['News_Headline'].apply(analyzer.get_sentiment_score)
    # Convert categorical facial sentiment to a numerical representation
    df['Facial_Sentiment_Code'] = df['Facial_Sentiment'].astype('category').cat.codes
    
    # --- Step 3: Define Features (X) and Target (y) ---
    numerical_features = ['Close_Price', 'Analyst_Buy_Ratio', 'News_Sentiment', 'Facial_Sentiment_Code']
    target_variable = ['Investor_Action']

    X = df[numerical_features]
    y = df[target_variable]

    # --- Step 4: Create Preprocessing Pipelines ---
    # Neural networks require scaled numerical inputs and one-hot encoded categorical outputs
    numeric_transformer = StandardScaler()
    target_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Create a preprocessor specifically for the input features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though we don't have them here
    )

    # --- Step 5: Split and Preprocess Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the preprocessor on the training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Fit the encoder on the training data and transform both sets
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)

    # --- Step 6: Define the Neural Network Architecture ---
    print("Building Neural Network model...")
    model = keras.Sequential([
        layers.Input(shape=(X_train_processed.shape[1],), name="input_layer"),
        layers.Dense(128, activation='relu', name="hidden_layer_1"),
        layers.Dropout(0.3, name="dropout_1"), # Helps prevent overfitting
        layers.Dense(64, activation='relu', name="hidden_layer_2"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Dense(y_train_encoded.shape[1], activation='softmax', name="output_layer")
    ])
    
    model.summary()

    # --- Step 7: Compile the Model ---
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # Standard for multi-class classification
        metrics=['accuracy']
    )

    # --- Step 8: Train the Model ---
    print("\nTraining the Neural Network...")
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train_processed,
        y_train_encoded,
        epochs=100,
        batch_size=32,
        validation_split=0.2, # Use 20% of training data for validation during training
        callbacks=[early_stopping],
        verbose=1
    )

    # --- Step 9: Evaluate the Model on the Test Set ---
    print("\nEvaluating model performance on unseen test data...")
    loss, accuracy = model.evaluate(X_test_processed, y_test_encoded)
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- Step 10: Save All Necessary Assets ---
    model.save('investor_nn_model.keras')
    joblib.dump(preprocessor, 'data_preprocessor.joblib')
    joblib.dump(target_encoder, 'target_encoder.joblib')
    print("\nNeural network model and data preprocessors saved successfully.")

if __name__ == "__main__":
    train_and_save_model()