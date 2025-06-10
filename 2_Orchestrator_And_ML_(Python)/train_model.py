import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
from sentiment_analyzer import SentimentAnalyzer

def train_and_save_model():
    print("Based computer vision??? More like fucking error exception generator jfc kill me please")
    
    training_data_file = 'interactive_training_app/backend/investor_decisions_with_vision.csv'
    try:
        df = pd.read_csv(training_data_file)
    except FileNotFoundError:
        print(f"FATAL. \n Just fatal. \n you *could* run an interactive session first.")
        return

    # Feature Engineering
    analyzer = SentimentAnalyzer()
    df['News_Sentiment'] = df['News_Headline'].apply(analyzer.get_sentiment_score)
    df['Facial_Sentiment_Code'] = df['Facial_Sentiment'].astype('category').cat.codes

    #these are scuffed, so I might want to add more? idk, consult about KPIs I should even include, the more I do the more complicated this model is going to be
    features = ['Close_Price', 'Analyst_Buy_Ratio', 'News_Sentiment', 'Facial_Sentiment_Code']
    X = df[features]
    y = df['Investor_Action']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #test size is completely arbitrary, perhaps play around with it? maybe copilot has a suggestion
    
    print("RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n checking if it works")
    y_pred = model.predict(X_test)
    labels = sorted(y.unique())
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    model_filename = 'investor_model.joblib'
    joblib.dump(model, model_filename)
    print(f"\nModel 'successfully' trained and saved to '{model_filename}'")

if __name__ == "__main__":
    train_and_save_model()
