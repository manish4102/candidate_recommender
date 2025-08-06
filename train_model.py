import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib
import sys
import os

def main():
    # Specify the exact path to your dataset
    dataset_path = "/Users/manish/Desktop/MANISH/Candidate_Recommendation_System/New_CRS/CRS_data.csv"
    
    # Verify the file exists
    if not os.path.exists(dataset_path):
        print(f"\nERROR: File not found at: {dataset_path}")
        print("Please verify the path to your CSV file")
        return
    
    try:
        print(f"\nLoading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        required_columns = ['Age', 'EdLevel', 'Gender', 'MainBranch',
                          'YearsCode', 'YearsCodePro', 'PreviousSalary',
                          'ComputerSkills', 'Employed']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"\nERROR: Missing required columns: {missing_cols}")
            return
        
        # Preprocessing
        print("\nPreprocessing data...")
        categorical_cols = ['Age', 'EdLevel', 'Gender', 'MainBranch']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Feature selection
        features = ['Age', 'EdLevel', 'Gender', 'MainBranch',
                   'YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']
        target = 'Employed'
        
        X = df[features]
        y = df[target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature scaling
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train model
        print("\nTraining Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        print("\nModel Evaluation:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Save artifacts
        print("\nSaving model artifacts...")
        joblib.dump(model, 'employability_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(label_encoders, 'label_encoders.pkl')
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {os.path.abspath('employability_model.pkl')}")
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")

if __name__ == "__main__":
    main()