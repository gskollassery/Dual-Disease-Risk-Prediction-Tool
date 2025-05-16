
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# Constants
DATA_PATH = "data/health_data.csv"
MODEL_PATH = "models/disease_model.pkl"
FEATURE_PATH = "models/features.pkl"

class DiseaseRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load and preprocess health data"""
        try:
            df = pd.read_csv(DATA_PATH)
            
            # Example preprocessing (adjust based on actual data)
            df = pd.get_dummies(df, columns=['gender', 'smoking_status'])
            
            # Handle missing values
            numerical_cols = df.select_dtypes(include=np.number).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            return df
        
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def train_model(self, df, target='disease_risk'):
        """Train logistic regression model"""
        try:
            X = df.drop(columns=[target])
            y = df[target]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            # Train model
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            sensitivity = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            print(f"Model trained successfully:")
            print(f"- Accuracy: {accuracy:.3f}")
            print(f"- Sensitivity: {sensitivity:.3f}")
            print(f"- ROC AUC: {roc_auc:.3f}")
            
            self.feature_names = X.columns
            
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def save_model(self):
        """Save trained model and features"""
        try:
            Path("models").mkdir(exist_ok=True)
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.feature_names
            }, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def visualize_risk_factors(self, top_n=10):
        """Visualize most important risk factors"""
        if self.model is None:
            print("Model not trained yet")
            return
            
        coefficients = self.model.coef_[0]
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Absolute_Impact': np.abs(coefficients)
        }).sort_values('Absolute_Impact', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x='Coefficient', 
            y='Feature', 
            data=importance.head(top_n),
            palette='coolwarm'
        )
        plt.title(f"Top {top_n} Disease Risk Factors")
        plt.xlabel("Impact on Risk")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig("visualization/risk_factors.png")
        plt.show()
    
    def create_dashboard(self):
        """Create interactive risk assessment dashboard"""
        if self.model is None:
            print("Train model first")
            return
            
        # Create interactive widgets
        inputs = {}
        for feature in self.feature_names:
            if feature.startswith(('age', 'bmi', 'blood_pressure')):
                inputs[feature] = widgets.FloatSlider(
                    description=feature,
                    min=0,
                    max=100 if feature == 'age' else 50,
                    step=0.1
                )
            elif feature.startswith(('gender_', 'smoking_status_')):
                inputs[feature] = widgets.Checkbox(
                    description=feature,
                    value=False
                )
            else:
                inputs[feature] = widgets.FloatText(
                    description=feature,
                    value=0
                )
        
        # Prediction function
        def predict_risk(**kwargs):
            input_df = pd.DataFrame([kwargs])
            input_scaled = self.scaler.transform(input_df)
            proba = self.model.predict_proba(input_scaled)[0, 1]
            print(f"\nPredicted Disease Risk: {proba:.1%}")
            if proba > 0.7:
                print("🟢 Low Risk")
            elif proba > 0.3:
                print("🟡 Moderate Risk")
            else:
                print("🔴 High Risk")
        
        # Display interactive widgets
        print("Adjust your health parameters:")
        interactive_plot = widgets.interactive(predict_risk, **inputs)
        display(interactive_plot)

def main():
    """Main execution function"""
    predictor = DiseaseRiskPredictor()
    
    # Load and prepare data
    print("Loading data...")
    health_data = predictor.load_data()
    if health_data is None:
        return
    
    # Train model
    print("\nTraining model...")
    if not predictor.train_model(health_data):
        return
    
    # Save model
    print("\nSaving model...")
    predictor.save_model()
    
    # Visualize risk factors
    print("\nGenerating visualizations...")
    predictor.visualize_risk_factors()
    
    # Create dashboard (for Jupyter)
    print("\nTo create interactive dashboard, run in Jupyter:")
    print("predictor = DiseaseRiskPredictor()")
    print("predictor.load_model()  # If loading existing model")
    print("predictor.create_dashboard()")

if __name__ == "__main__":
    main()