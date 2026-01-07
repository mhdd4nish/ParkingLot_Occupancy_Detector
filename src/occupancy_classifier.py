import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
from typing import List, Tuple

class OccupancyClassifier:

    
    def __init__(self, config, classifier_type='svm'):
        self.config = config
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        
        if classifier_type == 'svm':
            self.classifier = SVC(
                kernel=config.SVM_KERNEL,
                C=config.SVM_C,
                gamma=config.SVM_GAMMA,
                probability=True
            )

    
    def prepare_dataset(self, features_list: List[np.ndarray], 
                       labels: List[int]) -> Tuple:
        X = np.array(features_list)
        y = np.array(labels)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SPLIT_RATIO,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train, y_train)
        
        
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate the classifier"""
        y_pred = self.classifier.predict(X_test)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Empty', 'Occupied']))
        
        print("\nConfusion Matrix:")
        cm=confusion_matrix(y_test, y_pred)
        print(cm)
        
        accuracy = self.classifier.score(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        try:
            output_path = self.config.RESULTS_DIR / f"{self.classifier_type}_confusion_matrix.png"

            
            self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Empty', 'Occupied'],
                        yticklabels=['Empty', 'Occupied'])
            plt.title(f'{self.classifier_type.upper()} Confusion Matrix')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')

            plt.savefig(output_path)
            print(f"\nConfusion matrix plot saved to: {output_path}")

        except Exception as e:
            print(f"Error saving plot: {e}")

        try:
            
            y_probs = self.classifier.predict_proba(X_test)[:, 1]

            fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)

            output_path_roc = self.config.RESULTS_DIR / f"{self.classifier_type}_roc_curve.png"

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--') 
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{self.classifier_type.upper()} ROC Curve')
            plt.legend(loc="lower right")

            plt.savefig(output_path_roc)
            print(f"ROC curve plot saved to: {output_path_roc}")

        except Exception as e:
            print(f"Error saving ROC plot: {e}")

        return accuracy
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.classifier.predict(features_scaled)[0]
        confidence = self.classifier.predict_proba(features_scaled)[0]
        
        return prediction, confidence[prediction]
    
    def save_model(self, filename: str):
        
        model_path = self.config.MODELS_DIR / filename
        joblib.dump({
            'classifier': self.classifier,
            'scaler': self.scaler,
            'type': self.classifier_type
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename: str):
       
        model_path = self.config.MODELS_DIR / filename
        data = joblib.load(model_path)
        self.classifier = data['classifier']
        self.scaler = data['scaler']
        self.classifier_type = data['type']
        print(f"Model loaded from {model_path}")