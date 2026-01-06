import pandas as pd
import numpy as np
import joblib
import re
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, StackingRegressor
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report,
    r2_score, confusion_matrix, precision_recall_fscore_support,
    mean_absolute_error
)
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix, vstack
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
class ScoreCalibrator:
    """Calibrates regression predictions to full 0-10 range"""
    
    def __init__(self):
        self.class_stats = {}
        self.global_stats = {}
        
    def fit(self, y_score, y_class, label_encoder):
        """Learn class-conditioned statistics"""
        classes = label_encoder.classes_
        
        # Global statistics
        self.global_stats = {
            'mean': np.mean(y_score),
            'std': np.std(y_score),
            'min': np.min(y_score),
            'max': np.max(y_score)
        }
        
        # Class-conditioned statistics
        for i, class_name in enumerate(classes):
            mask = (y_class == i)
            scores_in_class = y_score[mask]
            
            if len(scores_in_class) > 0:
                self.class_stats[class_name] = {
                    'mean': np.mean(scores_in_class),
                    'std': np.std(scores_in_class) + 1e-6,  # prevent division by zero
                    'min': np.min(scores_in_class),
                    'max': np.max(scores_in_class),
                    'q05': np.percentile(scores_in_class, 5),
                    'q95': np.percentile(scores_in_class, 95)
                }
        
        # Define target ranges per class
        self.target_ranges = {
            'Easy': (1.0, 4.0),
            'Medium': (4.0, 7.0),
            'Hard': (7.0, 10.0)
        }
        
        return self
    
    def transform(self, y_pred_raw, predicted_classes, label_encoder):
        """Calibrate predictions to full range"""
        y_calibrated = np.zeros_like(y_pred_raw)
        
        for i, class_name in enumerate(label_encoder.classes_):
            mask = (predicted_classes == i)
            
            if not np.any(mask):
                continue
            
            scores = y_pred_raw[mask]
            stats = self.class_stats.get(class_name, self.global_stats)
            target_min, target_max = self.target_ranges[class_name]
            
            # Normalize to [0, 1] using training range
            if stats['max'] > stats['min']:
                normalized = (scores - stats['min']) / (stats['max'] - stats['min'])
            else:
                normalized = np.full_like(scores, 0.5)
            
            # Clip to avoid extrapolation
            normalized = np.clip(normalized, 0, 1)
            
            # Map to target range
            calibrated = target_min + normalized * (target_max - target_min)
            
            y_calibrated[mask] = calibrated
        
        # Final clip to [1, 10]
        y_calibrated = np.clip(y_calibrated, 1.0, 10.0)
        
        return y_calibrated
    
class RobustModelTrainer:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
        # Feature extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            binary=True
        )
        
        # Models
        self.classifiers = {}
        self.regressors = {}
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=50, random_state=seed)  # Reduced components
        self.score_calibrator = ScoreCalibrator()
        # Training history
        self.history = {
            'classifier_metrics': {},
            'regressor_metrics': {},
            'training_time': {}
        }
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
    
    def load_data(self, filepath='../data/dataset.jsonl'):
        """Load and validate dataset"""
        print(f"📂 Loading data from {filepath}...")
        
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        data.append(record)
                    except json.JSONDecodeError:
                        continue
            
            df = pd.DataFrame(data)
            
            if len(df) == 0:
                print("❌ No valid data found in file")
                return None
            
            print(f"✅ Loaded {len(df)} samples")
            
            # Validate required columns
            required_columns = ['title', 'description', 'problem_class', 'problem_score']
            for col in required_columns:
                if col not in df.columns:
                    print(f"❌ Missing required column: {col}")
                    return None
            
            # Handle missing values
            for col in required_columns:
                if df[col].isnull().any():
                    print(f"⚠️  Dropping rows with missing {col}")
                    df = df.dropna(subset=[col])
            
            # Validate and clean scores
            df['problem_score'] = pd.to_numeric(df['problem_score'], errors='coerce')
            df = df.dropna(subset=['problem_score'])
            df['problem_score'] = df['problem_score'].clip(1, 10)
            
            # Clean class labels
            df['problem_class'] = df['problem_class'].str.strip().str.title()
            
            print(f"✅ Cleaned dataset: {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_text(self, text):
        """Clean text for feature extraction"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace problematic characters
        text = re.sub(r'[^\w\s\+\-\*/=<>\(\)\[\]\{\}\.,;:!?\^&\|%#@$\n]', ' ', text)
        
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, text):
        """Extract meaningful features from text"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text.replace(' ', ''))
        
        sentences = re.split(r'[.!?]+', text)
        sentence_count = max(1, len([s for s in sentences if s.strip()]))
        features['sentence_count'] = sentence_count
        features['avg_sentence_length'] = features['word_count'] / sentence_count
        features['avg_word_length'] = features['char_count'] / max(1, features['word_count'])
        
        # Vocabulary richness
        words = text.lower().split()
        if words:
            unique_words = set(words)
            features['vocab_richness'] = len(unique_words) / len(words)
        else:
            features['vocab_richness'] = 0
        
        # Mathematical and programming indicators
        math_operators = r'[+\-*/=<>^&|%]'
        features['math_operators'] = len(re.findall(math_operators, text))
        
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        features['number_count'] = len(numbers)
        
        # Algorithm categories
        algorithm_categories = {
            'dp': ['dp', 'dynamic programming', 'memoization', 'knapsack'],
            'graph': ['graph', 'bfs', 'dfs', 'dijkstra', 'tree'],
            'search': ['binary search', 'linear search'],
            'sort': ['sort', 'quicksort', 'mergesort'],
            'ds': ['array', 'linked list', 'stack', 'queue', 'hash'],
            'math': ['prime', 'gcd', 'lcm', 'modulo']
        }
        
        text_lower = text.lower()
        for category, keywords in algorithm_categories.items():
            features[f'has_{category}'] = int(any(keyword in text_lower for keyword in keywords))
        
        # Complexity indicators
        complexity_terms = ['time complexity', 'space complexity', 'O(', 'efficient', 'optimize']
        features['complexity_mentions'] = int(any(term in text_lower for term in complexity_terms))
        
        # Problem structure
        features['has_input'] = int('input' in text_lower)
        features['has_output'] = int('output' in text_lower)
        features['has_example'] = int('example' in text_lower)
        features['has_constraint'] = int('constraint' in text_lower)
        
        # Ensure all features are finite
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    features[key] = 0
        
        return features
    
    def prepare_features(self, df):
        """Prepare features for training"""
        print("\n🔧 Preparing features...")
        
        # Clean and combine text
        print("Processing text...")
        df['combined_text'] = df.apply(
            lambda row: self.clean_text(f"{row.get('title', '')} {row.get('description', '')}"), 
            axis=1
        )
        
        # Extract engineered features
        print("Extracting engineered features...")
        engineered_features = []
        for i, text in enumerate(df['combined_text']):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i} samples...")
            features = self.extract_features(text)
            engineered_features.append(features)
        
        engineered_df = pd.DataFrame(engineered_features)
        
        # Clean engineered features - REMOVE INF AND NAN
        engineered_df = engineered_df.replace([np.inf, -np.inf], np.nan)
        engineered_df = engineered_df.fillna(0)
        
        # Check for any remaining non-finite values
        if not np.isfinite(engineered_df.values).all():
            print("⚠️  Warning: Non-finite values found in engineered features")
            engineered_df = engineered_df.replace([np.inf, -np.inf], 0)
            engineered_df = engineered_df.fillna(0)
        
        # Text vectorization
        print("Creating text features...")
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.fit_transform(df['combined_text'])
        
        # Count features
        count_features = self.count_vectorizer.fit_transform(df['combined_text'])
        
        # Combine sparse features
        print("Combining features...")
        sparse_features = hstack([tfidf_features, count_features])
        
        # Reduce dimensionality
        print("Reducing dimensionality...")
        sparse_reduced = self.svd.fit_transform(sparse_features)
        
        # Prepare labels
        y_class = self.label_encoder.fit_transform(df['problem_class'])
        y_score = df['problem_score'].values
        
        # Combine all features (dense array for stability)
        X_dense = np.hstack([
            sparse_reduced,
            engineered_df.values
        ])
        
        # Final check for non-finite values
        if not np.isfinite(X_dense).all():
            print("⚠️  Cleaning non-finite values in feature matrix...")
            X_dense = np.nan_to_num(X_dense, nan=0, posinf=0, neginf=0)
        
        print(f"\n📊 Feature Summary:")
        print(f"   Feature matrix shape: {X_dense.shape}")
        print(f"   Classes: {self.label_encoder.classes_}")
        print(f"   Score range: {y_score.min():.1f} - {y_score.max():.1f}")
        print(f"   Class distribution: {dict(pd.Series(y_class).value_counts())}")
        
        return X_dense, y_class, y_score
    
    def train_simple_models(self, X, y_class, y_score):
        """Train simple but effective models"""
        print("\n🤖 Training models...")
    
    # Split data
        X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
            X, y_class, y_score, 
            test_size=0.2, 
            random_state=self.seed, 
            stratify=y_class
        )
    
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
    
    # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
    
    # Train classifier - Using Random Forest (more robust)
        print("\n📈 Training classifier...")
        start_time = time.time()
    
    # Simple Random Forest classifier
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.seed,
            n_jobs=-1
        )
    
        classifier.fit(X_train_scaled, y_train_class)
        classifier_time = time.time() - start_time
    
    # Predictions
        y_class_pred = classifier.predict(X_test_scaled)
        y_class_proba = classifier.predict_proba(X_test_scaled)
    
    # Metrics
        accuracy = accuracy_score(y_test_class, y_class_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_class, y_class_pred, average='weighted'
        )
    
        print(f"\n📋 Classification Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Training time: {classifier_time:.2f}s")
    
    # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(
            y_test_class, y_class_pred, 
            target_names=self.label_encoder.classes_
        ))
    
    # Train regressor - Using Gradient Boosting
        print("\n📈 Training regressor...")
        start_time = time.time()
    
    # Gradient Boosting Regressor (more robust than XGBoost for this)
        regressor = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=self.seed
    )
    
        regressor.fit(X_train_scaled, y_train_score)
        regressor_time = time.time() - start_time
    
    # === CORRECTED CALIBRATION SECTION ===
    # Fit calibrator on training data
        y_train_pred_raw = regressor.predict(X_train_scaled)
        self.score_calibrator.fit(y_train_score, y_train_class, self.label_encoder)
    
    # Get raw test predictions
        y_score_pred_raw = regressor.predict(X_test_scaled)
    
    # Apply calibration using predicted classes
        y_score_pred = self.score_calibrator.transform(
            y_score_pred_raw, y_class_pred, self.label_encoder
    )
    
    # === ANALYSIS OUTPUT ===
        print(f"\n🔍 Prediction Range Analysis:")
        print(f"\n   RAW PREDICTIONS (uncalibrated):")
        print(f"     Min: {y_score_pred_raw.min():.2f}")
        print(f"     Max: {y_score_pred_raw.max():.2f}")
        print(f"     Range: {y_score_pred_raw.max() - y_score_pred_raw.min():.2f}")
    
        print(f"\n   CALIBRATED PREDICTIONS:")
        print(f"     Min: {y_score_pred.min():.2f}")
        print(f"     Max: {y_score_pred.max():.2f}")
        print(f"     Range: {y_score_pred.max() - y_score_pred.min():.2f}")
    
    # Class-specific ranges
        print(f"\n   CLASS-SPECIFIC RANGES (calibrated):")
        for i, class_name in enumerate(self.label_encoder.classes_):
            mask = (y_class_pred == i)
            if np.any(mask):
                class_preds = y_score_pred[mask]
                print(f"     {class_name:10s}: [{class_preds.min():.2f}, {class_preds.max():.2f}]")
    
    # Metrics
        mse = mean_squared_error(y_test_score, y_score_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_score, y_score_pred)
        r2 = r2_score(y_test_score, y_score_pred)
    
        print(f"\n📋 Regression Results:")
        print(f"   MSE: {mse:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Training time: {regressor_time:.2f}s")
    
    # Store models
        self.classifiers['main'] = classifier
        self.regressors['main'] = regressor
    
    # Store metrics
        self.history['classifier_metrics'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': classifier_time
    }
    
        self.history['regressor_metrics'] = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'training_time': regressor_time
    }
    
    # Show sample predictions
        print("\n📊 Sample predictions:")
        samples = min(5, len(y_test_score))
        for i in range(samples):
            actual_class = self.label_encoder.inverse_transform([y_test_class[i]])[0]
            predicted_class = self.label_encoder.inverse_transform([y_class_pred[i]])[0]
            print(f"   Sample {i+1}:")
            print(f"     Actual: {actual_class} ({y_test_score[i]:.2f})")
            print(f"     Predicted (raw): {predicted_class} ({y_score_pred_raw[i]:.2f})")
            print(f"     Predicted (calibrated): {predicted_class} ({y_score_pred[i]:.2f})")
            print(f"     Error: {abs(y_test_score[i] - y_score_pred[i]):.2f}")
    
        return {
        'classifier': classifier,
        'regressor': regressor,
        'X_test': X_test_scaled,
        'y_test_class': y_test_class,
        'y_test_score': y_test_score,
        'y_pred_class': y_class_pred,
        'y_pred_score': y_score_pred,
        'y_pred_score_raw': y_score_pred_raw  # Store raw for comparison
    }

    def create_visualizations(self, df, results):
        """Create visualizations"""
        print("\n🎨 Creating visualizations...")
        
        # Create directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Class Distribution
        plt.figure(figsize=(10, 6))
        class_counts = df['problem_class'].value_counts()
        colors = ['#28a745', '#ffc107', '#dc3545']
        bars = plt.bar(class_counts.index, class_counts.values, color=colors)
        plt.title('Difficulty Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add counts on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('visualizations/class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Score Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['problem_score'], bins=20, edgecolor='black', color='#4361ee', alpha=0.7)
        plt.axvline(df['problem_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["problem_score"].mean():.2f}')
        plt.title('Difficulty Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/score_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(results['y_test_class'], results['y_pred_class'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Regression Results
        plt.figure(figsize=(10, 6))
        plt.scatter(results['y_test_score'], results['y_pred_score'], alpha=0.6, color='#ff6b6b')
        plt.plot([1, 10], [1, 10], 'k--', alpha=0.5)
        plt.title('Actual vs Predicted Scores', fontsize=14, fontweight='bold')
        plt.xlabel('Actual Score', fontsize=12)
        plt.ylabel('Predicted Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = self.history['regressor_metrics']['r2']
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('visualizations/regression_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations saved to visualizations/ directory")
    
    def save_models(self):
        """Save all models and artifacts"""
        print(f"\n💾 Saving models and artifacts...")
        
        artifacts = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'scaler': self.scaler,
            'svd': self.svd,
            'label_encoder': self.label_encoder,
            'score_calibrator': self.score_calibrator,
            'classifier': self.classifiers['main'],
            'regressor': self.regressors['main']
        }
        
        for name, artifact in artifacts.items():
            try:
                filepath = f'models/{name}.pkl'
                joblib.dump(artifact, filepath, compress=3)
                size = os.path.getsize(filepath) / 1024 / 1024
                print(f"  ✅ {name:20s} - {size:.2f} MB")
            except Exception as e:
                print(f"  ❌ {name:20s} - Error: {str(e)[:50]}")
        
        # Save metadata
        metadata = {
            'model_version': '1.0.0',
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'classes': list(self.label_encoder.classes_),
            'metrics': {
                'classifier': self.history.get('classifier_metrics', {}),
                'regressor': self.history.get('regressor_metrics', {})
            },
            'config': {
                'seed': self.seed,
                'svd_components': self.svd.n_components,
                'tfidf_features': self.tfidf_vectorizer.max_features
            }
        }
        
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved to models/metadata.json")
        
        # Save training report
        report = f"""
        ===========================================
        ROBUST MODEL TRAINING REPORT
        ===========================================
        
        Training Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
        Model Version: 1.0.0
        
        CLASSIFICATION RESULTS:
        ---------------------
        Accuracy:  {self.history['classifier_metrics'].get('accuracy', 0):.4f}
        Precision: {self.history['classifier_metrics'].get('precision', 0):.4f}
        Recall:    {self.history['classifier_metrics'].get('recall', 0):.4f}
        F1-Score:  {self.history['classifier_metrics'].get('f1', 0):.4f}
        Time:      {self.history['classifier_metrics'].get('training_time', 0):.2f}s
        
        REGRESSION RESULTS:
        ------------------
        MSE:  {self.history['regressor_metrics'].get('mse', 0):.4f}
        RMSE: {self.history['regressor_metrics'].get('rmse', 0):.4f}
        MAE:  {self.history['regressor_metrics'].get('mae', 0):.4f}
        R²:   {self.history['regressor_metrics'].get('r2', 0):.4f}
        Time: {self.history['regressor_metrics'].get('training_time', 0):.2f}s
        
        MODEL ARCHITECTURE:
        ------------------
        Classifier: Random Forest
        Regressor: Gradient Boosting
        Feature Engineering: TF-IDF + Count Vectors + Custom Features
        Dimensionality Reduction: SVD (50 components)
        
        FILES GENERATED:
        ------------------
        • models/ - All trained models and artifacts
        • visualizations/ - Analysis plots
        
        NEXT STEPS:
        ----------
        1. Test the model with sample problems
        2. Deploy using app.py
        """
        
        with open('models/training_report.txt', 'w') as f:
            f.write(report)
        
        print(f"✅ Training report saved to models/training_report.txt")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("="*70)
        print("🤖 ROBUST CP DIFFICULTY PREDICTOR - TRAINING")
        print("="*70)
        
        total_start_time = time.time()
        
        # 1. Load data
        df = self.load_data('../data/dataset.jsonl')
        if df is None or len(df) < 10:
            print("❌ Insufficient data for training (need at least 10 samples)")
            print("\n📝 Creating a sample dataset for testing...")
            self.create_sample_dataset()
            df = self.load_data('../data/dataset.jsonl')
        
        if df is None or len(df) == 0:
            print("❌ Failed to load data. Exiting...")
            return
        
        # 2. Show dataset info
        print(f"\n📊 Dataset Information:")
        print(f"   Total samples: {len(df)}")
        print(f"   Classes: {df['problem_class'].unique().tolist()}")
        print(f"   Score range: {df['problem_score'].min():.1f} - {df['problem_score'].max():.1f}")
        print(f"   Class distribution:")
        for cls, count in df['problem_class'].value_counts().items():
            print(f"     {cls}: {count}")
        
        # 3. Prepare features
        X, y_class, y_score = self.prepare_features(df)
        
        # 4. Train models
        results = self.train_simple_models(X, y_class, y_score)
        
        # 5. Create visualizations
        self.create_visualizations(df, results)
        
        # 6. Save models
        self.save_models()
        
        total_time = time.time() - total_start_time
        
        # Final report
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        print(f"⏱️  Total Training Time: {total_time:.2f} seconds")
        print(f"📊 Final Performance Metrics:")
        print(f"\n📈 CLASSIFICATION:")
        print(f"   Accuracy:  {self.history['classifier_metrics']['accuracy']:.4f}")
        print(f"   F1-Score:  {self.history['classifier_metrics']['f1']:.4f}")
        
        print(f"\n📈 REGRESSION:")
        print(f"   RMSE: {self.history['regressor_metrics']['rmse']:.4f}")
        print(f"   R²:   {self.history['regressor_metrics']['r2']:.4f}")
        
        print(f"\n📁 Models saved to: models/")
        print(f"📈 Visualizations saved to: visualizations/")
        
        print(f"\n🚀 To test the model:")
        print(f"   1. Run: python test_model.py")
        print(f"   2. Or deploy with: python app.py")
        print("="*70)
    
    def create_sample_dataset(self):
        """Create a sample dataset if none exists"""
        print("Creating sample dataset...")
        
        sample_data = [
            {
                "title": "Sum of Two Numbers",
                "description": "Given two integers, calculate their sum.",
                "problem_class": "Easy",
                "problem_score": 2.0
            },
            {
                "title": "Find Maximum in Array",
                "description": "Given an array of integers, find the maximum value.",
                "problem_class": "Easy",
                "problem_score": 2.5
            },
            {
                "title": "Binary Search",
                "description": "Implement binary search to find an element in a sorted array.",
                "problem_class": "Medium",
                "problem_score": 5.0
            },
            {
                "title": "BFS Graph Traversal",
                "description": "Implement breadth-first search on a graph represented as adjacency list.",
                "problem_class": "Medium",
                "problem_score": 5.5
            },
            {
                "title": "Dynamic Programming - Knapsack",
                "description": "Solve the 0/1 knapsack problem using dynamic programming.",
                "problem_class": "Hard",
                "problem_score": 8.0
            },
            {
                "title": "Dijkstra Shortest Path",
                "description": "Find shortest path in weighted graph using Dijkstra's algorithm.",
                "problem_class": "Hard",
                "problem_score": 8.5
            }
        ]
        
        # Add more variations
        for i in range(4):
            sample_data.append({
                "title": f"Simple Problem {i+1}",
                "description": f"Basic programming problem for beginners. This is example {i+1}.",
                "problem_class": "Easy",
                "problem_score": 1.5 + i * 0.5
            })
        
        for i in range(3):
            sample_data.append({
                "title": f"Intermediate Problem {i+1}",
                "description": f"Medium difficulty problem involving algorithms and data structures. Requires O(n log n) solution.",
                "problem_class": "Medium",
                "problem_score": 4.5 + i * 0.5
            })
        
        for i in range(3):
            sample_data.append({
                "title": f"Advanced Problem {i+1}",
                "description": f"Hard problem requiring advanced algorithms like dynamic programming or graph theory. Constraints: n ≤ 10^5.",
                "problem_class": "Hard",
                "problem_score": 7.5 + i * 0.5
            })
        
        
        # Save to file
        with open('../data/dataset.jsonl', 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"✅ Created sample dataset with {len(sample_data)} problems")
        print("   You can add your own problems to dataset/dataset.jsonl")

def main():
    """Main execution function"""
    trainer = RobustModelTrainer(seed=42)
    trainer.run_training_pipeline()

if __name__ == '__main__':
    main()