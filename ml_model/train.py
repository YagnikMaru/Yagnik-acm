import pandas as pd
import numpy as np
import joblib
import re
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, using sklearn models only")

# Theoretical score ranges (soft guidance, not hard limits)
THEORETICAL_RANGES = {
    'Easy': (1, 3),
    'Medium': (3, 6),
    'Hard': (6, 9)
}


def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_score_discriminative_features(text):
    """Extract features that DIFFERENTIATE scores within same class"""
    features = {}
    
    if not text or not isinstance(text, str):
        text = ""
    
    words = text.split()
    
    # Basic text metrics
    features['text_length'] = len(text)
    features['word_count'] = len(words)
    features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
    
    word_lengths = [len(w) for w in words]
    features['avg_word_length'] = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    
    # Complexity depth indicators
    features['long_words'] = sum(1 for w in words if len(w) > 8)
    features['very_long_words'] = sum(1 for w in words if len(w) > 12)
    
    # Technical density (MORE GRANULAR)
    tech_basic = ['algorithm', 'function', 'array', 'string', 'number', 'sort']
    tech_intermediate = ['complexity', 'optimize', 'efficient', 'data structure', 'hash', 'tree']
    tech_advanced = ['dynamic programming', 'greedy', 'backtrack', 'divide conquer', 'bit manipulation']
    
    features['tech_basic_count'] = sum(term in text for term in tech_basic)
    features['tech_intermediate_count'] = sum(term in text for term in tech_intermediate)
    features['tech_advanced_count'] = sum(term in text for term in tech_advanced)
    features['tech_total_density'] = features['tech_basic_count'] + features['tech_intermediate_count'] + features['tech_advanced_count']
    
    # Constraint magnitude (CRITICAL for score discrimination)
    numbers = re.findall(r'\d+', text)
    if numbers:
        try:
            num_values = [int(n) for n in numbers if len(n) < 10]  # Avoid overflow
            num_values = [n for n in num_values if n > 0]
            
            if num_values:
                import math
                features['max_constraint'] = math.log10(max(num_values) + 1)
                features['avg_constraint'] = math.log10(sum(num_values) / len(num_values) + 1)
                features['constraint_count'] = len(num_values)
                features['large_constraints'] = sum(1 for n in num_values if n > 10000)
            else:
                features['max_constraint'] = 0.0
                features['avg_constraint'] = 0.0
                features['constraint_count'] = 0
                features['large_constraints'] = 0
        except (ValueError, OverflowError):
            features['max_constraint'] = 0.0
            features['avg_constraint'] = 0.0
            features['constraint_count'] = 0
            features['large_constraints'] = 0
    else:
        features['max_constraint'] = 0.0
        features['avg_constraint'] = 0.0
        features['constraint_count'] = 0
        features['large_constraints'] = 0
    
    # Problem scope indicators
    features['has_edge_cases'] = int(any(term in text for term in ['edge case', 'corner case', 'boundary']))
    features['has_optimization'] = int(any(term in text for term in ['optimize', 'minimize', 'maximize', 'efficient']))
    features['has_multiple_conditions'] = int(any(term in text for term in ['if', 'else', 'condition', 'case']))
    
    # Algorithmic complexity markers
    algo_patterns = ['time complexity', 'space complexity', 'o(n)', 'o(log', 'o(n^2)', 'o(nlogn)']
    features['complexity_mentions'] = sum(pattern in text for pattern in algo_patterns)
    
    # Data structure variety (more = harder)
    ds_basic = ['array', 'list', 'string']
    ds_intermediate = ['hash', 'map', 'set', 'dictionary', 'stack', 'queue']
    ds_advanced = ['tree', 'graph', 'heap', 'trie', 'segment tree']
    
    features['ds_basic'] = sum(ds in text for ds in ds_basic)
    features['ds_intermediate'] = sum(ds in text for ds in ds_intermediate)
    features['ds_advanced'] = sum(ds in text for ds in ds_advanced)
    features['ds_variety'] = features['ds_basic'] + features['ds_intermediate'] + features['ds_advanced']
    
    # Mathematical density
    math_terms = ['sum', 'product', 'modulo', 'prime', 'factorial', 'permutation', 'combination']
    features['math_density'] = sum(term in text for term in math_terms)
    
    # Explanation depth (longer detailed explanations = harder)
    features['explanation_depth'] = text.count('because') + text.count('since') + text.count('therefore')
    
    # Multiple test cases indicator
    features['test_case_count'] = text.count('example') + text.count('test case')
    
    return features


def load_dataset(file_path):
    """Load dataset from JSONL file"""
    print(f"\n📂 Loading dataset from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return None
    
    try:
        df = pd.read_json(file_path, lines=True)
        print(f"✓ Loaded {len(df)} samples")
        
        required_cols = ['problem_class', 'problem_score']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
            return None
        
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def prepare_features_global(df, vectorizer=None, count_vectorizer=None, fit=True):
    """Feature preparation for GLOBAL score regression"""
    print(f"\n{'='*70}")
    print(f"🔧 GLOBAL FEATURE PREPARATION (Class as Input Feature)")
    print(f"{'='*70}")
    
    df = df.copy()
    
    # Combine and clean text - handle NaN in source columns
    df['combined_text'] = df.apply(
        lambda row: f"{str(row.get('title', '') or '')} {str(row.get('description', '') or '')} "
                   f"{str(row.get('input', '') or '')} {str(row.get('output', '') or '')}",
        axis=1
    )
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Ensure no empty strings
    df['cleaned_text'] = df['cleaned_text'].replace('', 'empty')
    df['cleaned_text'] = df['cleaned_text'].fillna('empty')
    
    # TF-IDF features
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True
        )
        tfidf_features = vectorizer.fit_transform(df['cleaned_text'])
    else:
        tfidf_features = vectorizer.transform(df['cleaned_text'])
    
    # Count features
    if fit:
        count_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            binary=True,
            min_df=1
        )
        count_features = count_vectorizer.fit_transform(df['cleaned_text'])
    else:
        count_features = count_vectorizer.transform(df['cleaned_text'])
    
    # Score-discriminative engineered features
    score_features_list = []
    for text in df['cleaned_text']:
        score_features = extract_score_discriminative_features(text)
        score_features_list.append(score_features)
    
    score_df = pd.DataFrame(score_features_list)
    
    # Fill any NaN values with 0
    score_df = score_df.fillna(0)
    
    # Verify no NaN or inf values
    score_df = score_df.replace([np.inf, -np.inf], 0)
    
    # CRITICAL: Add class as ordinal feature (Easy=0, Medium=1, Hard=2)
    if 'problem_class' in df.columns:
        class_mapping = {'Easy': 0, 'Medium': 1, 'Hard': 2}
        score_df['class_ordinal'] = df['problem_class'].map(class_mapping).fillna(0).astype(int)
        
        # Also add one-hot encoding for class
        score_df['class_is_easy'] = (df['problem_class'] == 'Easy').astype(int)
        score_df['class_is_medium'] = (df['problem_class'] == 'Medium').astype(int)
        score_df['class_is_hard'] = (df['problem_class'] == 'Hard').astype(int)
    
    # Fill any remaining NaN values
    score_df = score_df.fillna(0)
    score_df = score_df.replace([np.inf, -np.inf], 0)
    
    # Interaction features (class × complexity)
    if 'class_ordinal' in score_df.columns:
        score_df['class_tech_interaction'] = score_df['class_ordinal'] * score_df['tech_total_density']
        score_df['class_constraint_interaction'] = score_df['class_ordinal'] * score_df['max_constraint']
        
        # Fill any NaN from interactions
        score_df['class_tech_interaction'] = score_df['class_tech_interaction'].fillna(0)
        score_df['class_constraint_interaction'] = score_df['class_constraint_interaction'].fillna(0)
    
    feature_names = score_df.columns.tolist()
    engineered_sparse = csr_matrix(score_df.values)
    
    # Combine all features
    X = hstack([tfidf_features, count_features, engineered_sparse])
    
    # CRITICAL: Remove any NaN or Inf values from sparse matrix
    X = X.tocsr()  # Ensure CSR format
    X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Verify no NaN values
    if np.any(np.isnan(X.data)):
        print("⚠️  WARNING: NaN values detected after cleaning, replacing with 0")
        X.data[np.isnan(X.data)] = 0.0
    
    print(f"✓ Total features: {X.shape[1]}")
    print(f"  - TF-IDF: {tfidf_features.shape[1]}")
    print(f"  - Count: {count_features.shape[1]}")
    print(f"  - Engineered (with class): {engineered_sparse.shape[1]}")
    print(f"✓ NaN check passed")
    
    y_class = None
    y_score = None
    
    if 'problem_class' in df.columns:
        label_encoder = LabelEncoder()
        y_class = label_encoder.fit_transform(df['problem_class'])
    
    if 'problem_score' in df.columns:
        y_score = df['problem_score'].values
    
    return X, y_class, y_score, vectorizer, count_vectorizer, feature_names


def calculate_variance_penalty(y_pred, y_true):
    """Penalize models with collapsed variance"""
    pred_std = y_pred.std()
    true_std = y_true.std()
    
    # Penalty if prediction std is much lower than true std
    variance_ratio = pred_std / max(true_std, 0.1)
    penalty = max(0, 1 - variance_ratio)  # 0 if good variance, 1 if collapsed
    
    return penalty


def composite_score(mse, mae, variance_penalty):
    """Composite scoring that penalizes low variance"""
    # Lower is better
    return 0.5 * mse + 0.3 * mae + 0.2 * variance_penalty


def train_classifier(X_train, y_class_train, X_test, y_class_test, class_names):
    """Train the difficulty class classifier"""
    print(f"\n{'='*70}")
    print(f"🎯 STAGE 1: CLASSIFIER TRAINING")
    print(f"{'='*70}")
    
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    classifier.fit(X_train, y_class_train)
    
    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(classifier, cv=3, method='sigmoid')
    calibrated.fit(X_train, y_class_train)
    
    # Evaluation
    y_pred = calibrated.predict(X_test)
    accuracy = accuracy_score(y_class_test, y_pred)
    
    print(f"✓ Classification Accuracy: {accuracy:.4f}")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_class_test, y_pred, target_names=class_names))
    
    return calibrated


def train_global_regressor(X_train, y_score_train, X_test, y_score_test):
    """Train GLOBAL regressor with variance-aware selection"""
    print(f"\n{'='*70}")
    print(f"📊 STAGE 2: GLOBAL REGRESSOR TRAINING")
    print(f"{'='*70}")
    
    print(f"\nTraining on ALL samples globally")
    print(f"Samples: {len(y_score_train)}")
    print(f"Score range: [{y_score_train.min():.2f}, {y_score_train.max():.2f}]")
    print(f"Score std: {y_score_train.std():.3f}")
    
    # Candidate models
    models = []
    
    # Gradient Boosting with less regularization
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        loss='huber'  # More robust than MSE
    )
    models.append(('GradientBoosting', gb_model))
    
    # Random Forest with less regularization
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    models.append(('RandomForest', rf_model))
    
    if XGBOOST_AVAILABLE:
        xgb_model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )
        models.append(('XGBoost', xgb_model))
    
    # Train and select with variance-aware scoring
    best_model = None
    best_composite = float('inf')
    best_name = None
    
    print(f"\nEvaluating models with variance-aware selection:")
    print(f"{'Model':<20} {'MSE':<10} {'MAE':<10} {'Var Penalty':<12} {'Composite':<10}")
    print(f"{'-'*70}")
    
    for name, model in models:
        try:
            # Train
            model.fit(X_train, y_score_train)
            
            # Predict on test
            y_pred_test = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_score_test, y_pred_test)
            mae = mean_absolute_error(y_score_test, y_pred_test)
            var_penalty = calculate_variance_penalty(y_pred_test, y_score_test)
            comp_score = composite_score(mse, mae, var_penalty)
            
            print(f"{name:<20} {mse:<10.4f} {mae:<10.4f} {var_penalty:<12.4f} {comp_score:<10.4f}")
            
            # Check for variance collapse
            pred_std = y_pred_test.std()
            if pred_std < 0.3:
                print(f"  ⚠️  WARNING: Low variance (std={pred_std:.3f})")
            
            if comp_score < best_composite:
                best_composite = comp_score
                best_model = model
                best_name = name
                
        except Exception as e:
            print(f"{name:<20} FAILED: {e}")
    
    if best_model is None:
        raise ValueError("All models failed!")
    
    print(f"\n✓ Selected: {best_name} (Composite={best_composite:.4f})")
    
    # Analyze predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    print(f"\n📊 Prediction Analysis:")
    print(f"  Training:")
    print(f"    Predicted range: [{y_pred_train.min():.2f}, {y_pred_train.max():.2f}]")
    print(f"    Predicted std: {y_pred_train.std():.3f}")
    print(f"  Testing:")
    print(f"    Predicted range: [{y_pred_test.min():.2f}, {y_pred_test.max():.2f}]")
    print(f"    Predicted std: {y_pred_test.std():.3f}")
    print(f"    True std: {y_score_test.std():.3f}")
    
    return best_model


def calibrate_scores_percentile(y_pred, y_true_train):
    """Calibrate predictions using percentile matching"""
    # Map predicted percentiles to true percentiles
    pred_percentiles = np.percentile(y_pred, range(0, 101))
    true_percentiles = np.percentile(y_true_train, range(0, 101))
    
    # Interpolate
    calibrated = np.interp(y_pred, pred_percentiles, true_percentiles)
    
    return calibrated


def soft_constrain_by_class(scores, classes, class_names):
    """Apply SOFT constraints based on class (not hard clipping)"""
    constrained = scores.copy()
    
    for class_idx, class_name in enumerate(class_names):
        mask = (classes == class_idx)
        if mask.sum() == 0:
            continue
        
        class_scores = scores[mask]
        target_range = THEORETICAL_RANGES[class_name]
        
        # Soft shrinkage towards range (not hard clip)
        center = (target_range[0] + target_range[1]) / 2
        width = (target_range[1] - target_range[0]) / 2
        
        # Shrink extreme deviations
        deviations = class_scores - center
        shrinkage_factor = 0.8  # Allow some overflow
        constrained[mask] = center + deviations * shrinkage_factor
        
        # Only clip extreme outliers (3x range)
        extreme_min = target_range[0] - width
        extreme_max = target_range[1] + width
        constrained[mask] = np.clip(constrained[mask], extreme_min, extreme_max)
    
    return constrained


def evaluate_model(classifier, regressor, X_test, y_class_test, y_score_test, class_names):
    """Evaluate the complete model"""
    print(f"\n{'='*70}")
    print(f"📈 MODEL EVALUATION")
    print(f"{'='*70}")
    
    # Classification
    y_class_pred = classifier.predict(X_test)
    class_acc = accuracy_score(y_class_test, y_class_pred)
    print(f"\n✓ Classification Accuracy: {class_acc:.4f}")
    
    # Regression
    y_score_pred_raw = regressor.predict(X_test)
    
    # Apply soft constraints
    y_score_pred = soft_constrain_by_class(y_score_pred_raw, y_class_pred, class_names)
    
    # Metrics
    mse = mean_squared_error(y_score_test, y_score_pred)
    mae = mean_absolute_error(y_score_test, y_score_pred)
    r2 = r2_score(y_score_test, y_score_pred)
    
    print(f"\n✓ Score Prediction:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²:  {r2:.4f}")
    
    # Variance analysis
    print(f"\n📊 Variance Analysis:")
    print(f"  True std: {y_score_test.std():.3f}")
    print(f"  Predicted std: {y_score_pred.std():.3f}")
    print(f"  Variance ratio: {y_score_pred.std() / y_score_test.std():.3f}")
    
    # Per-class analysis
    print(f"\n📊 Per-Class Performance:")
    for class_idx, class_name in enumerate(class_names):
        mask = (y_class_test == class_idx)
        if mask.sum() > 0:
            class_true = y_score_test[mask]
            class_pred = y_score_pred[mask]
            
            class_mse = mean_squared_error(class_true, class_pred)
            class_mae = mean_absolute_error(class_true, class_pred)
            
            print(f"\n  {class_name}:")
            print(f"    Samples: {mask.sum()}")
            print(f"    True range: [{class_true.min():.2f}, {class_true.max():.2f}]")
            print(f"    Pred range: [{class_pred.min():.2f}, {class_pred.max():.2f}]")
            print(f"    True std: {class_true.std():.3f}")
            print(f"    Pred std: {class_pred.std():.3f}")
            print(f"    MSE: {class_mse:.4f}")
            print(f"    MAE: {class_mae:.4f}")
    
    return {
        'classification_accuracy': class_acc,
        'score_mse': mse,
        'score_mae': mae,
        'score_r2': r2,
        'variance_ratio': y_score_pred.std() / y_score_test.std()
    }


def save_models(classifier, regressor, vectorizer, count_vectorizer,
                label_encoder, feature_names, metrics):
    """Save all models and metadata"""
    print(f"\n{'='*70}")
    print(f"💾 SAVING MODELS")
    print(f"{'='*70}")
    
    os.makedirs('../models', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save models
    joblib.dump(classifier, f'../models/classifier_{timestamp}.pkl')
    joblib.dump(regressor, f'../models/global_regressor_{timestamp}.pkl')
    joblib.dump(vectorizer, f'../models/tfidf_vectorizer_{timestamp}.pkl')
    joblib.dump(count_vectorizer, f'../models/count_vectorizer_{timestamp}.pkl')
    joblib.dump(label_encoder, f'../models/label_encoder_{timestamp}.pkl')
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'metrics': metrics,
        'theoretical_ranges': THEORETICAL_RANGES,
        'model_type': 'global_regressor'
    }
    
    with open(f'../models/metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ All models saved with timestamp: {timestamp}")


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🚀 FIXED TRAINING PIPELINE - DIVERSE SCORE PREDICTION")
    print("="*70)
    
    # Load dataset
    df = load_dataset('../data/dataset.jsonl')
    if df is None or len(df) < 20:
        print("\n❌ Insufficient data")
        return
    
    # Analyze score distribution
    print(f"\n📊 Score Distribution Analysis:")
    for class_name in ['Easy', 'Medium', 'Hard']:
        class_df = df[df['problem_class'] == class_name]
        if len(class_df) > 0:
            scores = class_df['problem_score'].values
            print(f"\n{class_name}:")
            print(f"  Samples: {len(class_df)}")
            print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
            print(f"  Score mean: {scores.mean():.2f}")
            print(f"  Score std: {scores.std():.3f}")
            print(f"  Unique scores: {len(np.unique(scores))}")
    
    # Prepare features WITH class as input
    X, y_class, y_score, vectorizer, count_vectorizer, feature_names = prepare_features_global(
        df, fit=True
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(['Easy', 'Medium', 'Hard'])
    class_names = label_encoder.classes_
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing:  {X_test.shape[0]} samples")
    
    # Train classifier
    classifier = train_classifier(X_train, y_class_train, X_test, y_class_test, class_names)
    
    # Train GLOBAL regressor
    regressor = train_global_regressor(X_train, y_score_train, X_test, y_score_test)
    
    # Evaluate
    metrics = evaluate_model(
        classifier, regressor,
        X_test, y_class_test, y_score_test, class_names
    )
    
    # Save models
    save_models(
        classifier, regressor,
        vectorizer, count_vectorizer, label_encoder,
        feature_names, metrics
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\n💡 Key Improvements:")
    print("   ✓ Global regressor with class as feature")
    print("   ✓ Variance-aware model selection")
    print("   ✓ Soft constraints instead of hard clipping")
    print("   ✓ Enhanced score-discriminative features")
    print("   ✓ Composite loss (MSE + MAE + Variance)")
    print("="*70)


if __name__ == '__main__':
    main()