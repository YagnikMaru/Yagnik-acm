"""
Production Flask API for CP Difficulty Predictor
COMPATIBLE WITH GLOBAL REGRESSOR TRAINING SCRIPT

ARCHITECTURE:
    STAGE 1: Classify into Easy/Medium/Hard with probabilities
    STAGE 2: GLOBAL regressor predicts score (uses class as feature)
    STAGE 3: Soft constraint applied based on predicted class
    
Author: Senior ML Engineer
Version: 5.0.0 - Fixed for Global Regressor
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import re
import os
import json
import traceback
import math
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# FLASK APP SETUP
# ============================================================================
app = Flask(__name__, 
            template_folder='../web_app/templates',
            static_folder='../web_app/static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


# ============================================================================
# GLOBAL MODEL ARTIFACTS
# ============================================================================
class ModelArtifacts:
    """Container for all trained model artifacts"""
    def __init__(self):
        # Preprocessing
        self.vectorizer = None
        self.count_vectorizer = None
        self.label_encoder = None
        
        # Models
        self.classifier = None
        self.regressor = None  # SINGLE GLOBAL REGRESSOR
        
        # Metadata
        self.class_names = []
        self.feature_names = []
        self.theoretical_ranges = {}
        self.timestamp = None
        
        # Status
        self.is_loaded = False
        self.load_timestamp = None

# Global instance
artifacts = ModelArtifacts()


# ============================================================================
# THEORETICAL SCORE RANGES (MATCHES TRAINING)
# ============================================================================
THEORETICAL_RANGES = {
    'Easy': (1, 3),
    'Medium': (3, 6),
    'Hard': (6, 9)
}


# ============================================================================
# TEXT PREPROCESSING - IDENTICAL TO TRAINING
# ============================================================================
def clean_text(text):
    """
    Clean and preprocess text.
    MUST be 100% identical to training version.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_score_discriminative_features(text):
    """
    Extract features that DIFFERENTIATE scores within same class.
    MUST be 100% identical to training version.
    """
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


# ============================================================================
# MODEL LOADING
# ============================================================================
def find_latest_models(base_dir='../models'):
    """
    Find the latest trained models by timestamp.
    Returns the timestamp string or None.
    """
    if not os.path.exists(base_dir):
        return None
    
    # Look for metadata files
    metadata_files = [f for f in os.listdir(base_dir) if f.startswith('metadata_') and f.endswith('.json')]
    
    if not metadata_files:
        return None
    
    # Sort by timestamp (filename format: metadata_YYYYMMDD_HHMMSS.json)
    metadata_files.sort(reverse=True)
    latest = metadata_files[0]
    
    # Extract timestamp
    timestamp = latest.replace('metadata_', '').replace('.json', '')
    return timestamp


def load_models(model_dir='../models', timestamp=None):
    """
    Load all trained model artifacts (GLOBAL REGRESSOR VERSION).
    
    Args:
        model_dir: Directory containing model files
        timestamp: Specific timestamp to load (format: YYYYMMDD_HHMMSS)
                  If None, loads the latest models
    """
    global artifacts
    
    print(f"\n{'='*70}")
    print(f"🔍 LOADING MODELS (Global Regressor Version)")
    print(f"{'='*70}")
    
    # Find model directory
    possible_paths = [
        model_dir,
        '../models',
        './models',
        '../ml_model/models',
        './ml_model/models',
        os.path.join(os.path.dirname(__file__), '..', 'models'),
        os.path.join(os.path.dirname(__file__), 'models')
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ Model directory not found!")
        print("   Searched in:")
        for path in possible_paths[:5]:
            print(f"   - {os.path.abspath(path)}")
        return False
    
    print(f"✓ Found model directory: {os.path.abspath(model_path)}")
    
    # Find timestamp
    if timestamp is None:
        timestamp = find_latest_models(model_path)
        if timestamp is None:
            print("❌ No trained models found!")
            print(f"   Please run: python train.py")
            return False
        print(f"✓ Using latest models: {timestamp}")
    else:
        print(f"✓ Using specified timestamp: {timestamp}")
    
    try:
        # Load metadata first
        print("\n📋 Loading metadata...")
        metadata_file = os.path.join(model_path, f'metadata_{timestamp}.json')
        
        if not os.path.exists(metadata_file):
            print(f"❌ Metadata file not found: {metadata_file}")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        artifacts.feature_names = metadata['feature_names']
        artifacts.theoretical_ranges = metadata.get('theoretical_ranges', THEORETICAL_RANGES)
        artifacts.timestamp = timestamp
        model_type = metadata.get('model_type', 'unknown')
        
        print(f"✓ Metadata loaded")
        print(f"  Features: {len(artifacts.feature_names)}")
        print(f"  Model type: {model_type}")
        
        if model_type != 'global_regressor':
            print(f"⚠️  WARNING: Expected 'global_regressor', got '{model_type}'")
            print(f"   This API requires models trained with the NEW global regressor script")
        
        # Load preprocessing
        print("\n📦 Loading preprocessing artifacts...")
        artifacts.vectorizer = joblib.load(
            os.path.join(model_path, f'tfidf_vectorizer_{timestamp}.pkl')
        )
        artifacts.count_vectorizer = joblib.load(
            os.path.join(model_path, f'count_vectorizer_{timestamp}.pkl')
        )
        artifacts.label_encoder = joblib.load(
            os.path.join(model_path, f'label_encoder_{timestamp}.pkl')
        )
        artifacts.class_names = list(artifacts.label_encoder.classes_)
        
        print(f"✓ Vectorizers and encoder loaded")
        print(f"  Classes: {artifacts.class_names}")
        
        # Load classifier
        print("\n🎯 Loading STAGE 1 (Classifier)...")
        artifacts.classifier = joblib.load(
            os.path.join(model_path, f'classifier_{timestamp}.pkl')
        )
        print(f"✓ Classifier: {artifacts.classifier.__class__.__name__}")
        
        # Verify predict_proba
        if not hasattr(artifacts.classifier, 'predict_proba'):
            print("⚠️  WARNING: Classifier missing predict_proba!")
            return False
        
        # Load GLOBAL regressor
        print("\n📊 Loading STAGE 2 (Global Regressor)...")
        regressor_file = os.path.join(model_path, f'global_regressor_{timestamp}.pkl')
        
        if os.path.exists(regressor_file):
            artifacts.regressor = joblib.load(regressor_file)
            print(f"✓ Global Regressor loaded: {artifacts.regressor.__class__.__name__}")
        else:
            print(f"❌ Global regressor file not found: {regressor_file}")
            print(f"   This API requires the NEW training script with global regressor")
            return False
        
        artifacts.is_loaded = True
        artifacts.load_timestamp = datetime.now()
        
        print(f"\n{'='*70}")
        print(f"✅ ALL MODELS LOADED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Timestamp: {timestamp}")
        print(f"Architecture: Global Regressor (class as feature)")
        print(f"Classes: {artifacts.class_names}")
        print(f"Features: {len(artifacts.feature_names)}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR loading models: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# PREDICTION PIPELINE - MATCHES NEW TRAINING
# ============================================================================
def prepare_features(title, description, input_desc="", output_desc="", predicted_class=None):
    """
    Prepare features exactly as in training.
    CRITICAL: Must include class features for global regressor.
    
    Args:
        predicted_class: The predicted class name (Easy/Medium/Hard) - REQUIRED for score prediction
    """
    # Combine text
    combined_text = f"{str(title or '')} {str(description or '')} {str(input_desc or '')} {str(output_desc or '')}"
    cleaned_text = clean_text(combined_text)
    
    # Handle empty strings
    if not cleaned_text:
        cleaned_text = 'empty'
    
    # Extract score-discriminative features
    score_dict = extract_score_discriminative_features(cleaned_text)
    
    # CRITICAL: Add class features if provided (needed for score prediction)
    if predicted_class is not None:
        class_mapping = {'Easy': 0, 'Medium': 1, 'Hard': 2}
        score_dict['class_ordinal'] = class_mapping.get(predicted_class, 0)
        score_dict['class_is_easy'] = int(predicted_class == 'Easy')
        score_dict['class_is_medium'] = int(predicted_class == 'Medium')
        score_dict['class_is_hard'] = int(predicted_class == 'Hard')
        
        # Interaction features
        score_dict['class_tech_interaction'] = score_dict['class_ordinal'] * score_dict['tech_total_density']
        score_dict['class_constraint_interaction'] = score_dict['class_ordinal'] * score_dict['max_constraint']
    
    # Create DataFrame with all features
    feature_df = pd.DataFrame([score_dict])
    
    # Fill missing features with 0
    for fname in artifacts.feature_names:
        if fname not in feature_df.columns:
            feature_df[fname] = 0.0
    
    # Fill NaN values
    feature_df = feature_df.fillna(0)
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    
    # CRITICAL: Reorder to match training
    feature_df = feature_df[artifacts.feature_names]
    
    # TF-IDF features
    tfidf_features = artifacts.vectorizer.transform([cleaned_text])
    
    # Count features
    count_features = artifacts.count_vectorizer.transform([cleaned_text])
    
    # Combine: [TF-IDF | Count | Engineered (with class)]
    engineered_sparse = csr_matrix(feature_df.values)
    X = hstack([tfidf_features, count_features, engineered_sparse])
    
    # Clean sparse matrix
    X = X.tocsr()
    X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, cleaned_text


def soft_constrain_by_class(score, predicted_class):
    """
    Apply SOFT constraints based on class (not hard clipping).
    Matches training logic.
    """
    target_range = THEORETICAL_RANGES[predicted_class]
    
    # Soft shrinkage towards range (not hard clip)
    center = (target_range[0] + target_range[1]) / 2
    width = (target_range[1] - target_range[0]) / 2
    
    # Shrink extreme deviations
    deviation = score - center
    shrinkage_factor = 0.8  # Allow some overflow
    constrained = center + deviation * shrinkage_factor
    
    # Only clip extreme outliers (3x range)
    extreme_min = target_range[0] - width
    extreme_max = target_range[1] + width
    constrained = np.clip(constrained, extreme_min, extreme_max)
    
    return float(constrained)


def predict_difficulty(title, description, input_desc="", output_desc=""):
    """
    Two-stage prediction pipeline with GLOBAL regressor.
    
    STAGE 1: Classify → Easy/Medium/Hard + probabilities
    STAGE 2: Global regressor → Score (uses class as feature)
    STAGE 3: Soft constraint → Final score
    """
    if not artifacts.is_loaded:
        raise RuntimeError("Models not loaded")
    
    # ========================================
    # STAGE 1: CLASSIFY (without class features)
    # ========================================
    X_classify, cleaned_text = prepare_features(title, description, input_desc, output_desc, predicted_class=None)
    
    class_probs = artifacts.classifier.predict_proba(X_classify)[0]
    predicted_class_idx = int(np.argmax(class_probs))
    predicted_class = artifacts.class_names[predicted_class_idx]
    confidence = float(class_probs[predicted_class_idx])
    
    # Build probability dict
    class_prob_dict = {
        artifacts.class_names[i]: round(float(class_probs[i]), 4)
        for i in range(len(artifacts.class_names))
    }
    
    # ========================================
    # STAGE 2: PREDICT SCORE (with class features)
    # ========================================
    # Prepare features WITH predicted class
    X_score, _ = prepare_features(title, description, input_desc, output_desc, predicted_class=predicted_class)
    
    try:
        # Global regressor prediction
        predicted_score_raw = artifacts.regressor.predict(X_score)[0]
        
        # Apply soft constraints
        predicted_score = soft_constrain_by_class(predicted_score_raw, predicted_class)
        
        regressor_used = True
        
    except Exception as e:
        print(f"⚠️  Regressor prediction failed: {e}")
        traceback.print_exc()
        
        # Fallback to class mean
        score_range = THEORETICAL_RANGES[predicted_class]
        predicted_score = (score_range[0] + score_range[1]) / 2.0
        confidence *= 0.7
        regressor_used = False
    
    # ========================================
    # BUILD RESPONSE
    # ========================================
    score_range = THEORETICAL_RANGES[predicted_class]
    
    response = {
        "problem_class": predicted_class,
        "problem_score": round(predicted_score, 2),
        "confidence": round(confidence, 4),
        "class_probabilities": class_prob_dict,
        "score_range": [float(score_range[0]), float(score_range[1])],
        "metadata": {
            "text_length": len(str(title) + str(description) + str(input_desc) + str(output_desc)),
            "word_count": len(cleaned_text.split()),
            "features_used": X_score.shape[1],
            "regressor_used": regressor_used,
            "model_timestamp": artifacts.timestamp,
            "architecture": "global_regressor"
        }
    }
    
    return response


# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route('/')
def index():
    """Render the main web interface"""
    try:
        return render_template('index.html')
    except:
        return jsonify({
            'service': 'CP Difficulty Predictor API',
            'version': '5.0.0',
            'architecture': 'Global Regressor',
            'status': 'healthy' if artifacts.is_loaded else 'not ready',
            'endpoints': {
                'GET /api': 'API information',
                'POST /predict': 'Make prediction',
                'GET /health': 'Health check',
                'GET /info': 'Model information'
            }
        })


@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        'service': 'CP Difficulty Predictor',
        'version': '5.0.0',
        'architecture': 'Global Regressor (class as feature)',
        'status': 'healthy' if artifacts.is_loaded else 'not ready',
        'models_loaded': artifacts.is_loaded,
        'model_timestamp': artifacts.timestamp,
        'endpoints': {
            'GET /': 'Web interface',
            'GET /api': 'API information',
            'POST /predict': 'Make prediction',
            'POST /api/predict': 'Make prediction (alias)',
            'GET /health': 'Health check',
            'GET /info': 'Model information'
        }
    })


@app.route('/health')
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy' if artifacts.is_loaded else 'unhealthy',
        'models_loaded': artifacts.is_loaded,
        'model_timestamp': artifacts.timestamp,
        'load_timestamp': artifacts.load_timestamp.isoformat() if artifacts.load_timestamp else None,
        'architecture': 'global_regressor',
        'classes': artifacts.class_names,
        'regressor_type': artifacts.regressor.__class__.__name__ if artifacts.regressor else None,
        'feature_count': len(artifacts.feature_names) if artifacts.feature_names else 0
    })


@app.route('/info')
def info():
    """Model information"""
    if not artifacts.is_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    return jsonify({
        'version': '5.0.0',
        'architecture': 'Two-Stage Global Regressor',
        'model_timestamp': artifacts.timestamp,
        'classes': artifacts.class_names,
        'theoretical_ranges': THEORETICAL_RANGES,
        'features': {
            'total': len(artifacts.feature_names),
            'includes_class_features': any('class' in f for f in artifacts.feature_names)
        },
        'models': {
            'classifier': artifacts.classifier.__class__.__name__,
            'regressor': artifacts.regressor.__class__.__name__ if artifacts.regressor else None
        },
        'training_approach': 'Global regressor trained on all samples with class as input feature'
    })


@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """
    Main prediction endpoint.
    
    Request JSON:
    {
        "title": "Problem title",
        "description": "Problem description",
        "input": "Input description (optional)",
        "output": "Output description (optional)"
    }
    """
    try:
        # Check if models loaded
        if not artifacts.is_loaded:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Please restart server or train models first'
            }), 503
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract fields
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        input_desc = data.get('input', '').strip()
        output_desc = data.get('output', '').strip()
        
        # Validate
        if not title:
            return jsonify({'error': 'Title is required'}), 400
        
        if not description:
            return jsonify({'error': 'Description is required'}), 400
        
        if len(description) < 20:
            return jsonify({'error': 'Description too short (min 20 characters)'}), 400
        
        # Make prediction
        result = predict_difficulty(title, description, input_desc, output_desc)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"\n❌ ERROR in prediction:")
        traceback.print_exc()
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if not artifacts.is_loaded:
            return jsonify({'error': 'Models not loaded'}), 503
        
        data = request.get_json()
        problems = data.get('problems', [])
        
        if not problems or not isinstance(problems, list):
            return jsonify({'error': 'Invalid request format'}), 400
        
        if len(problems) > 100:
            return jsonify({'error': 'Maximum 100 problems per batch'}), 400
        
        results = []
        for problem in problems:
            try:
                result = predict_difficulty(
                    problem.get('title', ''),
                    problem.get('description', ''),
                    problem.get('input', ''),
                    problem.get('output', '')
                )
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """404 handler"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """500 handler"""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    import sys
    
    print("\n" + "="*70)
    print("🚀 CP DIFFICULTY PREDICTOR - FLASK SERVER v5.0")
    print("="*70)
    print("\nArchitecture:")
    print("  STAGE 1: Classifier → Easy/Medium/Hard + probabilities")
    print("  STAGE 2: Global Regressor → Score (uses class as feature)")
    print("  STAGE 3: Soft Constraints → Final bounded score")
    print("\nTheoretical Score Ranges:")
    for class_name, (min_s, max_s) in THEORETICAL_RANGES.items():
        print(f"  {class_name:8s}: [{min_s:.1f}, {max_s:.1f}]")
    print("\n💡 Key Improvement:")
    print("  - Global regressor prevents score collapse")
    print("  - Scores vary meaningfully within each class")
    print("="*70)
    
    # Allow specifying model directory and timestamp
    model_dir = '../models'
    timestamp = None
    
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    if len(sys.argv) > 2:
        timestamp = sys.argv[2]
    
    # Load models
    success = load_models(model_dir, timestamp)
    
    if not success:
        print("\n❌ Failed to load models")
        print("   Server will start but predictions will fail")
        print("\n💡 Usage: python app.py [model_directory] [timestamp]")
        print("   Example: python app.py ../models 20250107_143022")
        print("\n⚠️  Make sure you've trained models with the NEW script:")
        print("   python train.py")
    
    # Start server
    print("\n🌐 Starting server on http://0.0.0.0:5000")
    print("\nEndpoints:")
    print("  GET  /          - Web interface / Service info")
    print("  GET  /api       - API information")
    print("  GET  /health    - Health check")
    print("  GET  /info      - Model info")
    print("  POST /predict   - Single prediction")
    print("  POST /batch     - Batch predictions")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)