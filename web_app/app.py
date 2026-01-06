from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import re
import os
import json
import time
import random
from datetime import datetime
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for models
tfidf_vectorizer = None
count_vectorizer = None
scaler = None
svd = None
label_encoder = None
classifier = None
regressor = None
is_loaded = False

def load_models():
    """Load ML models and artifacts"""
    global tfidf_vectorizer, count_vectorizer, scaler, svd, label_encoder, classifier, regressor, is_loaded
    
    try:
        print("🔍 Looking for models...")
        
        # Try multiple possible model locations
        possible_paths = [
            '../ml_model/models',  # One level up, then ml_model/models
            os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'models'),  # Absolute path
            'models',  # Current directory
            '../models',  # Parent directory
        ]
        
        models_dir = None
        for dir_path in possible_paths:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                models_dir = dir_path
                print(f"✅ Found models directory: {os.path.abspath(dir_path)}")
                break
        
        if not models_dir:
            print("❌ Could not find models directory!")
            print("   Searched in:")
            for dir_path in possible_paths:
                print(f"   - {os.path.abspath(dir_path) if os.path.exists(dir_path) else dir_path + ' (not found)'}")
            print("\n⚠️  Will use mock data for predictions")
            print("   To fix: Train models with: python ../ml_model/train.py")
            return False
        
        # Check for required files
        required_files = [
            'tfidf_vectorizer.pkl',
            'count_vectorizer.pkl',
            'scaler.pkl',
            'svd.pkl',
            'label_encoder.pkl',
            'classifier.pkl',
            'regressor.pkl'
        ]
        
        print("📦 Checking for model files...")
        all_files_exist = True
        
        for file_name in required_files:
            file_path = os.path.join(models_dir, file_name)
            if os.path.exists(file_path):
                print(f"   ✅ {file_name}")
            else:
                print(f"   ❌ {file_name}")
                all_files_exist = False
        
        if not all_files_exist:
            print(f"\n❌ Some model files are missing!")
            print("   Please train the models first by running: python ../ml_model/train.py")
            print("⚠️  Will use mock data for predictions")
            return False
        
        print("\n🚀 Loading models...")
        
        try:
            # Load each model
            tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
            print(f"   ✓ TF-IDF Vectorizer loaded")
            
            count_vectorizer = joblib.load(os.path.join(models_dir, 'count_vectorizer.pkl'))
            print(f"   ✓ Count Vectorizer loaded")
            
            scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            print(f"   ✓ Scaler loaded")
            
            svd = joblib.load(os.path.join(models_dir, 'svd.pkl'))
            print(f"   ✓ SVD loaded ({svd.n_components} components)")
            
            label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
            classes = list(label_encoder.classes_)
            print(f"   ✓ Label Encoder loaded (classes: {classes})")
            
            classifier = joblib.load(os.path.join(models_dir, 'classifier.pkl'))
            print(f"   ✓ Classifier loaded ({classifier.__class__.__name__})")
            
            regressor = joblib.load(os.path.join(models_dir, 'regressor.pkl'))
            print(f"   ✓ Regressor loaded ({regressor.__class__.__name__})")
            
            is_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading model files: {e}")
            print("⚠️  Models may be corrupted or incompatible")
            print("   Please run: python ../ml_model/train.py to retrain models")
            return False
        
        # Try to load metadata
        metadata_path = os.path.join(models_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   ✓ Metadata loaded (v{metadata.get('model_version', '1.0.0')})")
        else:
            print("   ⚠️  Metadata not found")
        
        print("\n" + "="*60)
        print("🎉 ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in load_models: {e}")
        import traceback
        traceback.print_exc()
        return False

# Load models when app starts
print("\n" + "="*60)
print("🤖 AutoJudge - AI Difficulty Predictor")
print("="*60)
load_models()

class TextFeatureExtractor:
    """Feature extractor matching the training code"""
    
    def __init__(self):
        pass
    
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
        """Extract features from text"""
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

# Initialize feature extractor
feature_extractor = TextFeatureExtractor()

# ========== ROUTES ==========

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', models_loaded=is_loaded)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if is_loaded else 'unhealthy',
        'models_loaded': is_loaded,
        'service': 'AutoJudge',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/sample', methods=['GET'])
def get_sample():
    """Get sample problems"""
    samples = [
        {
            'title': 'Two Sum',
            'description': 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order.',
            'input_description': 'First line contains an integer n, the size of array. Next line contains n space-separated integers. The last line contains the target sum.',
            'output_description': 'Print the indices of two numbers that sum to target.',
            'expected_difficulty': 'Easy'
        },
        {
            'title': 'Binary Tree Level Order Traversal',
            'description': 'Given the root of a binary tree, return the level order traversal of its nodes values. (i.e., from left to right, level by level).',
            'input_description': 'The input contains the tree nodes in level order format. Use -1 for null nodes.',
            'output_description': 'Print each level on a separate line.',
            'expected_difficulty': 'Medium'
        },
        {
            'title': 'Dynamic Programming: Coin Change',
            'description': 'Given an array of coin denominations and a target amount, return the minimum number of coins needed to make up that amount. If that amount cannot be made up, return -1. You may assume an infinite number of each kind of coin.',
            'input_description': 'First line contains n and amount. Second line contains n space-separated integers representing coin denominations.',
            'output_description': 'Print the minimum number of coins needed.',
            'expected_difficulty': 'Hard'
        }
    ]
    return jsonify({'samples': samples, 'count': len(samples)})

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            })
        
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        input_desc = data.get('input_description', '').strip()
        output_desc = data.get('output_description', '').strip()
        
        if not title or not description:
            return jsonify({
                'success': False,
                'error': 'Title and description are required'
            })
        
        print(f"\n📝 PREDICTION REQUEST:")
        print(f"   Title: {title[:50]}...")
        print(f"   Description: {len(description)} chars")
        
        # If models aren't loaded, return mock data
        if not is_loaded:
            print("⚠️  Models not loaded - returning mock data")
            return generate_mock_prediction(title, description, input_desc, output_desc)
        
        # Combine all text
        combined_text = f"{title} {description} {input_desc} {output_desc}"
        cleaned_text = feature_extractor.clean_text(combined_text)
        
        # Extract engineered features
        engineered_features = feature_extractor.extract_features(cleaned_text)
        engineered_df = pd.DataFrame([engineered_features])
        
        # Clean engineered features
        engineered_df = engineered_df.replace([np.inf, -np.inf], np.nan)
        engineered_df = engineered_df.fillna(0)
        engineered_array = engineered_df.values
        
        # Text vectorization
        tfidf_features = tfidf_vectorizer.transform([cleaned_text])
        count_features = count_vectorizer.transform([cleaned_text])
        
        # Combine sparse features and apply SVD
        sparse_features = hstack([tfidf_features, count_features])
        sparse_reduced = svd.transform(sparse_features)
        
        # Combine all features
        X_dense = np.hstack([sparse_reduced, engineered_array])
        
        # Final check for non-finite values
        if not np.isfinite(X_dense).all():
            X_dense = np.nan_to_num(X_dense, nan=0, posinf=0, neginf=0)
        
        # Scale features
        X_scaled = scaler.transform(X_dense)
        
        # Make predictions
        class_pred = classifier.predict(X_scaled)[0]
        class_label = label_encoder.inverse_transform([class_pred])[0]
        
        score_pred = regressor.predict(X_scaled)[0]
        score_pred = max(1.0, min(10.0, float(score_pred)))
        
        # Get probabilities if available
        if hasattr(classifier, 'predict_proba'):
            class_probs = classifier.predict_proba(X_scaled)[0]
            confidence_scores = {
                cls: float(prob) for cls, prob in zip(label_encoder.classes_, class_probs)
            }
            confidence = float(max(class_probs))
        else:
            confidence_scores = {cls: 0.0 for cls in label_encoder.classes_}
            confidence_scores[class_label] = 0.8
            confidence = 0.8
        
        print(f"✅ Prediction: {class_label} ({score_pred:.2f}/10)")
        
        # Prepare feature analysis
        feature_analysis = {
            'text_statistics': {
                'text_length': int(engineered_features.get('text_length', 0)),
                'word_count': int(engineered_features.get('word_count', 0)),
                'sentence_count': int(engineered_features.get('sentence_count', 0)),
                'char_count': int(engineered_features.get('char_count', 0)),
                'vocab_richness': round(float(engineered_features.get('vocab_richness', 0)), 3)
            },
            'algorithmic_features': {
                'has_dp': bool(engineered_features.get('has_dp', 0)),
                'has_graph': bool(engineered_features.get('has_graph', 0)),
                'has_search': bool(engineered_features.get('has_search', 0)),
                'has_sort': bool(engineered_features.get('has_sort', 0)),
                'has_ds': bool(engineered_features.get('has_ds', 0)),
                'has_math': bool(engineered_features.get('has_math', 0)),
                'total_algorithm_keywords': sum([engineered_features.get(f'has_{cat}', 0) 
                                                for cat in ['dp', 'graph', 'search', 'sort', 'ds', 'math']])
            },
            'complexity_indicators': {
                'math_operators': int(engineered_features.get('math_operators', 0)),
                'number_count': int(engineered_features.get('number_count', 0)),
                'complexity_mentions': bool(engineered_features.get('complexity_mentions', 0)),
                'has_constraints': bool(engineered_features.get('has_constraint', 0))
            },
            'feature_counts': {
                'total_features': X_dense.shape[1],
                'svd_components': sparse_reduced.shape[1],
                'engineered_features': engineered_array.shape[1]
            }
        }
        
        # Generate insights
        insights = generate_insights(class_label, score_pred, engineered_features)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'problem_class': class_label,
                'problem_score': round(score_pred, 2),
                'class_confidence': round(confidence, 3),
                'confidence_scores': confidence_scores,
                'feature_analysis': feature_analysis,
                'insights': insights,
                'metadata': {
                    'model_version': '1.0.0',
                    'model_architecture': {
                        'classifier': classifier.__class__.__name__,
                        'regressor': regressor.__class__.__name__
                    },
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_features_used': X_dense.shape[1],
                    'model_status': 'real'
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error with mock data for frontend
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback_data': generate_mock_prediction(
                data.get('title', '') if data else '',
                data.get('description', '') if data else '',
                error=True
            )['prediction']
        })

def generate_mock_prediction(title, description, input_desc="", output_desc="", error=False):
    """Generate mock prediction data when models aren't loaded"""
    
    # Combine all text
    combined_text = f"{title} {description} {input_desc} {output_desc}"
    
    # Extract features for better mock data
    try:
        engineered_features = feature_extractor.extract_features(combined_text)
    except:
        engineered_features = {}
    
    # Determine class based on text length and content
    text_len = len(description)
    word_count = len(description.split())
    
    if error:
        # For error cases
        difficulty = 'Medium'
        score = 5.0
        confidence = 0.7
    else:
        # Heuristic based on text
        if word_count < 50 or ('sum' in description.lower() and 'array' in description.lower()):
            difficulty = 'Easy'
            score = random.uniform(2.0, 4.0)
        elif word_count < 150 or ('tree' in description.lower() or 'graph' in description.lower()):
            difficulty = 'Medium'
            score = random.uniform(4.0, 7.0)
        else:
            difficulty = 'Hard'
            score = random.uniform(7.0, 9.5)
        
        # Adjust based on keywords
        if 'dynamic programming' in description.lower() or 'dp' in description.lower():
            difficulty = 'Hard'
            score = random.uniform(7.5, 9.5)
        
        confidence = random.uniform(0.7, 0.9)
    
    # Mock confidence scores
    classes = ['Easy', 'Medium', 'Hard']
    conf_scores = {cls: 0.1 for cls in classes}
    conf_scores[difficulty] = confidence
    # Distribute remaining probability
    remaining = 1.0 - confidence
    other_classes = [c for c in classes if c != difficulty]
    for cls in other_classes:
        conf_scores[cls] = remaining / len(other_classes)
    
    # Feature analysis
    feature_analysis = {
        'text_statistics': {
            'text_length': text_len,
            'word_count': word_count,
            'sentence_count': max(1, len(re.split(r'[.!?]+', description))),
            'avg_word_length': round(sum(len(w) for w in description.split()) / max(1, word_count), 2)
        },
        'algorithmic_features': {
            'has_dp': 'dynamic' in description.lower() or 'dp' in description.lower(),
            'has_graph': 'graph' in description.lower() or 'tree' in description.lower(),
            'has_search': 'search' in description.lower() or 'binary' in description.lower(),
            'has_math': 'sum' in description.lower() or 'calculate' in description.lower()
        },
        'complexity_indicators': {
            'math_operators': sum(1 for c in description if c in '+-*/='),
            'constraints_present': 'constraint' in description.lower() or 'limit' in description.lower(),
            'examples_count': description.lower().count('example')
        }
    }
    
    # Insights
    insights = [
        f"Predicted as {difficulty} difficulty",
        f"Score: {score:.1f}/10.0",
        "Note: Using mock data - train models for real predictions"
    ]
    
    if difficulty == 'Easy':
        insights.append("Suitable for beginners")
        insights.append("Focuses on basic programming concepts")
    elif difficulty == 'Medium':
        insights.append("Good for intermediate practice")
        insights.append("Requires algorithmic thinking")
    else:
        insights.append("Challenging problem")
        insights.append("Tests advanced concepts")
    
    # Add feature-based insights
    if feature_analysis['algorithmic_features']['has_dp']:
        insights.append("Dynamic programming problem detected")
    if feature_analysis['algorithmic_features']['has_graph']:
        insights.append("Graph/tree structure involved")
    if feature_analysis['complexity_indicators']['constraints_present']:
        insights.append("Contains specific constraints")
    
    return {
        'success': not error,
        'prediction': {
            'problem_class': difficulty,
            'problem_score': round(score, 2),
            'class_confidence': round(confidence, 3),
            'confidence_scores': conf_scores,
            'feature_analysis': feature_analysis,
            'insights': insights,
            'metadata': {
                'model_version': '1.0.0',
                'model_architecture': 'Mock Model',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_features_used': 50,
                'model_status': 'mock',
                'note': 'Train models with: python ../ml_model/train.py'
            }
        }
    }

def generate_insights(class_label, score, features):
    """Generate insights based on prediction"""
    insights = []
    
    # Class-based insights
    if class_label == 'Easy':
        insights.append("🎯 Suitable for beginners - focuses on fundamental concepts")
        insights.append("💡 Great for practicing basic programming skills")
    elif class_label == 'Medium':
        insights.append("🎯 Intermediate level - requires algorithmic thinking")
        insights.append("💡 Good for improving problem-solving skills")
    else:
        insights.append("🎯 Advanced challenge - tests deep understanding")
        insights.append("💡 Excellent for competition preparation")
    
    # Score-based insights
    if score < 3:
        insights.append("📊 Very basic problem - quick to solve")
    elif score < 6:
        insights.append("📊 Moderate difficulty - balanced challenge")
    elif score < 8:
        insights.append("📊 Challenging problem - requires careful optimization")
    else:
        insights.append("📊 Expert-level difficulty - tests advanced concepts")
    
    # Feature-based insights
    if features.get('has_dp', 0):
        insights.append("🔹 Dynamic programming detected - requires memoization/tabulation")
    
    if features.get('has_graph', 0):
        insights.append("🔹 Graph algorithms present - may need BFS/DFS/Dijkstra")
    
    if features.get('math_operators', 0) > 5:
        insights.append("🧮 Mathematical problem - requires numerical computation")
    
    if features.get('has_constraint', 0):
        insights.append("🔸 Constraints present - check boundary conditions")
    
    if features.get('complexity_mentions', 0):
        insights.append("⚡ Complexity mentioned - optimization is important")
    
    return insights

@app.route('/models', methods=['GET'])
def get_model_info():
    """Get model information"""
    try:
        # Try to find metadata
        metadata = {}
        possible_paths = [
            '../ml_model/models/metadata.json',
            'models/metadata.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    metadata = json.load(f)
                break
        
        model_info = {
            'success': True,
            'model_info': {
                'version': metadata.get('model_version', '1.0.0'),
                'training_date': metadata.get('training_date', 'N/A'),
                'classes': metadata.get('classes', ['Easy', 'Medium', 'Hard']),
                'classifier': classifier.__class__.__name__ if classifier else 'Not loaded',
                'regressor': regressor.__class__.__name__ if regressor else 'Not loaded',
                'status': 'loaded' if is_loaded else 'not loaded',
                'performance': metadata.get('metrics', {}),
                'using_mock_data': not is_loaded
            }
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/features', methods=['POST'])
def analyze_features():
    """Analyze features without making prediction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()
        
        if not description:
            return jsonify({'success': False, 'error': 'Description required'})
        
        # Clean and extract features
        combined_text = f"{title} {description}"
        cleaned_text = feature_extractor.clean_text(combined_text)
        features = feature_extractor.extract_features(cleaned_text)
        
        # Calculate derived metrics
        algorithm_count = sum([features.get(f'has_{cat}', 0) 
                             for cat in ['dp', 'graph', 'search', 'sort', 'ds', 'math']])
        
        analysis = {
            'text_metrics': {
                'length': features.get('text_length', 0),
                'words': features.get('word_count', 0),
                'sentences': features.get('sentence_count', 0),
                'avg_word_length': round(features.get('avg_word_length', 0), 2),
                'vocabulary_richness': round(features.get('vocab_richness', 0), 3)
            },
            'algorithm_indicators': {
                'total_algorithms': algorithm_count,
                'dynamic_programming': bool(features.get('has_dp', 0)),
                'graph_algorithms': bool(features.get('has_graph', 0)),
                'search_algorithms': bool(features.get('has_search', 0)),
                'sorting': bool(features.get('has_sort', 0)),
                'data_structures': bool(features.get('has_ds', 0)),
                'mathematics': bool(features.get('has_math', 0))
            },
            'complexity_hints': {
                'math_operators': features.get('math_operators', 0),
                'numbers_mentioned': features.get('number_count', 0),
                'complexity_discussed': bool(features.get('complexity_mentions', 0)),
                'has_constraints': bool(features.get('has_constraint', 0))
            }
        }
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'raw_features': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in features.items()}
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        'status': 'running',
        'models_loaded': is_loaded,
        'service': 'AutoJudge',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'GET /': 'Web interface',
            'GET /health': 'Health check',
            'GET /sample': 'Sample problems',
            'POST /predict': 'Make prediction',
            'GET /models': 'Model info',
            'POST /features': 'Feature analysis'
        }
    })

# Simple test endpoint
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'success': True,
        'message': 'AutoJudge API is running',
        'models_loaded': is_loaded,
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'path': request.path,
        'available_endpoints': [
            {'GET': '/'},
            {'GET': '/health'},
            {'GET': '/sample'},
            {'POST': '/predict'},
            {'GET': '/models'},
            {'POST': '/features'},
            {'GET': '/status'},
            {'GET': '/test'}
        ]
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print(f"\n🌐 Starting Flask server...")
    print(f"   Open http://localhost:5000 in your browser")
    print(f"\n📋 Available endpoints:")
    print(f"   GET  /               - Web interface")
    print(f"   GET  /health         - Health check")
    print(f"   GET  /status         - Status info")
    print(f"   GET  /sample         - Sample problems")
    print(f"   GET  /models         - Model information")
    print(f"   POST /predict        - Make prediction")
    print(f"   POST /features       - Feature analysis")
    print(f"   GET  /test           - Test endpoint")
    print(f"\n⚠️  Model status: {'LOADED ✅' if is_loaded else 'NOT LOADED (using mock data)'}")
    if not is_loaded:
        print(f"   To train models: cd ../ml_model && python train.py")
    print("="*60 + "\n")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)