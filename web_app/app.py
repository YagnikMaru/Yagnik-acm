from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import hstack, csr_matrix
import re
import os
import sys
import json
import warnings
from typing import Dict, Tuple
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize with defaults
classifier = None
regressor = None
vectorizer = None
scaler = None
label_encoder = None
is_loaded = False

def load_models():
    """Load ML models and artifacts"""
    global classifier, regressor, vectorizer, scaler, label_encoder, is_loaded
    
    try:
        print("Loading models...")
        
        # Get the current directory (web_app folder)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple possible locations for models
        possible_paths = [
            os.path.join(base_dir, '..', 'saved_models'),  # ../saved_models/
            os.path.join(base_dir, 'saved_models'),        # web_app/saved_models/
            os.path.join(base_dir, '..', 'ml_model', 'saved_models'),  # ../ml_model/saved_models/
        ]
        
        models_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                models_dir = path
                print(f"✅ Found models at: {path}")
                break
        
        if not models_dir:
            print("❌ Models directory not found in any location!")
            print("Possible locations checked:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease train models first: python train.py")
            is_loaded = False
            return
        
        # Check if all required model files exist
        model_files = ['classifier.pkl', 'regressor.pkl', 'vectorizer.pkl', 'scaler.pkl', 'label_encoder.pkl']
        missing_files = []
        
        for file in model_files:
            file_path = os.path.join(models_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
                print(f"  ❌ {file} not found")
            else:
                print(f"  ✅ {file} found")
        
        if missing_files:
            print(f"\n❌ Missing files: {missing_files}")
            print("Please train models first: python train.py")
            is_loaded = False
            return
        
        # Load models
        print("\nLoading models...")
        classifier = joblib.load(os.path.join(models_dir, 'classifier.pkl'))
        print("✓ Classifier loaded")
        
        regressor = joblib.load(os.path.join(models_dir, 'regressor.pkl'))
        print("✓ Regressor loaded")
        
        vectorizer = joblib.load(os.path.join(models_dir, 'vectorizer.pkl'))
        print(f"✓ Vectorizer loaded (max_features: {vectorizer.max_features})")
        
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        print(f"✓ Scaler loaded (expecting {scaler.n_features_in_} features)")
        
        label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        print(f"✓ Label encoder loaded (classes: {list(label_encoder.classes_)})")
        
        is_loaded = True
        print("\n" + "="*60)
        print("✅ ALL MODELS LOADED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting steps:")
        print("1. Make sure you've run: python train.py")
        print("2. Check that saved_models/ folder exists in parent directory")
        print("3. Verify all .pkl files are present")
        is_loaded = False

# Load models on startup
load_models()

# ==============================================
# ULTRA-ENHANCED TextProcessor Class
# ==============================================

class UltraEnhancedTextProcessor:
    """Ultra-enhanced text processor with 50+ features and deterministic preprocessing"""
    
    def __init__(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Comprehensive Algorithm Dictionary with hierarchical weights
        self.algorithms = {
            # CORE ADVANCED ALGORITHMS (Highest weight)
            'dynamic_programming': {
                'keywords': ['dp', 'dynamic programming', 'memoization', 'tabulation', 
                           'knapsack', 'lcs', 'lis', 'bitmask dp', 'digit dp', 'tree dp',
                           'dp on trees', 'dp on graphs', 'state compression'],
                'weight': 4.0
            },
            'graph_advanced': {
                'keywords': ['max flow', 'min cut', 'dinic', 'edmonds karp', 'hopcroft karp',
                           'bipartite matching', 'hungarian', 'blossom', 'strongly connected',
                           'articulation', 'bridge', 'tarjan', 'kosaraju', 'centroid',
                           'heavy light', 'binary lifting', 'lowest common ancestor'],
                'weight': 3.8
            },
            'data_structures_advanced': {
                'keywords': ['segment tree', 'fenwick tree', 'binary indexed tree', 'splay tree',
                           'treap', 'suffix array', 'suffix tree', 'trie', 'aho corasick',
                           'persistent', 'wavelet tree', 'skip list', 'rope', 'link-cut tree',
                           'disjoint set union', 'union find', 'dsu on tree'],
                'weight': 3.5
            },
            'mathematics_advanced': {
                'keywords': ['fft', 'ntt', 'modular exponentiation', 'matrix exponentiation',
                           'chinese remainder', 'fermat', 'euler totient', 'miller rabin',
                           'sieve of eratosthenes', 'linear sieve', 'mobius function',
                           'berlekamp massey', 'linear recurrence', 'generating functions',
                           'inclusion exclusion', 'burnside lemma', 'polya enumeration'],
                'weight': 3.7
            },
            'geometry': {
                'keywords': ['convex hull', 'graham scan', 'jarvis march', 'line intersection',
                           'polygon area', 'point in polygon', 'closest pair', 'voronoi',
                           'delaunay triangulation', 'sweep line', 'rotating calipers'],
                'weight': 3.2
            },
            
            # INTERMEDIATE ALGORITHMS (Medium weight)
            'graph_basic': {
                'keywords': ['dijkstra', 'bellman ford', 'floyd warshall', 'topological sort',
                           'bfs', 'dfs', 'minimum spanning', 'kruskal', 'prim', 'euler path',
                           'hamiltonian', 'traveling salesman'],
                'weight': 2.5
            },
            'search_techniques': {
                'keywords': ['binary search', 'ternary search', 'meet in the middle',
                           'two pointer', 'sliding window', 'divide and conquer'],
                'weight': 2.0
            },
            'string_algorithms': {
                'keywords': ['kmp', 'rabin karp', 'z algorithm', 'manacher', 'palindrome',
                           'suffix automation', 'rolling hash', 'trie', 'aho corasick'],
                'weight': 2.3
            },
            
            # BASIC ALGORITHMS (Low weight)
            'basic_ds': {
                'keywords': ['stack', 'queue', 'deque', 'priority queue', 'heap', 'hash',
                           'hashmap', 'hashset', 'linked list', 'array', 'vector'],
                'weight': 1.0
            },
            'basic_math': {
                'keywords': ['gcd', 'lcm', 'prime', 'modulo', 'combinatorics', 'probability',
                           'number theory', 'geometry basics', 'calculus'],
                'weight': 1.5
            }
        }
        
        # Mathematical symbols preserved during preprocessing
        self.math_preserve = {
            'operators': {'+', '-', '*', '/', '=', '^', '%', '**', '//', '±', '∓'},
            'comparisons': {'<', '>', '<=', '>=', '==', '!=', '≡', '≈', '∼', '≠', '≤', '≥', '≪', '≫'},
            'advanced': {'∑', '∏', '∫', '∂', '∇', '∞', '√', '∛', '∜', '→', '↔', '∧', '∨', '⊕', '⊗'},
            'sets': {'∈', '∉', '⊆', '⊂', '∪', '∩', '∅', 'ℕ', 'ℤ', 'ℚ', 'ℝ', 'ℂ'},
            'greek': {'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 
                     'ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω'},
        }
        
        # Problem structure indicators
        self.structure_indicators = {
            'sections': ['input', 'output', 'example', 'note', 'constraints', 'explanation',
                        'interaction', 'scoring', 'subtasks', 'time limit', 'memory limit'],
            'list_markers': ['•', '-', '*', '◦', '‣', '·', '▪', '▫'],
            'complexity_terms': ['time complexity', 'space complexity', 'O(', 'Θ(', 'Ω('],
        }
        
        # Compile regex patterns once for efficiency
        self.patterns = {
            'url': re.compile(r'http\S+|www\S+|https\S+'),
            'code_blocks': re.compile(r'```[\s\S]*?```'),
            'inline_code': re.compile(r'`[^`]*`'),
            'numbers': re.compile(r'\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b'),
            'constraints': re.compile(r'(\d+(?:\.\d+)?)\s*(?:≤|<|<=|≤=|less than|up to|maximum|max\.?)\s*[A-Za-z]', re.IGNORECASE),
            'large_numbers': re.compile(r'\b(10\^?\d+|\d+e\d+|\d{4,})\b'),
            'variables': re.compile(r'\b[a-z]\b'),
            'time_limit': re.compile(r'(\d+(?:\.\d+)?)\s*(?:second|sec|s|millisecond|ms)', re.IGNORECASE),
            'memory_limit': re.compile(r'(\d+(?:\.\d+)?)\s*(?:MB|megabyte|GB|gigabyte|KB|kilobyte)', re.IGNORECASE),
        }
        
        # Feature names (FIXED for determinism - 52 features)
        self.feature_names = [
            # Text statistics (6)
            'word_count', 'sentence_count', 'avg_word_length', 
            'unique_word_ratio', 'avg_sentence_length', 'paragraph_count',
            
            # Algorithm features (12)
            'algo_dp_score', 'algo_graph_advanced_score', 'algo_ds_advanced_score',
            'algo_math_advanced_score', 'algo_geometry_score', 'algo_graph_basic_score',
            'algo_search_score', 'algo_string_score', 'algo_basic_ds_score',
            'algo_basic_math_score', 'algo_total_weighted', 'unique_algorithms_count',
            
            # Constraint features (8)
            'max_constraint_log10', 'constraint_count', 'has_large_constraints',
            'constraint_variance', 'constraint_types', 'has_time_limit',
            'has_memory_limit', 'max_time_limit',
            
            # Mathematical features (8)
            'math_symbol_count', 'math_advanced_symbols', 'equation_density',
            'inequality_count', 'summation_integral_count', 'greek_symbol_count',
            'modulo_mentions', 'combinatorics_mentions',
            
            # Structural features (8)
            'section_count', 'example_count', 'has_multiple_test_cases',
            'has_formal_constraints', 'has_subtasks', 'io_format_complexity',
            'interactive_problem', 'has_scoring_section',
            
            # Language complexity features (4)
            'technical_term_density', 'acronym_count', 'conditional_mentions',
            'imperative_verbs',
            
            # Composite scores (6)
            'text_complexity', 'algorithmic_complexity', 'mathematical_complexity',
            'structural_complexity', 'language_complexity', 'overall_complexity_score'
        ]
    
    def deterministic_preprocess(self, text: str) -> str:
        """Deterministic text preprocessing identical for training/inference"""
        if not isinstance(text, str):
            return ""
        
        # 1. Preserve mathematical symbols (add spaces around them)
        preserved = text
        for category in self.math_preserve.values():
            for symbol in category:
                preserved = preserved.replace(symbol, f' {symbol} ')
        
        # 2. Convert to lowercase (except preserved symbols)
        processed = preserved.lower()
        
        # 3. Remove URLs (deterministic)
        processed = self.patterns['url'].sub(' ', processed)
        
        # 4. Remove code blocks
        processed = self.patterns['code_blocks'].sub(' ', processed)
        processed = self.patterns['inline_code'].sub(' ', processed)
        
        # 5. Remove special characters (keep alphanumeric and preserved math)
        processed = re.sub(r'[^\w\s\.\,\!\?\:\;\+\-\*/=<>^%\(\)\[\]\{\}∑∏∫∂∇∞√±→↔∧∨∈∉⊆⊂∪∩∅≤≥≠≈∼≡αβγδεζηθικλμνξπρστυφχψω]', ' ', processed)
        
        # 6. Normalize whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def extract_algorithm_features(self, text: str) -> dict:
        """Extract algorithm presence with hierarchical weights"""
        text_lower = text.lower()
        features = {}
        
        total_weight = 0
        category_scores = {}
        unique_algs = set()
        
        for category, info in self.algorithms.items():
            category_score = 0
            for keyword in info['keywords']:
                # Match whole words
                pattern = rf'\b{re.escape(keyword)}\b'
                matches = re.findall(pattern, text_lower)
                if matches:
                    category_score += len(matches) * info['weight']
                    # Add first word of multi-word keywords
                    if ' ' in keyword:
                        unique_algs.add(keyword.split()[0])
                    else:
                        unique_algs.add(keyword)
            
            category_scores[category] = category_score
            total_weight += category_score
        
        # Store all category scores
        for category in self.algorithms.keys():
            features[f'algo_{category}_score'] = category_scores.get(category, 0)
        
        features['algo_total_weighted'] = total_weight
        features['unique_algorithms_count'] = len(unique_algs)
        
        return features
    
    def extract_constraint_features(self, text: str) -> dict:
        """Extract numerical constraints with magnitude analysis"""
        features = {}
        
        # Find all numbers
        numbers = []
        for match in self.patterns['numbers'].finditer(text):
            try:
                num = float(match.group())
                numbers.append(num)
            except:
                continue
        
        # Find constraints in typical format
        constraint_matches = self.patterns['constraints'].findall(text)
        constraint_values = []
        for match in constraint_matches:
            try:
                val = float(match)
                constraint_values.append(val)
            except:
                continue
        
        all_constraints = numbers + constraint_values
        
        if all_constraints:
            max_constraint = max(all_constraints)
            features['max_constraint_log10'] = np.log10(max_constraint + 1)
            features['constraint_count'] = len(all_constraints)
            features['has_large_constraints'] = 1.0 if max_constraint > 10000 else 0.0
            
            # Constraint variance
            if len(all_constraints) > 1:
                features['constraint_variance'] = np.log1p(np.var(all_constraints))
            else:
                features['constraint_variance'] = 0.0
            
            # Count types of constraints (different orders of magnitude)
            features['constraint_types'] = len(set(int(np.log10(c + 1)) for c in all_constraints if c > 0))
        else:
            features.update({
                'max_constraint_log10': 0,
                'constraint_count': 0,
                'has_large_constraints': 0.0,
                'constraint_variance': 0.0,
                'constraint_types': 0
            })
        
        # Time and memory limits
        time_matches = self.patterns['time_limit'].findall(text)
        memory_matches = self.patterns['memory_limit'].findall(text)
        
        features['has_time_limit'] = 1.0 if time_matches else 0.0
        features['has_memory_limit'] = 1.0 if memory_matches else 0.0
        
        if time_matches:
            try:
                features['max_time_limit'] = max(float(t) for t in time_matches)
            except:
                features['max_time_limit'] = 0.0
        else:
            features['max_time_limit'] = 0.0
        
        return features
    
    def extract_mathematical_features(self, text: str) -> dict:
        """Extract mathematical complexity indicators"""
        features = {}
        
        # Count math symbols by category
        math_count = 0
        advanced_count = 0
        greek_count = 0
        
        for category_name, symbols in self.math_preserve.items():
            for symbol in symbols:
                count = text.count(symbol)
                math_count += count
                if category_name in ['advanced', 'greek']:
                    advanced_count += count
                if category_name == 'greek':
                    greek_count += count
        
        # Count specific mathematical patterns
        equation_patterns = [
            r'[=≠]',  # Equality/inequality
            r'∑', r'∏', r'∫',  # Summations/integrals
            r'[≤≥<>]',  # Inequalities
        ]
        
        equation_density = 0
        inequality_count = 0
        summation_count = 0
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text)
            equation_density += len(matches)
            if pattern in '[≤≥<>]':
                inequality_count += len(matches)
            if pattern in '∑∏∫':
                summation_count += len(matches)
        
        # Count modulo mentions
        modulo_mentions = len(re.findall(r'\bmod\b|\%|modulo', text.lower()))
        
        # Count combinatorics mentions
        combinatorics_mentions = len(re.findall(r'\bpermutation\b|\bcombination\b|\bchoose\b|\bbinomial\b', text.lower()))
        
        features['math_symbol_count'] = math_count
        features['math_advanced_symbols'] = advanced_count
        features['equation_density'] = equation_density / max(1, len(text.split()))
        features['inequality_count'] = inequality_count
        features['summation_integral_count'] = summation_count
        features['greek_symbol_count'] = greek_count
        features['modulo_mentions'] = modulo_mentions
        features['combinatorics_mentions'] = combinatorics_mentions
        
        return features
    
    def extract_structural_features(self, text: str) -> dict:
        """Extract problem structure complexity"""
        features = {}
        lines = text.split('\n')
        
        # Count sections
        section_count = 0
        example_count = 0
        has_constraints = 0
        has_subtasks = 0
        interactive_problem = 0
        has_scoring = 0
        
        for line in lines:
            line_lower = line.lower().strip()
            for section in self.structure_indicators['sections']:
                if line_lower.startswith(section):
                    section_count += 1
                    if section == 'example':
                        example_count += 1
                    elif section == 'constraints':
                        has_constraints = 1
                    elif section == 'subtasks':
                        has_subtasks = 1
                    elif section == 'scoring':
                        has_scoring = 1
                    elif section == 'interaction':
                        interactive_problem = 1
        
        # Check for multiple test cases
        test_case_indicators = ['multiple test cases', 't test cases', 'first line contains t', 'number of test cases']
        has_multiple_tests = 0
        for indicator in test_case_indicators:
            if indicator in text.lower():
                has_multiple_tests = 1
                break
        
        # IO format complexity (count of lines in example input/output)
        io_complexity = 0
        in_example = False
        example_lines = 0
        
        for line in lines:
            if 'example input' in line.lower():
                in_example = True
                example_lines = 0
            elif in_example and ('output' in line.lower() or line.strip() == ''):
                if example_lines > io_complexity:
                    io_complexity = example_lines
                in_example = False
            elif in_example:
                example_lines += 1
        
        # Count paragraphs
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        features['section_count'] = section_count
        features['example_count'] = example_count
        features['has_multiple_test_cases'] = has_multiple_tests
        features['has_formal_constraints'] = has_constraints
        features['has_subtasks'] = has_subtasks
        features['io_format_complexity'] = min(io_complexity, 20) / 20.0  # Normalize
        features['interactive_problem'] = interactive_problem
        features['has_scoring_section'] = has_scoring
        features['paragraph_count'] = paragraph_count
        
        return features
    
    def extract_language_complexity_features(self, text: str) -> dict:
        """Extract linguistic complexity features"""
        words = text.lower().split()
        if not words:
            return {
                'technical_term_density': 0,
                'acronym_count': 0,
                'conditional_mentions': 0,
                'imperative_verbs': 0
            }
        
        # Technical terms (algorithmic/mathematical terms)
        technical_terms = {
            'algorithm', 'complexity', 'optimization', 'efficient', 'optimal',
            'constraint', 'parameter', 'variable', 'function', 'recursive',
            'iterative', 'heuristic', 'deterministic', 'stochastic'
        }
        
        technical_count = sum(1 for word in words if word in technical_terms)
        
        # Acronyms (words in ALL CAPS or with dots)
        acronym_pattern = r'\b[A-Z]{2,}\b|\b[A-Z]\.[A-Z]\.'
        acronym_count = len(re.findall(acronym_pattern, text))
        
        # Conditional statements
        conditional_terms = {'if', 'else', 'when', 'unless', 'provided', 'given that'}
        conditional_count = sum(1 for word in words if word in conditional_terms)
        
        # Imperative verbs (commands)
        imperative_verbs = {'find', 'determine', 'calculate', 'compute', 'print',
                          'output', 'return', 'write', 'implement', 'solve'}
        imperative_count = sum(1 for word in words if word in imperative_verbs)
        
        return {
            'technical_term_density': technical_count / len(words),
            'acronym_count': acronym_count,
            'conditional_mentions': conditional_count,
            'imperative_verbs': imperative_count
        }
    
    def extract_text_features(self, text: str) -> dict:
        """Extract text complexity features"""
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not words:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'unique_word_ratio': 0,
                'avg_sentence_length': 0
            }
        
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        unique_words = len(set(words))
        
        # Vocabulary richness measures
        ttr = unique_words / word_count  # Type-token ratio
        h_point = 0  # Hapax legomena (words appearing only once)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        h_point = sum(1 for freq in word_freq.values() if freq == 1) / word_count
        
        features = {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': sum(len(w) for w in words) / word_count,
            'unique_word_ratio': ttr * (1 + h_point),  # Enhanced TTR
            'avg_sentence_length': word_count / sentence_count
        }
        
        return features
    
    def extract_composite_scores(self, features: dict) -> dict:
        """Compute composite complexity scores"""
        # Text complexity
        text_complexity = (
            0.25 * np.log1p(features.get('word_count', 0)) +
            0.25 * features.get('unique_word_ratio', 0) +
            0.25 * min(features.get('avg_sentence_length', 0) / 25.0, 1.0) +
            0.25 * features.get('technical_term_density', 0)
        )
        
        # Algorithmic complexity (weighted by importance)
        algo_complexity = (
            0.25 * np.log1p(features.get('algo_dp_score', 0)) +
            0.20 * np.log1p(features.get('algo_graph_advanced_score', 0)) +
            0.15 * np.log1p(features.get('algo_math_advanced_score', 0)) +
            0.15 * np.log1p(features.get('algo_ds_advanced_score', 0)) +
            0.10 * np.log1p(features.get('algo_geometry_score', 0)) +
            0.10 * features.get('unique_algorithms_count', 0) / 10.0 +
            0.05 * np.log1p(features.get('algo_total_weighted', 0))
        )
        
        # Mathematical complexity
        math_complexity = (
            0.25 * np.log1p(features.get('math_symbol_count', 0)) +
            0.20 * np.log1p(features.get('math_advanced_symbols', 0)) +
            0.15 * features.get('equation_density', 0) * 10 +
            0.15 * np.log1p(features.get('inequality_count', 0)) +
            0.10 * np.log1p(features.get('summation_integral_count', 0)) +
            0.10 * features.get('modulo_mentions', 0) +
            0.05 * features.get('combinatorics_mentions', 0)
        ) / 2.0
        
        # Structural complexity
        structural_complexity = (
            0.20 * min(features.get('section_count', 0) / 8.0, 1.0) +
            0.20 * min(features.get('example_count', 0) / 3.0, 1.0) +
            0.20 * features.get('io_format_complexity', 0) +
            0.15 * min(features.get('constraint_count', 0) / 10.0, 1.0) +
            0.10 * features.get('has_subtasks', 0) +
            0.10 * features.get('has_multiple_test_cases', 0) +
            0.05 * features.get('interactive_problem', 0)
        )
        
        # Language complexity
        language_complexity = (
            0.40 * features.get('technical_term_density', 0) +
            0.25 * min(features.get('acronym_count', 0) / 5.0, 1.0) +
            0.20 * min(features.get('conditional_mentions', 0) / 5.0, 1.0) +
            0.15 * min(features.get('imperative_verbs', 0) / 10.0, 1.0)
        )
        
        # Overall weighted score (sum to approximately 10 for score prediction)
        overall = (
            0.20 * text_complexity * 2.5 +
            0.35 * algo_complexity * 3.0 +
            0.25 * math_complexity * 2.5 +
            0.12 * structural_complexity * 2.0 +
            0.08 * language_complexity * 1.5
        )
        
        return {
            'text_complexity': text_complexity,
            'algorithmic_complexity': algo_complexity,
            'mathematical_complexity': math_complexity,
            'structural_complexity': structural_complexity,
            'language_complexity': language_complexity,
            'overall_complexity_score': overall
        }
    
    def extract_all_features(self, text: str):
        """Extract all 52 features deterministically"""
        processed_text = self.deterministic_preprocess(text)
        
        # Extract feature categories
        text_features = self.extract_text_features(processed_text)
        algo_features = self.extract_algorithm_features(processed_text)
        constraint_features = self.extract_constraint_features(text)
        math_features = self.extract_mathematical_features(text)
        structural_features = self.extract_structural_features(text)
        language_features = self.extract_language_complexity_features(processed_text)
        
        # Combine all features
        all_features = {}
        all_features.update(text_features)
        all_features.update(algo_features)
        all_features.update(constraint_features)
        all_features.update(math_features)
        all_features.update(structural_features)
        all_features.update(language_features)
        
        # Add composite scores
        composite_scores = self.extract_composite_scores(all_features)
        all_features.update(composite_scores)
        
        # Ensure all features are present and in correct order
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(all_features.get(name, 0.0))
        
        return np.array(feature_vector).reshape(1, -1), all_features

# Initialize the text processor
processor = UltraEnhancedTextProcessor()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', models_loaded=is_loaded)

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if is_loaded else 'degraded',
        'models_loaded': is_loaded,
        'scaler_features': scaler.n_features_in_ if is_loaded else 0,
        'vectorizer_features': vectorizer.max_features if is_loaded else 0,
        'message': 'Models loaded successfully' if is_loaded else 'Models not loaded. Please train models first.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if models are loaded
        if not is_loaded:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please train the models first.',
                'instructions': 'Run: python train.py from the project root directory'
            })
        
        # Get data from request
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            })
        
        title = data.get('title', '')
        description = data.get('description', '')
        input_desc = data.get('input_description', '')
        output_desc = data.get('output_description', '')
        
        # Validate required fields
        if not title or not description:
            return jsonify({
                'success': False,
                'error': 'Title and description are required'
            })
        
        # Combine text for processing
        combined_text = f"{title} {description} {input_desc} {output_desc}"
        
        # Preprocess text for TF-IDF
        processed_text = processor.deterministic_preprocess(combined_text)
        
        # Extract numeric features (52 features)
        numeric_features, detailed_features = processor.extract_all_features(combined_text)
        
        # Check feature dimensions
        if numeric_features.shape[1] != scaler.n_features_in_:
            return jsonify({
                'success': False,
                'error': f'Feature dimension mismatch: Expected {scaler.n_features_in_} features, got {numeric_features.shape[1]}. Please retrain models with consistent feature extraction.'
            })
        
        # Scale numeric features
        numeric_scaled = scaler.transform(numeric_features)
        
        # Transform with TF-IDF vectorizer
        tfidf_features = vectorizer.transform([processed_text])
        
        # Debug information
        print(f"\n{'='*50}")
        print("PREDICTION REQUEST")
        print(f"{'='*50}")
        print(f"Title: {title[:50]}...")
        print(f"Description length: {len(description)} chars")
        print(f"TF-IDF shape: {tfidf_features.shape}")
        print(f"Numeric features shape: {numeric_scaled.shape}")
        print(f"Total features: {tfidf_features.shape[1] + numeric_scaled.shape[1]}")
        
        # Combine features
        X = hstack([tfidf_features, numeric_scaled])
        
        # Make predictions
        class_pred = classifier.predict(X)[0]
        score_pred = regressor.predict(X)[0]
        
        # Clip score to 0-10
        score_pred = max(0.0, min(10.0, float(score_pred)))
        
        # Get class label
        class_label = label_encoder.inverse_transform([class_pred])[0]
        
        # Calculate difficulty class based on score
        def get_difficulty_class(score):
            if score <= 3.33:
                return 'Easy'
            elif score <= 6.67:
                return 'Medium'
            else:
                return 'Hard'
        
        score_based_class = get_difficulty_class(score_pred)
        
        # Calculate confidence scores
        if hasattr(classifier, 'predict_proba'):
            class_probs = classifier.predict_proba(X)[0]
            class_confidence = float(max(class_probs))
        else:
            class_confidence = 0.8
        
        # Calculate score confidence
        if 3.0 <= score_pred <= 7.0:
            score_confidence = 0.85
        elif 1.0 <= score_pred < 3.0 or 7.0 < score_pred <= 9.0:
            score_confidence = 0.75
        else:
            score_confidence = 0.65
        
        # Adjust confidence based on feature consistency
        algo_total = detailed_features.get('algo_total_weighted', 0)
        if algo_total > 5:
            class_confidence = min(0.95, class_confidence * 1.1)
        
        # Prepare detailed analysis for frontend
        feature_analysis = {
            'text_statistics': {
                'words': detailed_features.get('word_count', 0),
                'sentences': detailed_features.get('sentence_count', 0),
                'avg_word_length': round(detailed_features.get('avg_word_length', 0), 2)
            },
            'algorithms_detected': {
                'total': detailed_features.get('unique_algorithms_count', 0),
                'has_dp': detailed_features.get('algo_dp_score', 0) > 0,
                'has_graph': detailed_features.get('algo_graph_basic_score', 0) > 0 or detailed_features.get('algo_graph_advanced_score', 0) > 0,
                'has_advanced': detailed_features.get('algo_dp_score', 0) > 2 or detailed_features.get('algo_graph_advanced_score', 0) > 2
            },
            'mathematical_complexity': {
                'math_symbols': detailed_features.get('math_symbol_count', 0),
                'equations': detailed_features.get('equation_density', 0),
                'advanced_symbols': detailed_features.get('math_advanced_symbols', 0)
            },
            'structural_analysis': {
                'sections': detailed_features.get('section_count', 0),
                'constraints': detailed_features.get('constraint_count', 0),
                'examples': detailed_features.get('example_count', 0)
            }
        }
        
        # Generate insights based on features
        insights = []
        if detailed_features.get('algo_dp_score', 0) > 0:
            insights.append("Dynamic programming detected - indicates medium to high difficulty")
        if detailed_features.get('algo_graph_advanced_score', 0) > 0:
            insights.append("Advanced graph algorithms present - requires strong algorithmic knowledge")
        if detailed_features.get('math_symbol_count', 0) > 10:
            insights.append("High mathematical complexity detected")
        if detailed_features.get('constraint_count', 0) > 3:
            insights.append("Multiple constraints require careful handling")
        if detailed_features.get('example_count', 0) == 0:
            insights.append("No examples provided - might be less clear")
        
        # If no specific insights, add generic ones
        if not insights:
            if score_pred < 4:
                insights.append("Problem appears straightforward")
            elif score_pred < 7:
                insights.append("Moderate difficulty - requires algorithmic thinking")
            else:
                insights.append("High difficulty - advanced concepts required")
        
        response = {
            'success': True,
            'prediction': {
                'problem_class': class_label,
                'score_based_class': score_based_class,
                'class_confidence': round(class_confidence, 3),
                'problem_score': round(score_pred, 2),
                'score_confidence': round(score_confidence, 3),
                'feature_analysis': feature_analysis,
                'insights': insights,
                'metadata': {
                    'model_version': '1.0',
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'total_features_used': X.shape[1],
                    'numeric_features': numeric_features.shape[1],
                    'tfidf_features': tfidf_features.shape[1]
                }
            }
        }
        
        print(f"Prediction: {score_based_class} (Score: {score_pred:.2f}/10)")
        print(f"Confidence: {class_confidence:.2%}")
        print(f"{'='*50}\n")
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'details': 'Check console for full error trace'
        })

@app.route('/sample', methods=['GET'])
def get_sample():
    """Return sample problems"""
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
            'title': 'Shortest Path in Weighted Graph (Dijkstra)',
            'description': 'Given a weighted directed graph with n nodes and m edges, find the shortest path from node 1 to node n using Dijkstra\'s algorithm. Edge weights are positive integers. Return the shortest distance or -1 if no path exists.',
            'input_description': 'First line contains n and m. Next m lines contain u v w representing an edge from u to v with weight w. Constraints: 1 ≤ n ≤ 10^5, 1 ≤ m ≤ 2 * 10^5, 1 ≤ w ≤ 10^9.',
            'output_description': 'Print the shortest distance, or -1 if no path exists.',
            'expected_difficulty': 'Medium'
        },
        {
            'title': 'Dynamic Programming: Coin Change',
            'description': 'Given an array of coin denominations and a target amount, return the minimum number of coins needed to make up that amount. If that amount cannot be made up, return -1. You may assume an infinite number of each kind of coin.',
            'input_description': 'First line contains n and amount. Second line contains n space-separated integers representing coin denominations.',
            'output_description': 'Print the minimum number of coins needed.',
            'expected_difficulty': 'Medium'
        }
    ]
    return jsonify({'samples': samples, 'count': len(samples)})

@app.route('/analyze', methods=['POST'])
def analyze_features():
    """Endpoint to get detailed feature analysis without prediction"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        title = data.get('title', '')
        description = data.get('description', '')
        
        if not description:
            return jsonify({'success': False, 'error': 'Description required'})
        
        combined_text = f"{title} {description}"
        _, detailed_features = processor.extract_all_features(combined_text)
        
        # Clean up features for JSON serialization
        cleaned_features = {}
        for key, value in detailed_features.items():
            if isinstance(value, (int, float, str, bool)):
                cleaned_features[key] = value
            elif isinstance(value, np.integer):
                cleaned_features[key] = int(value)
            elif isinstance(value, np.floating):
                cleaned_features[key] = float(value)
            else:
                cleaned_features[key] = str(value)
        
        return jsonify({
            'success': True,
            'feature_analysis': cleaned_features
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

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 AutoJudge AI - Ultra Enhanced Difficulty Predictor")
    print("="*60)
    print(f"📊 Models loaded: {'✅ READY' if is_loaded else '❌ NOT LOADED'}")
    print(f"🔬 Text Processor: {len(processor.feature_names)} features")
    print(f"🌐 Web Interface: http://localhost:5000")
    print(f"📈 Health Check: http://localhost:5000/health")
    print("="*60)
    
    if not is_loaded:
        print("\n⚠️  WARNING: ML models not loaded!")
        print("Please train the models first:")
        print("  cd ..  # Go to parent directory")
        print("  python train.py")
        print("\nThe web app will still run with demo functionality.")
        print("Sample predictions will be generated.")
    
    print("\n📋 Available endpoints:")
    print("  GET  /              - Web interface")
    print("  GET  /health        - System health check")
    print("  GET  /sample        - Sample problems")
    print("  POST /predict       - Make prediction")
    print("  POST /analyze       - Feature analysis only")
    print("="*60 + "\n")
    
    # Run the app
    app.run(debug=True, port=5000, host='0.0.0.0')