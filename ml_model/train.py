import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# DETERMINISM & REPRODUCIBILITY
# ==============================================
SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif

# ==============================================
# ULTRA-ENHANCED TextProcessor Class
# ==============================================

class UltraEnhancedTextProcessor:
    """Ultra-enhanced text processor with 50+ features and deterministic preprocessing"""
    
    def __init__(self):
        # Set random seed for reproducibility
        np.random.seed(SEED)
        
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
    
    def extract_algorithm_features(self, text: str) -> Dict[str, float]:
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
    
    def extract_constraint_features(self, text: str) -> Dict[str, float]:
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
    
    def extract_mathematical_features(self, text: str) -> Dict[str, float]:
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
    
    def extract_structural_features(self, text: str) -> Dict[str, float]:
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
    
    def extract_language_complexity_features(self, text: str) -> Dict[str, float]:
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
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
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
    
    def extract_composite_scores(self, features: Dict[str, float]) -> Dict[str, float]:
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
    
    def extract_all_features(self, text: str) -> Tuple[np.ndarray, Dict[str, float]]:
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

# ==============================================
# ADVANCED DATA QUALITY ENFORCER - CORRECTED
# ==============================================

class AdvancedDataQualityEnforcer:
    """Enforce data quality rules with advanced validation"""
    
    def __init__(self):
        self.corrections = []
        self.removals = []
        self.stats = {}
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset statistics"""
        stats = {
            'total_samples': len(df),
            'class_distribution': df['problem_class'].value_counts().to_dict(),
            'score_stats': {
                'mean': df['problem_score'].mean(),
                'std': df['problem_score'].std(),
                'min': df['problem_score'].min(),
                'max': df['problem_score'].max()
            },
            'text_length_stats': {}
        }
        
        # Analyze text lengths
        text_lengths = []
        for idx, row in df.iterrows():
            combined = f"{row['title']} {row['description']} {row['input_description']} {row['output_description']}"
            text_lengths.append(len(combined.split()))
        
        stats['text_length_stats'] = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': min(text_lengths),
            'max': max(text_lengths)
        }
        
        return stats
    
    def enforce_score_class_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce strict boundaries with intelligent corrections - FIXED FOR LOWERCASE"""
        df_clean = df.copy()
        
        # First normalize all class names to Title Case
        df_clean['problem_class'] = df_clean['problem_class'].apply(
            lambda x: str(x).strip().title() if pd.notnull(x) else 'Medium'
        )
        
        # Define strict boundaries with buffer zones
        boundaries = {
            'Easy': (0, 4),    # score ≤ 4
            'Medium': (4, 7),  # 4 < score ≤ 7
            'Hard': (7, 10)    # score > 7
        }
        
        buffer_zones = {
            'Easy-Medium': (3.8, 4.2),
            'Medium-Hard': (6.8, 7.2)
        }
        
        class_order = ['Easy', 'Medium', 'Hard']
        
        for idx, row in df_clean.iterrows():
            score = row['problem_score']
            actual_class = row['problem_class']
            
            # Ensure actual_class is in the expected format
            if actual_class not in class_order:
                # Try to match case-insensitively
                actual_class_lower = actual_class.lower()
                if actual_class_lower == 'easy':
                    actual_class = 'Easy'
                elif actual_class_lower == 'medium':
                    actual_class = 'Medium'
                elif actual_class_lower == 'hard' or actual_class_lower == 'difficult':
                    actual_class = 'Hard'
                else:
                    # Default to Medium if unknown
                    actual_class = 'Medium'
                df_clean.at[idx, 'problem_class'] = actual_class
            
            # Determine expected class based on score
            expected_class = None
            for cls, (low, high) in boundaries.items():
                if low < score <= high:
                    expected_class = cls
                    break
            
            # Handle edge cases
            if score <= 4:
                expected_class = 'Easy'
            elif score > 7:
                expected_class = 'Hard'
            elif score == 4:
                expected_class = 'Easy'
            elif score == 7:
                expected_class = 'Medium'
            
            # Check if in buffer zone
            in_buffer = False
            buffer_name = None
            for buffer, (low, high) in buffer_zones.items():
                if low <= score <= high:
                    in_buffer = True
                    buffer_name = buffer
                    break
            
            # Handle inconsistencies
            if expected_class and actual_class != expected_class:
                if in_buffer:
                    # In buffer zone, we can be more lenient
                    actual_idx = class_order.index(actual_class)
                    expected_idx = class_order.index(expected_class)
                    
                    if abs(actual_idx - expected_idx) == 1:
                        # Adjacent classes in buffer zone - keep as is
                        pass
                    else:
                        # Non-adjacent classes in buffer zone - correct
                        df_clean.at[idx, 'problem_class'] = expected_class
                        self.corrections.append({
                            'index': idx,
                            'old_class': actual_class,
                            'new_class': expected_class,
                            'score': score,
                            'buffer_zone': buffer_name,
                            'action': 'corrected_buffer'
                        })
                else:
                    # Not in buffer zone - enforce correction
                    df_clean.at[idx, 'problem_class'] = expected_class
                    self.corrections.append({
                        'index': idx,
                        'old_class': actual_class,
                        'new_class': expected_class,
                        'score': score,
                        'action': 'corrected'
                    })
        
        return df_clean
    
    def remove_noisy_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove samples with poor quality text"""
        df_clean = df.copy()
        
        to_remove = []
        for idx, row in df_clean.iterrows():
            combined = f"{row['title']} {row['description']} {row['input_description']} {row['output_description']}"
            words = combined.split()
            
            # Rule 1: Too short
            if len(words) < 15:
                to_remove.append(idx)
                self.removals.append({
                    'index': idx,
                    'reason': 'text_too_short',
                    'word_count': len(words)
                })
                continue
            
            # Rule 2: Too long (potential noise)
            if len(words) > 2000:
                to_remove.append(idx)
                self.removals.append({
                    'index': idx,
                    'reason': 'text_too_long',
                    'word_count': len(words)
                })
                continue
            
            # Rule 3: High repetition
            if len(words) > 50:
                trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
                unique_trigrams = len(set(trigrams))
                if unique_trigrams / len(trigrams) < 0.25:
                    to_remove.append(idx)
                    self.removals.append({
                        'index': idx,
                        'reason': 'high_repetition',
                        'unique_trigram_ratio': unique_trigrams / len(trigrams)
                    })
                    continue
            
            # Rule 4: Missing critical sections
            combined_lower = combined.lower()
            has_input = 'input' in combined_lower
            has_output = 'output' in combined_lower
            if not (has_input and has_output):
                to_remove.append(idx)
                self.removals.append({
                    'index': idx,
                    'reason': 'missing_critical_sections',
                    'has_input': has_input,
                    'has_output': has_output
                })
                continue
        
        df_clean = df_clean.drop(to_remove).reset_index(drop=True)
        return df_clean
    
    def balance_classes_smart(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart class balancing using stratified sampling"""
        # First normalize class names
        df_clean = df.copy()
        df_clean['problem_class'] = df_clean['problem_class'].apply(
            lambda x: str(x).strip().title() if pd.notnull(x) else 'Medium'
        )
        
        class_counts = df_clean['problem_class'].value_counts()
        
        # Target balanced distribution
        target_count = int(class_counts.mean())
        
        balanced_dfs = []
        for cls in df_clean['problem_class'].unique():
            cls_df = df_clean[df_clean['problem_class'] == cls]
            
            if len(cls_df) > target_count:
                # Use stratified sampling based on score distribution
                # Create bins based on score
                cls_df['score_bin'] = pd.cut(cls_df['problem_score'], bins=5)
                
                # Sample proportionally from each bin
                sampled = cls_df.groupby('score_bin', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), max(1, int(target_count / 5))), 
                                      random_state=SEED, replace=False)
                )
                
                # If we didn't get enough samples, sample more from largest bins
                if len(sampled) < target_count:
                    additional_needed = target_count - len(sampled)
                    remaining = cls_df.drop(sampled.index)
                    additional = remaining.sample(n=min(additional_needed, len(remaining)), 
                                                 random_state=SEED)
                    sampled = pd.concat([sampled, additional])
                
                cls_df = sampled.drop(columns=['score_bin'])
            balanced_dfs.append(cls_df)
        
        return pd.concat(balanced_dfs).reset_index(drop=True)
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps with detailed reporting"""
        print("  Step 1: Analyzing dataset...")
        self.stats = self.analyze_dataset(df)
        print(f"    Initial stats: {self.stats}")
        
        print("  Step 2: Enforcing score-class consistency...")
        df_clean = self.enforce_score_class_consistency(df)
        
        print("  Step 3: Removing noisy samples...")
        df_clean = self.remove_noisy_samples(df_clean)
        
        print("  Step 4: Balancing classes...")
        df_clean = self.balance_classes_smart(df_clean)
        
        # Final stats
        final_stats = self.analyze_dataset(df_clean)
        
        print(f"\n  Cleaning Summary:")
        print(f"    Initial samples: {self.stats['total_samples']}")
        print(f"    Corrections made: {len(self.corrections)}")
        print(f"    Samples removed: {len(self.removals)}")
        print(f"    Final dataset size: {len(df_clean)}")
        print(f"    Class distribution: {final_stats['class_distribution']}")
        print(f"    Score statistics: Mean={final_stats['score_stats']['mean']:.2f}, "
              f"Std={final_stats['score_stats']['std']:.2f}")
        
        return df_clean

# ==============================================
# MAIN TRAINING PIPELINE
# ==============================================

def main():
    print("="*80)
    print("AUTOJUDGE - ADVANCED TRAINING PIPELINE")
    print("="*80)
    print("\nDesigned for: Maximum Accuracy, Determinism, and Explainability")
    print(f"Random Seed: {SEED}")
    
    # Step 1: Load Data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    try:
        data = []
        with open('../data/dataset.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        
        # Rename columns to match expected format
        column_mapping = {
            'input': 'input_description',
            'output': 'output_description',
            'difficulty': 'problem_class',
            'score': 'problem_score'
        }
        df = df.rename(columns=column_mapping)
        
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        print(f"Class distribution: {df['problem_class'].value_counts().to_dict()}")
        
    except FileNotFoundError:
        print("❌ Error: dataset.jsonl not found at ../data/dataset.jsonl")
        print("Please create the dataset first or check the path.")
        return
    
    # Step 2: Data Quality Enhancement
    print("\n" + "="*60)
    print("STEP 2: DATA QUALITY ENHANCEMENT")
    print("="*60)
    
    enforcer = AdvancedDataQualityEnforcer()
    df_clean = enforcer.clean_dataset(df)
    
    # Step 3: Initialize Text Processor
    print("\n" + "="*60)
    print("STEP 3: INITIALIZING TEXT PROCESSOR")
    print("="*60)
    
    processor = UltraEnhancedTextProcessor()
    print(f"Processor initialized with {len(processor.feature_names)} features")
    print(f"Feature categories: Text({6}), Algorithm({12}), Constraint({8}), "
          f"Math({8}), Structural({8}), Language({4}), Composite({6})")
    
    # Step 4: Train Models with simplified trainer
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING (SIMPLIFIED)")
    print("="*60)
    
    # Use simplified trainer to avoid cross-validation errors
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data
    label_encoder = LabelEncoder()
    y_class = label_encoder.fit_transform(df_clean['problem_class'])
    y_score = df_clean['problem_score'].values
    
    # Create features
    print("    Creating features...")
    text_features_list = []
    numeric_features_list = []
    
    for idx, row in df_clean.iterrows():
        combined_text = (
            f"{row['title']} {row['description']} "
            f"{row['input_description']} {row['output_description']}"
        )
        
        # Extract numeric features
        feature_vector, _ = processor.extract_all_features(combined_text)
        numeric_features_list.append(feature_vector.flatten())
        
        # Store text for TF-IDF
        processed_text = processor.deterministic_preprocess(combined_text)
        text_features_list.append(processed_text)
    
    # TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True,
        analyzer='word'
    )
    X_tfidf = vectorizer.fit_transform(text_features_list)
    
    # Numeric features
    X_numeric = np.array(numeric_features_list)
    print(f"    Numeric features shape: {X_numeric.shape}")
    
    # Scale numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    # Combine features
    X = hstack([X_tfidf, X_numeric_scaled])
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2,
        random_state=SEED, stratify=y_class
    )
    
    # Train classifier
    print("\n5. Training classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=-1
    )
    classifier.fit(X_train, y_class_train)
    
    # Train regressor
    print("6. Training regressor...")
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=SEED,
        n_jobs=-1
    )
    regressor.fit(X_train, y_score_train)
    
    # Predict on test set
    print("\n7. Making predictions...")
    y_class_pred = classifier.predict(X_test)
    y_class_proba = classifier.predict_proba(X_test)
    y_score_pred = regressor.predict(X_test)
    
    # Enforce consistency
    y_class_pred_labels = label_encoder.inverse_transform(y_class_pred)
    y_class_pred_labels = np.array([cls.title() for cls in y_class_pred_labels])
    
    # Adjust scores based on class
    for i, (cls, score) in enumerate(zip(y_class_pred_labels, y_score_pred)):
        if cls == 'Easy' and score > 4:
            y_score_pred[i] = 3.5
        elif cls == 'Medium' and (score <= 4 or score > 7):
            y_score_pred[i] = 5.5
        elif cls == 'Hard' and score <= 7:
            y_score_pred[i] = 8.0
    
    y_score_pred = np.clip(y_score_pred, 0, 10)
    
    # Step 5: Evaluate Models
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    
    # Classification evaluation
    print("\nCLASSIFICATION EVALUATION")
    print("="*60)
    
    accuracy = accuracy_score(y_class_test, y_class_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_class_test, y_class_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_class_test, y_class_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('../web_app/static/confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("\nConfusion matrix saved to web_app/static/confusion_matrix.png")
    
    # Regression evaluation
    print("\nREGRESSION EVALUATION")
    print("="*60)
    
    mae = mean_absolute_error(y_score_test, y_score_pred)
    rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
    r2 = r2_score(y_score_test, y_score_pred)
    
    print(f"\nMAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Score distribution plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_score_test, y_score_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    plt.plot([0, 10], [0, 10], 'r--', alpha=0.5, label='Perfect Prediction')
    plt.axvline(x=4, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=7, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=4, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=7, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Actual vs Predicted Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../web_app/static/regression_plot.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Regression plot saved to web_app/static/regression_plot.png")
    
    # Step 6: Save Models and Artifacts
    print("\n" + "="*60)
    print("STEP 6: SAVING MODELS AND ARTIFACTS")
    print("="*60)
    
    # Create directory
    os.makedirs('saved_models', exist_ok=True)
    
    # Save all artifacts
    artifacts = {
        'classifier': classifier,
        'regressor': regressor,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'text_processor': processor,
        'training_info': {
            'feature_names': processor.feature_names,
            'class_names': list(label_encoder.classes_),
            'num_features': len(processor.feature_names),
            'tfidf_features': vectorizer.max_features,
            'random_seed': SEED,
            'training_date': pd.Timestamp.now().isoformat(),
            'model_version': '2.0.0'
        }
    }
    
    # Save each artifact
    for name, artifact in artifacts.items():
        if artifact is not None:
            path = f'saved_models/{name}.pkl'
            joblib.dump(artifact, path, compress=3)
            print(f"✅ Saved {name} to {path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\n" + "="*60)
    print("SYSTEM SUMMARY")
    print("="*60)
    print("✅ 52 engineered features with clear interpretation")
    print("✅ Deterministic preprocessing (identical training/inference)")
    print("✅ Advanced algorithm detection with hierarchical weights")
    print("✅ Strict class-score consistency enforcement")
    print("✅ Full reproducibility guaranteed")
    print("✅ Production-ready models saved")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Classification Accuracy: {accuracy:.4f}")
    print(f"   Regression R² Score: {r2:.4f}")
    
    print(f"\n📁 Models saved to: saved_models/")
    print(f"📈 Visualizations saved to: web_app/static/")
    
    print("\n🚀 System is ready for deployment with guaranteed determinism!")

if __name__ == "__main__":
    main()