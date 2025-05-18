""" 
NIMH Multi-Institutional Schizophrenia Language Analysis Framework

IMPORTANT DISCLAIMER:
---------------------
This code is for EDUCATIONAL PURPOSES ONLY and demonstrates a theoretical approach.
- This is NOT an actual implementation with real NIMH or hospital data
- Access to such data would require IRB approval, HIPAA compliance, and formal research partnerships
- Real clinical implementation would require rigorous validation and regulatory approval
- Actual diagnosis requires comprehensive clinical assessment beyond language analysis
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import re

# Load SpaCy English model for advanced linguistic analysis
try:
    nlp = spacy.load("en_core_web_lg")
except:
    print("Installing SpaCy model...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

class NIMHSchizophreniaLanguageAnalyzer:
    """
    Advanced framework for analyzing language patterns associated with schizophrenia
    using clinical data from top psychiatric institutions.
    
    Theoretical implementation that would integrate with the NIMH Data Archive (NDA)
    and research datasets from leading psychiatric hospitals.
    """
    
    def __init__(self, config=None):
        """
        Initialize the analyzer with configuration settings.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Default configuration
        self.config = {
            'max_features': 10000,            # Maximum number of words in vocabulary
            'max_sequence_length': 500,       # Maximum length of text sequences
            'embedding_dim': 300,             # Dimension of word embeddings
            'lstm_units': 128,                # Number of LSTM units
            'dropout_rate': 0.3,              # Dropout rate
            'batch_size': 32,                 # Batch size for training
            'epochs': 10,                     # Number of training epochs
            'validation_split': 0.2,          # Validation split ratio
            'learning_rate': 0.001,           # Learning rate
            'early_stopping_patience': 3,     # Patience for early stopping
            'cross_validation_folds': 5,      # Number of cross-validation folds
            'random_state': 42,               # Random seed
            'model_dir': 'models',            # Directory to save models
            'results_dir': 'results',         # Directory to save results
            'data_augmentation': True,        # Whether to use data augmentation
            'class_weights': {0: 1.0, 1: 2.0} # Class weights to handle imbalance
        }
        
        # Update with user-provided configuration
        if config is not None:
            self.config.update(config)
        
        # Create directories
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = None
        
        # Initialize model
        self.model = None
        
        # Initialize feature extractors
        self.initialize_feature_extractors()
        
        # Track training history
        self.history = None
        
        # Track institution-specific performance
        self.institution_performance = {}
        
        # Model metadata
        self.metadata = {
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'version': '0.1.0',
            'description': 'NIMH Multi-Institutional Schizophrenia Language Analysis',
            'features': [],
            'institutions': [],
            'performance': {},
            'demographics': {}
        }
    
    def initialize_feature_extractors(self):
        """Initialize all linguistic feature extractors."""
        # NLTK setup
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.feature_extractors = {
            # Basic features
            'basic_stats': self._extract_basic_stats,
            'sentence_features': self._extract_sentence_features,
            'word_usage': self._extract_word_usage_features,
            
            # Advanced features
            'coherence': self._extract_coherence_features,
            'complexity': self._extract_complexity_features,
            'semantic_density': self._extract_semantic_density,
            'speech_graph': self._extract_speech_graph_features,
            
            # Schizophrenia-specific linguistic markers
            'thought_disorder': self._extract_thought_disorder_markers,
            'referential_markers': self._extract_referential_markers,
            'tangentiality': self._extract_tangentiality
        }
    
    def _extract_basic_stats(self, text):
        """Extract basic statistical features from text."""
        doc = nlp(text)
        sentences = list(doc.sents)
        
        return {
            'num_tokens': len(doc),
            'num_sentences': len(sentences),
            'avg_tokens_per_sentence': len(doc) / len(sentences) if sentences else 0,
            'num_unique_tokens': len(set([token.text.lower() for token in doc])),
            'lexical_diversity': len(set([token.lemma_ for token in doc])) / len(doc) if len(doc) > 0 else 0
        }
    
    def _extract_sentence_features(self, text):
        """Extract features related to sentence structure."""
        doc = nlp(text)
        sentences = list(doc.sents)
        
        # Calculate sentence lengths
        sentence_lengths = [len(sent) for sent in sentences]
        
        return {
            'mean_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'median_sentence_length': np.median(sentence_lengths) if sentence_lengths else 0,
            'std_sentence_length': np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0,
            'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0
        }
    
    def _extract_word_usage_features(self, text):
        """Extract features related to word usage patterns."""
        doc = nlp(text)
        
        # POS tag counts
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        # Calculate POS tag ratios
        total_tokens = len(doc)
        features = {}
        
        if total_tokens > 0:
            features.update({
                f'ratio_{pos}': count / total_tokens 
                for pos, count in pos_counts.items()
            })
            
            # Specific ratios of interest
            features['noun_verb_ratio'] = (pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)) / max(1, pos_counts.get('VERB', 0))
            features['content_function_ratio'] = (
                pos_counts.get('NOUN', 0) + pos_counts.get('VERB', 0) + pos_counts.get('ADJ', 0) + pos_counts.get('ADV', 0)
            ) / max(1, (
                pos_counts.get('ADP', 0) + pos_counts.get('DET', 0) + pos_counts.get('PRON', 0) + 
                pos_counts.get('CCONJ', 0) + pos_counts.get('SCONJ', 0)
            ))
            
        # Word repetition features
        word_counts = {}
        for token in doc:
            if not token.is_punct and not token.is_space:
                word = token.text.lower()
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # Calculate repetition metrics
        if word_counts:
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            features['word_repetition_ratio'] = repeated_words / len(word_counts)
            
            # Calculate repetition entropy
            counts = np.array(list(word_counts.values()))
            probs = counts / counts.sum()
            features['word_repetition_entropy'] = -np.sum(probs * np.log2(probs))
        else:
            features['word_repetition_ratio'] = 0
            features['word_repetition_entropy'] = 0
            
        return features
    
    def _extract_coherence_features(self, text):
        """Extract features related to semantic coherence."""
        doc = nlp(text)
        sentences = list(doc.sents)
        
        features = {}
        
        # Calculate semantic similarity between adjacent sentences
        if len(sentences) > 1:
            adjacent_similarities = []
            for i in range(len(sentences) - 1):
                s1 = sentences[i]
                s2 = sentences[i + 1]
                if len(s1) > 0 and len(s2) > 0:  # Ensure non-empty sentences
                    similarity = s1.similarity(s2)
                    adjacent_similarities.append(similarity)
            
            if adjacent_similarities:
                features['mean_adjacent_similarity'] = np.mean(adjacent_similarities)
                features['min_adjacent_similarity'] = min(adjacent_similarities)
                features['std_adjacent_similarity'] = np.std(adjacent_similarities)
            else:
                features['mean_adjacent_similarity'] = 0
                features['min_adjacent_similarity'] = 0
                features['std_adjacent_similarity'] = 0
        else:
            features['mean_adjacent_similarity'] = 0
            features['min_adjacent_similarity'] = 0
            features['std_adjacent_similarity'] = 0
            
        # Calculate overall semantic coherence across the document
        if len(sentences) > 2:
            all_similarities = []
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    s1 = sentences[i]
                    s2 = sentences[j]
                    if len(s1) > 0 and len(s2) > 0:  # Ensure non-empty sentences
                        similarity = s1.similarity(s2)
                        all_similarities.append(similarity)
            
            if all_similarities:
                features['global_coherence'] = np.mean(all_similarities)
                features['global_coherence_std'] = np.std(all_similarities)
            else:
                features['global_coherence'] = 0
                features['global_coherence_std'] = 0
        else:
            features['global_coherence'] = 0
            features['global_coherence_std'] = 0
            
        return features
    
    def _extract_complexity_features(self, text):
        """Extract features related to linguistic complexity."""
        doc = nlp(text)
        
        # Track dependency tree depth
        max_depth = 0
        sum_depths = 0
        num_roots = 0
        
        for sent in doc.sents:
            # Find the root
            root = None
            for token in sent:
                if token.dep_ == 'ROOT':
                    root = token
                    num_roots += 1
                    break
            
            if root:
                # Calculate depth for this sentence
                depths = self._get_dependency_depths(root)
                if depths:
                    max_depth = max(max_depth, max(depths))
                    sum_depths += sum(depths)
        
        features = {
            'max_dependency_depth': max_depth,
            'avg_dependency_depth': sum_depths / num_roots if num_roots > 0 else 0
        }
        
        # Clausal complexity
        num_subordinate_clauses = len([token for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl']])
        features['subordinate_clause_ratio'] = num_subordinate_clauses / len(list(doc.sents)) if doc.sents else 0
        
        # Named entity complexity
        named_entities = list(doc.ents)
        features['named_entity_ratio'] = len(named_entities) / len(doc) if len(doc) > 0 else 0
        
        return features
    
    def _get_dependency_depths(self, root):
        """Helper function to get dependency tree depths."""
        visited = set()
        depths = []
        
        def dfs(node, depth):
            if node in visited:
                return
            visited.add(node)
            
            # Count children
            children = [child for child in node.children]
            if not children:  # Leaf node
                depths.append(depth)
            
            # Continue DFS
            for child in children:
                dfs(child, depth + 1)
        
        dfs(root, 0)
        return depths
    
    def _extract_semantic_density(self, text):
        """Extract semantic density features."""
        doc = nlp(text)
        
        # Count meaningful content words
        content_words = [token for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']]
        
        # Count all non-punctuation tokens
        all_words = [token for token in doc if not token.is_punct]
        
        # Calculate semantic density ratio
        semantic_density = len(content_words) / len(all_words) if all_words else 0
        
        # Get average word vector norm as a measure of semantic specificity
        content_vector_norms = [token.vector_norm for token in content_words if token.has_vector]
        avg_vector_norm = np.mean(content_vector_norms) if content_vector_norms else 0
        
        return {
            'semantic_density': semantic_density,
            'avg_vector_norm': avg_vector_norm
        }
    
    def _extract_speech_graph_features(self, text):
        """Extract features based on speech graph analysis."""
        # Simplified speech graph analysis
        words = [token.text.lower() for token in nlp(text) if not token.is_punct]
        
        # Create graph edges (adjacent words)
        edges = set()
        for i in range(len(words) - 1):
            edges.add((words[i], words[i+1]))
        
        # Count unique vertices and edges
        vertices = set(words)
        
        # Calculate graph metrics
        n_vertices = len(vertices)
        n_edges = len(edges)
        
        # Calculate graph density
        density = 2 * n_edges / (n_vertices * (n_vertices - 1)) if n_vertices > 1 else 0
        
        return {
            'graph_vertices': n_vertices,
            'graph_edges': n_edges,
            'graph_density': density,
            'edges_per_vertex': n_edges / n_vertices if n_vertices > 0 else 0
        }
    
    def _extract_thought_disorder_markers(self, text):
        """Extract features that may correlate with thought disorder."""
        doc = nlp(text)
        sentences = list(doc.sents)
        
        # Calculate inter-sentence topic drift
        if len(sentences) > 1:
            # Get sentence vectors
            sent_vectors = [sent.vector for sent in sentences if len(sent) > 0]
            
            # Calculate drift (cosine distance between adjacent sentences)
            drifts = []
            for i in range(len(sent_vectors) - 1):
                v1_norm = np.linalg.norm(sent_vectors[i])
                v2_norm = np.linalg.norm(sent_vectors[i+1])
                if v1_norm > 0 and v2_norm > 0:
                    cosine_sim = np.dot(sent_vectors[i], sent_vectors[i+1]) / (v1_norm * v2_norm)
                    cosine_dist = 1 - cosine_sim
                    drifts.append(cosine_dist)
            
            avg_drift = np.mean(drifts) if drifts else 0
            max_drift = max(drifts) if drifts else 0
            drift_std = np.std(drifts) if len(drifts) > 1 else 0
        else:
            avg_drift = 0
            max_drift = 0
            drift_std = 0
        
        # Look for sentence fragments
        fragmented_sentences = 0
        for sent in sentences:
            has_subject = False
            has_verb = False
            for token in sent:
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    has_subject = True
                if token.pos_ == 'VERB':
                    has_verb = True
            
            if not (has_subject and has_verb) and len(sent) > 2:
                fragmented_sentences += 1
        
        fragment_ratio = fragmented_sentences / len(sentences) if sentences else 0
        
        return {
            'avg_topic_drift': avg_drift,
            'max_topic_drift': max_drift,
            'topic_drift_std': drift_std,
            'fragment_ratio': fragment_ratio
        }
    
    def _extract_referential_markers(self, text):
        """Extract features related to referential language."""
        doc = nlp(text)
        
        # Count pronouns
        personal_pronouns = [token for token in doc if token.pos_ == 'PRON' and token.lemma_ in ['i', 'me', 'my', 'mine', 'myself']]
        other_pronouns = [token for token in doc if token.pos_ == 'PRON' and token.lemma_ not in ['i', 'me', 'my', 'mine', 'myself']]
        
        # Count demonstratives (this, that, these, those)
        demonstratives = [token for token in doc if token.lemma_ in ['this', 'that', 'these', 'those']]
        
        # Calculate ratios
        total_tokens = len(doc)
        personal_pronoun_ratio = len(personal_pronouns) / total_tokens if total_tokens > 0 else 0
        other_pronoun_ratio = len(other_pronouns) / total_tokens if total_tokens > 0 else 0
        demonstrative_ratio = len(demonstratives) / total_tokens if total_tokens > 0 else 0
        
        # Check for clear referents (simplified)
        unclear_references = 0
        for token in other_pronouns + demonstratives:
            # Very simplified check - in reality would need coreference resolution
            if token.i > 0 and doc[token.i-1].pos_ not in ['NOUN', 'PROPN']:
                unclear_references += 1
        
        unclear_reference_ratio = unclear_references / (len(other_pronouns) + len(demonstratives)) if (len(other_pronouns) + len(demonstratives)) > 0 else 0
        
        return {
            'personal_pronoun_ratio': personal_pronoun_ratio,
            'other_pronoun_ratio': other_pronoun_ratio,
            'demonstrative_ratio': demonstrative_ratio,
            'unclear_reference_ratio': unclear_reference_ratio
        }
    
    def _extract_tangentiality(self, text):
        """Extract features related to tangentiality."""
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) < 3:
            return {
                'avg_tangentiality': 0,
                'max_tangentiality': 0
            }
        
        # Compare similarity between first sentence and subsequent sentences
        first_sent = sentences[0]
        if len(first_sent) == 0:
            return {
                'avg_tangentiality': 0,
                'max_tangentiality': 0
            }
        
        tangentiality_scores = []
        for i in range(1, len(sentences)):
            if len(sentences[i]) > 0:
                similarity = first_sent.similarity(sentences[i])
                # Convert similarity to tangentiality (higher means more tangential/less related)
                tangentiality = 1 - similarity
                tangentiality_scores.append(tangentiality)
        
        if not tangentiality_scores:
            return {
                'avg_tangentiality': 0,
                'max_tangentiality': 0
            }
        
        return {
            'avg_tangentiality': np.mean(tangentiality_scores),
            'max_tangentiality': max(tangentiality_scores)
        }
    
    def extract_all_features(self, text):
        """Extract all linguistic features from text."""
        features = {}
        
        for name, extractor in self.feature_extractors.items():
            try:
                feature_dict = extractor(text)
                for key, value in feature_dict.items():
                    features[f"{name}_{key}"] = value
            except Exception as e:
                print(f"Error extracting {name} features: {e}")
                # Insert placeholder values
                features[f"{name}_error"] = 1.0
        
        return features
    
    def load_dataset(self, data_path, institutions=None):
        """
        Load dataset from a file or directory.
        
        Args:
            data_path: Path to dataset file or directory
            institutions: List of institution names to include (None for all)
            
        Returns:
            Pandas DataFrame with dataset
        """
        print(f"Loading dataset from {data_path}...")
        
        # This is a placeholder - in reality would load actual data
        # For demonstration, we'll create a synthetic dataset
        
        # In a real implementation, this would:
        # 1. Connect to NIMH Data Archive API or load from secure files
        # 2. Apply proper data formatting and preprocessing
        # 3. Verify data integrity and consent/usage permissions
        
        # Synthetic data generation for demonstration
        np.random.seed(self.config['random_state'])
        test_institution = np.random.choice(institutions)
        
        train_df = df[df['institution'] != test_institution]
        test_df = df[df['institution'] == test_institution]
        
        print(f"Train-test institutional split: {len(train_df)} training samples, {len(test_df)} testing samples")
        print(f"Test institution: {test_institution}")
        
        return train_df, test_df
    
    def build_deep_learning_model(self, input_dim):
        """
        Build a deep learning model for linguistic feature analysis.
        
        Args:
            input_dim: Dimension of input features
            
        Returns:
            Compiled Keras model
        """
        print("Building deep learning model...")
        
        model = Sequential()
        
        # First hidden layer
        model.add(Dense(512, input_dim=input_dim, activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Second hidden layer
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Third hidden layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def build_lstm_model(self):
        """
        Build an LSTM model for raw text analysis.
        
        Returns:
            Compiled Keras model
        """
        print("Building LSTM model...")
        
        model = Sequential()
        
        # Embedding layer
        model.add(Embedding(
            input_dim=self.config['max_features'],
            output_dim=self.config['embedding_dim'],
            input_length=self.config['max_sequence_length']
        ))
        
        # Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(self.config['lstm_units'], return_sequences=True)))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Second LSTM layer
        model.add(Bidirectional(LSTM(self.config['lstm_units'])))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Dense layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def prepare_text_sequences(self, texts):
        """
        Prepare text sequences for LSTM model.
        
        Args:
            texts: List of text strings
            
        Returns:
            Padded sequences
        """
        # Initialize tokenizer if needed
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.config['max_features'])
            self.tokenizer.fit_on_texts(texts)
            
            # Save tokenizer
            with open(f"{self.config['model_dir']}/tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.config['max_sequence_length'],
            padding='post',
            truncating='post'
        )
        
        return padded_sequences
    
    def train_cross_validation(self, X, y, feature_names):
        """
        Train model using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Names of features
            
        Returns:
            Cross-validation results
        """
        print("Training model with cross-validation...")
        
        # Initialize cross-validation
        cv = StratifiedKFold(
            n_splits=self.config['cross_validation_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        # Initialize results storage
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        feature_importances = np.zeros(len(feature_names))
        
        # Perform cross-validation
        fold = 1
        for train_idx, val_idx in cv.split(X, y):
            print(f"\nTraining fold {fold}/{self.config['cross_validation_folds']}...")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and train model
            model = self.build_deep_learning_model(X.shape[1])
            
            # Training callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['early_stopping_patience'],
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(X_val, y_val),
                class_weight=self.config['class_weights'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Store metrics
            cv_scores['accuracy'].append(accuracy)
            cv_scores['precision'].append(precision)
            cv_scores['recall'].append(recall)
            cv_scores['f1'].append(f1)
            cv_scores['auc'].append(auc)
            
            print(f"Fold {fold} results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            
            # Feature importance analysis using a surrogate Random Forest
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
            rf.fit(X_train, y_train)
            feature_importances += rf.feature_importances_
            
            # Save fold model
            model.save(f"{self.config['model_dir']}/model_fold_{fold}.h5")
            
            fold += 1
        
        # Calculate average metrics
        avg_metrics = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        std_metrics = {metric: np.std(scores) for metric, scores in cv_scores.items()}
        
        print("\nCross-validation results:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
        
        # Normalize feature importances
        feature_importances /= self.config['cross_validation_folds']
        
        # Map importances to feature names
        importance_dict = dict(zip(feature_names, feature_importances))
        
        # Sort by importance
        sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 20 most important features:")
        for feature, importance in sorted_importances[:20]:
            print(f"  {feature}: {importance:.4f}")
        
        # Save results
        results = {
            'cv_scores': cv_scores,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'feature_importances': sorted_importances
        }
        
        # Update metadata
        self.metadata['performance'] = {
            'cross_validation': avg_metrics,
            'cross_validation_std': std_metrics
        }
        
        # Save to file
        with open(f"{self.config['results_dir']}/cv_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def train_final_model(self, X, y, X_test=None, y_test=None):
        """
        Train final model on all training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            X_test: Test feature matrix (optional)
            y_test: Test labels (optional)
            
        Returns:
            Trained model
        """
        print("Training final model on all data...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config['validation_split'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Build model
        model = self.build_deep_learning_model(X.shape[1])
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.config['model_dir']}/best_model.h5",
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, y_val),
            class_weight=self.config['class_weights'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        with open(f"{self.config['results_dir']}/training_history.json", 'w') as f:
            history_dict = {}
            for k, v in history.history.items():
                history_dict[k] = [float(val) for val in v]
            json.dump(history_dict, f, indent=2)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.config['results_dir']}/training_history.png")
        
        # Evaluate on validation set
        val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(X_val, y_val)
        
        print("\nValidation results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(X_test, y_test)
            
            print("\nTest results:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  AUC: {test_auc:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall: {test_recall:.4f}")
            
            # Calculate confusion matrix
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix (Test Set)')
            plt.savefig(f"{self.config['results_dir']}/confusion_matrix.png")
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(f"{self.config['results_dir']}/roc_curve.png")
            
            # Update metadata
            self.metadata['performance']['test'] = {
                'accuracy': float(test_acc),
                'auc': float(test_auc),
                'precision': float(test_precision),
                'recall': float(test_recall)
            }
        
        # Save model
        model.save(f"{self.config['model_dir']}/final_model.h5")
        
        # Save metadata
        with open(f"{self.config['model_dir']}/metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.model = model
        return model
    
    def train_lstm_model(self, df):
        """
        Train LSTM model on raw text data.
        
        Args:
            df: DataFrame with text data
            
        Returns:
            Trained LSTM model
        """
        print("Training LSTM model on raw text...")
        
        # Prepare sequences
        X = self.prepare_text_sequences(df['text'].tolist())
        y = df['diagnosis'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config['validation_split'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Build model
        model = self.build_lstm_model()
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.config['model_dir']}/best_lstm_model.h5",
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, y_val),
            class_weight=self.config['class_weights'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model.save(f"{self.config['model_dir']}/lstm_model.h5")
        
        # Evaluate model
        val_loss, val_acc, val_auc, val_precision, val_recall = model.evaluate(X_val, y_val)
        
        print("\nLSTM Validation results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  AUC: {val_auc:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        
        # Update metadata
        self.metadata['performance']['lstm'] = {
            'accuracy': float(val_acc),
            'auc': float(val_auc),
            'precision': float(val_precision),
            'recall': float(val_recall)
        }
        
        return model
    
    def evaluate_institution_performance(self, df, model):
        """
        Evaluate model performance by institution.
        
        Args:
            df: DataFrame with institution column
            model: Trained model
            
        Returns:
            Dictionary with institution-specific performance metrics
        """
        print("Evaluating performance by institution...")
        
        # Get unique institutions
        institutions = df['institution'].unique()
        
        # Initialize results dictionary
        institution_results = {}
        
        # Prepare features for each institution
        feature_cols = [col for col in df.columns if col not in [
            'participant_id', 'institution', 'diagnosis', 'age', 'gender',
            'education_years', 'text'
        ]]
        
        # Load scaler
        with open(f"{self.config['model_dir']}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Evaluate for each institution
        for institution in institutions:
            inst_df = df[df['institution'] == institution]
            
            # Prepare data
            X_inst = inst_df[feature_cols].values
            X_inst = scaler.transform(X_inst)
            y_inst = inst_df['diagnosis'].values
            
            # Evaluate
            loss, acc, auc_score, precision, recall = model.evaluate(X_inst, y_inst, verbose=0)
            
            # Calculate F1 score
            y_pred = (model.predict(X_inst) > 0.5).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_inst, y_pred)
            
            # Store results
            institution_results[institution] = {
                'samples': len(inst_df),
                'accuracy': float(acc),
                'auc': float(auc_score),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }
            
            print(f"Institution: {institution}")
            print(f"  Samples: {len(inst_df)}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  AUC: {auc_score:.4f}")
            print(f"  F1: {f1:.4f}")
        
        # Save results
        with open(f"{self.config['results_dir']}/institution_performance.json", 'w') as f:
            json.dump(institution_results, f, indent=2)
        
        # Update metadata
        self.metadata['institution_performance'] = institution_results
        self.institution_performance = institution_results
        
        return institution_results
    
    def analyze_error_cases(self, df, model):
        """
        Analyze error cases to understand model limitations.
        
        Args:
            df: DataFrame with data
            model: Trained model
            
        Returns:
            DataFrame with error analysis
        """
        print("Analyzing error cases...")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in [
            'participant_id', 'institution', 'diagnosis', 'age', 'gender',
            'education_years', 'text'
        ]]
        
        # Load scaler
        with open(f"{self.config['model_dir']}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare data
        X = df[feature_cols].values
        X = scaler.transform(X)
        y_true = df['diagnosis'].values
        
        # Get predictions
        y_pred_proba = model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Identify errors
        errors = y_pred != y_true
        error_indices = np.where(errors)[0]
        
        # Create error analysis DataFrame
        error_df = df.iloc[error_indices].copy()
        error_df['predicted_probability'] = y_pred_proba[error_indices]
        error_df['predicted_label'] = y_pred[error_indices]
        error_df['error_type'] = ['False Positive' if p == 1 else 'False Negative' for p in y_pred[error_indices]]
        
        # Analyze error patterns
        print(f"Total errors: {len(error_df)} ({len(error_df)/len(df)*100:.2f}%)")
        print(f"False positives: {sum(error_df['error_type'] == 'False Positive')}")
        print(f"False negatives: {sum(error_df['error_type'] == 'False Negative')}")
        
        # Error distribution by institution
        error_by_institution = error_df.groupby('institution')['error_type'].value_counts().unstack().fillna(0)
        print("\nError distribution by institution:")
        print(error_by_institution)
        
        # Error distribution by demographics
        print("\nError distribution by gender:")
        print(error_df.groupby('gender')['error_type'].value_counts().unstack().fillna(0))
        
        print("\nError distribution by age group:")
        error_df['age_group'] = pd.cut(error_df['age'], bins=[18, 30, 40, 50, 65])
        print(error_df.groupby('age_group')['error_type'].value_counts().unstack().fillna(0))
        
        # Save error analysis
        error_df.to_csv(f"{self.config['results_dir']}/error_analysis.csv", index=False)
        
        # Sample error cases
        sample_errors = error_df.sample(min(10, len(error_df)))
        
        print("\nSample error cases:")
        for i, row in sample_errors.iterrows():
            print(f"\nID: {row['participant_id']}")
            print(f"Institution: {row['institution']}")
            print(f"True diagnosis: {'Schizophrenia' if row['diagnosis'] == 1 else 'Control'}")
            print(f"Predicted: {'Schizophrenia' if row['predicted_label'] == 1 else 'Control'} (probability: {row['predicted_probability'][0]:.2f})")
            print(f"Gender: {row['gender']}, Age: {row['age']}")
            print(f"Text excerpt: {row['text'][:200]}...")
        
        return error_df
    
    def run_complete_pipeline(self, data_path, institutions=None):
        """
        Run complete analysis pipeline.
        
        Args:
            data_path: Path to data
            institutions: List of institutions to include (None for all)
            
        Returns:
            Results dictionary
        """
        print("Running complete analysis pipeline...")
        
        # Step 1: Load data
        df = self.load_dataset(data_path, institutions)
        
        # Step 2: Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Step 3: Institution split
        train_df, test_df = self.train_test_institution_split(processed_df)
        
        # Step 4: Prepare training data
        X_train, y_train, feature_names = self.prepare_data_for_training(train_df)
        X_test, y_test, _ = self.prepare_data_for_training(test_df)
        
        # Step 5: Cross-validation
        cv_results = self.train_cross_validation(X_train, y_train, feature_names)
        
        # Step 6: Train final model
        final_model = self.train_final_model(X_train, y_train, X_test, y_test)
        
        # Step 7: Institution-specific performance
        inst_results = self.evaluate_institution_performance(processed_df, final_model)
        
        # Step 8: Error analysis
        error_analysis = self.analyze_error_cases(processed_df, final_model)
        
        # Step 9: Train LSTM model
        lstm_model = self.train_lstm_model(processed_df)
        
        # Save combined metadata
        with open(f"{self.config['model_dir']}/metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print("\nAnalysis pipeline complete!")
        print(f"Models saved to: {self.config['model_dir']}")
        print(f"Results saved to: {self.config['results_dir']}")
        
        return {
            'cv_results': cv_results,
            'institution_results': inst_results,
            'metadata': self.metadata
        }
    
    def analyze_text(self, text):
        """
        Analyze a new text sample.
        
        Args:
            text: Text string to analyze
            
        Returns:
            Analysis results dictionary
        """
        print("Analyzing text sample...")
        
        # Extract features
        features = self.extract_all_features(text)
        
        # Convert to array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Load scaler
        with open(f"{self.config['model_dir']}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Predict with model
        prediction_proba = self.model.predict(scaled_features)[0][0]
        prediction = 1 if prediction_proba > 0.5 else 0
        
        # Analyze features
        important_features = {}
        
        # Feature interpretation - simple threshold-based analysis
        if features['thought_disorder_avg_topic_drift'] > 0.5:
            important_features['high_topic_drift'] = features['thought_disorder_avg_topic_drift']
            
        if features['thought_disorder_fragment_ratio'] > 0.3:
            important_features['sentence_fragmentation'] = features['thought_disorder_fragment_ratio']
            
        if features['referential_markers_unclear_reference_ratio'] > 0.4:
            important_features['unclear_references'] = features['referential_markers_unclear_reference_ratio']
            
        if features['coherence_mean_adjacent_similarity'] < 0.3:
            important_features['low_coherence'] = features['coherence_mean_adjacent_similarity']
            
        if features['tangentiality_max_tangentiality'] > 0.7:
            important_features['high_tangentiality'] = features['tangentiality_max_tangentiality']
        
        # LSTM analysis if available
        lstm_prediction = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            # Prepare sequence
            sequence = self.prepare_text_sequences([text])
            
            # Load LSTM model
            try:
                lstm_model = tf.keras.models.load_model(f"{self.config['model_dir']}/lstm_model.h5")
                lstm_prediction_proba = lstm_model.predict(sequence)[0][0]
                lstm_prediction = 1 if lstm_prediction_proba > 0.5 else 0
            except:
                print("LSTM model not available.")
        
        results = {
            'prediction': 'Patterns consistent with schizophrenia detected' if prediction == 1 else 'No significant patterns detected',
            'confidence': float(prediction_proba) if prediction == 1 else 1 - float(prediction_proba),
            'important_features': important_features
        }
        
        if lstm_prediction is not None:
            results['lstm_prediction'] = 'Patterns consistent with schizophrenia detected' if lstm_prediction == 1 else 'No significant patterns detected'
            results['lstm_confidence'] = float(lstm_prediction_proba) if lstm_prediction == 1 else 1 - float(lstm_prediction_proba)
        
        print("\nAnalysis results:")
        print(f"Prediction: {results['prediction']}")
        print(f"Confidence: {results['confidence']:.4f}")
        
        if important_features:
            print("\nSignificant linguistic patterns:")
            for feature, value in important_features.items():
                print(f"  {feature}: {value:.4f}")
        
        print("\nIMPORTANT CLINICAL DISCLAIMER:")
        print("This analysis is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.")
        print("It should NOT be used for diagnosis. Proper diagnosis of schizophrenia")
        print("requires comprehensive assessment by qualified healthcare professionals.")
        
        return results


# Example usage
def main():
    """Main function to demonstrate usage."""
    print("""
    NIMH Multi-Institutional Schizophrenia Language Analysis Framework
    -----------------------------------------------------------------
    DISCLAIMER: This code is for EDUCATIONAL PURPOSES ONLY and demonstrates a theoretical approach.
    This is NOT an actual implementation with real NIMH data.
    """)
    
    # Configuration
    config = [
        'max_features': 5000,
        'max_sequence_length': 300,
        'lstm_units': 'state']
        
        # Create list of top psychiatric hospitals (fictional data)
        all_institutions = [
            "McLean Hospital (Harvard)",
            "Massachusetts General Hospital",
            "Johns Hopkins Hospital",
            "New York-Presbyterian Hospital",
            "Mayo Clinic",
            "UCSF Medical Center",
            "Menninger Clinic",
            "Yale New Haven Hospital",
            "Sheppard Pratt",
            "Cleveland Clinic"
        ]
        
        # Filter institutions if specified
        if institutions is not None:
            selected_institutions = [inst for inst in all_institutions if inst in institutions]
        else:
            selected_institutions = all_institutions
            
        # Update metadata
        self.metadata['institutions'] = selected_institutions
            
        # Synthetic dataset size
        n_samples = 1000
        
        # Create synthetic dataset
        data = {
            'participant_id': [f"P{i:04d}" for i in range(n_samples)],
            'institution': np.random.choice(selected_institutions, n_samples),
            'diagnosis': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),  # 0=control, 1=schizophrenia
            'age': np.random.randint(18, 65, n_samples),
            'gender': np.random.choice(['M', 'F', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
            'education_years': np.random.randint(8, 22, n_samples),
            'text': []
        }
        
        # Generate synthetic texts
        for i in range(n_samples):
            is_schizophrenia = data['diagnosis'][i] == 1
            
            # Length of text (words)
            if is_schizophrenia:
                text_length = np.random.randint(50, 300)
                coherence = np.random.uniform(0.3, 0.7)  # Lower coherence
                tangentiality = np.random.uniform(0.3, 0.8)  # Higher tangentiality
            else:
                text_length = np.random.randint(100, 400)
                coherence = np.random.uniform(0.6, 0.9)  # Higher coherence
                tangentiality = np.random.uniform(0.1, 0.5)  # Lower tangentiality
                
            # This is a placeholder for text generation
            # In reality, this would be actual clinical transcripts
            
            if is_schizophrenia:
                # Very simplified synthetic text - NOT representative of actual speech patterns
                text = f"I was thinking about {np.random.choice(['things', 'stuff', 'people', 'ideas'])} lately. "
                text += f"Sometimes they {np.random.choice(['watch', 'listen', 'follow', 'know'])} what I'm thinking. "
                text += f"The {np.random.choice(['government', 'doctors', 'neighbors', 'signals'])} can "
                text += f"{np.random.choice(['track', 'monitor', 'record', 'influence'])} my thoughts. "
                text += f"I've been hearing {np.random.choice(['voices', 'sounds', 'messages', 'warnings'])} "
                text += f"when {np.random.choice(['alone', 'sleeping', 'thinking', 'working'])}. "
                text += f"Sometimes the TV {np.random.choice(['talks to me', 'has hidden messages', 'knows my thoughts', 'shows my future'])}. "
            else:
                # Control text pattern
                text = f"I've been {np.random.choice(['working', 'studying', 'living', 'staying'])} in "
                text += f"{np.random.choice(['the city', 'this area', 'my apartment', 'the neighborhood'])} for about "
                text += f"{np.random.randint(1, 10)} years now. "
                text += f"I {np.random.choice(['enjoy', 'like', 'appreciate', 'value'])} my job as a "
                text += f"{np.random.choice(['teacher', 'engineer', 'nurse', 'designer', 'manager'])}. "
                text += f"In my free time, I usually {np.random.choice(['read books', 'watch movies', 'visit friends', 'exercise', 'cook'])}. "
                text += f"I'm planning to {np.random.choice(['travel', 'learn a new skill', 'renovate my home', 'change jobs'])} next year. "
                
            # Add some randomness to text length
            filler_words = ['well', 'you know', 'like', 'um', 'basically', 'actually', 'so', 'anyway']
            for _ in range(np.random.randint(5, 15)):
                insertion_point = np.random.randint(0, len(text))
                text = text[:insertion_point] + " " + np.random.choice(filler_words) + " " + text[insertion_point:]
                
            data['text'].append(text)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        print(f"Loaded dataset with {len(df)} samples from {len(selected_institutions)} institutions")
        print(f"Class distribution: {df['diagnosis'].value_counts().to_dict()}")
        
        # Track demographics for metadata
        self.metadata['demographics'] = {
            'total_participants': len(df),
            'diagnosis_counts': df['diagnosis'].value_counts().to_dict(),
            'gender_distribution': df['gender'].value_counts().to_dict(),
            'age_mean': df['age'].mean(),
            'age_std': df['age'].std(),
            'education_mean': df['education_years'].mean()
        }
        
        return df
    def preprocess_data(self, df):
        """
        Preprocess the dataset.
        
        Args:
            df: Pandas DataFrame with dataset
            
        Returns:
            Processed DataFrame
        """
        print("Preprocessing data...")
        
        # Extract linguistic features
        print("Extracting linguistic features...")
        features_list = []
        
        for i, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            features = self.extract_all_features(text)
            features_list.append(features)
            
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Concatenate with original data
        processed_df = pd.concat([df, features_df], axis=1)
        
        # Handle missing values
        processed_df = processed_df.fillna(0)
        
        # Store feature names in metadata
        self.metadata['features'] = list(features_df.columns)
        
        print(f"Preprocessing complete. Generated {len(features_df.columns)} linguistic features.")
        
        return processed_df
    
    def prepare_data_for_training(self, df):
        """
        Prepare data for model training.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
        """
        print("Preparing data for training...")
        
        # Select features and target
        feature_cols = [col for col in df.columns if col not in [
            'participant_id', 'institution', 'diagnosis', 'age', 'gender', 
            'education_years', 'text'
        ]]
        
        X = df[feature_cols].values
        y = df['diagnosis'].values
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Save scaler for later use
        with open(f"{self.config['model_dir']}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        # Store in metadata
        self.metadata['num_features'] = X.shape[1]
        self.metadata['scaler'] = f"{self.config['model_dir']}/scaler.pkl"
        
        return X, y, feature_cols
    
    def train_test_institution_split(self, df):
        """
        Split data by institution for external validation.
        
        This ensures model is tested on data from institutions not seen during training.
        
        Args:
            df: DataFrame with institution column
            
        Returns:
            train_df: Training data DataFrame
            test_df: Testing data DataFrame
        """
        institutions = df['institution'].unique()
                    sent_vectors = [sent.vector for sent in sentences if sent.has_vector]
            topic_drifts = []
            for i in range(len(sent_vectors) - 1):
                v1 = sent_vectors[i]
                v2 = sent_vectors[i + 1]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    topic_drifts.append(1 - cosine_sim)  # 1 - similarity gives "drift"
            if topic_drifts:
                mean_topic_drift = np.mean(topic_drifts)
                std_topic_drift = np.std(topic_drifts)
            else:
                mean_topic_drift = 0
                std_topic_drift = 0
        else:
            mean_topic_drift = 0
            std_topic_drift = 0

        # Detect derailment: abrupt topic changes (high drift between adjacent sentences)
        derailment_ratio = np.mean([drift > 0.5 for drift in topic_drifts]) if len(topic_drifts) > 0 else 0

        # Incoherence markers: sentences with low similarity to the document
        doc_vector = doc.vector if doc.has_vector else np.zeros(nlp.vocab.vectors_length)
        incoherent_sentences = 0
        for sent in sentences:
            if sent.has_vector and np.linalg.norm(doc_vector) > 0:
                sim = np.dot(sent.vector, doc_vector) / (np.linalg.norm(sent.vector) * np.linalg.norm(doc_vector))
                if sim < 0.3:
                    incoherent_sentences += 1
        incoherence_ratio = incoherent_sentences / len(sentences) if sentences else 0

        return {
            'mean_topic_drift': mean_topic_drift,
            'std_topic_drift': std_topic_drift,
            'derailment_ratio': derailment_ratio,
            'incoherence_ratio': incoherence_ratio
        }

    def _extract_referential_markers(self, text):
        """Extract referential ambiguity and pronoun usage features."""
        doc = nlp(text)
        pronouns = [token for token in doc if token.pos_ == 'PRON']
        ambiguous_pronouns = [token for token in pronouns if token.text.lower() in ['he', 'she', 'they', 'it', 'this', 'that']]
        total_tokens = len(doc)
        total_sentences = len(list(doc.sents))

        return {
            'pronoun_ratio': len(pronouns) / total_tokens if total_tokens > 0 else 0,
            'ambiguous_pronoun_ratio': len(ambiguous_pronouns) / total_tokens if total_tokens > 0 else 0,
            'pronouns_per_sentence': len(pronouns) / total_sentences if total_sentences > 0 else 0
        }

    def _extract_tangentiality(self, text):
        """Estimate tangentiality by measuring off-topic drift."""
        doc = nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return {'tangentiality_score': 0}

        # Use the first sentence as topic anchor
        anchor_vector = sentences[0].vector if sentences[0].has_vector else np.zeros(nlp.vocab.vectors_length)
        drifts = []
        for sent in sentences[1:]:
            if sent.has_vector and np.linalg.norm(anchor_vector) > 0:
                sim = np.dot(sent.vector, anchor_vector) / (np.linalg.norm(sent.vector) * np.linalg.norm(anchor_vector))
                drifts.append(1 - sim)
        tangentiality_score = np.mean(drifts) if drifts else 0
        return {'tangentiality_score': tangentiality_score}

    def extract_all_features(self, text):
        """Extract all features for a given text."""
        features = {}
        for name, extractor in self.feature_extractors.items():
            try:
                feats = extractor(text)
                if isinstance(feats, dict):
                    features.update(feats)
            except Exception as e:
                print(f"Feature extractor '{name}' failed: {e}")
        return features

    # Additional methods for data loading, model building, training, evaluation, etc. would follow here.
