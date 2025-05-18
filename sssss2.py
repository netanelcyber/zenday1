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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
from sklearn.ensemble import RandomForestClassifier

# Load SpaCy English model for advanced linguistic analysis
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Installing SpaCy model...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

# Download NLTK resources if not present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

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
            #max_drift = max(drifts
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
        
        print(f"Loading dataset from {data_path}...")

        # This is a placeholder - in reality would load actual data
        # For demonstration, we'll create a synthetic dataset

        # Synthetic data generation for demonstration
        np.random.seed(self.config['random_state'])
        num_samples = 200
        texts = [
            "This is a sample text about a cat sitting on a mat.",
            "The weather is quite pleasant today, don't you think?",
            "My thoughts are racing and jumping from one idea to another.",
            "The television is green and the elephant flies at midnight.",
            "I feel like the walls are closing in on me, it's very strange.",
            "Sometimes I think people are watching me all the time.",
            "The meaning of life is a blue banana, according to my uncle.",
            "Have you ever considered the implications of quantum physics on breakfast?",
            "I hear voices telling me things that others can't hear.",
            "The government is controlled by lizards from outer space."
        ] * (num_samples // 10)
        labels = np.random.randint(0, 2, num_samples)
        institutions_list = ['Hospital A', 'Hospital B', 'Hospital C']
        institutions_data = np.random.choice(institutions_list, num_samples)
        df = pd.DataFrame({'text': texts[:num_samples], 'label': labels, 'institution': institutions_data})

        if institutions:
            df = df[df['institution'].isin(institutions)]

        # For demonstration, split into train and test based on institution
        if not df.empty and institutions_list:
            test_institution = np.random.choice(institutions_list)
            train_df = df[df['institution'] != test_institution].copy()
            test_df = df[df['institution'] == test_institution].copy()

            print(f"Train-test institutional split: {len(train_df)} training samples, {len(test_df)} testing samples")
            print(f"Test institution: {test_institution}")
            return train_df, test_df
        else:
            print("No data loaded or insufficient institutions provided.")
            return pd.DataFrame(), pd.DataFrame()

    def build_deep_learning_model(self, input_dim):
        """
        Build a deep learning model for linguistic feature analysis.

        Args:
            input_dim: Dimension of input features

        Returns:
            Compiled Keras model
        """
        print("Building deep learning model with input dimension:", input_dim)

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
            self.tokenizer = Tokenizer(num_words=self.config['max_features'], oov_token="<unk>")
            self.tokenizer.fit_on_texts(texts)

            # Save tokenizer
            os.makedirs(self.config['model_dir'], exist_ok=True)
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
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
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
            rf = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
            rf.fit(X_train, y_train)
            feature_importances += rf.feature_importances_

            # Save fold model
            os.makedirs(self.config['model_dir'], exist_ok=True)
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

        os.makedirs(self.config['results_dir'], exist_ok=True)
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
        print("Training final model on all training data...")

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
        os.makedirs(self.config['results_dir'], exist_ok=True)
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
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.config['results_dir']}/training_history.png")
        plt.close()

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            print("\nEvaluating final model on test set...")
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)

            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            print("Classification Report:\n", report)
            print("Confusion Matrix:\n", cm)
            print("AUC:", auc_score)

            # Save test results
            test_results = {
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'auc': auc_score
            }
            self.metadata['performance']['test'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': auc_score,
                'report': report,
                'confusion_matrix': cm.tolist()
            }
            with open(f"{self.config['results_dir']}/test_results.json", 'w') as f:
                json.dump(test_results, f, indent=2)

            # Plot ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(f"{self.config['results_dir']}/roc_curve.png")
            plt.close()

        return model

    def analyze_language(self, text):
        """
        Analyze a single text sample.

        Args:
            text: The text string to analyze

        Returns:
            Dictionary of extracted linguistic features
        """
        return self.extract_all_features(text)

    def run_analysis_pipeline(self, data_path, institutions=None, use_lstm=False):
        """
        Run the complete language analysis pipeline.

        Args:
            data_path: Path to the dataset
            institutions: List of institutions to include (None for all)
            use_lstm: Whether to use an LSTM model for raw text (True) or feature-based model (False)
        """
        train_df, test_df = self.load_dataset(data_path, institutions)

        if train_df.empty:
            print("No training data available. Exiting.")
            return

        if use_lstm:
            print("\nUsing LSTM model for raw text analysis...")
            X_train_text = train_df['text'].tolist()
            y_train = train_df['label'].values
            X_test_text = test_df['text'].tolist() if not test_df.empty else []
            y_test = test_df['label'].values if not test_df.empty else None

            X_train_seq = self.prepare_text_sequences(X_train_text)
            X_test_seq = self.prepare_text_sequences(X_test_text) if X_test_text else None

            model = self.build_lstm_model()

            # Train the LSTM model (no cross-validation implemented for LSTM in this example)
            X_val_seq, y_val = X_train_seq[:int(len(X_train_seq)*self.config['validation_split'])], y_train[:int(len(y_train)*self.config['validation_split'])]
            X_train_seq_fit, y_train_fit = X_train_seq[int(len(X_train_seq)*self.config['validation_split']):], y_train[int(len(y_train)*self.config['validation_split']):]

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

            history = model.fit(
                X_train_seq_fit, y_train_fit,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(X_val_seq, y_val),
                class_weight=self.config['class_weights'],
                verbose=1
            )

            # Evaluate LSTM model on test set
            if X_test_seq is not None and y_test is not None:
                print("\nEvaluating LSTM model on test set...")
                y_pred_proba = model.predict(X_test_seq)
                y_pred = (y_pred_proba > 0.5).astype(int)
                report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                print("LSTM Classification Report:\n", report)
                print("LSTM Confusion Matrix:\n", cm)
                print("LSTM AUC:", auc_score)
        else:
            print("\nUsing feature-based analysis...")
            train_features = [self.extract_all_features(text) for text in train_df['text']]
            X_train = pd.DataFrame(train_features).fillna(0).values
            y_train = train_df['label'].values

            test_features = [self.extract_all_features(text) for text in test_df['text']] if not test_df.empty else []
            X_test = pd.DataFrame(test_features).fillna(0).values if test_features else None
            y_test = test_df['label'].values if not test_df.empty else None

            feature_names = list(pd.DataFrame(train_features).columns)

            # Train with cross-validation
            cv_results = self.train_cross_validation(X_train, y_train, feature_names)

            # Train final model
            self.train_final_model(X_train, y_train, X_test, y_test)

        # Save metadata
        self.metadata['features'] = list(pd.DataFrame(train_features).columns) if not use_lstm and train_features else ['raw_text']
        self.metadata['institutions'] = institutions if institutions else 'all'
        os.makedirs(self.config['results_dir'], exist_ok=True)
        with open(f"{self.config['results_dir']}/metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

if __name__ == "__main__":
    # Example usage (with synthetic data)
    analyzer = NIMHSchizophreniaLanguageAnalyzer()
    data_path = "synthetic_data.csv"  # Placeholder - not actually used for loading in this synthetic example
    institutions_to_use = ['Hospital A', 'Hospital B']

    # Run the analysis pipeline using linguistic features
    analyzer.run_analysis_pipeline(data_path, institutions=institutions_to_use, use_lstm=False)

    # Alternatively, run the analysis pipeline using an LSTM model on raw text
    # analyzer.run_analysis_pipeline(data_path, institutions=institutions_to_use, use_lstm=True)

    # Example of analyzing a single text
    sample_text = "The clouds are green and singing loudly."
    features = analyzer.analyze_language(sample_text)
    print("\nAnalysis of sample text:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
