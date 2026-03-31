import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import joblib

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class TextPreprocessor:
    """
    Text preprocessing class for cleaning academic abstracts.
    """
    def __init__(self, use_stemming=False, use_lemmatization=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        if use_stemming:
            self.stemmer = nltk.PorterStemmer()
        if use_lemmatization:
            self.lemmatizer = nltk.WordNetLemmatizer()
        if remove_stopwords:
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove LaTeX math expressions: $...$ and \[...\]
        text = re.sub(r'\$.*?\$', '', text)
        text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
        
        # 3. Remove LaTeX commands (e.g., \frac, \text, \alpha)
        # We remove the command but try to keep the text inside curly braces if it exists
        # e.g., \textbf{hello} -> hello
        text = re.sub(r'\\[a-zA-Z]+\{(.*?)\}', r'\1', text)
        # Remove any remaining standalone commands like \alpha or \frac without braces
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # 4. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 5. Remove citations like [1], [2,3]
        text = re.sub(r'\[[0-9, ]+\]', '', text)
        
        # 6. Keep letters, numbers, hyphens, and underscores
        # This addresses the "3d" and "real-world" issues from your chat
        text = re.sub(r'[^a-z0-9\-_ ]', ' ', text)
        
        # 7. Cleanup whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_wordnet_pos(self, word):
        """
        Map POS tag to first character lemmatize() accepts
        """
        if not self.use_lemmatization:
            return 'n'  # Default to noun if not using lemmatization
        
        # This is a simplified version; for better results, you'd use pos_tag
        # but that requires tokenization first
        return 'n'  # Default to noun for simplicity
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        words = text.split()
        
        # 7. FILTER: Remove stopwords AND all single letters (length > 1)
        # This ensures 'e' and 'g' are removed even if stopwords are being processed
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words and len(w) > 1]
        else:
            words = [w for w in words if len(w) > 1]
        
        # Apply stemming or lemmatization
        if self.use_stemming:
            words = [self.stemmer.stem(w) for w in words]
        elif self.use_lemmatization:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return ' '.join(words)

class DataPreprocessor:
    """
    A comprehensive class for data preprocessing, text cleaning, 
    TF-IDF vectorization, and feature selection.
    """
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        """
        Initialize the DataPreprocessor with paths and configurations.
        
        Args:
            data_dir: Path to the data directory
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.random_state = random_state
        self.train_df = None
        self.test_df = None
        self.taxonomy_df = None
        self.sample_submission = None
        
        # Preprocessing components
        self.text_preprocessor = None
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizers = {}
        self.fs_data = {}
        
        # Set pandas display options
        pd.set_option("display.max_colwidth", 160)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all required datasets from the data directory.
        
        Returns:
            Tuple of (train_df, test_df, taxonomy_df)
        """
        # Define file paths
        train_path = self.data_dir / "train.csv"
        test_path = self.data_dir / "test.csv"
        taxonomy_path = self.data_dir / "label_taxonomy.csv"
        sample_submission_path = self.data_dir / "sample_submission_random_guess.csv"
        
        # Verify files exist
        required_paths = [train_path, test_path, taxonomy_path]
        missing_files = [str(p) for p in required_paths if not p.exists()]
        
        if missing_files:
            print("Missing required files:")
            for file in missing_files:
                print("-", file)
            raise FileNotFoundError("Please add the dataset CSV files to the data folder.")
        
        # Load data with tab separator
        self.train_df = pd.read_csv(train_path, sep="\t")
        self.test_df = pd.read_csv(test_path, sep="\t")
        self.taxonomy_df = pd.read_csv(taxonomy_path, sep="\t")
        
        # Load sample submission if it exists
        if sample_submission_path.exists():
            self.sample_submission = pd.read_csv(sample_submission_path)
        
        print("\n✓ Data loaded successfully!")
        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print(f"Taxonomy shape: {self.taxonomy_df.shape}")
        
        return self.train_df, self.test_df, self.taxonomy_df
    
    def explore_data(self) -> Dict:
        """
        Perform exploratory data analysis and visualize class distribution.
        
        Returns:
            Dictionary with data statistics
        """
        if self.train_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'missing_train': self.train_df.isnull().sum().sum(),
            'missing_test': self.test_df.isnull().sum().sum(),
            'n_unique_labels': self.train_df['label_id'].nunique(),
            'label_counts': self.train_df['label_id'].value_counts().to_dict()
        }
        
        print(f"\nMissing values in train: {stats['missing_train']}")
        print(f"Missing values in test: {stats['missing_test']}")
        print(f"\nNumber of unique labels: {stats['n_unique_labels']}")
        
        print("\nTop 10 most frequent labels:")
        label_counts = self.train_df['label_id'].value_counts()
        for label, count in label_counts.head(10).items():
            # Get category name from taxonomy
            category = self.taxonomy_df[
                self.taxonomy_df['label_id'] == label
            ]['category'].values[0] if label in self.taxonomy_df['label_id'].values else "Unknown"
            percentage = (count / len(self.train_df)) * 100
            print(f"  Label {label} ({category}): {count:,} samples ({percentage:.1f}%)")
            stats[f'label_{label}_count'] = count
            stats[f'label_{label}_pct'] = percentage
        
        # Visualize class distribution
        self._plot_class_distribution(label_counts)
        
        return stats
    
    def _plot_class_distribution(self, label_counts: pd.Series, top_n: int = 20):
        """
        Plot class distribution for top N labels.
        
        Args:
            label_counts: Series with label counts
            top_n: Number of top labels to plot
        """
        plt.figure(figsize=(12, 6))
        label_counts.head(top_n).plot(kind='bar')
        plt.title(f'Top {top_n} Most Frequent Labels in Training Data')
        plt.xlabel('Label ID')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.data_dir / 'class_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()

    def create_text_preprocessor(self, use_stemming: bool = False, 
                                  use_lemmatization: bool = True, 
                                  remove_stopwords: bool = True):
        """
        Create and initialize the text preprocessor.
        
        Args:
            use_stemming: Whether to apply stemming
            use_lemmatization: Whether to apply lemmatization
            remove_stopwords: Whether to remove stopwords
        """
        self.text_preprocessor = TextPreprocessor(
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization,
            remove_stopwords=remove_stopwords
        )
    
    def preprocess_texts(self):
        """
        Apply text preprocessing to train and test abstracts.
        """
        if self.text_preprocessor is None:
            raise ValueError("Text preprocessor not created. Call create_text_preprocessor() first.")
        
        if self.train_df is None or self.test_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Preprocessing training abstracts...")
        self.train_df['cleaned_abstract'] = self.train_df['abstract'].apply(
            self.text_preprocessor.preprocess
        )
        
        print("Preprocessing test abstracts...")
        self.test_df['cleaned_abstract'] = self.test_df['abstract'].apply(
            self.text_preprocessor.preprocess
        )
        
        # Show sample
        print("\nSample before preprocessing:")
        print(self.train_df['abstract'].iloc[0][:200])
        print("\nSample after preprocessing:")
        print(self.train_df['cleaned_abstract'].iloc[0][:200])
    
    def split_data(self, test_size: float = 0.2, stratify: bool = True) -> Tuple:
        """
        Split data into train and validation sets.
        
        Args:
            test_size: Proportion of data for validation
            stratify: Whether to stratify by label
            
        Returns:
            Tuple of (X_train_raw, X_val_raw, y_train, y_val)
        """
        if self.train_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stratify_col = self.train_df['label_id'] if stratify else None
        
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            self.train_df['cleaned_abstract'],
            self.train_df['label_id'],
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        print(f"\nTraining samples: {len(X_train_raw)}")
        print(f"Validation samples: {len(X_val_raw)}")
        
        return X_train_raw, X_val_raw, y_train, y_val
    
    def create_tfidf_vectorizer(self, max_features: int = 5000, 
                               ngram_range: tuple = (1, 2),
                               min_df: int = 2,
                               max_df: float = 0.8,
                               sublinear_tf: bool = True) -> TfidfVectorizer:
        """
        Create a TF-IDF vectorizer with specified parameters.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            sublinear_tf: Whether to use sublinear TF scaling
            
        Returns:
            Configured TfidfVectorizer
        """
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            stop_words='english'
        )
    
    def fit_transform_tfidf(self, X_train_raw: pd.Series, 
                           X_val_raw: pd.Series,
                           X_test_raw: pd.Series,
                           max_features: int = 5000,
                           vectorizer_name: str = 'default') -> Dict:
        """
        Fit TF-IDF on training data and transform all sets.
        
        Args:
            X_train_raw: Training text data
            X_val_raw: Validation text data
            X_test_raw: Test text data
            max_features: Maximum number of features
            vectorizer_name: Name to store the vectorizer
            
        Returns:
            Dictionary with transformed data
        """
        tfidf = self.create_tfidf_vectorizer(max_features=max_features)
        
        print(f"Fitting TF-IDF with {max_features} features...")
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        X_val_tfidf = tfidf.transform(X_val_raw)
        X_test_tfidf = tfidf.transform(X_test_raw)
        
        # Store vectorizer
        self.tfidf_vectorizers[vectorizer_name] = tfidf
        
        result = {
            'train': X_train_tfidf,
            'val': X_val_tfidf,
            'test': X_test_tfidf,
            'vectorizer': tfidf
        }
        
        print(f"Train shape: {X_train_tfidf.shape}")
        print(f"Validation shape: {X_val_tfidf.shape}")
        print(f"Test shape: {X_test_tfidf.shape}")
        
        return result
    
    def feature_selection_experiment(self, 
                                     feature_sizes: List[int] = [2000, 1000, 500, 100],
                                     ngram_range: tuple = (1, 2)) -> Dict:
        """
        Run feature selection experiment with different feature sizes.
        
        Args:
            feature_sizes: List of feature sizes to try
            ngram_range: Range of n-grams to consider
            
        Returns:
            Dictionary with results for each feature size
        """
        if self.X_train_raw is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        self.fs_data = {}
        
        for size in feature_sizes:
            print(f"\nCreating TF-IDF with {size} features...")
            tfidf_fs = self.create_tfidf_vectorizer(
                max_features=size,
                ngram_range=ngram_range
            )
            
            X_train_fs = tfidf_fs.fit_transform(self.X_train_raw)
            X_val_fs = tfidf_fs.transform(self.X_val_raw)
            X_test_fs = tfidf_fs.transform(self.test_df['cleaned_abstract'])
            
            self.fs_data[size] = {
                'train': X_train_fs,
                'val': X_val_fs,
                'test': X_test_fs,
                'vectorizer': tfidf_fs
            }
            
            print(f"  Train shape: {X_train_fs.shape}")
            print(f"  Val shape: {X_val_fs.shape}")
            print(f"  Test shape: {X_test_fs.shape}")
        
        return self.fs_data
    
    def get_feature_names(self, vectorizer_name: str = 'default') -> List[str]:
        """
        Get feature names from a stored vectorizer.
        
        Args:
            vectorizer_name: Name of the vectorizer
            
        Returns: 
            List of feature names
        """
        if vectorizer_name not in self.tfidf_vectorizers:
            raise KeyError(f"Vectorizer '{vectorizer_name}' not found. Available: {list(self.tfidf_vectorizers.keys())}")
        return self.tfidf_vectorizers[vectorizer_name].get_feature_names_out()
    
    def get_fs_feature_names(self, size: int) -> List[str]:
        """
        Get feature names from a feature selection vectorizer.
        
        Args:
            size: Feature size used in experiment
            
        Returns:
            List of feature names
        """
        if size not in self.fs_data:
            raise KeyError(f"Feature size '{size}' not found. Available: {list(self.fs_data.keys())}")
        
        return self.fs_data[size]['vectorizer'].get_feature_names_out()
    
    def save_processed_data(self, output_dir: str = 'processed_data'):
        """
        Save processed data and vectorizers for later use.
        
        Args:
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save dataframes
        if self.train_df is not None:
            self.train_df.to_csv(output_path / 'train_processed.csv', index=False)
            self.test_df.to_csv(output_path / 'test_processed.csv', index=False)
        
        # Save train/validation splits
        if self.X_train_raw is not None:
            pd.DataFrame({
                'cleaned_abstract': self.X_train_raw,
                'label_id': self.y_train
            }).to_csv(output_path / 'train_split.csv', index=False)
            
            pd.DataFrame({
                'cleaned_abstract': self.X_val_raw,
                'label_id': self.y_val
            }).to_csv(output_path / 'val_split.csv', index=False)
            print(f"  ✓ Saved train_split.csv and val_split.csv")

        # Save vectorizers
        for name, vectorizer in self.tfidf_vectorizers.items():
            joblib.dump(vectorizer, output_path / f'tfidf_{name}.pkl')
        print(f"  ✓ Saved {len(self.tfidf_vectorizers)} TF-IDF vectorizer(s)")
        
        # Save feature selection vectorizers
        for size, data in self.fs_data.items():
            joblib.dump(data['vectorizer'], output_path / f'tfidf_fs_{size}.pkl')
            joblib.dump(data['train'], output_path / f'X_train_fs_{size}.pkl')
            joblib.dump(data['val'], output_path / f'X_val_fs_{size}.pkl')
            joblib.dump(data['test'], output_path / f'X_test_fs_{size}.pkl')
        print(f"  ✓ Saved {len(self.fs_data)} feature selection sets")
        
        print(f"\n✓ All data saved to {output_path}/")
    
    def load_processed_data(self, input_dir: str = 'processed_data'):
        """
        Load previously saved processed data and vectorizers.
        
        Args:
            input_dir: Directory containing saved data
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Directory {input_path} not found.")
        
        # Load dataframes
        train_path = input_path / 'train_processed.csv'
        test_path = input_path / 'test_processed.csv'
        
        if train_path.exists():
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            print(f"Loaded train data: {self.train_df.shape}")
            print(f"Loaded test data: {self.test_df.shape}")
        
         # Load train/validation splits
        train_split_path = input_path / 'train_split.csv'
        val_split_path = input_path / 'val_split.csv'
        
        if train_split_path.exists():
            train_split = pd.read_csv(train_split_path)
            val_split = pd.read_csv(val_split_path)
            self.X_train_raw = train_split['cleaned_abstract']
            self.y_train = train_split['label_id']
            self.X_val_raw = val_split['cleaned_abstract']
            self.y_val = val_split['label_id']
            print(f"  ✓ Loaded train/validation splits")
        
        # Clear existing data
        self.tfidf_vectorizers = {}
        self.fs_data = {}
        
        # Load vectorizers
        for p in input_path.glob('tfidf_*.pkl'):
            if 'tfidf_fs' not in p.name:
                name = p.stem.replace('tfidf_', '')
                try:
                    self.tfidf_vectorizers[name] = joblib.load(p)
                    print(f"Loaded vectorizer: {name}")
                except Exception as e:
                    print(f"Error loading {p.name}: {e}")

        # Load feature selection vectorizers
        for p in input_path.glob('tfidf_fs_*.pkl'):
            if 'X_train_fs' not in p.name and 'X_val_fs' not in p.name and 'X_test_fs' not in p.name:
                size = int(p.stem.replace('tfidf_fs_', ''))
                try:
                    size = int(p.stem.replace('tfidf_fs_', ''))
                    vectorizer = joblib.load(p)
                    self.fs_data[size] = {'vectorizer': vectorizer}
                    self.fs_data[size]['train'] = joblib.load(input_path / f'X_train_fs_{size}.pkl')
                    self.fs_data[size]['val'] = joblib.load(input_path / f'X_val_fs_{size}.pkl')
                    self.fs_data[size]['test'] = joblib.load(input_path / f'X_test_fs_{size}.pkl')
                except Exception as e:
                    print(f"Error loading feature selection data for size {size}: {e}")


# Initialize preprocessor
preprocessor = DataPreprocessor(data_dir="data", random_state=42)
# Load data
train_df, test_df, taxonomy_df = preprocessor.load_data()
# Explore data
data_stats = preprocessor.explore_data()
# Create text preprocessor
preprocessor.create_text_preprocessor(
    use_stemming=False,
    use_lemmatization=True,
    remove_stopwords=True
)
preprocessor.preprocess_texts()
# Split data
X_train_raw, X_val_raw, y_train, y_val = preprocessor.split_data(test_size=0.2, stratify=True)
preprocessor.X_train_raw = X_train_raw
preprocessor.X_val_raw = X_val_raw
preprocessor.y_train = y_train
preprocessor.y_val = y_val
# Fit TF-IDF
tfidf_results = preprocessor.fit_transform_tfidf(X_train_raw=preprocessor.X_train_raw,
                                                X_val_raw=preprocessor.X_val_raw, 
                                                X_test_raw=preprocessor.test_df['cleaned_abstract'], 
                                                max_features=5000, 
                                                vectorizer_name='default')
# Run feature selection experiment
fs_results = preprocessor.feature_selection_experiment(
    feature_sizes=[2000, 1000, 500, 100], 
    ngram_range=(1, 2)
)
# Save processed data
preprocessor.save_processed_data(output_dir='processed_data')