# %% [markdown]
# # Data Preprocessing

# %% [markdown]
# ### 1. Data Loading

# %%
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

# %%
from pathlib import Path

pd.set_option("display.max_colwidth", 160)

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"

# Define file paths - NOTE: using train.csv, not train_abstracts.csv
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TAXONOMY_PATH = DATA_DIR / "label_taxonomy.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission_random_guess.csv"

# %%
# Verify files exist
required_paths = [TRAIN_PATH, TEST_PATH, TAXONOMY_PATH]
missing_files = [str(p) for p in required_paths if not p.exists()]

if missing_files:
    print("Missing required files:")
    for file in missing_files:
        print("-", file)
    raise FileNotFoundError("Please add the dataset CSV files to the data folder.")

# Load data with tab separator
train_df = pd.read_csv(TRAIN_PATH, sep="\t")
test_df = pd.read_csv(TEST_PATH, sep="\t")
taxonomy_df = pd.read_csv(TAXONOMY_PATH, sep="\t")

# Load sample submission if it exists
if SAMPLE_SUBMISSION_PATH.exists():
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
else:
    sample_submission = None

print("\n✓ Data loaded successfully!")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Taxonomy shape: {taxonomy_df.shape}")
print(f"Sample submission shape: {sample_submission.shape if sample_submission is not None else 'N/A'}")

# Display first few rows
print("\nFirst few rows of train data:")
display(train_df.head())

print("\nFirst few rows of test data:")
display(test_df.head())

print("\nLabel taxonomy:")
display(taxonomy_df.head())

print("\n Sample submission:")
display(sample_submission.head())

# %%
# Check for missing values
print(f"\nMissing values in train: {train_df.isnull().sum().sum()}")
print(f"Missing values in test: {test_df.isnull().sum().sum()}")

# %%
# Check class distribution
print(f"\nNumber of unique labels: {train_df['label_id'].nunique()}")
print("\nTop 10 most frequent labels:")
label_counts = train_df['label_id'].value_counts()
for label, count in label_counts.head(10).items():
    # Get category name from taxonomy
    category = taxonomy_df[taxonomy_df['label_id'] == label]['category'].values[0] if label in taxonomy_df['label_id'].values else "Unknown"
    percentage = (count / len(train_df)) * 100
    print(f"  Label {label} ({category}): {count:,} samples ({percentage:.1f}%)")

# Visualize class distribution
plt.figure(figsize=(12, 6))
label_counts.head(20).plot(kind='bar')
plt.title('Top 20 Most Frequent Labels in Training Data')
plt.xlabel('Label ID')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(DATA_DIR / 'class_distribution.png', dpi=100, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2. Text Cleaning

# %%
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class TextPreprocessor:
    def __init__(self, use_stemming=False, use_lemmatization=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        
        if use_stemming:
            self.stemmer = PorterStemmer()
        if use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
    
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

# Apply preprocessing to abstracts
preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)

train_df['cleaned_abstract'] = train_df['abstract'].apply(preprocessor.preprocess)
test_df['cleaned_abstract'] = test_df['abstract'].apply(preprocessor.preprocess)

print("Sample before preprocessing:")
print(train_df['abstract'].iloc[0][:200])
print("\nSample after preprocessing:")
print(train_df['cleaned_abstract'].iloc[0][:200])
display(train_df.head(100)[['abstract', 'cleaned_abstract']])

# %% [markdown]
# ### 3. Prepare Data for All Tasks

# %%
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    train_df['cleaned_abstract'],
    train_df['label_id'],
    test_size=0.2,
    random_state=42,
    stratify=train_df['label_id']  # Important for imbalanced data
)

print(f"Training samples: {len(X_train_raw)}")
print(f"Validation samples: {len(X_val_raw)}")

# %% [markdown]
# ### 4. TF-IDF Vectorization

# %%
tfidf_task1 = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # Unigrams and bigrams
    min_df=2,                # Ignore terms appearing in < 2 docs
    max_df=0.8,              # Ignore terms appearing in > 80% of docs
    sublinear_tf=True,       # Use 1+log(tf)
    stop_words='english'     # Remove common English stop words
)

# Fit on training data only
print("Fitting TF-IDF on training data...")
X_train_tfidf = tfidf_task1.fit_transform(X_train_raw)
X_val_tfidf = tfidf_task1.transform(X_val_raw)
X_test_tfidf = tfidf_task1.transform(test_df['cleaned_abstract'])

print(f"Train TF-IDF shape: {X_train_tfidf.shape}")
print(f"Validation TF-IDF shape: {X_val_tfidf.shape}")
print(f"Test TF-IDF shape: {X_test_tfidf.shape}")

# %% [markdown]
# ### 5. Feature Selection

# %%
feature_sizes = [2000, 1000, 500, 100]
fs_data = {}  # Feature Selection data

for size in feature_sizes:
    print(f"\nCreating TF-IDF with {size} features...")
    tfidf_fs = TfidfVectorizer(
        max_features=size,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
        stop_words='english'
    )
    
    X_train_fs = tfidf_fs.fit_transform(X_train_raw)
    X_val_fs = tfidf_fs.transform(X_val_raw)
    X_test_fs = tfidf_fs.transform(test_df['cleaned_abstract'])
    
    fs_data[size] = {
        'train': X_train_fs,
        'val': X_val_fs,
        'test': X_test_fs,
        'vectorizer': tfidf_fs
    }
    
    print(f"  Train shape: {X_train_fs.shape}")
    print(f"  Val shape: {X_val_fs.shape}")
    print(f"  Test shape: {X_test_fs.shape}")


