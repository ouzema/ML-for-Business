import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
import re
from collections import Counter
import torch

# Configure for Mac Silicon GPU (MPS - Metal Performance Shaders)
if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using Mac Silicon GPU (MPS) for acceleration!")
elif torch.cuda.is_available():
    device = "cuda"
    print("✓ Using CUDA GPU for acceleration!")
else:
    device = "cpu"
    print("⚠ Using CPU (GPU not available)")

print(f"Device: {device}")

def extract_keywords_from_descriptions(df, text_col='Transaction Comments', 
                                       top_n=50):
    """Extract most common keywords from transaction descriptions."""
    print(f"Extracting keywords from {text_col}...")
    
    # Combine all text
    all_text = ' '.join(df[text_col].fillna('').astype(str))
    
    # Simple keyword extraction (remove common words)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                 'do', 'does', 'did', 'will', 'would', 'should', 'could',
                 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
                 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    # Extract words
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    words = [w for w in words if w not in stopwords]
    
    # Count frequencies
    word_freq = Counter(words)
    top_keywords = word_freq.most_common(top_n)
    
    print(f"Top 20 keywords: {top_keywords[:20]}")
    
    return [word for word, count in top_keywords]

def preprocess_data(filepath):
    print("Loading enriched data...")
    df = pd.read_csv(filepath)
    
    print(f"Initial shape: {df.shape}")
    
    # Extract and display keywords
    keywords = extract_keywords_from_descriptions(df)
    
    # 1. Handling Missing Values
    missing_percent = df.isnull().mean()
    print("\nMissing value percentages for key columns:")
    print(missing_percent[missing_percent > 0].head(20))
    
    # 2. Feature Selection
    # Numeric features: Transaction value + deal rationale scores + text metrics
    numeric_features = [
        'Total Transaction Value ($USDmm, Historical rate)',
        'rationale_operational',
        'rationale_financial',
        'rationale_regulatory',
        'rationale_technology',
        'rationale_market_expansion',
        'mentioned_revenue',
        'mentioned_ebitda',
        'has_valuation_ratio',
        'has_advisor',
        'num_advisors'
    ]
    
    categorical_features = [
        'Country/Region of Incorporation [Target/Issuer]',
        'Transaction Status',
        'Sector',
        'Industry'
    ]
    
    text_feature = 'Transaction Comments'
    
    # Filter to existing columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): "
          f"{categorical_features}")
    
    # Log-transform Transaction Value
    if 'Total Transaction Value ($USDmm, Historical rate)' in df.columns:
        df['Total Transaction Value ($USDmm, Historical rate)'] = np.log1p(
            df['Total Transaction Value ($USDmm, Historical rate)'])
        print("Applied log transformation to Transaction Value")
    
    # 3. Preprocessing Pipelines
    
    # Numeric: Impute 0 for rationale scores, median for others, Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical: Impute 'Unknown', OneHot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    print("\nApplying numeric and categorical transformations...")
    X_structured = preprocessor.fit_transform(df)
    print(f"Structured data shape: {X_structured.shape}")
    
    # 4. Text Embeddings with GPU acceleration
    print("\nGenerating text embeddings with GPU acceleration...")
    df[text_feature] = df[text_feature].fillna("")
    
    # Load model and move to GPU
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Encode with larger batch size for GPU efficiency
    batch_size = 128 if device in ["mps", "cuda"] else 32
    print(f"Using batch size: {batch_size} on {device}")
    
    embeddings = model.encode(
        df[text_feature].tolist(), 
        show_progress_bar=True,
        batch_size=batch_size,
        device=device,
        convert_to_numpy=True
    )
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 5. Combine all features
    X_final = np.hstack([X_structured, embeddings])
    print(f"\nFinal dataset shape: {X_final.shape}")
    
    # Save processed data
    np.save("X_final.npy", X_final)
    print("Saved processed matrix to X_final.npy")
    
    # Save keywords for later analysis
    with open("extracted_keywords.txt", 'w') as f:
        f.write("Top Keywords from Transaction Descriptions:\\n")
        f.write("=" * 50 + "\\n")
        for i, word in enumerate(keywords[:50], 1):
            f.write(f"{i}. {word}\\n")
    print("Saved keywords to extracted_keywords.txt")
    
    return X_final

if __name__ == "__main__":
    preprocess_data("eurozone_transactions_enriched.csv")
