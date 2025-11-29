# M&A Unsupervised Learning Project

**Machine Learning Approaches to Measuring the Impact of Mergers and Acquisitions**

This project applies **unsupervised machine learning** to cluster M&A transactions in the Eurozone, identifying patterns, relationships, and strategic rationales behind deals. The analysis helps identify potential trading opportunities and understand deal dynamics similar to relationships like Nvidia-OpenAI (strategic partnerships involving investments and commercial relationships).

## 🎯 Project Objectives

1. **Cluster M&A transactions** based on key characteristics and deal rationale
2. **Extract business insights** from transaction descriptions  
3. **Identify company relationships** (acquirers, targets, strategic partners)
4. **Analyze deal rationale** (operational, financial, regulatory, technology, market expansion)
5. **Find trading opportunities** by understanding cluster patterns

## 🚀 Setup

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn yfinance sentence-transformers openpyxl
```

Or use the provided requirements:
```bash
pip install -r requirements.txt
```

### Data
Place your M&A data file in the project directory:
- **Required**: `Project/MA Transactions Over 50M.xlsx`

The system will automatically filter for Eurozone transactions and process them.

## 📊 Pipeline Overview

The project follows the **CRISP-DM** framework:

### 1. **Data Loading & Enrichment** (`data_loader.py`)
```bash
python data_loader.py
```

**What it does:**
- Filters transactions for Eurozone countries
- Samples 300 transactions (if > 300 available)
- **Extracts deal rationale** from descriptions:
  - Operational (synergies, efficiency)
  - Financial (diversification, valuation)
  - Regulatory (compliance, licenses)
  - Technology (R&D, innovation)
  - Market Expansion (geographic, customer base)
- **Extracts financial metrics** from text (revenue, EBITDA mentions)
- Fetches market data from Yahoo Finance for public companies
- Identifies financial advisors and deal complexity

**Output:** `eurozone_transactions_enriched.csv`

### 2. **Preprocessing** (`preprocessing.py`)
```bash
python preprocessing.py
```

**What it does:**
- **Keyword extraction** from transaction descriptions
- Handles missing values intelligently
- Creates features from:
  - Transaction values (log-transformed)
  - Deal rationale scores (5 categories)
  - Text-derived metrics
  - Geographic and sector information
- **Text embeddings** using Sentence-BERT (captures semantic meaning)
- Scales and normalizes all features

**Outputs:**
- `X_final.npy` - Processed feature matrix
- `extracted_keywords.txt` - Top keywords from deals

### 3. **Modeling** (`modeling.py`)
```bash
python modeling.py
```

**What it does:**
- **PCA dimensionality reduction** (preserving 95% variance)
- **K-Means clustering** with automatic K selection
- Generates clusters for K=3 and K=4
- **t-SNE visualization** of clusters
- Saves clustered datasets

**Outputs:**
- `eurozone_transactions_clustered_k3.csv`
- `eurozone_transactions_clustered_k4.csv`  
- `eurozone_transactions_clustered.csv` (default K=3)
- `clusters_visualization_k3.png`
- `clusters_visualization_k4.png`

### 4. **Evaluation & Analysis** (`evaluation.py`)
```bash
python evaluation.py
```

**What it does:**
- Comprehensive cluster statistics
- Quantitative analysis (deal sizes, rationale scores)
- Qualitative analysis (sectors, countries, status)
- **Business insights** for each cluster:
  - Dominant deal rationale
  - Top keywords and themes
  - Geographic concentration
  - Sector focus
  - Company relationships (frequent acquirers, dual-role companies)
- Creates detailed visualizations

**Outputs:**
- Detailed console reports
- `*_analysis.png` - Multi-panel cluster analysis charts

### 5. **Advanced Cluster Analysis** (`cluster_analyzer.py`)
```bash
python cluster_analyzer.py eurozone_transactions_clustered_k3.csv
```

**What it does:**
- **Identifies company relationships** within clusters:
  - Companies appearing as both buyer and target
  - Frequent acquirers (multiple deals)
  - Potential strategic partnerships
- Extracts cluster-specific keywords
- Analyzes geographic and sector patterns
- Generates business-focused summaries

## 📈 Key Features

### Deal Rationale Analysis
The system automatically categorizes deals into:
- **Operational**: Economies of scale, synergies, efficiency gains
- **Financial**: Diversification, capital access, valuation arbitrage
- **Regulatory**: Licensing, compliance, market access
- **Technology**: R&D acquisition, innovation, digital transformation
- **Market Expansion**: Geographic expansion, customer base growth

### Company Relationship Detection
Identifies patterns like:
- Strategic investors (e.g., Nvidia investing in OpenAI)
- Roll-up strategies (serial acquirers in same cluster)
- Cross-border partnerships
- Vertical integration patterns

### Semantic Analysis
Uses state-of-the-art NLP (Sentence-BERT) to:
- Understand deal context beyond keywords
- Group similar transaction narratives
- Capture strategic intent from descriptions

## 🔍 Example Insights

**Cluster Example:**
```
CLUSTER 0: Technology & Innovation Deals
- Size: 120 transactions
- Avg Deal Value: $245M USD
- Dominant Rationale: Technology (R&D, innovation)
- Keywords: technology, platform, digital, innovation, software
- Geographic Focus: France (35%), Germany (28%), Netherlands (15%)
- Top Sectors: Technology, Financial Services
- Active Acquirers: Company X (5 deals), Company Y (4 deals)
```

## 📁 Project Structure
```
├── data_loader.py           # Data ingestion and enrichment
├── preprocessing.py          # Feature engineering and text processing  
├── modeling.py              # Clustering algorithms
├── evaluation.py            # Comprehensive evaluation
├── cluster_analyzer.py      # Advanced business insights
├── inspect_data.py          # Data exploration utilities
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── Project/
    └── ML for Business III Unsupervised Learning.ipynb
```

## 🎓 Academic Context

This project demonstrates:
- **CRISP-DM methodology** for data mining
- **Unsupervised learning** techniques (PCA, K-Means, t-SNE)
- **NLP and text mining** for financial documents
- **Feature engineering** from unstructured data
- **Business intelligence** from ML models

## 📊 Evaluation Metrics

- Silhouette Score (cluster quality)
- Within-cluster homogeneity
- Cluster size distribution
- Business interpretability
- Deal rationale distribution

## 💡 Usage Tips

1. **Start with the full pipeline:**
   ```bash
   python data_loader.py && python preprocessing.py && python modeling.py && python evaluation.py
   ```

2. **For quick analysis of existing clusters:**
   ```bash
   python cluster_analyzer.py eurozone_transactions_clustered_k3.csv
   ```

3. **To extract keywords only:**
   Check `extracted_keywords.txt` after running preprocessing

4. **To compare different K values:**
   Review both `_k3` and `_k4` outputs from evaluation

## 🔬 Finance Background

This project applies ML to classic M&A analysis questions:
- What drives deal multiples in different segments?
- Which companies are strategic consolidators?
- What geographic patterns emerge in European M&A?
- How do deal rationales cluster?
- Where are cross-border partnership opportunities?

## 📝 Notes

- The analysis focuses on **Eurozone targets** only
- Transaction values are log-transformed for modeling
- Missing financial data is handled via intelligent imputation
- Text embeddings capture semantic meaning beyond keywords

## 🙏 Acknowledgments

Based on academic research in M&A performance measurement and the CRISP-DM framework for data mining projects.

---

**Author**: Oussema  
**Framework**: CRISP-DM  
**Focus**: Unsupervised Learning for M&A Analysis

