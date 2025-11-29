"""
Advanced cluster analysis for M&A transactions.
Provides detailed insights into cluster characteristics and business patterns.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re


def analyze_cluster_rationale(df, cluster_id):
    """Analyze the dominant deal rationale for a cluster."""
    cluster_data = df[df['Cluster'] == cluster_id]
    
    rationale_cols = ['rationale_operational', 'rationale_financial', 
                      'rationale_regulatory', 'rationale_technology', 
                      'rationale_market_expansion']
    
    existing_cols = [c for c in rationale_cols if c in df.columns]
    
    if not existing_cols:
        return "No rationale data available"
    
    rationale_means = cluster_data[existing_cols].mean()
    dominant = rationale_means.idxmax()
    
    rationale_map = {
        'rationale_operational': 'Operational (economies of scale, efficiency)',
        'rationale_financial': 'Financial (diversification, capital access)',
        'rationale_regulatory': 'Regulatory (licenses, compliance)',
        'rationale_technology': 'Technology (R&D, innovation)',
        'rationale_market_expansion': 'Market Expansion (geographic, customer)'
    }
    
    return rationale_map.get(dominant, dominant)


def extract_cluster_keywords(df, cluster_id, top_n=20):
    """Extract most common keywords for transactions in a cluster."""
    cluster_data = df[df['Cluster'] == cluster_id]
    
    if 'Transaction Comments' not in df.columns:
        return []
    
    # Combine all transaction comments in cluster
    all_text = ' '.join(cluster_data['Transaction Comments'].fillna('').astype(str))
    
    # Stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'also', 'its', 'their', 'which', 'who',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very'
    }
    
    # Extract words (3+ characters)
    words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
    words = [w for w in words if w not in stopwords]
    
    # Count and return top keywords
    word_freq = Counter(words)
    return [word for word, count in word_freq.most_common(top_n)]


def find_company_relationships(df, cluster_id):
    """
    Identify potential company relationships within a cluster.
    Similar to Nvidia-OpenAI example: companies that acquire and are acquired,
    or appear multiple times in deals.
    """
    cluster_data = df[df['Cluster'] == cluster_id]
    
    # Extract all company names
    targets = cluster_data['Target/Issuer'].value_counts()
    buyers = cluster_data['Buyers/Investors'].value_counts()
    
    # Find companies appearing as both buyer and target
    target_set = set(targets.index)
    buyer_set = set(buyers.index)
    both_roles = target_set.intersection(buyer_set)
    
    # Find companies with multiple transactions
    frequent_buyers = buyers[buyers > 1].to_dict()
    frequent_targets = targets[targets > 1].to_dict()
    
    relationships = {
        'dual_role_companies': list(both_roles)[:5],
        'frequent_buyers': frequent_buyers,
        'frequent_targets': frequent_targets
    }
    
    return relationships


def analyze_cluster_geography(df, cluster_id):
    """Analyze geographic distribution of transactions in cluster."""
    cluster_data = df[df['Cluster'] == cluster_id]
    
    if 'Country/Region of Incorporation [Target/Issuer]' in df.columns:
        geo_dist = cluster_data[
            'Country/Region of Incorporation [Target/Issuer]'
        ].value_counts()
        return geo_dist.head(5).to_dict()
    
    return {}


def analyze_cluster_sectors(df, cluster_id):
    """Analyze sector distribution in cluster."""
    cluster_data = df[df['Cluster'] == cluster_id]
    
    if 'Sector' in df.columns:
        sector_dist = cluster_data['Sector'].value_counts()
        return sector_dist.head(5).to_dict()
    
    return {}


def generate_cluster_summary(df, cluster_id):
    """Generate comprehensive summary for a cluster."""
    cluster_data = df[df['Cluster'] == cluster_id]
    n_transactions = len(cluster_data)
    
    # Basic statistics
    avg_value = cluster_data[
        'Total Transaction Value ($USDmm, Historical rate)'
    ].mean() if 'Total Transaction Value ($USDmm, Historical rate)' in df.columns else None
    
    # Convert back from log if needed
    if avg_value is not None:
        avg_value = np.expm1(avg_value)  # Reverse log1p transformation
    
    # Deal rationale
    rationale = analyze_cluster_rationale(df, cluster_id)
    
    # Keywords
    keywords = extract_cluster_keywords(df, cluster_id, top_n=10)
    
    # Relationships
    relationships = find_company_relationships(df, cluster_id)
    
    # Geography
    geography = analyze_cluster_geography(df, cluster_id)
    
    # Sectors
    sectors = analyze_cluster_sectors(df, cluster_id)
    
    summary = {
        'cluster_id': cluster_id,
        'n_transactions': n_transactions,
        'avg_transaction_value_usd_mm': avg_value,
        'dominant_rationale': rationale,
        'top_keywords': keywords,
        'company_relationships': relationships,
        'top_countries': geography,
        'top_sectors': sectors
    }
    
    return summary


def print_cluster_report(summary):
    """Print a formatted report for a cluster."""
    print(f"\n{'='*70}")
    print(f"CLUSTER {summary['cluster_id']} ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nSize: {summary['n_transactions']} transactions")
    
    if summary['avg_transaction_value_usd_mm']:
        print(f"Average Deal Value: ${summary['avg_transaction_value_usd_mm']:.2f}M USD")
    
    print(f"\nDominant Deal Rationale:")
    print(f"  → {summary['dominant_rationale']}")
    
    if summary['top_keywords']:
        print(f"\nTop Keywords:")
        print(f"  {', '.join(summary['top_keywords'][:15])}")
    
    if summary['top_countries']:
        print(f"\nGeographic Focus:")
        for country, count in list(summary['top_countries'].items())[:5]:
            print(f"  → {country}: {count} deals")
    
    if summary['top_sectors']:
        print(f"\nKey Sectors:")
        for sector, count in list(summary['top_sectors'].items())[:5]:
            if pd.notna(sector):
                print(f"  → {sector}: {count} deals")
    
    if summary['company_relationships']['frequent_buyers']:
        print(f"\nActive Acquirers:")
        for company, count in list(
            summary['company_relationships']['frequent_buyers'].items()
        )[:5]:
            print(f"  → {company}: {count} acquisitions")
    
    if summary['company_relationships']['dual_role_companies']:
        print(f"\nCompanies in Dual Roles (Both Buyer & Target):")
        for company in summary['company_relationships']['dual_role_companies'][:3]:
            print(f"  → {company}")


def create_full_cluster_report(clustered_csv_path):
    """Create full analysis report for all clusters."""
    df = pd.read_csv(clustered_csv_path)
    
    print(f"\n{'#'*70}")
    print(f"# M&A TRANSACTION CLUSTERING ANALYSIS")
    print(f"{'#'*70}")
    print(f"\nTotal Transactions: {len(df)}")
    print(f"Number of Clusters: {df['Cluster'].nunique()}")
    
    # Analyze each cluster
    for cluster_id in sorted(df['Cluster'].unique()):
        summary = generate_cluster_summary(df, cluster_id)
        print_cluster_report(summary)
    
    return df


if __name__ == "__main__":
    import sys
    
    # Default to K=3 analysis
    csv_file = "eurozone_transactions_clustered_k3.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    create_full_cluster_report(csv_file)
