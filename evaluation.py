import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cluster_analyzer import (
    generate_cluster_summary, 
    print_cluster_report,
    create_full_cluster_report
)


def evaluate_clusters(filepath):
    print(f"\n{'='*70}")
    print(f"Loading clustered data from: {filepath}")
    print(f"{'='*70}")
    df = pd.read_csv(filepath)
    
    # Cluster counts
    print("\nCluster Distribution:")
    print(df['Cluster'].value_counts().sort_index())
    
    # Analyze numeric features by cluster
    numeric_cols = [
        'Total Transaction Value ($USDmm, Historical rate)',
        'rationale_operational',
        'rationale_financial',
        'rationale_regulatory',
        'rationale_technology',
        'rationale_market_expansion',
        'mentioned_revenue',
        'mentioned_ebitda',
        'Market Cap', 
        'Revenue', 
        'EBITDA', 
        'Enterprise Value'
    ]
    existing_num = [c for c in numeric_cols if c in df.columns]
    
    if existing_num:
        print("\n" + "="*70)
        print("QUANTITATIVE CLUSTER CHARACTERISTICS")
        print("="*70)
        cluster_means = df.groupby('Cluster')[existing_num].mean()
        
        # Reverse log transformation for transaction value
        if 'Total Transaction Value ($USDmm, Historical rate)' in cluster_means.columns:
            cluster_means['Avg Deal Value (Original Scale)'] = np.expm1(
                cluster_means['Total Transaction Value ($USDmm, Historical rate)']
            )
        
        print(cluster_means.round(2))
    
    # Analyze categorical features by cluster
    print("\n" + "="*70)
    print("QUALITATIVE CLUSTER CHARACTERISTICS")
    print("="*70)
    
    # Top Sectors per Cluster
    if 'Sector' in df.columns:
        print("\nTop 3 Sectors per Cluster:")
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster]
            print(f"\n  Cluster {cluster}:")
            sector_counts = cluster_data['Sector'].value_counts().head(3)
            for sector, count in sector_counts.items():
                if pd.notna(sector):
                    pct = (count / len(cluster_data)) * 100
                    print(f"    → {sector}: {count} ({pct:.1f}%)")
    
    # Top Countries per Cluster
    if 'Country/Region of Incorporation [Target/Issuer]' in df.columns:
        print("\nTop 3 Countries per Cluster:")
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster]
            print(f"\n  Cluster {cluster}:")
            country_counts = cluster_data[
                'Country/Region of Incorporation [Target/Issuer]'
            ].value_counts().head(3)
            for country, count in country_counts.items():
                pct = (count / len(cluster_data)) * 100
                print(f"    → {country}: {count} ({pct:.1f}%)")
    
    # Transaction Status
    if 'Transaction Status' in df.columns:
        print("\nTransaction Status per Cluster:")
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster]
            print(f"\n  Cluster {cluster}:")
            status_counts = cluster_data['Transaction Status'].value_counts()
            for status, count in status_counts.items():
                pct = (count / len(cluster_data)) * 100
                print(f"    → {status}: {count} ({pct:.1f}%)")
    
    # Run detailed cluster analysis
    print("\n" + "#"*70)
    print("# DETAILED CLUSTER BUSINESS INSIGHTS")
    print("#"*70)
    
    for cluster_id in sorted(df['Cluster'].unique()):
        summary = generate_cluster_summary(df, cluster_id)
        print_cluster_report(summary)
    
    # Create visualization
    create_cluster_visualizations(df, filepath.replace('.csv', ''))
    
    return df


def create_cluster_visualizations(df, output_prefix):
    """Create additional visualizations for cluster analysis."""
    
    # 1. Cluster size pie chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cluster distribution
    cluster_counts = df['Cluster'].value_counts().sort_index()
    axes[0, 0].pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Cluster Distribution')
    
    # Average deal value by cluster
    if 'Total Transaction Value ($USDmm, Historical rate)' in df.columns:
        avg_values = df.groupby('Cluster')[
            'Total Transaction Value ($USDmm, Historical rate)'
        ].mean()
        avg_values_orig = np.expm1(avg_values)
        axes[0, 1].bar(avg_values_orig.index, avg_values_orig.values, 
                       color='skyblue')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Average Deal Value ($M USD)')
        axes[0, 1].set_title('Average Transaction Value by Cluster')
        axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Deal rationale heatmap
    rationale_cols = [c for c in df.columns if c.startswith('rationale_')]
    if rationale_cols:
        rationale_means = df.groupby('Cluster')[rationale_cols].mean()
        rationale_means.columns = [c.replace('rationale_', '').replace('_', ' ').title() 
                                   for c in rationale_means.columns]
        sns.heatmap(rationale_means.T, annot=True, fmt='.2f', 
                   cmap='YlOrRd', ax=axes[1, 0])
        axes[1, 0].set_title('Deal Rationale Intensity by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Rationale Type')
    
    # Top countries distribution
    if 'Country/Region of Incorporation [Target/Issuer]' in df.columns:
        top_countries = df['Country/Region of Incorporation [Target/Issuer]'].value_counts().head(8)
        axes[1, 1].barh(range(len(top_countries)), top_countries.values, 
                       color='lightcoral')
        axes[1, 1].set_yticks(range(len(top_countries)))
        axes[1, 1].set_yticklabels(top_countries.index)
        axes[1, 1].set_xlabel('Number of Transactions')
        axes[1, 1].set_title('Top 8 Target Countries')
        axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\nSaved detailed analysis to {output_prefix}_analysis.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# M&A CLUSTERING EVALUATION REPORT")
    print("#"*70)
    
    print("\n" + "="*70)
    print("EVALUATING K=3 CLUSTERING")
    print("="*70)
    df_k3 = evaluate_clusters("eurozone_transactions_clustered_k3.csv")
    
    print("\n\n" + "="*70)
    print("EVALUATING K=4 CLUSTERING")
    print("="*70)
    df_k4 = evaluate_clusters("eurozone_transactions_clustered_k4.csv")
    
    print("\n" + "#"*70)
    print("# EVALUATION COMPLETE")
    print("#"*70)

