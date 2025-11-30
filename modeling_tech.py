import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# For large datasets, use MiniBatchKMeans for speed
USE_MINIBATCH = True  # Set to False for standard KMeans

def run_modeling(data_path, original_data_path):
    print("Loading processed data...")
    X = np.load(data_path)
    df_orig = pd.read_csv(original_data_path)
    
    print(f"Data shape: {X.shape}")
    
    # 1. Dimensionality Reduction
    # High dimensionality is bad for clustering (curse of dimensionality).
    # Use PCA to reduce to a reasonable number of components, capturing 95% variance.
    
    print(f"Applying PCA (this may take a moment for {X.shape[0]} transactions)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"✓ PCA complete: {X.shape[1]} → {X_pca.shape[1]} dimensions")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # 2. Clustering
    # We'll try K-Means and determine optimal K using Silhouette Score.
    
    print(f"\nFinding optimal K for K-Means ({X_pca.shape[0]} transactions)...")
    print(f"Using {'MiniBatchKMeans (faster)' if USE_MINIBATCH else 'Standard KMeans'}...")
    
    silhouette_scores = []
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        if USE_MINIBATCH and X_pca.shape[0] > 1000:
            # Use MiniBatchKMeans for large datasets (much faster)
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, 
                                    batch_size=256, n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        labels = kmeans.fit_predict(X_pca)
        
        # Sample for silhouette score if dataset is very large (for speed)
        if X_pca.shape[0] > 5000:
            sample_size = 5000
            sample_indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
            score = silhouette_score(X_pca[sample_indices], labels[sample_indices])
            print(f"K={k}, Silhouette={score:.4f} (sampled), Inertia={kmeans.inertia_:.2f}")
        else:
            score = silhouette_score(X_pca, labels)
            print(f"K={k}, Silhouette={score:.4f}, Inertia={kmeans.inertia_:.2f}")
        
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)
        
    # Pick best K (Original logic)
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"Best K based on Silhouette Score: {best_k}")
    
    # Generate results for K=3 and K=4 as requested
    for k in [3, 4]:
        print(f"\nGenerating results for K={k}...")
        if USE_MINIBATCH and X_pca.shape[0] > 1000:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, 
                                    batch_size=256, n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
        
        # t-SNE Visualization (sample if too many points for visualization)
        print(f"Running t-SNE for K={k}...")
        if X_pca.shape[0] > 2000:
            # Sample for visualization to keep it readable
            sample_size = 2000
            sample_indices = np.random.choice(X_pca.shape[0], sample_size, replace=False)
            print(f"  Sampling {sample_size} points for visualization clarity...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne_sample = tsne.fit_transform(X_pca[sample_indices])
            
            # Create visualization with sampled data
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_tsne_sample[:, 0], X_tsne_sample[:, 1], 
                                c=cluster_labels[sample_indices], 
                                cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f't-SNE Visualization (K={k}, {sample_size} sampled from {X_pca.shape[0]} transactions)')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.savefig(f"tech_clusters_visualization_k{k}.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_pca)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                c=cluster_labels, cmap='viridis', 
                                alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f't-SNE Visualization of M&A Transactions (K={k})')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.savefig(f"tech_clusters_visualization_k{k}.png", dpi=150, bbox_inches='tight')
            plt.close()
        print(f"✓ Saved visualization to tech_clusters_visualization_k{k}.png")
        
        # Save Clustered Data
        df_clustered = df_orig.copy()
        df_clustered['Cluster'] = cluster_labels
        output_file = f"tech_transactions_clustered_k{k}.csv"
        df_clustered.to_csv(output_file, index=False)
        print(f"Saved clustered data to {output_file}")

    # Save K=3 as the default result
    print("\nSaving default output files with K=3 results...")
    if USE_MINIBATCH and X_pca.shape[0] > 1000:
        kmeans_final = MiniBatchKMeans(n_clusters=3, random_state=42, 
                                       batch_size=256, n_init=10)
    else:
        kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    cluster_labels = kmeans_final.fit_predict(X_pca)
    
    df_clustered = df_orig.copy()
    df_clustered['Cluster'] = cluster_labels
    df_clustered.to_csv("tech_transactions_clustered.csv", index=False)
    print(f"✓ Saved default clustered data (K=3) to tech_transactions_clustered.csv")

    # 5. Cluster Interpretation (Basic)
    print("\nCluster Interpretation (Mean values of key metrics):")
    # Select some key columns to inspect
    key_cols = ['Total Transaction Value ($USDmm, Historical rate)', 'Market Cap', 'Revenue', 'EBITDA']
    # Ensure they exist
    existing_cols = [c for c in key_cols if c in df_orig.columns]
    
    if existing_cols:
        print(df_clustered.groupby('Cluster')[existing_cols].mean())
    
    return df_orig

if __name__ == "__main__":
    run_modeling("X_tech_final.npy", "tech_transactions_enriched.csv")
