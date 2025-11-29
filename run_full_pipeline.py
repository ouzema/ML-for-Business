"""
Complete M&A Clustering Pipeline Runner
Executes all steps from data loading to final evaluation.
"""

import sys
import os
from datetime import datetime


def run_pipeline(data_file="Project/MA Transactions Over 50M.xlsx"):
    """Run the complete M&A clustering pipeline."""
    
    print("\n" + "="*70)
    print("M&A TRANSACTION CLUSTERING PIPELINE")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Step 1: Data Loading (if needed)
    print("\n" + "#"*70)
    print("# STEP 1: DATA LOADING & ENRICHMENT")
    print("#"*70)
    
    if os.path.exists("eurozone_transactions_enriched.csv"):
        print("✓ Enriched data already exists, skipping data_loader.py")
        print("  (Delete eurozone_transactions_enriched.csv to re-run)")
    else:
        print("Running data_loader.py...")
        import data_loader
        try:
            df = data_loader.load_and_filter_data(data_file)
            df.to_csv("eurozone_transactions.csv", index=False)
            df_enriched = data_loader.enrich_data(df)
            df_enriched.to_csv("eurozone_transactions_enriched.csv", 
                             index=False)
            print("✓ Data loading and enrichment complete!")
        except Exception as e:
            print(f"✗ Error in data loading: {e}")
            return False
    
    # Step 2: Preprocessing
    print("\n" + "#"*70)
    print("# STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("#"*70)
    
    if os.path.exists("X_final.npy"):
        print("✓ Processed features already exist, skipping preprocessing.py")
        print("  (Delete X_final.npy to re-run)")
    else:
        print("Running preprocessing.py...")
        import preprocessing
        try:
            X_final = preprocessing.preprocess_data(
                "eurozone_transactions_enriched.csv"
            )
            print("✓ Preprocessing complete!")
        except Exception as e:
            print(f"✗ Error in preprocessing: {e}")
            return False
    
    # Step 3: Modeling
    print("\n" + "#"*70)
    print("# STEP 3: CLUSTERING & DIMENSIONALITY REDUCTION")
    print("#"*70)
    
    print("Running modeling.py...")
    import modeling
    try:
        modeling.run_modeling("X_final.npy", 
                            "eurozone_transactions_enriched.csv")
        print("✓ Modeling complete!")
    except Exception as e:
        print(f"✗ Error in modeling: {e}")
        return False
    
    # Step 4: Evaluation
    print("\n" + "#"*70)
    print("# STEP 4: CLUSTER EVALUATION & ANALYSIS")
    print("#"*70)
    
    print("Running evaluation.py...")
    import evaluation
    try:
        print("\n--- K=3 Clustering Analysis ---")
        evaluation.evaluate_clusters("eurozone_transactions_clustered_k3.csv")
        
        print("\n\n--- K=4 Clustering Analysis ---")
        evaluation.evaluate_clusters("eurozone_transactions_clustered_k4.csv")
        
        print("✓ Evaluation complete!")
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        return False
    
    # Step 5: Advanced Analysis
    print("\n" + "#"*70)
    print("# STEP 5: ADVANCED CLUSTER BUSINESS INSIGHTS")
    print("#"*70)
    
    print("Running cluster_analyzer.py...")
    try:
        from cluster_analyzer import create_full_cluster_report
        create_full_cluster_report("eurozone_transactions_clustered_k3.csv")
        print("✓ Advanced analysis complete!")
    except Exception as e:
        print(f"✗ Error in cluster analysis: {e}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated Files:")
    print("  - eurozone_transactions_enriched.csv")
    print("  - X_final.npy")
    print("  - extracted_keywords.txt")
    print("  - eurozone_transactions_clustered_k3.csv")
    print("  - eurozone_transactions_clustered_k4.csv")
    print("  - clusters_visualization_k3.png")
    print("  - clusters_visualization_k4.png")
    print("  - eurozone_transactions_clustered_k3_analysis.png")
    print("  - eurozone_transactions_clustered_k4_analysis.png")
    print("\nNext Steps:")
    print("  1. Review cluster visualizations (PNG files)")
    print("  2. Examine extracted_keywords.txt for deal themes")
    print("  3. Analyze clustered CSV files for specific transactions")
    print("  4. Use insights for trading opportunity identification")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    # Check if data file exists
    data_file = "Project/MA Transactions Over 50M.xlsx"
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    if not os.path.exists(data_file):
        print(f"\n✗ Error: Data file not found: {data_file}")
        print("\nUsage: python run_full_pipeline.py [data_file.xlsx]")
        print("\nExpected file locations:")
        print("  - Project/MA Transactions Over 50M.xlsx")
        sys.exit(1)
    
    success = run_pipeline(data_file)
    sys.exit(0 if success else 1)
