#!/usr/bin/env python3
"""
Tech M&A Clustering Pipeline
Analyzes technology sector M&A transactions only
"""
import subprocess
import sys
from datetime import datetime

def run_step(step_num, description, script, skip_if_exists=None):
    """Run a pipeline step."""
    print("\n" + "#"*70)
    print(f"# STEP {step_num}: {description.upper()}")
    print("#"*70)
    
    if skip_if_exists:
        import os
        if os.path.exists(skip_if_exists):
            print(f"✓ {description} already exists, skipping {script}")
            print(f"  (Delete {skip_if_exists} to re-run)")
            return True
    
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"✗ Error in {script}")
        return False
    
    print(f"✓ {description} complete!")
    return True

def main():
    """Run the complete tech M&A clustering pipeline."""
    
    print("="*70)
    print("TECH M&A TRANSACTION CLUSTERING PIPELINE")
    print("="*70)
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    # Pipeline steps
    steps = [
        (1, "Data Loading & Enrichment (Tech Only)", "data_loader_tech.py", "tech_transactions_enriched.csv"),
        (2, "Preprocessing & Feature Engineering", "preprocessing_tech.py", "X_tech_final.npy"),
        (3, "Clustering & Dimensionality Reduction", "modeling_tech.py", None),
        (4, "Cluster Evaluation & Analysis", "evaluation_tech.py", None),
        (5, "Advanced Cluster Business Insights", "cluster_analyzer_tech.py", None),
    ]
    
    for step_num, description, script, skip_file in steps:
        success = run_step(step_num, description, script, skip_file)
        if not success:
            print("\n" + "="*70)
            print("PIPELINE FAILED!")
            print("="*70)
            sys.exit(1)
    
    # Success
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Generated Files:")
    print("  - tech_transactions_enriched.csv")
    print("  - X_tech_final.npy")
    print("  - extracted_keywords_tech.txt")
    print("  - tech_transactions_clustered_k3.csv")
    print("  - tech_transactions_clustered_k4.csv")
    print("  - tech_clusters_visualization_k3.png")
    print("  - tech_clusters_visualization_k4.png")
    print("  - tech_transactions_clustered_k3_analysis.png")
    print("  - tech_transactions_clustered_k4_analysis.png")
    print()
    print("Next Steps:")
    print("  1. Review cluster visualizations (PNG files)")
    print("  2. Examine extracted_keywords_tech.txt for tech deal themes")
    print("  3. Analyze clustered CSV files for specific transactions")
    print("  4. Compare with full Eurozone analysis for tech insights")
    print("="*70)

if __name__ == "__main__":
    main()
