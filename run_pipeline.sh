#!/bin/bash

# M&A Clustering Project - Quick Start Script
# This script runs the complete pipeline

echo "=========================================="
echo "M&A Transaction Clustering Pipeline"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if data file exists
if [ ! -f "Project/MA Transactions Over 50M.xlsx" ]; then
    echo "❌ Data file not found: Project/MA Transactions Over 50M.xlsx"
    echo "Please ensure the data file is in the correct location"
    exit 1
fi

echo "✓ Data file found"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip3 install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "⚠ Warning: Some dependencies may have failed to install"
fi

echo ""
echo "=========================================="
echo "Running Pipeline..."
echo "=========================================="
echo ""

# Run the pipeline
python3 run_full_pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    echo "  📊 eurozone_transactions_clustered_k3.csv"
    echo "  📊 eurozone_transactions_clustered_k4.csv"
    echo "  📈 clusters_visualization_k3.png"
    echo "  📈 clusters_visualization_k4.png"
    echo "  📝 extracted_keywords.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Open the PNG files to view cluster visualizations"
    echo "  2. Review the CSV files for detailed transaction data"
    echo "  3. Check extracted_keywords.txt for deal themes"
    echo "  4. Run: python3 cluster_analyzer.py eurozone_transactions_clustered_k3.csv"
    echo ""
else
    echo ""
    echo "❌ Pipeline failed. Check error messages above."
    exit 1
fi
