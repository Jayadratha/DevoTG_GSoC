#!/usr/bin/env python3
"""
Main script for running comprehensive C. elegans connectome analysis.

This script integrates all components of the DevoTG framework to:
1. Download and process connectome datasets
2. Perform network analysis
3. Generate comprehensive visualizations
4. Create interactive animations
Usage:
    python run_connectome_analysis.py [-h] [--skip-download] [--skip-analysis] [--skip-visualization] [--output-dir OUTPUT_DIR]
Options:
  -h, --help            Show this help message and exit
  --skip-download       Skip dataset download if files already exist
  --skip-analysis       Skip network analysis if already completed
  --skip-visualization  Skip static visualizations if already generated
  --output-dir OUTPUT_DIR
                        Base output directory (default: current directory)
Example:
  python scripts/run_connectome_analysis.py --skip-download
  python scripts/run_connectome_analysis.py --skip-download --output-dir ./outputs/connectome_analysis/connectome_analysis
  python scripts/run_connectome_analysis.py --output-dir ./outputs/connectome_analysis/connectome_analysis
Requirements:
- Internet connection for dataset download (unless --skip-download is used)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path to import devotg
sys.path.insert(0, str(Path(__file__).parent.parent))

from devotg.data import load_connectome_datasets
from devotg.analysis import analyze_connectome_network
from devotg.visualization import (
    create_comprehensive_visualizations,
    create_neural_network_animation,
    generate_network_summary
)

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/connectome_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary output directories."""
    directories = [
        "data/connectome_datasets",
        "data/processed_datasets",
        "outputs/connectome_analysis/statistics",
        "outputs/connectome_analysis/visualizations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive C. elegans connectome analysis"
    )
    parser.add_argument(
        "--skip-download", 
        action="store_true",
        help="Skip dataset download if files already exist"
    )
    parser.add_argument(
        "--skip-analysis", 
        action="store_true",
        help="Skip network analysis if already completed"
    )
    parser.add_argument(
        "--skip-visualization", 
        action="store_true",
        help="Skip static visualizations if already generated"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/connectome_analysis",
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Starting DevoTG Connectome Analysis Pipeline")
    logger.info("="*60)
    
    # Setup directories
    setup_directories()
    
    # Step 1: Download and process datasets
    if not args.skip_download:
        logger.info("\n🔄 Step 1: Downloading and processing connectome datasets...")
        try:
            loader = load_connectome_datasets(
                data_dir="data/connectome_datasets",
                output_dir="data/processed_datasets"
            )
            logger.info("✅ Dataset download and processing complete")
        except Exception as e:
            logger.error(f"❌ Error in dataset processing: {e}")
            return 1
    else:
        logger.info("⏭️ Skipping dataset download (using existing files)")
    
    # Check if required files exist
    required_files = [
        "data/processed_datasets/dtdg_nodes.csv",
        "data/processed_datasets/dtdg_edges_temporal.csv",
        "data/processed_datasets/dtdg_summary_statistics.csv"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        logger.error("Please run without --skip-download to generate these files")
        return 1
    
    # Step 2: Network analysis
    if not args.skip_analysis:
        logger.info("\n📊 Step 2: Performing network analysis...")
        try:
            analyzer = analyze_connectome_network(
                edges_csv_path="data/processed_datasets/dtdg_edges_temporal.csv",
                nodes_csv_path="data/processed_datasets/dtdg_nodes.csv", 
                output_dir="outputs/connectome_analysis/statistics"
            )
            logger.info("✅ Network analysis complete")
        except Exception as e:
            logger.error(f"❌ Error in network analysis: {e}")
            return 1
    else:
        logger.info("⏭️ Skipping network analysis (using existing results)")
    
    # Step 3: Static visualizations
    if not args.skip_visualization:
        logger.info("\n📈 Step 3: Creating static visualizations...")
        try:
            visualizations = create_comprehensive_visualizations(
                edges_csv_path="data/processed_datasets/dtdg_edges_temporal.csv",
                nodes_csv_path="data/processed_datasets/dtdg_nodes.csv",
                summary_csv_path="data/processed_datasets/dtdg_summary_statistics.csv",
                output_dir="outputs/connectome_analysis/visualizations"
            )
            logger.info("✅ Static visualizations complete")
        except Exception as e:
            logger.error(f"❌ Error in visualization creation: {e}")
            return 1
    else:
        logger.info("⏭️ Skipping static visualizations (using existing files)")
    
    # Step 4: Interactive animation
    logger.info("\n🎬 Step 4: Creating interactive neural network animation...")
    try:
        animation_fig = create_neural_network_animation(
            nodes_csv_path="data/processed_datasets/dtdg_nodes.csv",
            edges_csv_path="data/processed_datasets/dtdg_edges_temporal.csv",
            summary_csv_path="data/processed_datasets/dtdg_summary_statistics.csv",
            output_path="outputs/connectome_analysis/visualizations/neural_development_animation.html"
        )
        logger.info("✅ Interactive animation complete")
    except Exception as e:
        logger.error(f"❌ Error in animation creation: {e}")
        return 1
    
    # Step 5: Generate summary report
    logger.info("\n📋 Step 5: Generating summary report...")
    try:
        import pandas as pd
        nodes_df = pd.read_csv("data/processed_datasets/dtdg_nodes.csv")
        edges_df = pd.read_csv("data/processed_datasets/dtdg_edges_temporal.csv")
        
        summary_stats = generate_network_summary(nodes_df, edges_df)
        
        # Save summary to file
        import json
        with open("outputs/connectome_analysis/statistics/network_summary.json", "w") as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        logger.info("✅ Summary report generated")
    except Exception as e:
        logger.error(f"❌ Error in summary generation: {e}")
        return 1
    
    # Final report
    logger.info("\n" + "="*60)
    logger.info("🎉 DevoTG Connectome Analysis Complete!")
    logger.info("="*60)
    logger.info("\n📁 Generated files:")
    logger.info("   📊 Dataset Processing:")
    logger.info("      - data/processed_datasets/dtdg_*.csv")
    logger.info("      - data/processed_datasets/dtdg_celegans_development.json")
    logger.info("\n   📈 Analysis Results:")
    logger.info("      - outputs/connectome_analysis/statistics/*.csv")
    logger.info("      - outputs/connectome_analysis/statistics/comprehensive_network_report.json")
    logger.info("      - outputs/connectome_analysis/statistics/network_summary.json")
    logger.info("\n   🎨 Visualizations:")
    logger.info("      - outputs/connectome_analysis/visualizations/*.png (static plots)")
    logger.info("      - outputs/connectome_analysis/visualizations/*.html (interactive plots)")
    logger.info("      - outputs/connectome_analysis/visualizations/network_development_animation.mp4")
    logger.info("      - outputs/connectome_analysis/visualizations/neural_development_animation.html")
    logger.info("\n🚀 Open the HTML files in a web browser to explore interactive visualizations!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
