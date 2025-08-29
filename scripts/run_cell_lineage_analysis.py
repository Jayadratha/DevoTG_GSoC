#!/usr/bin/env python3
"""
Complete Analysis Pipeline Runner

This script runs the complete DevoTG analysis pipeline including:
- Data loading and validation
- Statistical analysis
- Threshold calculation
- Visualization generation
- Report creation

Usage:
    python scripts/run_cell_lineage_analysis.py [--csv_path PATH] [--output_dir PATH] [--config PATH]
        [--threshold_method METHOD] [--verbose]
Options:
  --csv_path PATH         Path to cell division CSV file (default: data/cell_lineage_datasets/cells_birth_and_pos.csv)
  --output_dir PATH       Output directory for results (default: outputs/lineage_analysis)
  --config PATH           Path to configuration file (JSON/YAML) (default: config.yaml)
  --threshold_method METHOD
                          Method for threshold calculation: '1sigma', '2sigma', 'percentile' (default: '1sigma')
  --verbose               Enable verbose output
Example:
  python scripts/run_cell_lineage_analysis.py
  python scripts/run_cell_lineage_analysis.py --csv_path data/cell_lineage_datasets/cells_birth_and_pos.csv --output_dir outputs/lineage_analysis --threshold_method 1sigma --verbose
"""

import argparse
import sys
from pathlib import Path
import json
import time
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from devotg.data import DatasetLoader, load_sample_data
from devotg.utils import ThresholdCalculator
from devotg.analysis import StatisticalAnalyzer, generate_comprehensive_report
from devotg.visualization import CellDivisionVisualizer

# Configure logging
import logging

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_directories(output_dir: Path) -> dict:
    """Create output directories and return paths."""
    directories = {
        'base': output_dir,
        'statistics': output_dir / 'statistics',
        'visualizations': output_dir / 'visualizations', 
        'dataset_processing': output_dir / 'dataset_processing'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {path}")
    
    return directories


def load_and_validate_data(csv_path: str) -> tuple:
    """Load and validate dataset."""
    logger.info(f"📊 Loading dataset from: {csv_path}")
    
    try:
        loader = DatasetLoader(default_csv_path=csv_path)
        df = loader.load_csv()
        is_valid = loader.validate_dataset(df)
        
        if is_valid:
            logger.info("✅ Dataset loaded and validated successfully!")
            return df, loader, True
        else:
            logger.info("⚠️  Dataset loaded but validation issues found.")
            return df, loader, False
            
    except FileNotFoundError:
        logger.error(f"❌ Dataset file not found: {csv_path}")
        logger.info("🔄 Using sample data for demonstration...")
        df = load_sample_data()
        loader = DatasetLoader()
        return df, loader, True
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        raise


def perform_statistical_analysis(df, directories: dict) -> dict:
    """Perform comprehensive statistical analysis."""
    logger.info("\n📈 Performing statistical analysis...")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(df)
    
    # Run all analyses
    results = analyzer.run_all_analyses()
    
    # Save detailed results
    analyzer.save_results(directories['statistics'] / 'detailed_analysis_results.json')
    
    # Generate comprehensive report
    report_path = directories['statistics'] / 'comprehensive_analysis_report.json'
    comprehensive_report = generate_comprehensive_report(df, save_path=str(report_path))
    
    logger.info("✅ Statistical analysis completed!")
    return results


def calculate_thresholds(df, directories: dict, method: str = "1sigma") -> dict:
    """Calculate thresholds for visualization."""
    logger.info(f"\n🎯 Calculating thresholds using {method} method...")
    
    # Calculate cell size if not present
    if 'cell_size' not in df.columns:
        import numpy as np
        df['cell_size'] = np.sqrt(df['parent_x']**2 + df['parent_y']**2 + df['parent_z']**2)
    
    # Initialize threshold calculator
    threshold_calc = ThresholdCalculator(method=method)
    thresholds = threshold_calc.calculate_thresholds(df)
    
    # Print summary
    threshold_calc.print_summary(df)
    logger.info("✅ Threshold calculation completed!")
    
    # Save thresholds
    threshold_path = directories['dataset_processing'] / 'calculated_thresholds.json'
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    logger.info(f"💾 Thresholds saved to: {threshold_path}")
    return thresholds


def create_visualizations(df, thresholds: dict, directories: dict) -> dict:
    """Create comprehensive visualizations."""
    logger.info("\n🎨 Creating visualizations...")
    
    # Initialize visualizer
    visualizer = CellDivisionVisualizer(
        csv_data=df,
        size_threshold_small=thresholds['size_threshold_small'],
        size_threshold_large=thresholds['size_threshold_large'],
        birth_time_threshold_low=thresholds['birth_time_threshold_low'],
        birth_time_threshold_high=thresholds['birth_time_threshold_high']
    )
    
    viz_results = {}
    output_dir = directories['visualizations']
    
    # Create static plots
    logger.info("📊 Creating static plots...")
    for color_scheme in ['generation', 'size', 'time']:
        try:
            fig, ax = visualizer.create_static_plot(color_by=color_scheme)
            fig_path = output_dir / f'static_plot_{color_scheme}.png'
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"  ✅ Saved: {fig_path.name}")
            viz_results[f'static_{color_scheme}'] = str(fig_path)
        except Exception as e:
            logger.error(f"  ❌ Error creating static plot ({color_scheme}): {e}")
    
    # Create division pattern analysis
    logger.info("📈 Creating division pattern analysis...")
    try:
        fig_patterns = visualizer.analyze_division_patterns()
        patterns_path = output_dir / 'division_patterns_analysis.png'
        fig_patterns.savefig(patterns_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✅ Saved: {patterns_path.name}")
        viz_results['patterns'] = str(patterns_path)
    except Exception as e:
        logger.info(f"  ❌ Error creating pattern analysis: {e}")
    
    # Create interactive plots
    logger.info("🎮 Creating interactive plots...")
    for color_scheme in ['generation', 'size', 'time']:
        try:
            fig_interactive = visualizer.create_interactive_plot(color_by=color_scheme)
            html_path = output_dir / f'interactive_plot_{color_scheme}.html'
            fig_interactive.write_html(str(html_path))
            logger.info(f"  ✅ Saved: {html_path.name}")
            viz_results[f'interactive_{color_scheme}'] = str(html_path)
        except Exception as e:
            logger.error(f"  ❌ Error creating interactive plot ({color_scheme}): {e}")
    
    # Export processed data
    logger.info("💾 Exporting processed data...")
    try:
        export_path = directories['dataset_processing'] / 'visualization_ready_data.csv'
        exported_data = visualizer.export_data(str(export_path))
        logger.info(f"  ✅ Exported {len(exported_data)} records")
        viz_results['processed_data'] = str(export_path)
    except Exception as e:
        logger.error(f"  ❌ Error exporting data: {e}")
    
    logger.info("✅ Visualization creation completed!")
    return viz_results


def generate_summary_report(directories: dict, results: dict, viz_results: dict, 
                          thresholds: dict, runtime: float) -> dict:
    """Generate final summary report."""
    logger.info("\n📋 Generating summary report...")
    
    summary = {
        'metadata': {
            'generation_timestamp': datetime.now().isoformat(),
            'runtime_seconds': runtime,
            'devotg_version': '0.1.0'
        },
        'directories': {k: str(v) for k, v in directories.items()},
        'analysis_results': {
            'statistical_analysis': 'completed' if results else 'failed',
            'threshold_calculation': 'completed' if thresholds else 'failed',
            'visualization_creation': f"{len(viz_results)} items created"
        },
        'generated_files': {
            'visualizations': viz_results,
            'thresholds': thresholds
        },
        'next_steps': [
            "Explore interactive HTML visualizations in web browser",
            "Review statistical analysis results in statistics/ directory", 
            "Use processed data for custom analysis",
            "Run TGN training notebook for machine learning analysis"
        ]
    }
    
    # Save summary
    summary_path = directories['base'] / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"📋 Summary report saved: {summary_path}")
    return summary


def main():
    """Main analysis pipeline function."""
    parser = argparse.ArgumentParser(
        description="Run complete DevoTG analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--csv_path', 
        type=str, 
        default='data/cell_lineage_datasets/cells_birth_and_pos.csv',
        help='Path to cell division CSV file'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/lineage_analysis',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--threshold_method',
        type=str, 
        default='1sigma',
        choices=['1sigma', '2sigma', 'percentile'],
        help='Method for threshold calculation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (JSON/YAML)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Record start time
    start_time = time.time()
    
    logger.info("🚀 DevoTG Complete Analysis Pipeline")
    logger.info("=" * 50)
    logger.info(f"CSV Path: {args.csv_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Threshold Method: {args.threshold_method}")
    logger.info("=" * 50)
    
    try:
        # 1. Setup directories
        directories = setup_directories(Path(args.output_dir))
        
        # 2. Load and validate data
        df, loader, is_valid = load_and_validate_data(args.csv_path)
        
        if args.verbose:
            logger.info("\nDataset Info:")
            loader.print_dataset_info(df)
        
        # 3. Perform statistical analysis
        analysis_results = perform_statistical_analysis(df, directories)
        
        # 4. Calculate thresholds
        thresholds = calculate_thresholds(df, directories, args.threshold_method)
        
        # 5. Create visualizations
        viz_results = create_visualizations(df, thresholds, directories)
        
        # 6. Calculate runtime
        runtime = time.time() - start_time
        
        # 7. Generate summary report
        summary = generate_summary_report(
            directories, analysis_results, viz_results, thresholds, runtime
        )
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"⏱️  Total runtime: {runtime:.1f} seconds")
        logger.info(f"📁 Results saved to: {directories['base']}")
        logger.info(f"📊 Generated {len(viz_results)} visualization files")
        logger.info(f"📈 Statistical analysis: {'✅ Complete' if analysis_results else '❌ Failed'}")
        logger.info(f"🎯 Threshold calculation: {'✅ Complete' if thresholds else '❌ Failed'}")
        
        logger.info("\n💡 Next steps:")
        for step in summary['next_steps']:
            logger.info(f"   • {step}")
        
        logger.info(f"\n📋 Full summary available at: {directories['base'] / 'analysis_summary.json'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Analysis pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())