#!/usr/bin/env python3
"""
TGN Training Script

This script trains a Temporal Graph Network on C. elegans cell division data.

Usage:
    python scripts/train_tgn.py [--csv_path PATH] [--epochs N] [--output_dir PATH]
        [--memory_dim N] [--embedding_dim N] [--batch_size N] [--learning_rate LR]
        [--val_ratio R] [--test_ratio R] [--device DEVICE] [--seed N]
        [--save_model] [--verbose]

Options:
  --csv_path PATH
  --epochs N
  --output_dir PATH
  --memory_dim N
  --embedding_dim N
  --batch_size N
  --learning_rate LR
  --val_ratio R
  --test_ratio R
  --device DEVICE
  --seed N
  --save_model
  --verbose
Example:
  python scripts/train_tgn.py --csv_path data/cell_lineage_datasets/cells_birth_and_pos.csv --epochs 30 --output_dir outputs/models --save_model --verbose
Outputs:
- Trained model weights in specified output directory (if --save_model is used)
- Training history plots in specified output directory
- Training summary JSON file in specified output directory
"""

import argparse
import sys
from pathlib import Path
import json
import time
import torch
import matplotlib.pyplot as plt

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from devotg.data import build_cell_ctdg, load_sample_data
from devotg.models import TGNModel

# Configure logging
import logging

# Ensure log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tgn_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TGN model on cell division data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='data/cell_lineage_datasets/cells_birth_and_pos.csv',
        help='Path to cell division CSV file'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/models',
        help='Output directory for model and results'
    )
    
    parser.add_argument(
        '--memory_dim',
        type=int,
        default=100,
        help='Memory dimension for TGN'
    )
    
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=100,
        help='Embedding dimension for TGN'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=200,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for optimization'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation split ratio'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test split ratio'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save trained model weights'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def setup_reproducibility(seed: int):
    """Setup random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For numpy if used
    import numpy as np
    np.random.seed(seed)
    
    logger.info(f"🌱 Random seed set to: {seed}")


def load_temporal_data(csv_path: str, verbose: bool = False):
    """Load and prepare temporal graph data."""
    logger.info(f"📊 Loading temporal graph data from: {csv_path}")
    
    try:
        temporal_data = build_cell_ctdg(
            csv_path=csv_path,
            feature_dim=172
        )
        
        if verbose:
            logger.info(f"   Nodes: {temporal_data.num_nodes}")
            logger.info(f"   Events: {temporal_data.num_events}")
            logger.info(f"   Node features: {temporal_data.x.shape}")
            logger.info(f"   Edge features: {temporal_data.msg.shape}")
        
        return temporal_data
        
    except FileNotFoundError:
        logger.error(f"❌ Dataset file not found: {csv_path}")
        logger.info("🔄 Using sample data for demonstration...")
        
        # Create sample data
        sample_df = load_sample_data()
        temp_path = Path('temp_sample_data.csv')
        sample_df.to_csv(temp_path, index=False)
        
        try:
            temporal_data = build_cell_ctdg(str(temp_path), feature_dim=172)
            temp_path.unlink()  # Clean up
            return temporal_data
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    except Exception as e:
        logger.error(f"❌ Error loading temporal data: {e}")
        raise


def create_model(temporal_data, args) -> TGNModel:
    """Create and initialize TGN model."""
    logger.info("🧠 Initializing TGN model...")
    
    model_config = {
        'memory_dim': args.memory_dim,
        'time_dim': args.memory_dim,  # Same as memory_dim
        'embedding_dim': args.embedding_dim,
        'device': args.device
    }
    
    tgn_model = TGNModel(
        num_nodes=temporal_data.num_nodes,
        msg_dim=temporal_data.msg.size(-1),
        **model_config
    )
    
    # Update learning rate if different from default
    if args.learning_rate != 0.001:
        for param_group in tgn_model.optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        logger.info(f"   Learning rate set to: {args.learning_rate}")
    
    if args.verbose:
        model_info = tgn_model.get_model_info()
        logger.info("   Model configuration:")
        for key, value in model_info.items():
            logger.info(f"     {key}: {value}")
    
    return tgn_model, model_config


def train_model(tgn_model, temporal_data, args) -> dict:
    """Train the TGN model."""
    logger.info("🚀 Starting TGN training...")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Device: {tgn_model.device}")
    logger.info("\n" + "=" * 60)
    
    # Training configuration
    training_config = {
        'epochs': args.epochs,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'batch_size': args.batch_size,
        'verbose': args.verbose
    }
    
    # Start training
    start_time = time.time()
    history = tgn_model.train_model(temporal_data, **training_config)
    training_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Training completed in {training_time:.1f} seconds")
    
    return history, training_time


def save_results(tgn_model, history, training_time, model_config, args, output_dir: Path):
    """Save training results and model."""
    logger.info(f"💾 Saving results to: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Training loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation metrics
    axes[0, 1].plot(epochs, history['val_ap'], 'g-', label='AP', linewidth=2)
    axes[0, 1].plot(epochs, history['val_auc'], 'r-', label='AUC', linewidth=2)
    axes[0, 1].set_title('Validation Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Test metrics
    axes[1, 0].plot(epochs, history['test_ap'], 'g--', label='AP', linewidth=2)
    axes[1, 0].plot(epochs, history['test_auc'], 'r--', label='AUC', linewidth=2)
    axes[1, 0].set_title('Test Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Combined view
    axes[1, 1].plot(epochs, history['val_ap'], 'g-', label='Val AP', linewidth=2)
    axes[1, 1].plot(epochs, history['test_ap'], 'g--', label='Test AP', linewidth=2)
    axes[1, 1].plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    axes[1, 1].plot(epochs, history['test_auc'], 'r--', label='Test AUC', linewidth=2)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'training_history.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"   📊 Training plot saved: {plot_path}")
    
    # Save performance summary
    final_metrics = {
        'final_train_loss': float(history['train_loss'][-1]),
        'best_val_ap': float(max(history['val_ap'])),
        'best_val_auc': float(max(history['val_auc'])),
        'final_test_ap': float(history['test_ap'][-1]),
        'final_test_auc': float(history['test_auc'][-1]),
        'training_time_seconds': training_time
    }
    
    performance_summary = {
        'model_config': model_config,
        'training_args': vars(args),
        'final_metrics': final_metrics,
        'best_epoch': int(np.argmax(history['val_ap']) + 1),
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_ap': [float(x) for x in history['val_ap']],
            'val_auc': [float(x) for x in history['val_auc']],
            'test_ap': [float(x) for x in history['test_ap']],
            'test_auc': [float(x) for x in history['test_auc']]
        }
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(performance_summary, f, indent=2)
    logger.info(f"   📋 Training summary saved: {summary_path}")
    
    # Save model if requested
    if args.save_model:
        model_path = output_dir / 'trained_tgn_model.pth'
        tgn_model.save_model(str(model_path))
        logger.info(f"   🧠 Model weights saved: {model_path}")
        return model_path
    
    return None


def print_info_final_summary(history, training_time, model_path, output_dir):
    """logger.info final training summary."""
    import numpy as np
    
    final_metrics = {
        'Final Train Loss': history['train_loss'][-1],
        'Best Val AP': max(history['val_ap']),
        'Best Val AUC': max(history['val_auc']),
        'Final Test AP': history['test_ap'][-1],
        'Final Test AUC': history['test_auc'][-1]
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 TGN TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    
    logger.info(f"\n⏱️  Training Time: {training_time:.1f} seconds")
    logger.info("📊 Final Performance Metrics:")
    for metric, value in final_metrics.items():
        logger.info(f"   • {metric}: {value:.4f}")
    
    best_epoch = np.argmax(history['val_ap']) + 1
    logger.info(f"\n🏆 Best performance at epoch {best_epoch}")
    
    # Performance assessment
    final_auc = history['test_auc'][-1]
    if final_auc > 0.9:
        performance = "Excellent"
    elif final_auc > 0.8:
        performance = "Good"
    elif final_auc > 0.7:
        performance = "Fair"
    else:
        performance = "Poor"
    
    logger.info(f"🎯 Model Performance: {performance} (Test AUC: {final_auc:.3f})")
    
    logger.info(f"\n📁 Results saved to: {output_dir}")
    if model_path:
        logger.info(f"💾 Model available at: {model_path}")
    
    logger.info("\n💡 Next Steps:")
    logger.info("   • Review training plots and metrics")
    logger.info("   • Analyze model performance on specific data subsets")
    logger.info("   • Experiment with different hyperparameters")
    logger.info("   • Use trained model for downstream tasks")


def main():
    """Main training function."""
    args = parse_arguments()
    
    logger.info("🚀 DevoTG TGN Training Script")
    logger.info("=" * 50)
    logger.info(f"CSV Path: {args.csv_path}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 50)
    
    try:
        # Setup
        setup_reproducibility(args.seed)
        output_dir = Path(args.output_dir)
        
        # Load data
        temporal_data = load_temporal_data(args.csv_path, args.verbose)
        
        # Create model
        tgn_model, model_config = create_model(temporal_data, args)
        
        # Train model
        history, training_time = train_model(tgn_model, temporal_data, args)
        
        # Save results
        model_path = save_results(
            tgn_model, history, training_time, model_config, args, output_dir
        )
        
        # print info summary
        print_info_final_summary(history, training_time, model_path, output_dir)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import numpy as np  # Import here for the main function
    sys.exit(main())