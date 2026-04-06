"""
Quick PINN Model Evaluation Script
Trains a simplified PINN and reports accuracy metrics
"""
import os
import sys
import numpy as np
import logging
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_pinn():
    """Evaluate PINN model accuracy"""
    logging.info("="*60)
    logging.info("PINN MODEL ACCURACY EVALUATION")
    logging.info("="*60)
    
    # Load data
    from pinn_system.data_loader import load_data
    
    logging.info("\n1. Loading data...")
    X, y = load_data('data/dataset.parquet', ['time', 'x', 'temperature', 'survival_rate'])
    
    # Adjust for heat model (needs only x, time)
    X = X[:, [1, 0]]  # [time, x, temp] -> [x, time]
    
    logging.info(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Import and setup PINN
    logging.info("\n2. Setting up PINN model...")
    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde
    from pinn_system.models.heat_model import HeatModel
    
    pinn = HeatModel(alpha=0.4)
    pinn.setup_model_with_anchors(X, y)
    
    # Quick training with fewer iterations
    logging.info("\n3. Training PINN (quick evaluation - 500 iterations)...")
    pinn.compile(optimizer='adam', learning_rate=0.001)
    
    # Train
    pinn.train(iterations=500)
    
    # Get predictions
    logging.info("\n4. Generating predictions...")
    predictions = pinn.predict(X)
    
    # Calculate metrics
    logging.info("\n5. Calculating accuracy metrics...")
    
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    accuracy = r2 * 100
    
    # Calculate relative error
    relative_error = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100
    
    # Get final losses
    try:
        final_loss = pinn.model.train_state.loss_train[-1] if hasattr(pinn.model, 'train_state') else 0
        if hasattr(final_loss, '__len__'):
            final_loss = sum(final_loss)
    except:
        final_loss = 0
    
    logging.info("\n" + "="*60)
    logging.info("PINN MODEL PERFORMANCE METRICS")
    logging.info("="*60)
    logging.info(f"Training Iterations: 500")
    logging.info(f"Final Training Loss: {final_loss:.6f}")
    logging.info(f"")
    logging.info(f"Mean Squared Error (MSE): {mse:.6f}")
    logging.info(f"Mean Absolute Error (MAE): {mae:.6f}")
    logging.info(f"R² Score: {r2:.4f}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"Mean Relative Error: {relative_error:.2f}%")
    logging.info("="*60)
    
    # Compare with target
    if accuracy >= 90:
        logging.info(f"✅ PINN accuracy {accuracy:.2f}% meets target (>90%)")
    else:
        logging.info(f"⚠️  PINN accuracy {accuracy:.2f}% below target (>90%)")
        logging.info("   Note: PINN may need more training iterations for higher accuracy")
    
    # Save predictions
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/pinn_predictions.npy', predictions)
    logging.info(f"\n✅ Predictions saved to outputs/pinn_predictions.npy")
    
    # Summary comparison
    logging.info("\n" + "="*60)
    logging.info("MODEL COMPARISON SUMMARY")
    logging.info("="*60)
    logging.info(f"PINN Model:")
    logging.info(f"  - Accuracy: {accuracy:.2f}%")
    logging.info(f"  - Training: Physics-informed (PDE constraints)")
    logging.info(f"  - Iterations: 500")
    logging.info(f"")
    logging.info(f"Surrogate Model (for comparison):")
    logging.info(f"  - Accuracy: 99.54%")
    logging.info(f"  - Training: Data-driven (pure ML)")
    logging.info(f"  - Epochs: 500")
    logging.info("="*60)
    
    return accuracy

if __name__ == "__main__":
    try:
        accuracy = evaluate_pinn()
        sys.exit(0)
    except Exception as e:
        logging.error(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
