"""
Complete Pipeline Validation Script
Validates that the entire PINN + Surrogate system uses ONLY real data
"""
import os
import sys
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_dataset():
    """Validate that dataset.parquet exists and contains real, meaningful data"""
    logging.info("="*60)
    logging.info("STEP 1: VALIDATING DATASET")
    logging.info("="*60)
    
    dataset_path = "data/dataset.parquet"
    
    if not os.path.exists(dataset_path):
        logging.error(f"❌ Dataset not found at {dataset_path}")
        logging.info("Run: python data/prepare_materials_data.py")
        return False
    
    df = pd.read_parquet(dataset_path)
    
    logging.info(f"✅ Dataset loaded: {df.shape}")
    logging.info(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    required_cols = ['time', 'x', 'temperature', 'survival_rate']
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"❌ Missing required column: {col}")
            return False
    
    logging.info("✅ All required columns present")
    
    # Check for NaN values
    if df.isnull().sum().sum() > 0:
        logging.error(f"❌ Dataset contains NaN values:\n{df.isnull().sum()}")
        return False
    
    logging.info("✅ No NaN values")
    
    # Check dataset size
    if len(df) < 200:
        logging.error(f"❌ Dataset too small: {len(df)} rows (need >200)")
        return False
    
    logging.info(f"✅ Dataset size adequate: {len(df)} rows")
    
    # Check survival_rate variance
    survival_std = df['survival_rate'].std()
    if survival_std < 0.01:
        logging.error(f"❌ survival_rate variance too low: {survival_std}")
        return False
    
    logging.info(f"✅ survival_rate has good variance: std={survival_std:.4f}")
    
    # Check correlations
    corr = df.corr()
    survival_corr = corr['survival_rate'].drop('survival_rate')
    max_corr = survival_corr.abs().max()
    
    if max_corr < 0.1:
        logging.error(f"❌ survival_rate not correlated with inputs: max_corr={max_corr:.4f}")
        return False
    
    logging.info(f"✅ survival_rate correlates with inputs: max_corr={max_corr:.4f}")
    
    # Display statistics
    logging.info("\nDataset Statistics:")
    logging.info(f"\n{df.describe()}")
    
    logging.info("\nCorrelation Matrix:")
    logging.info(f"\n{corr}")
    
    logging.info("\n✅ DATASET VALIDATION PASSED")
    return True

def check_for_dummy_data():
    """Search for any remaining dummy data usage in the codebase"""
    logging.info("="*60)
    logging.info("STEP 2: CHECKING FOR DUMMY DATA USAGE")
    logging.info("="*60)
    
    files_to_check = [
        'pinn_system/train.py',
        'pinn_system/surrogate/surrogate_trainer.py',
        'pinn_system/data_loader.py',
        'data/prepare_materials_data.py'
    ]
    
    issues_found = []
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            logging.warning(f"⚠️  File not found: {filepath}")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for random data generation
        if 'np.random.rand' in content or 'np.random.uniform' in content:
            # Check if it's in a comment or actual code
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if ('np.random.rand' in line or 'np.random.uniform' in line) and not line.strip().startswith('#'):
                    issues_found.append(f"{filepath}:{i} - {line.strip()}")
    
    if issues_found:
        logging.error("❌ Found potential dummy data usage:")
        for issue in issues_found:
            logging.error(f"   {issue}")
        return False
    
    logging.info("✅ No dummy data usage found in key files")
    return True

def validate_data_loader():
    """Test that data_loader.py works correctly"""
    logging.info("="*60)
    logging.info("STEP 3: VALIDATING DATA LOADER")
    logging.info("="*60)
    
    try:
        from pinn_system.data_loader import load_data
        
        X, y = load_data('data/dataset.parquet', ['time', 'x', 'temperature', 'survival_rate'])
        
        logging.info(f"✅ Data loader works: X={X.shape}, y={y.shape}")
        
        # Validate loaded data
        assert X.shape[0] > 0, "X is empty"
        assert y.shape[0] > 0, "y is empty"
        assert not np.isnan(X).any(), "X contains NaN"
        assert not np.isnan(y).any(), "y contains NaN"
        assert y.std() > 0.01, f"y variance too low: {y.std()}"
        
        logging.info("✅ Loaded data passes validation")
        return True
        
    except Exception as e:
        logging.error(f"❌ Data loader validation failed: {e}")
        return False

def validate_surrogate_trainer():
    """Test that surrogate trainer can be imported and has correct structure"""
    logging.info("="*60)
    logging.info("STEP 4: VALIDATING SURROGATE TRAINER")
    logging.info("="*60)
    
    try:
        from pinn_system.surrogate.surrogate_trainer import train_surrogate, SurrogateMLP
        
        logging.info("✅ Surrogate trainer imports successfully")
        
        # Check that main function doesn't use dummy data
        import inspect
        source = inspect.getsource(train_surrogate)
        
        if 'np.random.rand' in source or 'np.random.uniform' in source:
            logging.error("❌ train_surrogate function contains random data generation")
            return False
        
        logging.info("✅ train_surrogate function clean (no dummy data)")
        return True
        
    except Exception as e:
        logging.error(f"❌ Surrogate trainer validation failed: {e}")
        return False

def main():
    """Run all validation checks"""
    logging.info("\n" + "="*60)
    logging.info("PHYSICS-INFORMED NEURAL NETWORK PIPELINE VALIDATION")
    logging.info("="*60 + "\n")
    
    results = {
        'Dataset Validation': validate_dataset(),
        'Dummy Data Check': check_for_dummy_data(),
        'Data Loader': validate_data_loader(),
        'Surrogate Trainer': validate_surrogate_trainer()
    }
    
    logging.info("\n" + "="*60)
    logging.info("VALIDATION SUMMARY")
    logging.info("="*60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logging.info(f"{check}: {status}")
        if not passed:
            all_passed = False
    
    logging.info("="*60)
    
    if all_passed:
        logging.info("\n🎉 ALL VALIDATION CHECKS PASSED!")
        logging.info("\nNext steps:")
        logging.info("1. Train PINN: python pinn_system/train.py --epochs 1000")
        logging.info("2. Train Surrogate: python pinn_system/surrogate/surrogate_trainer.py --epochs 1000")
        return 0
    else:
        logging.error("\n❌ SOME VALIDATION CHECKS FAILED")
        logging.error("Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
