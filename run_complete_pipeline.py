#!/usr/bin/env python
"""
Complete Pipeline Execution Script
Runs the entire PINN + Surrogate pipeline from start to finish
"""
import os
import sys
import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(cmd, description):
    """Run a command and handle errors"""
    logging.info("="*60)
    logging.info(f"RUNNING: {description}")
    logging.info(f"Command: {cmd}")
    logging.info("="*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        logging.info(f"✅ {description} completed in {elapsed:.2f}s")
        
        if result.stdout:
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logging.error(f"❌ {description} failed after {elapsed:.2f}s")
        logging.error(f"Error: {e}")
        
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        
        return False

def main():
    """Run complete pipeline"""
    logging.info("\n" + "="*60)
    logging.info("PHYSICS-INFORMED NEURAL NETWORK - COMPLETE PIPELINE")
    logging.info("="*60 + "\n")
    
    pipeline_start = time.time()
    
    # Step 1: Validate pipeline
    logging.info("STEP 1: Validating Pipeline")
    if not run_command("python validate_full_pipeline.py", "Pipeline Validation"):
        logging.error("❌ Validation failed. Please fix issues before proceeding.")
        return 1
    
    # Step 2: Check if dataset needs regeneration
    if not os.path.exists("data/dataset.parquet"):
        logging.info("\nSTEP 2: Generating Dataset (not found)")
        if not run_command("python data/prepare_materials_data.py", "Dataset Generation"):
            logging.error("❌ Dataset generation failed.")
            return 1
    else:
        logging.info("\nSTEP 2: Dataset exists, skipping generation")
    
    # Step 3: Train surrogate model
    logging.info("\nSTEP 3: Training Surrogate Model")
    if not run_command(
        "python pinn_system/surrogate/surrogate_trainer.py --epochs 500",
        "Surrogate Model Training"
    ):
        logging.error("❌ Surrogate training failed.")
        return 1
    
    # Step 4: Optional - Train PINN (commented out by default as it takes longer)
    train_pinn = input("\nDo you want to train the PINN model? (takes 5-10 min) [y/N]: ").lower().strip()
    
    if train_pinn == 'y':
        logging.info("\nSTEP 4: Training PINN Model")
        if not run_command(
            "python pinn_system/train.py --epochs 1000",
            "PINN Model Training"
        ):
            logging.error("❌ PINN training failed.")
            return 1
    else:
        logging.info("\nSTEP 4: Skipping PINN training (user choice)")
    
    # Summary
    total_time = time.time() - pipeline_start
    
    logging.info("\n" + "="*60)
    logging.info("PIPELINE EXECUTION SUMMARY")
    logging.info("="*60)
    logging.info(f"Total Time: {total_time:.2f}s")
    logging.info("")
    logging.info("✅ Validation: PASSED")
    logging.info("✅ Dataset: READY")
    logging.info("✅ Surrogate Model: TRAINED")
    
    if train_pinn == 'y':
        logging.info("✅ PINN Model: TRAINED")
    else:
        logging.info("⏭️  PINN Model: SKIPPED")
    
    logging.info("")
    logging.info("📁 Output Files:")
    
    if os.path.exists("data/dataset.parquet"):
        logging.info("   ✅ data/dataset.parquet")
    
    if os.path.exists("pinn_system/surrogate/surrogate_model.pth"):
        logging.info("   ✅ pinn_system/surrogate/surrogate_model.pth")
    
    if os.path.exists("outputs/predictions.npy"):
        logging.info("   ✅ outputs/predictions.npy")
    
    logging.info("")
    logging.info("="*60)
    logging.info("🎉 PIPELINE EXECUTION COMPLETE!")
    logging.info("="*60)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warning("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
