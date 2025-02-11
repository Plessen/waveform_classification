import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a complex CNN model.")

    # Data arguments
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data HDF5 file.")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data HDF5 file.")
    parser.add_argument("--train_batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--val_batch_size", type=int, default=256, help="Batch size for validation.")
    parser.add_argument("--test_batch_size", type=int, required=True, help="Batch size for testing. Remembet to set to the same number of test signals per SNR!")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for data loading.")

    # Model arguments
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")

    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of training data to use for validation.")
    parser.add_argument("--patience", type=int, default=30, help="Patience for early stopping.")

    # Logging arguments
    parser.add_argument("--architecture", type=str, required=True, help="Architecture of the model (realcnn, complexcnn).")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for logging.")
    parser.add_argument("--version", type=int, default=0, help="Version of the model for logging.")
    
    # Other
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file.")
    parser.add_argument("--pretrained_model_name", type=str, default=None, help="Name of the pretrained model to use.")
    parser.add_argument("--freeze", action="store_true", default=False, help="Freeze the pretrained model.")
    
    return parser.parse_args()

