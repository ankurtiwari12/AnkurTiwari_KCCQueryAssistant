import os
import sys
from pathlib import Path
import pickle
import json
import pandas as pd

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.append(str(src_dir))

from data_preprocessing import DataPreprocessor
from model import OllamaModel
import uvicorn
from api import app

# Constants
DATA_PATH = "kcc_dataset.csv"
PROCESSED_DATA_PATH = "processed_data.pkl"
VECTOR_STORE_PATH = "vector_store.pkl"
METADATA_PATH = "metadata.json"
MAX_SAMPLES = 30000
RANDOM_SEED = 42

def save_processed_data(df, metadata):
    """Save processed dataframe and metadata"""
    print("Saving processed data...")
    df.to_pickle(PROCESSED_DATA_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f)
    print(f"Data saved to {PROCESSED_DATA_PATH} and {METADATA_PATH}")

def load_processed_data():
    """Load processed dataframe if exists"""
    if os.path.exists(PROCESSED_DATA_PATH) and os.path.exists(METADATA_PATH):
        print("Loading preprocessed data...")
        df = pd.read_pickle(PROCESSED_DATA_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        # Check if the saved data matches our current settings
        if metadata.get('max_samples') != MAX_SAMPLES:
            print(f"Saved data has different sample size ({metadata.get('max_samples')})")
            print(f"Reprocessing with {MAX_SAMPLES} samples...")
            return None, None
        return df, metadata
    return None, None

def select_text_columns(df):
    """Helper function to identify and select appropriate text columns"""
    # Print column information
    print("\nDataset columns and sample values:")
    for col in df.columns:
        sample_value = df[col].iloc[0] if len(df) > 0 else None
        dtype = df[col].dtype
        print(f"\nColumn: {col}")
        print(f"Type: {dtype}")
        print(f"Sample value: {sample_value}")
    
    # Try to automatically identify text columns
    text_columns = []
    for col in df.columns:
        # Check if column contains string data
        if df[col].dtype == 'object':
            # Check if the column has substantial text content
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(sample, str) and len(sample.split()) > 3:  # More than 3 words
                text_columns.append(col)
    
    if text_columns:
        print(f"\nAutomatically identified text columns: {text_columns}")
    else:
        print("\nNo text columns automatically identified.")
        # If no text columns found, ask for manual input
        print("\nPlease enter the column names you want to use (comma-separated):")
        user_input = input().strip()
        text_columns = [col.strip() for col in user_input.split(',')]
    
    return text_columns

def main():
    # Check for preprocessed data
    df, metadata = load_processed_data()
    
    if df is None:
        # Initialize data preprocessor
        print("Preprocessed data not found. Processing data...")
        preprocessor = DataPreprocessor(DATA_PATH, max_samples=MAX_SAMPLES, random_seed=RANDOM_SEED)
        
        # Load and process the data
        print("Loading and preprocessing data...")
        df = preprocessor.load_data()
        
        # Clean the data
        df = preprocessor.clean_data()
        
        # Select text columns
        text_columns = select_text_columns(df)
        
        # Save processed data and metadata
        metadata = {
            "columns": df.columns.tolist(),
            "text_columns": text_columns,
            "shape": df.shape,
            "max_samples": MAX_SAMPLES,
            "random_seed": RANDOM_SEED,
            "preprocessing_date": str(pd.Timestamp.now())
        }
        save_processed_data(df, metadata)
    else:
        print("Using preprocessed data from:", metadata["preprocessing_date"])
        print("Dataset shape:", metadata["shape"])
        print(f"Number of samples: {metadata['max_samples']:,}")
        text_columns = metadata.get("text_columns", select_text_columns(df))
    
    # Initialize the model
    print("\nInitializing model...")
    model = OllamaModel(vector_store_path=VECTOR_STORE_PATH)
    
    # Initialize RAG with the dataset
    print("Initializing RAG system...")
    print(f"Using text columns: {text_columns}")
    
    try:
        model.initialize_rag(df, text_columns)
    except ValueError as e:
        print(f"\nError: {e}")
        print("Available columns:", df.columns.tolist())
        print("\nPlease update the text_columns list with correct column names.")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return
    
    # Check if model is available
    model_info = model.get_model_info()
    if model_info:
        print(f"Model loaded successfully: {model_info}")
    else:
        print("Warning: Could not get model information. Make sure Ollama is running.")
    
    # Start the API server
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 