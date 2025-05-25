import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, data_path, max_samples=30000, random_seed=42):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.max_samples = max_samples
        self.random_seed = random_seed
        
    def load_data(self):
        """Load the KCC dataset with size limitation"""
        print("Loading dataset...")
        # Read the first row to get column names
        df_sample = pd.read_csv(self.data_path, nrows=1)
        total_rows = sum(1 for _ in open(self.data_path)) - 1  # Subtract header row
        
        print(f"Total rows in dataset: {total_rows}")
        print(f"Loading {min(self.max_samples, total_rows):,} samples...")
        
        if total_rows > self.max_samples:
            # Calculate the number of rows to skip
            skip_rows = sorted(np.random.RandomState(self.random_seed).choice(
                range(1, total_rows + 1), 
                total_rows - self.max_samples, 
                replace=False
            ))
            self.df = pd.read_csv(self.data_path, skiprows=skip_rows)
        else:
            self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset loaded with shape: {self.df.shape}")
        return self.df
    
    def clean_data(self):
        """Clean the dataset by removing nulls and duplicates"""
        print("Cleaning dataset...")
        initial_shape = self.df.shape
        
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)
        after_duplicates = self.df.shape
        
        # Remove rows with null values
        self.df.dropna(inplace=True)
        after_nulls = self.df.shape
        
        # Print cleaning statistics
        print(f"Initial shape: {initial_shape}")
        print(f"After removing duplicates: {after_duplicates}")
        print(f"After removing nulls: {after_nulls}")
        print(f"Rows removed: {initial_shape[0] - after_nulls[0]}")
        
        return self.df
    
    def prepare_features(self):
        """Prepare features for model training"""
        print("Preparing features...")
        # Encode categorical variables if any
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.df[col] = self.label_encoder.fit_transform(self.df[col])
            
        return self.df
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """Split data into train, validation and test sets"""
        print("Splitting dataset...")
        # First split into train and test
        train_data, test_data = train_test_split(self.df, test_size=test_size, random_state=self.random_seed)
        
        # Then split train into train and validation
        val_size_adjusted = val_size / (1 - test_size)
        train_data, val_data = train_test_split(train_data, test_size=val_size_adjusted, random_state=self.random_seed)
        
        print(f"Train set shape: {train_data.shape}")
        print(f"Validation set shape: {val_data.shape}")
        print(f"Test set shape: {test_data.shape}")
        
        return train_data, val_data, test_data
    
    def process_pipeline(self):
        """Run the complete preprocessing pipeline"""
        self.load_data()
        self.clean_data()
        self.prepare_features()
        return self.split_data() 