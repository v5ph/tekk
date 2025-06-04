import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
from app.auth.models import SubscriptionPlan
from app.config import settings
from app.models.project import DatasetPreview
import os

class DataProcessor:
    """Service for processing and analyzing uploaded datasets"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json', '.txt']
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load a file into a pandas DataFrame"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file with any supported encoding")
                    
            elif file_ext == '.xlsx':
                df = pd.read_excel(file_path)
                
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Try to flatten nested structure
                    df = pd.json_normalize(data)
                else:
                    raise ValueError("Unsupported JSON structure")
                    
            elif file_ext == '.txt':
                # Assume tab-separated or comma-separated
                try:
                    df = pd.read_csv(file_path, sep='\t')
                except:
                    df = pd.read_csv(file_path, sep=',')
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Basic data cleaning
            df = self._clean_dataframe(df)
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading file: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations"""
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = df.columns.astype(str)
        df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def create_preview(self, df: pd.DataFrame) -> DatasetPreview:
        """Create a preview of the dataset"""
        # Get sample rows (first 5)
        sample_rows = df.head(5).fillna("").to_dict('records')
        
        # Convert numpy types to Python types for JSON serialization
        for row in sample_rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    row[key] = value.item()
                elif isinstance(value, np.bool_):
                    row[key] = bool(value)
                else:
                    row[key] = str(value)
        
        return DatasetPreview(
            columns=df.columns.tolist(),
            sample_rows=sample_rows,
            total_rows=len(df),
            file_size=df.memory_usage(deep=True).sum()
        )
    
    def get_row_limit(self, subscription_plan: SubscriptionPlan) -> int:
        """Get row limit based on subscription plan"""
        if subscription_plan == SubscriptionPlan.FREE:
            return settings.FREE_PLAN_ROWS_LIMIT
        elif subscription_plan == SubscriptionPlan.PLUS:
            return settings.PLUS_PLAN_ROWS_LIMIT
        elif subscription_plan == SubscriptionPlan.PRO:
            return settings.PRO_PLAN_ROWS_LIMIT
        return settings.FREE_PLAN_ROWS_LIMIT
    
    def analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column types and characteristics"""
        analysis = {}
        
        for column in df.columns:
            col_data = df[column]
            
            # Basic statistics
            analysis[column] = {
                'dtype': str(col_data.dtype),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'is_numeric': pd.api.types.is_numeric_dtype(col_data),
                'is_categorical': pd.api.types.is_categorical_dtype(col_data) or col_data.nunique() < 20,
                'is_datetime': pd.api.types.is_datetime64_any_dtype(col_data)
            }
            
            # Additional statistics for numeric columns
            if analysis[column]['is_numeric']:
                analysis[column].update({
                    'mean': col_data.mean() if not col_data.empty else None,
                    'std': col_data.std() if not col_data.empty else None,
                    'min': col_data.min() if not col_data.empty else None,
                    'max': col_data.max() if not col_data.empty else None
                })
        
        return analysis
    
    def suggest_target_columns(self, df: pd.DataFrame) -> List[str]:
        """Suggest potential target columns for ML tasks"""
        suggestions = []
        
        for column in df.columns:
            col_data = df[column]
            
            # Skip columns with too many nulls
            if col_data.isnull().sum() / len(col_data) > 0.5:
                continue
            
            # Numeric columns with reasonable range
            if pd.api.types.is_numeric_dtype(col_data):
                unique_ratio = col_data.nunique() / len(col_data)
                if 0.01 < unique_ratio < 0.95:  # Not too few, not too many unique values
                    suggestions.append(column)
            
            # Categorical columns with reasonable categories
            elif col_data.nunique() < 20 and col_data.nunique() > 1:
                suggestions.append(column)
        
        return suggestions[:5]  # Return top 5 suggestions 