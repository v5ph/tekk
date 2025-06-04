import openai
from typing import Dict, Any, List
import json
import pandas as pd
from app.config import settings
from app.auth.models import SubscriptionPlan
from app.models.project import TaskType

class AIAnalyzer:
    """Service for AI-powered analysis using OpenAI GPT-4"""
    
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def analyze_prompt(self, prompt: str, columns: List[str], sample_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user prompt to determine ML task type and target column"""
        
        # Prepare sample data for context
        sample_str = sample_data.head(3).to_string()
        columns_str = ", ".join(columns)
        
        system_prompt = """You are an expert data scientist. Analyze the user's request and dataset to determine:
1. The type of machine learning task (classification, regression, clustering, forecasting, anomaly_detection)
2. The target column (if applicable)
3. Brief reasoning

Respond with valid JSON only:
{
    "task_type": "classification|regression|clustering|forecasting|anomaly_detection",
    "target_column": "column_name or null",
    "reasoning": "brief explanation"
}

Guidelines:
- Classification: predicting categories/labels
- Regression: predicting continuous numbers
- Clustering: grouping similar data points
- Forecasting: predicting future values in time series
- Anomaly Detection: finding outliers

If the prompt mentions "predict", "forecast", "classify", "cluster", use that as a strong signal.
"""
        
        user_prompt = f"""
User Request: "{prompt}"

Dataset Columns: {columns_str}

Sample Data:
{sample_str}

Analyze this request and determine the ML task type and target column.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate task type
            valid_tasks = [task.value for task in TaskType]
            if result["task_type"] not in valid_tasks:
                result["task_type"] = TaskType.CLASSIFICATION.value
            
            # Validate target column
            if result["target_column"] and result["target_column"] not in columns:
                result["target_column"] = columns[0] if columns else None
            
            return result
            
        except Exception as e:
            # Fallback analysis
            return self._fallback_analysis(prompt, columns)
    
    def _fallback_analysis(self, prompt: str, columns: List[str]) -> Dict[str, Any]:
        """Fallback analysis when OpenAI fails"""
        prompt_lower = prompt.lower()
        
        # Simple keyword-based analysis
        if any(word in prompt_lower for word in ['classify', 'classification', 'category', 'class']):
            task_type = TaskType.CLASSIFICATION.value
        elif any(word in prompt_lower for word in ['predict', 'regression', 'price', 'value', 'amount']):
            task_type = TaskType.REGRESSION.value
        elif any(word in prompt_lower for word in ['cluster', 'group', 'segment']):
            task_type = TaskType.CLUSTERING.value
        elif any(word in prompt_lower for word in ['forecast', 'time series', 'future', 'trend']):
            task_type = TaskType.FORECASTING.value
        else:
            task_type = TaskType.CLASSIFICATION.value
        
        # Try to find target column from prompt
        target_column = None
        for col in columns:
            if col.lower() in prompt_lower:
                target_column = col
                break
        
        if not target_column and columns:
            target_column = columns[-1]  # Default to last column
        
        return {
            "task_type": task_type,
            "target_column": target_column,
            "reasoning": "Fallback analysis based on keywords"
        }
    
    async def generate_summary(self, ml_results: Dict[str, Any], subscription_plan: SubscriptionPlan) -> str:
        """Generate AI-powered summary of ML results"""
        
        # Determine summary length based on subscription
        if subscription_plan == SubscriptionPlan.FREE:
            max_tokens = 200
            detail_level = "brief"
        elif subscription_plan == SubscriptionPlan.PLUS:
            max_tokens = 500
            detail_level = "detailed"
        else:  # PRO
            max_tokens = 800
            detail_level = "comprehensive"
        
        system_prompt = f"""You are an expert data scientist writing a {detail_level} analysis report. 
Create a clear, professional summary of the machine learning results for business users.

Focus on:
- Key insights and findings
- Model performance in business terms
- Actionable recommendations
- Important patterns or trends

Write in a professional but accessible tone. Avoid technical jargon.
"""
        
        # Prepare results summary
        task_type = ml_results.get('task_type', 'Unknown')
        model_name = ml_results.get('model_name', 'Unknown')
        metrics = ml_results.get('metrics', {})
        target_column = ml_results.get('target_column', 'target variable')
        
        user_prompt = f"""
Analyze these machine learning results:

Task Type: {task_type}
Model: {model_name}
Target Variable: {target_column}
Performance Metrics: {json.dumps(metrics, indent=2)}

Feature Importance: {json.dumps(ml_results.get('feature_importance', {}), indent=2)}

Generate a {detail_level} business-focused summary of these results.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            summary = response.choices[0].message.content
            
            # Add watermark for free plan
            if subscription_plan == SubscriptionPlan.FREE:
                summary += "\n\n*Generated with Tekk Free Plan - Upgrade for detailed reports*"
            
            return summary
            
        except Exception as e:
            return self._generate_fallback_summary(ml_results, subscription_plan)
    
    def _generate_fallback_summary(self, ml_results: Dict[str, Any], subscription_plan: SubscriptionPlan) -> str:
        """Generate fallback summary when OpenAI fails"""
        task_type = ml_results.get('task_type', 'analysis')
        model_name = ml_results.get('model_name', 'machine learning model')
        metrics = ml_results.get('metrics', {})
        
        summary = f"## {task_type.title()} Analysis Results\n\n"
        summary += f"Successfully trained a {model_name} model for your dataset.\n\n"
        
        if metrics:
            summary += "**Performance Metrics:**\n"
            for metric, value in metrics.items():
                if isinstance(value, float):
                    summary += f"- {metric}: {value:.3f}\n"
                else:
                    summary += f"- {metric}: {value}\n"
        
        summary += "\nThe model has been trained and is ready for predictions."
        
        if subscription_plan == SubscriptionPlan.FREE:
            summary += "\n\n*Upgrade to Tekk Plus for detailed AI-generated insights and recommendations.*"
        
        return summary
    
    async def generate_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate data insights using AI"""
        
        # Basic data statistics
        stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_data': df.isnull().sum().sum(),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        system_prompt = """You are a data analyst. Generate 3-5 key insights about this dataset and analysis.
Focus on practical, actionable insights that would be valuable for business decisions.
Return as a JSON array of strings."""
        
        user_prompt = f"""
Dataset Statistics: {json.dumps(stats)}
Analysis Results: {json.dumps(analysis_results, default=str)}

Generate key insights about this data and analysis.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=400
            )
            
            insights = json.loads(response.choices[0].message.content)
            return insights if isinstance(insights, list) else [insights]
            
        except Exception:
            return [
                f"Dataset contains {stats['rows']} rows and {stats['columns']} columns",
                f"Found {stats['missing_data']} missing values that were handled during preprocessing",
                f"Analysis identified key patterns in the {analysis_results.get('target_column', 'target')} variable",
                "Model performance indicates good predictive capability for this dataset"
            ] 