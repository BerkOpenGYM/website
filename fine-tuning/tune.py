#!/usr/bin/env python
"""
GPT-4o Fine-tuning Script for OpenGYM Project
This script prepares and manages fine-tuning jobs for GPT-4o using the OpenAI API.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class FineTuner:
    def __init__(self, dataset_path, model_name="gpt-4o", suffix=None):
        """
        Initialize the fine-tuning process
        
        Args:
            dataset_path (str): Path to the dataset file in JSONL format
            model_name (str): Base model to fine-tune
            suffix (str, optional): Custom suffix for the fine-tuned model
        """
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.suffix = suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a directory for logs if it doesn't exist
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Ensure the dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
        # Validate the dataset format
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate that the dataset is in the correct JSONL format for OpenAI fine-tuning"""
        print(f"Validating dataset: {self.dataset_path}")
        
        try:
            # Count valid examples and check format
            valid_examples = 0
            total_examples = 0
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    total_examples += 1
                    try:
                        example = json.loads(line)
                        
                        # Check required fields for chat fine-tuning
                        if "messages" not in example:
                            print(f"Error in example {total_examples}: Missing 'messages' field")
                            continue
                            
                        messages = example["messages"]
                        
                        # Check if messages is a list
                        if not isinstance(messages, list) or len(messages) < 1:
                            print(f"Error in example {total_examples}: 'messages' must be a non-empty list")
                            continue
                            
                        # Check if each message has 'role' and 'content'
                        valid_message = True
                        for message in messages:
                            if not isinstance(message, dict):
                                print(f"Error in example {total_examples}: Each message must be a dictionary")
                                valid_message = False
                                break
                                
                            if "role" not in message or "content" not in message:
                                print(f"Error in example {total_examples}: Each message must have 'role' and 'content' fields")
                                valid_message = False
                                break
                                
                            # Check if role is valid
                            if message["role"] not in ["system", "user", "assistant"]:
                                print(f"Error in example {total_examples}: Invalid role '{message['role']}'. Must be 'system', 'user', or 'assistant'")
                                valid_message = False
                                break
                        
                        if valid_message:
                            valid_examples += 1
                            
                    except json.JSONDecodeError:
                        print(f"Error in example {total_examples}: Invalid JSON format")
            
            print(f"Dataset validation complete: {valid_examples}/{total_examples} valid examples")
            
            if valid_examples == 0:
                raise ValueError("No valid examples found in dataset")
            
            if valid_examples < total_examples:
                print(f"Warning: {total_examples - valid_examples} examples are invalid")
                
            # Calculate and print token estimate
            self._estimate_tokens()
                
        except Exception as e:
            print(f"Dataset validation failed: {str(e)}")
            raise
    
    def _estimate_tokens(self):
        """Estimate the number of tokens in the dataset"""
        try:
            # Simple estimation based on words (rough approximation)
            total_words = 0
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    for message in example["messages"]:
                        total_words += len(message["content"].split())
            
            # Rough estimate: average of 1.3 tokens per word
            estimated_tokens = int(total_words * 1.3)
            print(f"Estimated tokens in dataset: ~{estimated_tokens:,}")
            
            # Calculate potential cost (very rough estimate)
            # For gpt-4o, fine-tuning costs are subject to change
            estimated_cost_usd = estimated_tokens * 0.00015  # Example rate, actual may differ
            print(f"Estimated fine-tuning cost: ~${estimated_cost_usd:.2f} USD (approximate)")
            
        except Exception as e:
            print(f"Failed to estimate tokens: {str(e)}")
    
    def upload_file(self):
        """Upload the dataset file to OpenAI"""
        print(f"Uploading dataset to OpenAI: {self.dataset_path}")
        
        try:
            with open(self.dataset_path, "rb") as file:
                response = client.files.create(
                    file=file,
                    purpose="fine-tune"
                )
            
            self.file_id = response.id
            print(f"File uploaded successfully with ID: {self.file_id}")
            return self.file_id
            
        except Exception as e:
            print(f"File upload failed: {str(e)}")
            raise
    
    def create_fine_tuning_job(self):
        """Create a fine-tuning job"""
        print(f"Creating fine-tuning job for model {self.model_name}...")
        
        try:
            response = client.fine_tuning.jobs.create(
                training_file=self.file_id,
                model=self.model_name,
                suffix=self.suffix
            )
            
            self.job_id = response.id
            print(f"Fine-tuning job created with ID: {self.job_id}")
            
            # Log the job details
            job_log = {
                "job_id": self.job_id,
                "model": self.model_name,
                "file_id": self.file_id,
                "created_at": datetime.now().isoformat(),
                "suffix": self.suffix
            }
            
            with open(self.logs_dir / f"job_{self.job_id}.json", "w") as f:
                json.dump(job_log, f, indent=2)
            
            return self.job_id
            
        except Exception as e:
            print(f"Fine-tuning job creation failed: {str(e)}")
            raise
    
    def monitor_job_progress(self, interval=30):
        """Monitor the progress of the fine-tuning job"""
        print(f"Monitoring fine-tuning job {self.job_id}...")
        
        try:
            status = "validating_files"
            
            while status not in ["succeeded", "failed", "cancelled"]:
                response = client.fine_tuning.jobs.retrieve(fine_tuning_job_id=self.job_id)
                
                status = response.status
                print(f"Job status: {status}")
                
                if hasattr(response, 'fine_tuned_model') and response.fine_tuned_model:
                    print(f"Fine-tuned model ID: {response.fine_tuned_model}")
                
                # Print training metrics if available
                if hasattr(response, 'training_metrics') and response.training_metrics:
                    metrics = response.training_metrics
                    print(f"Training loss: {metrics.training_loss}")
                    print(f"Validation loss: {metrics.validation_loss}")
                    print(f"Epoch: {metrics.training_epoch}")
                
                if status in ["succeeded", "failed", "cancelled"]:
                    break
                    
                print(f"Checking again in {interval} seconds...")
                time.sleep(interval)
            
            # Log final job status
            job_log = {
                "job_id": self.job_id,
                "model": self.model_name,
                "file_id": self.file_id,
                "status": status,
                "completed_at": datetime.now().isoformat(),
                "fine_tuned_model": getattr(response, 'fine_tuned_model', None)
            }
            
            with open(self.logs_dir / f"job_{self.job_id}_completed.json", "w") as f:
                json.dump(job_log, f, indent=2)
            
            if status == "succeeded":
                print(f"Fine-tuning completed successfully! Model ID: {response.fine_tuned_model}")
                return response.fine_tuned_model
            else:
                print(f"Fine-tuning job completed with status: {status}")
                return None
                
        except Exception as e:
            print(f"Job monitoring failed: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        try:
            # 1. Upload dataset
            self.upload_file()
            
            # 2. Create fine-tuning job
            self.create_fine_tuning_job()
            
            # 3. Monitor progress
            fine_tuned_model = self.monitor_job_progress()
            
            if fine_tuned_model:
                print(f"\nFine-tuning complete! Your model is ready to use.")
                print(f"Model ID: {fine_tuned_model}")
                print("\nExample usage:")
                print(f"""
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="{fine_tuned_model}",
    messages=[
        {{"role": "system", "content": "You are an expert fitness assistant for OpenGYM."}},
        {{"role": "user", "content": "Can you suggest a workout routine for beginners?"}}
    ]
)
print(response.choices[0].message.content)
                """)
            else:
                print("\nFine-tuning was not successful. Please check the logs.")
                
            return fine_tuned_model
                
        except Exception as e:
            print(f"Fine-tuning pipeline failed: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-4o for OpenGYM')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset file (JSONL format)')
    parser.add_argument('--convert-csv', action='store_true', help='Convert CSV to JSONL format')
    parser.add_argument('--model', type=str, default='gpt-4o', help='Base model to fine-tune')
    parser.add_argument('--suffix', type=str, help='Custom suffix for the fine-tuned model')
    
    args = parser.parse_args()
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables or .env file")
        print("Please set your OpenAI API key before running this script.")
        sys.exit(1)
    
    # Create and run the fine-tuning pipeline
    fine_tuner = FineTuner(
        dataset_path=args.dataset,
        model_name=args.model,
        suffix=args.suffix
    )
    
    fine_tuner.run_pipeline()

if __name__ == "__main__":
    main()