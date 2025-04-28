import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Union, Tuple
import wandb
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistillationTrainer:
    def __init__(
        self,
        teacher_model_name: str = "gpt2-medium",
        student_model_name: str = "gpt2",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        batch_size: int = 4,
        max_length: int = 128,
        alpha: float = 0.7,  # KL loss weight
        beta: float = 0.3,   # CE loss weight
        temperature: float = 2.0,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        output_dir: str = "distilled_model"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        
        # Initialize models and tokenizer
        logger.info("Loading models and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(student_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.teacher = GPT2LMHeadModel.from_pretrained(teacher_model_name).to(self.device)
        self.student = GPT2LMHeadModel.from_pretrained(student_model_name).to(self.device)
        self.teacher.eval()
        
        # Load and prepare dataset
        logger.info("Loading and preparing dataset...")
        self.dataset = load_dataset(dataset_name, dataset_config)
        
        def tokenize(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        
        self.tokenized_datasets = self.dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )
        self.tokenized_datasets.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"]
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            self.tokenized_datasets["train"],
            batch_size=batch_size,
            shuffle=True
        )
        
        self.optimizer = AdamW(self.student.parameters(), lr=learning_rate)
        
        # Initialize wandb
        wandb.init(
            project=f"gpt2-distillation-{self.num_epochs}",
            config={
                "teacher_model": teacher_model_name,
                "student_model": student_model_name,
                "batch_size": batch_size,
                "max_length": max_length,
                "alpha": alpha,
                "beta": beta,
                "temperature": temperature,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs
            }
        )
    
    def distillation_loss(self, student_logits, teacher_logits):
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
    
    def train(self):
        logger.info("Starting distillation training...")
        
        for epoch in range(self.num_epochs):
            self.student.train()
            total_loss = 0
            total_kl_loss = 0
            total_ce_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Teacher forward
                with torch.no_grad():
                    teacher_outputs = self.teacher(
                        input_ids,
                        attention_mask=attention_mask
                    )
                    teacher_logits = teacher_outputs.logits
                
                # Student forward
                student_outputs = self.student(
                    input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.logits
                
                # Calculate losses
                loss_kl = self.distillation_loss(student_logits, teacher_logits)
                loss_ce = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    input_ids.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                loss = self.alpha * loss_kl + self.beta * loss_ce
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Accumulate losses
                total_loss += loss.item()
                total_kl_loss += loss_kl.item()
                total_ce_loss += loss_ce.item()
            
            # Calculate average losses
            avg_loss = total_loss / len(self.train_loader)
            avg_kl_loss = total_kl_loss / len(self.train_loader)
            avg_ce_loss = total_ce_loss / len(self.train_loader)
            
            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_kl_loss": avg_kl_loss,
                "avg_ce_loss": avg_ce_loss
            })
            
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, "
                       f"Avg KL Loss: {avg_kl_loss:.4f}, "
                       f"Avg CE Loss: {avg_ce_loss:.4f}")
    
    def evaluate(self) -> Dict[str, float]:
        logger.info("Evaluating model...")
        self.student.eval()
        
        metrics = {
            "loss": [],
            "perplexity": [],
            "inference_time": []
        }
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                start_time = time.time()
                outputs = self.student(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                inference_time = time.time() - start_time
                
                loss = outputs.loss
                perplexity = torch.exp(loss)
                
                metrics["loss"].append(loss.item())
                metrics["perplexity"].append(perplexity.item())
                metrics["inference_time"].append(inference_time)
        
        # Calculate average metrics
        results = {
            "avg_loss": np.mean(metrics["loss"]),
            "avg_perplexity": np.mean(metrics["perplexity"]),
            "avg_inference_time": np.mean(metrics["inference_time"]),
            "std_loss": np.std(metrics["loss"]),
            "std_perplexity": np.std(metrics["perplexity"]),
            "std_inference_time": np.std(metrics["inference_time"])
        }
        
        # Log evaluation results
        wandb.log(results)
        
        logger.info("\nEvaluation Results:")
        logger.info(f"Average Loss: {results['avg_loss']:.4f} ± {results['std_loss']:.4f}")
        logger.info(f"Average Perplexity: {results['avg_perplexity']:.4f} ± {results['std_perplexity']:.4f}")
        logger.info(f"Average Inference Time: {results['avg_inference_time']:.4f} ± {results['std_inference_time']:.4f} seconds")
        
        return results
    
    def save_model(self):
        logger.info(f"Saving model to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.student.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training configuration
        config = {
            "teacher_model": "gpt2-medium",
            "student_model": "gpt2",
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "alpha": self.alpha,
            "beta": self.beta,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs
        }
        
        with open(os.path.join(self.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        logger.info("Model saved successfully!")

def main():
    trainer = DistillationTrainer()
    trainer.train()
    results = trainer.evaluate()
    trainer.save_model()
    wandb.finish()

if __name__ == "__main__":
    main() 