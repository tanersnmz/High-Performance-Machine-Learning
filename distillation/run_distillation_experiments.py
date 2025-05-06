import torch
import torch.nn.functional as F
from distillation.distillation_eval import DistillationTrainer
import wandb
import json
import os
from typing import Dict, List
import logging
import shutil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressiveDistillationTrainer(DistillationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_epochs = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        self.results = {}
    
    def save_checkpoint(self, epoch: int, results: Dict):
        """Save model checkpoint and results at specific epochs"""
        checkpoint_dir = os.path.join(self.output_dir, f"epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.student.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
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
            "current_epoch": epoch
        }
        
        with open(os.path.join(checkpoint_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        # Save evaluation results
        with open(os.path.join(checkpoint_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        self.results[f"epoch_{epoch}"] = results
        logger.info(f"Checkpoint saved at epoch {epoch}")
    
    def train(self):
        logger.info("Starting progressive distillation training...")
        
        for epoch in range(self.num_epochs):
            self.student.train()
            total_loss = 0
            total_kl_loss = 0
            total_ce_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in progress_bar:
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
            
            # Save checkpoint if at target epoch
            if (epoch + 1) in self.checkpoint_epochs:
                results = self.evaluate()
                self.save_checkpoint(epoch + 1, results)
    
    def print_comparison(self):
        """Print comparison of results at different checkpoints"""
        logger.info("\nProgressive Training Results Comparison:")
        logger.info("=" * 80)
        logger.info(f"{'Epochs':<10} {'Avg Loss':<15} {'Avg Perplexity':<20} {'Avg Inference Time (s)':<25}")
        logger.info("-" * 80)
        
        for epoch in self.checkpoint_epochs:
            results = self.results[f"epoch_{epoch}"]
            logger.info(
                f"{epoch:<10} "
                f"{results['avg_loss']:.4f} ± {results['std_loss']:.4f} "
                f"{results['avg_perplexity']:.4f} ± {results['std_perplexity']:.4f} "
                f"{results['avg_inference_time']:.4f} ± {results['std_inference_time']:.4f}"
            )

def main():
    # Initialize wandb
    wandb.init(
        project="gpt2-progressive-distillation",
        config={
            "teacher_model": "gpt2-medium",
            "student_model": "gpt2",
            "checkpoint_epochs": [3, 5, 10]
        }
    )
    
    # Initialize trainer
    trainer = ProgressiveDistillationTrainer(
        num_epochs=50,  # Train for 10 epochs total
        output_dir="progressive_distillation"
    )
    
    # Run training with checkpoints
    trainer.train()
    
    # Print comparison of results
    trainer.print_comparison()
    
    # Save final results
    with open(os.path.join(trainer.output_dir, "all_results.json"), "w") as f:
        json.dump(trainer.results, f, indent=4)
    
    wandb.finish()

if __name__ == "__main__":
    main() 