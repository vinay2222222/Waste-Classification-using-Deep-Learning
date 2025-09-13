"""
train.py - RWC-Net Model Training Module
=======================================

This module handles training, validation, and optimization of the RWC-Net model
for recyclable waste classification. Includes features for:

- Training with auxiliary loss supervision
- Learning rate scheduling and optimization
- Early stopping and model checkpointing
- Cross-validation training
- Training metrics and visualization
- Hyperparameter optimization support

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import os
from collections import defaultdict
import json
from datetime import datetime

# Import custom modules
from dataset import DataLoaderFactory, DatasetDownloader
from model import RWCNet, RWCNetLoss, ModelUtils


class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self):
        # Model parameters
        self.num_classes = 6
        self.pretrained = True
        self.dropout_rate = 0.5
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 1e-5
        self.weight_decay = 1e-4
        
        # Loss parameters
        self.aux1_weight = 0.5
        self.aux2_weight = 0.25
        self.use_class_weights = True
        
        # Optimization parameters
        self.optimizer_type = 'adam'  # 'adam', 'sgd', 'adamw'
        self.scheduler_type = 'step'   # 'step', 'plateau', 'cosine'
        self.step_size = 20
        self.gamma = 0.1
        
        # Early stopping
        self.patience = 10
        self.min_delta = 1e-4
        
        # Data parameters
        self.val_split = 0.2
        self.test_split = 0.1
        self.num_workers = 4
        
        # Paths
        self.dataset_path = './data/dataset-resized'
        self.checkpoint_dir = './checkpoints'
        self.log_dir = './logs'
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Logging
        self.print_freq = 20
        self.save_freq = 5
        
    def save_config(self, filepath):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            setattr(self, key, value)
        print(f"Configuration loaded from {filepath}")


class Trainer:
    """Main training class for RWC-Net model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_main_loss': [],
            'train_aux1_loss': [],
            'train_aux2_loss': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
    def setup_model(self, class_weights=None):
        """Initialize model, loss, and optimizer"""
        # Create model
        self.model = RWCNet(
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Create loss function
        self.criterion = RWCNetLoss(
            aux1_weight=self.config.aux1_weight,
            aux2_weight=self.config.aux2_weight,
            class_weights=class_weights.to(self.device) if class_weights is not None else None
        )
        
        # Create optimizer
        if self.config.optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        
        # Create scheduler
        if self.config.scheduler_type.lower() == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.gamma,
                patience=self.config.step_size // 2,
                verbose=True
            )
        elif self.config.scheduler_type.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=1e-7
            )
        
        # Print model info
        ModelUtils.print_model_summary(self.model)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_main_loss = 0.0
        running_aux1_loss = 0.0
        running_aux2_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            main_output, aux1_output, aux2_output = self.model(inputs)
            outputs = (main_output, aux1_output, aux2_output)
            
            # Calculate losses
            losses = self.criterion(outputs, targets)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            running_main_loss += losses['main_loss'].item()
            running_aux1_loss += losses['aux1_loss'].item()
            running_aux2_loss += losses['aux2_loss'].item()
            
            # Calculate accuracy (using main output)
            _, predicted = torch.max(main_output.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Print progress
            if batch_idx % self.config.print_freq == 0:
                print(f'Batch {batch_idx:>4}/{len(train_loader):>4} '
                      f'Loss: {total_loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_main_loss = running_main_loss / len(train_loader)
        epoch_aux1_loss = running_aux1_loss / len(train_loader)
        epoch_aux2_loss = running_aux2_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'loss': epoch_loss,
            'main_loss': epoch_main_loss,
            'aux1_loss': epoch_aux1_loss,
            'aux2_loss': epoch_aux2_loss,
            'accuracy': epoch_acc
        }
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                main_output, aux1_output, aux2_output = self.model(inputs)
                outputs = (main_output, aux1_output, aux2_output)
                
                # Calculate losses
                losses = self.criterion(outputs, targets)
                total_loss = losses['total_loss']
                
                # Statistics
                running_loss += total_loss.item()
                
                # Calculate accuracy (using main output)
                _, predicted = torch.max(main_output.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            print(f'\nEpoch {epoch+1}/{self.config.num_epochs}')
            print('-' * 40)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.config.scheduler_type.lower() == 'plateau':
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Record current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_main_loss'].append(train_metrics['main_loss'])
            self.history['train_aux1_loss'].append(train_metrics['aux1_loss'])
            self.history['train_aux2_loss'].append(train_metrics['aux2_loss'])
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f'Train Loss: {train_metrics["loss"]:.4f}, '
                  f'Train Acc: {train_metrics["accuracy"]:.2f}%')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.2f}%')
            print(f'LR: {current_lr:.2e}, Time: {epoch_time:.2f}s')
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_val_acc + self.config.min_delta:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f'New best model! Val Acc: {self.best_val_acc:.2f}%')
                
            else:
                self.patience_counter += 1
                
            # Early stopping check
            if self.patience_counter >= self.config.patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
                
            # Regular checkpoint saving
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f'\nLoaded best model with validation accuracy: {self.best_val_acc:.2f}%')
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.2f} minutes')
        
        return self.model
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config.__dict__
        }
        
        if is_best:
            filepath = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, filepath)
            print(f'Best model saved to {filepath}')
        else:
            filepath = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}.pth')
            torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f'Checkpoint loaded from {filepath}')
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Auxiliary losses plot
        axes[1, 0].plot(self.history['train_main_loss'], label='Main Loss', color='blue')
        axes[1, 0].plot(self.history['train_aux1_loss'], label='Aux1 Loss', color='green')
        axes[1, 0].plot(self.history['train_aux2_loss'], label='Aux2 Loss', color='orange')
        axes[1, 0].set_title('Training Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 1].plot(self.history['learning_rates'], color='purple')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Training history plot saved to {save_path}')
        
        plt.show()


class CrossValidator:
    """Cross-validation training class"""
    
    def __init__(self, config, k_folds=5):
        self.config = config
        self.k_folds = k_folds
        self.device = torch.device(config.device)
        self.results = []
        
    def run_cross_validation(self, dataset_path):
        """Run k-fold cross-validation"""
        print(f"\n{'='*60}")
        print(f"STARTING {self.k_folds}-FOLD CROSS-VALIDATION")
        print(f"{'='*60}")
        
        # Create k-fold data loaders
        fold_loaders = DataLoaderFactory.create_kfold_loaders(
            dataset_path, 
            k_folds=self.k_folds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        fold_results = []
        
        for fold, train_loader, val_loader in fold_loaders:
            print(f"\n{'='*50}")
            print(f"FOLD {fold}/{self.k_folds}")
            print(f"{'='*50}")
            
            # Create trainer for this fold
            fold_config = copy.deepcopy(self.config)
            fold_config.num_epochs = min(20, self.config.num_epochs)  # Reduced epochs for CV
            
            trainer = Trainer(fold_config)
            
            # Calculate class weights for this fold
            class_counts = defaultdict(int)
            for _, labels in train_loader:
                for label in labels:
                    class_counts[label.item()] += 1
            
            total_samples = sum(class_counts.values())
            class_weights = torch.zeros(self.config.num_classes)
            for class_idx in range(self.config.num_classes):
                if class_counts[class_idx] > 0:
                    class_weights[class_idx] = total_samples / (self.config.num_classes * class_counts[class_idx])
                else:
                    class_weights[class_idx] = 1.0
            
            # Setup and train model
            trainer.setup_model(class_weights)
            trained_model = trainer.train(train_loader, val_loader)
            
            # Record results
            fold_result = {
                'fold': fold,
                'best_val_acc': trainer.best_val_acc,
                'final_train_loss': trainer.history['train_loss'][-1],
                'final_val_loss': trainer.history['val_loss'][-1],
                'training_epochs': len(trainer.history['train_loss'])
            }
            
            fold_results.append(fold_result)
            
            print(f"Fold {fold} completed - Best Val Acc: {trainer.best_val_acc:.2f}%")
        
        # Calculate and print summary statistics
        self._print_cv_summary(fold_results)
        
        return fold_results
    
    def _print_cv_summary(self, fold_results):
        """Print cross-validation summary statistics"""
        accuracies = [result['best_val_acc'] for result in fold_results]
        
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION RESULTS SUMMARY")
        print(f"{'='*70}")
        
        for result in fold_results:
            print(f"Fold {result['fold']:>2}: "
                  f"Val Acc = {result['best_val_acc']:>6.2f}%, "
                  f"Epochs = {result['training_epochs']:>2}")
        
        print(f"{'-'*70}")
        print(f"Mean Val Accuracy: {np.mean(accuracies):>6.2f}% ± {np.std(accuracies):>5.2f}%")
        print(f"Best Val Accuracy: {np.max(accuracies):>6.2f}%")
        print(f"Worst Val Accuracy: {np.min(accuracies):>5.2f}%")
        print(f"{'='*70}")


def main():
    """Main training function"""
    # Create configuration
    config = TrainingConfig()
    
    # Download dataset if needed
    if not os.path.exists(config.dataset_path):
        print("Dataset not found. Downloading...")
        dataset_path = DatasetDownloader.download_trashnet('./data')
        if not dataset_path:
            print("Failed to download dataset!")
            return
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = DataLoaderFactory.create_data_loaders(
        config.dataset_path,
        batch_size=config.batch_size,
        val_split=config.val_split,
        test_split=config.test_split,
        num_workers=config.num_workers
    )
    
    # Save configuration
    config.save_config(os.path.join(config.log_dir, 'training_config.json'))
    
    # Create trainer
    trainer = Trainer(config)
    
    # Setup model with class weights
    if config.use_class_weights:
        trainer.setup_model(class_weights)
        print(f"Using class weights: {class_weights}")
    else:
        trainer.setup_model()
        print("Training without class weights")
    
    # Train model
    trained_model = trainer.train(train_loader, val_loader)
    
    # Plot training history
    plot_path = os.path.join(config.log_dir, 'training_history.png')
    trainer.plot_training_history(save_path=plot_path)
    
    # Save final model
    final_model_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
    ModelUtils.save_model(
        trained_model, 
        final_model_path,
        additional_info={
            'best_val_acc': trainer.best_val_acc,
            'training_history': trainer.history,
            'class_names': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        }
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Model saved to: {final_model_path}")
    
    return trained_model, trainer


def train_with_cv(k_folds=5):
    """Train with cross-validation"""
    # Create configuration
    config = TrainingConfig()
    
    # Download dataset if needed
    if not os.path.exists(config.dataset_path):
        print("Dataset not found. Downloading...")
        dataset_path = DatasetDownloader.download_trashnet('./data')
        if not dataset_path:
            print("Failed to download dataset!")
            return
    
    # Run cross-validation
    cv = CrossValidator(config, k_folds=k_folds)
    cv_results = cv.run_cross_validation(config.dataset_path)
    
    # Save CV results
    cv_results_path = os.path.join(config.log_dir, 'cv_results.json')
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"Cross-validation results saved to: {cv_results_path}")
    
    return cv_results


def hyperparameter_search():
    """Simple hyperparameter search"""
    # Define hyperparameter ranges
    learning_rates = [1e-5, 5e-5, 1e-4]
    dropout_rates = [0.3, 0.5, 0.7]
    batch_sizes = [16, 32, 64]
    
    best_config = None
    best_score = 0.0
    results = []
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH")
    print(f"{'='*60}")
    
    for lr in learning_rates:
        for dropout in dropout_rates:
            for batch_size in batch_sizes:
                print(f"\nTesting: LR={lr}, Dropout={dropout}, Batch={batch_size}")
                
                # Create config
                config = TrainingConfig()
                config.learning_rate = lr
                config.dropout_rate = dropout
                config.batch_size = batch_size
                config.num_epochs = 15  # Reduced for hyperparameter search
                config.patience = 5
                
                try:
                    # Create data loaders
                    train_loader, val_loader, _, class_weights = DataLoaderFactory.create_data_loaders(
                        config.dataset_path,
                        batch_size=batch_size,
                        num_workers=2  # Reduced workers
                    )
                    
                    # Train model
                    trainer = Trainer(config)
                    trainer.setup_model(class_weights)
                    trained_model = trainer.train(train_loader, val_loader)
                    
                    # Record results
                    result = {
                        'learning_rate': lr,
                        'dropout_rate': dropout,
                        'batch_size': batch_size,
                        'best_val_acc': trainer.best_val_acc,
                        'final_train_loss': trainer.history['train_loss'][-1],
                        'final_val_loss': trainer.history['val_loss'][-1]
                    }
                    
                    results.append(result)
                    
                    # Check if best
                    if trainer.best_val_acc > best_score:
                        best_score = trainer.best_val_acc
                        best_config = result.copy()
                    
                    print(f"Result: Val Acc = {trainer.best_val_acc:.2f}%")
                    
                except Exception as e:
                    print(f"Failed with error: {e}")
                    continue
    
    # Print results
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*60}")
    
    # Sort results by validation accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    for i, result in enumerate(results[:5]):  # Top 5 results
        print(f"Rank {i+1}: LR={result['learning_rate']:.0e}, "
              f"Dropout={result['dropout_rate']}, "
              f"Batch={result['batch_size']}, "
              f"Val Acc={result['best_val_acc']:.2f}%")
    
    print(f"\nBest configuration:")
    if best_config:
        for key, value in best_config.items():
            print(f"  {key}: {value}")
    
    # Save results
    results_path = os.path.join('./logs', 'hyperparameter_search.json')
    os.makedirs('./logs', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nHyperparameter search results saved to: {results_path}")
    
    return results, best_config


def resume_training(checkpoint_path):
    """Resume training from checkpoint"""
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Recreate config
    config = TrainingConfig()
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create data loaders
    train_loader, val_loader, _, class_weights = DataLoaderFactory.create_data_loaders(
        config.dataset_path,
        batch_size=config.batch_size,
        val_split=config.val_split,
        test_split=config.test_split,
        num_workers=config.num_workers
    )
    
    # Create trainer and load checkpoint
    trainer = Trainer(config)
    trainer.setup_model(class_weights if config.use_class_weights else None)
    start_epoch = trainer.load_checkpoint(checkpoint_path)
    
    print(f"Resuming training from epoch {start_epoch + 1}")
    
    # Continue training
    trained_model = trainer.train(train_loader, val_loader)
    
    return trained_model, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RWC-Net model')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'cv', 'hypersearch', 'resume'],
                       help='Training mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for resume mode')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting standard training...")
        trained_model, trainer = main()
        
    elif args.mode == 'cv':
        print(f"Starting {args.folds}-fold cross-validation...")
        cv_results = train_with_cv(k_folds=args.folds)
        
    elif args.mode == 'hypersearch':
        print("Starting hyperparameter search...")
        results, best_config = hyperparameter_search()
        
    elif args.mode == 'resume':
        if args.checkpoint is None:
            print("Error: --checkpoint path required for resume mode")
        else:
            print("Resuming training from checkpoint...")
            trained_model, trainer = resume_training(args.checkpoint)
    
    print("Training script completed!")