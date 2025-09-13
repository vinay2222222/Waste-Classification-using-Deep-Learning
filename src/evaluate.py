"""
evaluate.py - RWC-Net Model Evaluation Module
============================================

This module provides comprehensive evaluation tools for the RWC-Net model including:

- Performance metrics calculation (accuracy, precision, recall, F1-score)
- Confusion matrix analysis and visualization
- Per-class performance analysis
- ROC curves and AUC scores
- Model interpretability with attention visualization
- Feature analysis and dimensionality reduction
- Prediction visualization and error analysis
- Model comparison utilities

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

import os
from collections import defaultdict
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from dataset import DataLoaderFactory, DataTransforms
from model import RWCNet, ModelUtils, FeatureExtractor


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.num_classes = len(self.class_names)
        
        # Evaluation results storage
        self.results = {
            'predictions': [],
            'true_labels': [],
            'probabilities': [],
            'features': None,
            'metrics': {}
        }
        
    def evaluate_dataset(self, dataloader, save_features=False):
        """
        Evaluate model on a dataset
        
        Args:
            dataloader: DataLoader for evaluation
            save_features (bool): Whether to save extracted features
            
        Returns:
            dict: Evaluation metrics and results
        """
        self.model.eval()
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        all_features = [] if save_features else None
        
        print("Evaluating model on dataset...")
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                main_output, aux1_output, aux2_output = self.model(inputs)
                
                # Get predictions and probabilities
                probabilities = F.softmax(main_output, dim=1)
                _, predicted = torch.max(main_output, 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Extract features if requested
                if save_features:
                    densenet_feat, mobilenet_feat, combined_feat = self.model.forward_features(inputs)
                    all_features.extend(combined_feat.cpu().numpy())
                
                # Print progress
                if batch_idx % 20 == 0:
                    print(f"Processed {batch_idx}/{len(dataloader)} batches")
        
        # Store results
        self.results['predictions'] = np.array(all_predictions)
        self.results['true_labels'] = np.array(all_true_labels)
        self.results['probabilities'] = np.array(all_probabilities)
        if save_features:
            self.results['features'] = np.array(all_features)
        
        # Calculate metrics
        self._calculate_metrics()
        
        print("Evaluation completed!")
        return self.results
    
    def _calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        y_true = self.results['true_labels']
        y_pred = self.results['predictions']
        y_prob = self.results['probabilities']
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Store metrics
        self.results['metrics'] = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist()
        }
        
        # Calculate ROC AUC for multiclass
        try:
            # Binarize labels for multiclass ROC
            y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
            
            # Calculate ROC AUC for each class
            roc_auc_per_class = []
            for i in range(self.num_classes):
                if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists in true labels
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    roc_auc_per_class.append(auc(fpr, tpr))
                else:
                    roc_auc_per_class.append(0.0)
            
            self.results['metrics']['roc_auc_per_class'] = roc_auc_per_class
            self.results['metrics']['roc_auc_macro'] = np.mean(roc_auc_per_class)
            
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC - {e}")
            self.results['metrics']['roc_auc_per_class'] = [0.0] * self.num_classes
            self.results['metrics']['roc_auc_macro'] = 0.0
    
    def print_metrics_summary(self):
        """Print comprehensive metrics summary"""
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        # Overall metrics
        print(f"{'Accuracy':<25}: {metrics['accuracy']:.4f}")
        print(f"{'Precision (Macro)':<25}: {metrics['precision_macro']:.4f}")
        print(f"{'Precision (Weighted)':<25}: {metrics['precision_weighted']:.4f}")
        print(f"{'Recall (Macro)':<25}: {metrics['recall_macro']:.4f}")
        print(f"{'Recall (Weighted)':<25}: {metrics['recall_weighted']:.4f}")
        print(f"{'F1-Score (Macro)':<25}: {metrics['f1_macro']:.4f}")
        print(f"{'F1-Score (Weighted)':<25}: {metrics['f1_weighted']:.4f}")
        print(f"{'ROC AUC (Macro)':<25}: {metrics['roc_auc_macro']:.4f}")
        
        print("\n" + "-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}")
        print("-"*60)
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            roc_auc = metrics['roc_auc_per_class'][i]
            
            print(f"{class_name:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {roc_auc:<10.4f}")
        
        print("="*60)
    
    def plot_confusion_matrix(self, save_path=None, figsize=(10, 8)):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.results['true_labels'], self.results['predictions'])
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        # Print normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("\nNormalized Confusion Matrix:")
        print("-" * 50)
        print(f"{'True\\Pred':<10}", end='')
        for class_name in self.class_names:
            print(f"{class_name[:8]:>8}", end='')
        print()
        print("-" * 50)
        
        for i, true_class in enumerate(self.class_names):
            print(f"{true_class[:8]:<10}", end='')
            for j in range(len(self.class_names)):
                print(f"{cm_normalized[i, j]:>8.3f}", end='')
            print()
    
    def plot_roc_curves(self, save_path=None, figsize=(12, 8)):
        """Plot ROC curves for each class"""
        y_true = self.results['true_labels']
        y_prob = self.results['probabilities']
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each class
        for i in range(self.num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-Class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, save_path=None, figsize=(12, 8)):
        """Plot Precision-Recall curves for each class"""
        y_true = self.results['true_labels']
        y_prob = self.results['probabilities']
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        plt.figure(figsize=figsize)
        
        # Plot PR curve for each class
        for i in range(self.num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(recall, precision, lw=2, 
                        label=f'{self.class_names[i]} (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Multi-Class Classification')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curves saved to {save_path}")
        
        plt.show()
    
    def analyze_classification_errors(self, dataloader, num_samples=16):
        """Analyze and visualize classification errors"""
        # Find misclassified samples
        misclassified_indices = np.where(
            self.results['predictions'] != self.results['true_labels']
        )[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassified samples found!")
            return
        
        print(f"Found {len(misclassified_indices)} misclassified samples")
        
        # Get a subset of misclassified samples
        sample_indices = np.random.choice(
            misclassified_indices, 
            size=min(num_samples, len(misclassified_indices)), 
            replace=False
        )
        
        # Extract images for visualization
        all_images = []
        all_labels = []
        
        for inputs, labels in dataloader:
            all_images.extend(inputs)
            all_labels.extend(labels)
        
        # Plot misclassified samples
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_indices):
            if i >= 16:  # Limit to 16 samples
                break
                
            img = all_images[idx]
            true_label = self.results['true_labels'][idx]
            pred_label = self.results['predictions'][idx]
            confidence = np.max(self.results['probabilities'][idx])
            
            # Denormalize image for display
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            # Plot
            axes[i].imshow(img_denorm.permute(1, 2, 0))
            axes[i].set_title(
                f'True: {self.class_names[true_label]}\n'
                f'Pred: {self.class_names[pred_label]}\n'
                f'Conf: {confidence:.3f}',
                color='red', fontsize=10
            )
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(sample_indices), 16):
            axes[i].axis('off')
        
        plt.suptitle('Misclassified Samples Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Error analysis by class
        self._analyze_error_patterns()
    
    def _analyze_error_patterns(self):
        """Analyze error patterns across classes"""
        y_true = self.results['true_labels']
        y_pred = self.results['predictions']
        
        print("\n" + "="*60)
        print("ERROR PATTERN ANALYSIS")
        print("="*60)
        
        # Confusion matrix for detailed analysis
        cm = confusion_matrix(y_true, y_pred)
        
        # Find most confused class pairs
        confused_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((
                        self.class_names[i], 
                        self.class_names[j], 
                        cm[i, j],
                        cm[i, j] / cm[i, :].sum()  # Error rate
                    ))
        
        # Sort by number of errors
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("Most Frequent Misclassifications:")
        print(f"{'True Class':<12} {'Predicted As':<12} {'Count':<8} {'Error Rate'}")
        print("-" * 50)
        
        for true_class, pred_class, count, error_rate in confused_pairs[:10]:
            print(f"{true_class:<12} {pred_class:<12} {count:<8} {error_rate:.3f}")
    
    def visualize_features(self, method='tsne', save_path=None, figsize=(10, 8)):
        """
        Visualize extracted features using dimensionality reduction
        
        Args:
            method (str): 'tsne' or 'pca'
            save_path (str): Path to save the plot
            figsize (tuple): Figure size
        """
        if self.results['features'] is None:
            print("No features available. Run evaluation with save_features=True first.")
            return
        
        features = self.results['features']
        labels = self.results['true_labels']
        
        print(f"Visualizing features using {method.upper()}...")
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        features_2d = reducer.fit_transform(features)
        
        # Plot
        plt.figure(figsize=figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=class_name, alpha=0.7, s=50)
        
        plt.title(f'Feature Visualization - {method.upper()}')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature visualization saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, save_path=None):
        """Generate detailed classification report"""
        y_true = self.results['true_labels']
        y_pred = self.results['predictions']
        
        # Generate sklearn classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better formatting
        df_report = pd.DataFrame(report).transpose()
        
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print(df_report.round(4))
        
        if save_path:
            # Save as CSV
            df_report.to_csv(save_path.replace('.txt', '.csv'))
            
            # Save as text
            with open(save_path, 'w') as f:
                f.write("RWC-Net Model Classification Report\n")
                f.write("="*50 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("Overall Metrics:\n")
                f.write("-" * 20 + "\n")
                
                metrics = self.results['metrics']
                for key, value in metrics.items():
                    if not isinstance(value, list):
                        f.write(f"{key}: {value:.4f}\n")
                
                f.write(f"\nDetailed Report:\n")
                f.write("-" * 20 + "\n")
                f.write(df_report.to_string())
            
            print(f"Classification report saved to {save_path}")
    
    def predict_single_image(self, image_path, transform=None):
        """
        Predict class for a single image
        
        Args:
            image_path (str): Path to image file
            transform: Image transform to apply
            
        Returns:
            dict: Prediction results
        """
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        if transform is None:
            transform = DataTransforms.get_val_transforms()
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            main_output, _, _ = self.model(image_tensor)
            probabilities = F.softmax(main_output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get top-k predictions
        top_k = min(3, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        top_predictions = []
        for i in range(top_k):
            top_predictions.append({
                'class': self.class_names[top_indices[i]],
                'probability': top_probs[i].item()
            })
        
        result = {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_probabilities': probabilities[0].cpu().numpy()
        }
        
        return result
    
    def save_results(self, save_dir):
        """Save all evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'true_label': [self.class_names[i] for i in self.results['true_labels']],
            'predicted_label': [self.class_names[i] for i in self.results['predictions']],
            'true_label_idx': self.results['true_labels'],
            'predicted_label_idx': self.results['predictions']
        })
        
        # Add probability columns
        for i, class_name in enumerate(self.class_names):
            predictions_df[f'prob_{class_name}'] = self.results['probabilities'][:, i]
        
        predictions_path = os.path.join(save_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        
        # Save features if available
        if self.results['features'] is not None:
            features_path = os.path.join(save_dir, 'extracted_features.npy')
            np.save(features_path, self.results['features'])
        
        print(f"Results saved to {save_dir}")


class ModelComparison:
    """Compare multiple models"""
    
    def __init__(self, models_dict, device='cpu'):
        """
        Args:
            models_dict (dict): Dictionary of {model_name: model}
            device (str): Device to run evaluation
        """
        self.models = models_dict
        self.device = torch.device(device)
        self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.comparison_results = {}
    
    def compare_models(self, test_loader):
        """Compare all models on test set"""
        print("Comparing models...")
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            evaluator = ModelEvaluator(model, self.device)
            results = evaluator.evaluate_dataset(test_loader)
            
            self.comparison_results[model_name] = {
                'metrics': results['metrics'],
                'evaluator': evaluator
            }
        
        self._print_comparison_summary()
        return self.comparison_results
    
    def _print_comparison_summary(self):
        """Print model comparison summary"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Prepare comparison table
        metrics_to_compare = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_macro']
        
        print(f"{'Model':<20}", end='')
        for metric in metrics_to_compare:
            print(f"{metric.replace('_', ' ').title():<15}", end='')
        print()
        print("-" * 95)
        
        for model_name, results in self.comparison_results.items():
            print(f"{model_name:<20}", end='')
            for metric in metrics_to_compare:
                value = results['metrics'].get(metric, 0.0)
                print(f"{value:<15.4f}", end='')
            print()
        
        print("="*80)
    
    def plot_comparison_chart(self, save_path=None):
        """Plot model comparison chart"""
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        model_names = list(self.comparison_results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.comparison_results[name]['metrics'][metric] for name in model_names]
            
            bars = axes[i].bar(model_names, values, alpha=0.7, color=plt.cm.Set3(np.arange(len(model_names))))
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison chart saved to {save_path}")
        
        plt.show()


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate class activation map"""
        self.model.eval()
        
        # Forward pass
        main_output, _, _ = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(main_output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        main_output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        