"""
model.py - RWC-Net Model Architecture Module
===========================================

This module implements the RWC-Net (Recyclable Waste Classification Network),
a hybrid deep learning model combining DenseNet201 and MobileNetV2 with 
auxiliary outputs for enhanced waste classification performance.

Architecture:
- Backbone: DenseNet201 + MobileNetV2 (pretrained)
- Features: Global Average Pooling + Feature Fusion
- Outputs: Main classifier + 2 auxiliary classifiers
- Loss: Weighted combination of main + auxiliary losses

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet201, mobilenet_v2
import math


class RWCNet(nn.Module):
    """
    RWC-Net: Hybrid DenseNet201 + MobileNetV2 model with auxiliary outputs
    
    This model combines features from two pretrained backbones and uses
    auxiliary supervision for better training convergence and performance.
    
    Args:
        num_classes (int): Number of output classes (default: 6 for TrashNet)
        pretrained (bool): Whether to use pretrained backbones (default: True)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    """
    
    def __init__(self, num_classes=6, pretrained=True, dropout_rate=0.5):
        super(RWCNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pretrained backbones
        self.densenet = densenet201(pretrained=pretrained)
        self.mobilenet = mobilenet_v2(pretrained=pretrained)
        
        # Extract feature extractors (remove final classifiers)
        self.densenet_features = self.densenet.features
        self.mobilenet_features = self.mobilenet.features
        
        # Feature dimensions after global average pooling
        self.densenet_out_features = 1920  # DenseNet201 output features
        self.mobilenet_out_features = 1280  # MobileNetV2 output features
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Auxiliary classifiers for intermediate supervision
        self.aux_classifier1 = self._make_auxiliary_classifier(
            self.densenet_out_features, num_classes
        )
        
        self.aux_classifier2 = self._make_auxiliary_classifier(
            self.mobilenet_out_features, num_classes
        )
        
        # Feature fusion dimensions
        combined_features = self.densenet_out_features + self.mobilenet_out_features
        
        # Main classifier with feature fusion
        self.main_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(combined_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_auxiliary_classifier(self, input_features, num_classes):
        """Create auxiliary classifier for intermediate supervision"""
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize weights for newly added layers"""
        for module in [self.aux_classifier1, self.aux_classifier2, self.main_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the RWC-Net model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            tuple: (main_output, aux1_output, aux2_output)
                - main_output: Main classification logits
                - aux1_output: DenseNet auxiliary output
                - aux2_output: MobileNet auxiliary output
        """
        
        # Extract features from both backbones
        densenet_feat = self.densenet_features(x)
        mobilenet_feat = self.mobilenet_features(x)
        
        # Global Average Pooling
        densenet_pooled = self.gap(densenet_feat).view(densenet_feat.size(0), -1)
        mobilenet_pooled = self.gap(mobilenet_feat).view(mobilenet_feat.size(0), -1)
        
        # Auxiliary outputs for intermediate supervision
        aux1 = self.aux_classifier1(densenet_pooled)
        aux2 = self.aux_classifier2(mobilenet_pooled)
        
        # Feature fusion: concatenate features from both backbones
        combined_features = torch.cat([densenet_pooled, mobilenet_pooled], dim=1)
        
        # Main classification output
        main_output = self.main_classifier(combined_features)
        
        return main_output, aux1, aux2
    
    def forward_features(self, x):
        """
        Extract features without classification (useful for feature analysis)
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (densenet_features, mobilenet_features, combined_features)
        """
        # Extract features from both backbones
        densenet_feat = self.densenet_features(x)
        mobilenet_feat = self.mobilenet_features(x)
        
        # Global Average Pooling
        densenet_pooled = self.gap(densenet_feat).view(densenet_feat.size(0), -1)
        mobilenet_pooled = self.gap(mobilenet_feat).view(mobilenet_feat.size(0), -1)
        
        # Combined features
        combined_features = torch.cat([densenet_pooled, mobilenet_pooled], dim=1)
        
        return densenet_pooled, mobilenet_pooled, combined_features
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': 'RWC-Net',
            'backbone': 'DenseNet201 + MobileNetV2',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'densenet_features': self.densenet_out_features,
            'mobilenet_features': self.mobilenet_out_features,
            'combined_features': self.densenet_out_features + self.mobilenet_out_features,
            'dropout_rate': self.dropout_rate
        }
        
        return info


class RWCNetLoss(nn.Module):
    """
    Custom loss function for RWC-Net with auxiliary loss weighting
    
    Combines main loss with auxiliary losses using decay factors as described
    in the original paper implementation.
    
    Args:
        aux1_weight (float): Weight for first auxiliary loss (default: 0.5)
        aux2_weight (float): Weight for second auxiliary loss (default: 0.25)
        class_weights (torch.Tensor, optional): Class weights for balanced training
    """
    
    def __init__(self, aux1_weight=0.5, aux2_weight=0.25, class_weights=None):
        super(RWCNetLoss, self).__init__()
        
        self.aux1_weight = aux1_weight
        self.aux2_weight = aux2_weight
        
        # Create loss function with optional class weighting
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, outputs, targets):
        """
        Calculate combined loss
        
        Args:
            outputs (tuple): (main_output, aux1_output, aux2_output)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            dict: Dictionary containing individual and total losses
        """
        main_output, aux1_output, aux2_output = outputs
        
        # Calculate individual losses
        main_loss = self.criterion(main_output, targets)
        aux1_loss = self.criterion(aux1_output, targets)
        aux2_loss = self.criterion(aux2_output, targets)
        
        # Combined loss with auxiliary weighting
        total_loss = main_loss + self.aux1_weight * aux1_loss + self.aux2_weight * aux2_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'aux1_loss': aux1_loss,
            'aux2_loss': aux2_loss
        }


class ModelFactory:
    """Factory class for creating different model configurations"""
    
    @staticmethod
    def create_rwc_net(num_classes=6, pretrained=True, dropout_rate=0.5):
        """Create standard RWC-Net model"""
        return RWCNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    
    @staticmethod
    def create_rwc_net_light(num_classes=6, pretrained=True):
        """Create lightweight version with reduced dropout"""
        return RWCNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=0.3
        )
    
    @staticmethod
    def create_rwc_net_heavy(num_classes=6, pretrained=True):
        """Create heavy regularization version"""
        return RWCNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=0.7
        )


class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def count_parameters(model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    @staticmethod
    def freeze_backbone(model, freeze_densenet=True, freeze_mobilenet=True):
        """
        Freeze backbone parameters for fine-tuning
        
        Args:
            model (RWCNet): Model to modify
            freeze_densenet (bool): Whether to freeze DenseNet backbone
            freeze_mobilenet (bool): Whether to freeze MobileNet backbone
        """
        if freeze_densenet:
            for param in model.densenet_features.parameters():
                param.requires_grad = False
            print("DenseNet backbone frozen")
        
        if freeze_mobilenet:
            for param in model.mobilenet_features.parameters():
                param.requires_grad = False
            print("MobileNet backbone frozen")
    
    @staticmethod
    def unfreeze_backbone(model):
        """Unfreeze all backbone parameters"""
        for param in model.densenet_features.parameters():
            param.requires_grad = True
        for param in model.mobilenet_features.parameters():
            param.requires_grad = True
        print("All backbone parameters unfrozen")
    
    @staticmethod
    def save_model(model, filepath, additional_info=None):
        """
        Save model with additional metadata
        
        Args:
            model (RWCNet): Model to save
            filepath (str): Save path
            additional_info (dict, optional): Additional information to save
        """
        model_info = model.get_model_info()
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_info': model_info,
            'model_class': model.__class__.__name__
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath, num_classes=6, device='cpu'):
        """
        Load saved model
        
        Args:
            filepath (str): Path to saved model
            num_classes (int): Number of classes
            device (str): Device to load model on
            
        Returns:
            RWCNet: Loaded model
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model
        model = RWCNet(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from {filepath}")
        if 'model_info' in checkpoint:
            print("Model info:", checkpoint['model_info'])
        
        return model
    
    @staticmethod
    def print_model_summary(model, input_size=(3, 224, 224)):
        """
        Print detailed model summary
        
        Args:
            model (RWCNet): Model to analyze
            input_size (tuple): Input tensor size
        """
        model_info = model.get_model_info()
        
        print("\n" + "="*60)
        print("RWC-NET MODEL SUMMARY")
        print("="*60)
        
        for key, value in model_info.items():
            if isinstance(value, int):
                print(f"{key.replace('_', ' ').title():>25}: {value:,}")
            else:
                print(f"{key.replace('_', ' ').title():>25}: {value}")
        
        print("="*60)
        
        # Test forward pass to get output shapes
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_size)
            main_out, aux1_out, aux2_out = model(dummy_input)
            
            print(f"{'Input Shape':>25}: {tuple(dummy_input.shape)}")
            print(f"{'Main Output Shape':>25}: {tuple(main_out.shape)}")
            print(f"{'Aux1 Output Shape':>25}: {tuple(aux1_out.shape)}")
            print(f"{'Aux2 Output Shape':>25}: {tuple(aux2_out.shape)}")
        
        print("="*60)


class FeatureExtractor:
    """Extract and analyze features from RWC-Net model"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def extract_features(self, dataloader, device='cpu'):
        """
        Extract features from all samples in dataloader
        
        Args:
            dataloader: DataLoader with samples
            device (str): Device to run extraction on
            
        Returns:
            dict: Dictionary with extracted features and labels
        """
        all_densenet_features = []
        all_mobilenet_features = []
        all_combined_features = []
        all_labels = []
        
        self.model.to(device)
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                
                # Extract features
                densenet_feat, mobilenet_feat, combined_feat = self.model.forward_features(inputs)
                
                all_densenet_features.append(densenet_feat.cpu())
                all_mobilenet_features.append(mobilenet_feat.cpu())
                all_combined_features.append(combined_feat.cpu())
                all_labels.append(labels)
        
        return {
            'densenet_features': torch.cat(all_densenet_features, dim=0),
            'mobilenet_features': torch.cat(all_mobilenet_features, dim=0),
            'combined_features': torch.cat(all_combined_features, dim=0),
            'labels': torch.cat(all_labels, dim=0)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing RWC-Net Model Module")
    print("=" * 50)
    
    # Test model creation
    model = ModelFactory.create_rwc_net(num_classes=6, pretrained=False)
    
    # Print model summary
    ModelUtils.print_model_summary(model)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    main_out, aux1_out, aux2_out = model(dummy_input)
    
    print(f"\nForward pass test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Main output shape: {main_out.shape}")
    print(f"Aux1 output shape: {aux1_out.shape}")
    print(f"Aux2 output shape: {aux2_out.shape}")
    
    # Test loss function
    criterion = RWCNetLoss()
    dummy_targets = torch.randint(0, 6, (4,))
    
    losses = criterion((main_out, aux1_out, aux2_out), dummy_targets)
    print(f"\nLoss calculation test:")
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.item():.4f}")
    
    # Test feature extraction
    feature_extractor = FeatureExtractor(model)
    densenet_feat, mobilenet_feat, combined_feat = model.forward_features(dummy_input)
    
    print(f"\nFeature extraction test:")
    print(f"DenseNet features shape: {densenet_feat.shape}")
    print(f"MobileNet features shape: {mobilenet_feat.shape}")
    print(f"Combined features shape: {combined_feat.shape}")
    
    print("\nModel module test completed successfully!")