"""
dataset.py - TrashNet Dataset Handling Module
==============================================

This module handles downloading, preprocessing, and loading the TrashNet dataset
for recyclable waste classification using the RWC-Net model.

Dataset: TrashNet (6 classes: cardboard, glass, metal, paper, plastic, trash)
"""

import os
import zipfile
import requests
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class TrashNetDataset(Dataset):
    """
    Custom Dataset class for TrashNet recyclable waste classification
    
    Args:
        root_dir (str): Path to dataset root directory
        transform (callable, optional): Transform to apply to images
        split (str): Dataset split identifier ('train', 'val', 'test')
    """
    
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Define class names and mapping
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.samples = []
        self.class_counts = {cls: 0 for cls in self.classes}
        
        self._load_samples()
        
    def _load_samples(self):
        """Load and validate all image samples"""
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory '{class_name}' not found!")
                continue
                
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for img_name in image_files:
                img_path = os.path.join(class_dir, img_name)
                # Verify image can be opened
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify image integrity
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    self.class_counts[class_name] += 1
                except Exception as e:
                    print(f"Skipping corrupted image: {img_path}")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {self.root_dir}")
        
        print(f"\nLoaded {len(self.samples)} valid images for {self.split}")
        self._print_class_distribution()
        
    def _print_class_distribution(self):
        """Print class distribution statistics"""
        print("Class distribution:")
        for class_name, count in self.class_counts.items():
            percentage = (count / len(self.samples)) * 100
            print(f"  {class_name:>10}: {count:>4} images ({percentage:>5.1f}%)")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        img_path, label = self.samples[idx]
        
        try:
            # Load image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms if provided
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image with correct label as fallback
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        total_samples = sum(self.class_counts.values())
        weights = []
        
        for class_name in self.classes:
            if self.class_counts[class_name] > 0:
                weight = total_samples / (len(self.classes) * self.class_counts[class_name])
                weights.append(weight)
            else:
                weights.append(1.0)
        
        return torch.FloatTensor(weights)
    
    def plot_class_distribution(self):
        """Plot class distribution as bar chart"""
        plt.figure(figsize=(10, 6))
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        bars = plt.bar(classes, counts, color='steelblue', alpha=0.7)
        plt.title(f'Class Distribution - {self.split.capitalize()} Set')
        plt.xlabel('Waste Categories')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


class TrashNetSubset(Dataset):
    """
    Subset of TrashNet dataset with specific transform
    
    Args:
        dataset (TrashNetDataset): Base dataset
        indices (list): Indices to include in subset
        transform (callable, optional): Transform to apply
    """
    
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get original sample
        original_idx = self.indices[idx]
        img_path, label = self.dataset.samples[original_idx]
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return blank image as fallback
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label


class DatasetDownloader:
    """Handle TrashNet dataset downloading and extraction"""
    
    @staticmethod
    def download_trashnet(download_path='./data'):
        """
        Download and extract TrashNet dataset
        
        Args:
            download_path (str): Path to download dataset
            
        Returns:
            str: Path to extracted dataset or None if failed
        """
        dataset_path = os.path.join(download_path, 'dataset-resized')
        
        # Check if already exists
        if os.path.exists(dataset_path):
            print(f"Dataset already exists at {dataset_path}")
            return dataset_path
        
        # Create download directory
        os.makedirs(download_path, exist_ok=True)
        
        # Direct link to the resized dataset
        url = "https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip?raw=true"
        zip_path = os.path.join(download_path, 'dataset-resized.zip')
        
        print("Downloading TrashNet dataset (resized version)...")
        
        try:
            # Download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
            
            print("\nExtracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            
            # Clean up zip file
            os.remove(zip_path)
            
            # Verify extraction
            if os.path.exists(dataset_path):
                print("Dataset successfully downloaded and extracted!")
                DatasetDownloader.print_dataset_stats(dataset_path)
                return dataset_path
            else:
                raise Exception("Dataset extraction failed")
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download from: https://github.com/garythung/trashnet")
            return None
    
    @staticmethod
    def print_dataset_stats(dataset_path):
        """Print statistics about the downloaded dataset"""
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        total_images = 0
        
        print("\n" + "="*40)
        print("DATASET STATISTICS")
        print("="*40)
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total_images += count
                print(f"{class_name.capitalize():>10}: {count:>4} images")
            else:
                print(f"{class_name.capitalize():>10}: {0:>4} images (missing)")
        
        print("-" * 40)
        print(f"{'Total':>10}: {total_images:>4} images")
        print("="*40)


class DataTransforms:
    """Define data augmentation and preprocessing transforms"""
    
    @staticmethod
    def get_train_transforms(input_size=224):
        """
        Get training transforms with data augmentation
        
        Args:
            input_size (int): Input image size
            
        Returns:
            torchvision.transforms.Compose: Training transforms
        """
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms(input_size=224):
        """
        Get validation/test transforms (no augmentation)
        
        Args:
            input_size (int): Input image size
            
        Returns:
            torchvision.transforms.Compose: Validation transforms
        """
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class DataLoaderFactory:
    """Factory class for creating data loaders with stratified splitting"""
    
    @staticmethod
    def create_data_loaders(dataset_path, batch_size=32, val_split=0.2, 
                          test_split=0.1, num_workers=4, random_state=42):
        """
        Create train, validation, and test data loaders with stratified splitting
        
        Args:
            dataset_path (str): Path to dataset
            batch_size (int): Batch size for data loaders
            val_split (float): Validation split ratio
            test_split (float): Test split ratio
            num_workers (int): Number of worker processes
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (train_loader, val_loader, test_loader, class_weights)
        """
        
        # Get transforms
        train_transform = DataTransforms.get_train_transforms()
        val_transform = DataTransforms.get_val_transforms()
        
        # Load full dataset without transforms
        full_dataset = TrashNetDataset(dataset_path, transform=None, split='full')
        
        # Get class weights for balanced training
        class_weights = full_dataset.get_class_weights()
        
        # Stratified splitting
        labels = [sample[1] for sample in full_dataset.samples]
        indices = list(range(len(full_dataset.samples)))
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_split, stratify=labels, random_state=random_state
        )
        
        # Second split: separate train and validation
        train_val_labels = [labels[i] for i in train_val_indices]
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_split/(1-test_split), 
            stratify=train_val_labels, random_state=random_state
        )
        
        # Create datasets with appropriate transforms
        train_dataset = TrashNetSubset(full_dataset, train_indices, train_transform)
        val_dataset = TrashNetSubset(full_dataset, val_indices, val_transform)
        test_dataset = TrashNetSubset(full_dataset, test_indices, val_transform)
        
        # Update split info for datasets
        train_dataset.dataset.split = 'train'
        val_dataset.dataset.split = 'validation'
        test_dataset.dataset.split = 'test'
        
        # Create data loaders
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
        
        print(f"\n" + "="*50)
        print("DATA LOADER SUMMARY")
        print("="*50)
        print(f"Training samples:   {len(train_dataset):>6}")
        print(f"Validation samples: {len(val_dataset):>6}") 
        print(f"Test samples:       {len(test_dataset):>6}")
        print(f"Batch size:         {batch_size:>6}")
        print(f"Number of workers:  {num_workers:>6}")
        print("="*50)
        
        return train_loader, val_loader, test_loader, class_weights
    
    @staticmethod
    def create_kfold_loaders(dataset_path, k_folds=5, batch_size=32, 
                           num_workers=4, random_state=42):
        """
        Create K-fold cross-validation data loaders
        
        Args:
            dataset_path (str): Path to dataset
            k_folds (int): Number of folds
            batch_size (int): Batch size
            num_workers (int): Number of workers
            random_state (int): Random seed
            
        Returns:
            generator: Yields (fold, train_loader, val_loader) tuples
        """
        # Load full dataset
        full_dataset = TrashNetDataset(dataset_path, transform=None, split='cv')
        
        # Get transforms
        train_transform = DataTransforms.get_train_transforms()
        val_transform = DataTransforms.get_val_transforms()
        
        # Get labels for stratification
        labels = [sample[1] for sample in full_dataset.samples]
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        
        pin_memory = torch.cuda.is_available()
        
        for fold, (train_ids, val_ids) in enumerate(skf.split(range(len(full_dataset)), labels)):
            # Create datasets for this fold
            train_dataset = TrashNetSubset(full_dataset, train_ids, train_transform)
            val_dataset = TrashNetSubset(full_dataset, val_ids, val_transform)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=num_workers, pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )
            
            yield fold + 1, train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    # Test dataset downloading and loading
    print("Testing TrashNet Dataset Module")
    print("=" * 50)
    
    # Download dataset
    dataset_path = DatasetDownloader.download_trashnet('./data')
    
    if dataset_path:
        # Create data loaders
        train_loader, val_loader, test_loader, class_weights = DataLoaderFactory.create_data_loaders(
            dataset_path, batch_size=16, num_workers=2
        )
        
        print(f"\nClass weights: {class_weights}")
        
        # Test data loading
        print("\nTesting data loading...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
            if batch_idx >= 2:  # Test first few batches
                break
        
        print("Dataset module test completed successfully!")
    else:
        print("Dataset download failed!")