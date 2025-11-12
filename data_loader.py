import os
import glob
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_dataset


class BonnEEGDataset(Dataset):
    
    
    def __init__(self, segments, labels, file_indices):
        
        self.segments = torch.FloatTensor(segments)
        self.labels = torch.LongTensor(labels)
        self.file_indices = file_indices
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        
        segment = self.segments[idx].unsqueeze(0)  # Add channel dimension
        label = self.labels[idx]
        return segment, label


def load_bonn_files(data_dir, set_name):
    
    set_dir = os.path.join(data_dir, set_name)
    
    if set_name == 'A':
        pattern = 'Z*.txt'
    elif set_name == 'B':
        pattern = 'O*.txt'
    elif set_name == 'E':
        pattern = 'S*.txt'
    else:
        raise ValueError(f"Unknown set name: {set_name}")
    
    file_paths = sorted(glob.glob(os.path.join(set_dir, pattern)))
    
    if len(file_paths) == 0:
        raise ValueError(f"No files found in {set_dir} with pattern {pattern}")
    
    return file_paths


def load_experiment_data(data_dir, set1_name, set2_name, 
                         original_fs=173.61, target_fs=100, 
                         segment_length=100, apply_filtering=True):
    
    # Load files for both sets
    set1_files = load_bonn_files(data_dir, set1_name)
    set2_files = load_bonn_files(data_dir, set2_name)
    
    print(f"Loading {len(set1_files)} files from set {set1_name}")
    print(f"Loading {len(set2_files)} files from set {set2_name}")
    
    # Preprocess set 1 (label = 0, non-seizure)
    set1_segments, set1_file_indices = preprocess_dataset(
        set1_files, original_fs, target_fs, segment_length, apply_filtering
    )
    set1_labels = np.zeros(len(set1_segments), dtype=np.int64)
    
    # Preprocess set 2 (label = 1, seizure)
    set2_segments, set2_file_indices = preprocess_dataset(
        set2_files, original_fs, target_fs, segment_length, apply_filtering
    )
    # Adjust file indices for set 2
    set2_file_indices = set2_file_indices + len(set1_files)
    set2_labels = np.ones(len(set2_segments), dtype=np.int64)
    
    # Combine both sets
    all_segments = np.vstack([set1_segments, set2_segments])
    all_labels = np.concatenate([set1_labels, set2_labels])
    all_file_indices = np.concatenate([set1_file_indices, set2_file_indices])
    
    # Create file to set mapping
    file_to_set_map = {}
    for i in range(len(set1_files)):
        file_to_set_map[i] = set1_name
    for i in range(len(set2_files)):
        file_to_set_map[len(set1_files) + i] = set2_name
    
    print(f"Total segments: {len(all_segments)}")
    print(f"Set {set1_name}: {len(set1_segments)} segments")
    print(f"Set {set2_name}: {len(set2_segments)} segments")
    
    return all_segments, all_labels, all_file_indices, file_to_set_map


def create_participant_level_splits(segments, labels, file_indices, n_splits=3, random_state=42):
    
    # Get unique file indices
    unique_files = np.unique(file_indices)
    
    # Create K-Fold split on files
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_file_idx, test_file_idx) in enumerate(kf.split(unique_files)):
        train_files = unique_files[train_file_idx]
        test_files = unique_files[test_file_idx]
        
        # Get segments belonging to train and test files
        train_mask = np.isin(file_indices, train_files)
        test_mask = np.isin(file_indices, test_files)
        
        train_segments = segments[train_mask]
        train_labels = labels[train_mask]
        test_segments = segments[test_mask]
        test_labels = labels[test_mask]
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train files: {len(train_files)}, segments: {len(train_segments)}")
        print(f"  Test files: {len(test_files)}, segments: {len(test_segments)}")
        print(f"  Train class distribution: {np.bincount(train_labels)}")
        print(f"  Test class distribution: {np.bincount(test_labels)}")
        
        yield train_segments, train_labels, test_segments, test_labels, test_file_idx


def create_data_loaders(train_segments, train_labels, train_file_indices,
                       test_segments, test_labels, test_file_indices,
                       batch_size=12, num_workers=0):
    
    train_dataset = BonnEEGDataset(train_segments, train_labels, train_file_indices)
    test_dataset = BonnEEGDataset(test_segments, test_labels, test_file_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_experiment_data_loaders(data_dir, experiment_type='A_vs_E', 
                               fold_idx=0, n_splits=3, batch_size=12,
                               random_state=42):
    
    # Parse experiment type
    if experiment_type == 'A_vs_E':
        set1, set2 = 'A', 'E'
    elif experiment_type == 'B_vs_E':
        set1, set2 = 'B', 'E'
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Load data
    segments, labels, file_indices, file_to_set_map = load_experiment_data(
        data_dir, set1, set2
    )
    
    # Get the specific fold
    splits = list(create_participant_level_splits(
        segments, labels, file_indices, n_splits, random_state
    ))
    
    train_segments, train_labels, test_segments, test_labels, _ = splits[fold_idx]
    
    # Note: We need to recreate file_indices for train/test
    # For simplicity, we'll use dummy indices since we only need them for dataset
    train_file_indices = np.arange(len(train_segments))
    test_file_indices = np.arange(len(test_segments))
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_segments, train_labels, train_file_indices,
        test_segments, test_labels, test_file_indices,
        batch_size=batch_size
    )
    
    return train_loader, test_loader
