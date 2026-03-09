"""
Training module for the audio stress detection model.
Handles dataset creation, model training, evaluation, and cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from typing import Tuple, List, Dict, Any, Optional
import time

from .config import CONFIG
from .stresslstm import EnhancedStressLSTM
from .features import AudioFeatures
from .augmentation import AudioAugmenter
from .embedding import AudioEmbedding
from .utils import ensure_audio_contiguous, validate_audio_chunk


class StressDataset(Dataset):
    """
    Enhanced stress dataset with audio augmentation capabilities for training.
    
    Loads audio data from CSV files and additional folders, extracts features,
    and applies data augmentation to improve model robustness.
    """
    
    def __init__(self, 
                 csv_path: str = None,
                 audio_folder: str = None,
                 additional_folders: Dict[str, str] = None,
                 chunk_duration: float = None,
                 sample_rate: int = None,
                 augment_data: bool = None,
                 augmentation_factor: int = None,
                 feature_set: str = None):
        """
        Initialize the stress dataset.
        
        Args:
            csv_path: Path to CSV file with audio metadata
            audio_folder: Path to folder containing audio files
            additional_folders: Dictionary mapping labels to additional audio folders
            chunk_duration: Duration of each audio chunk in seconds
            sample_rate: Sampling rate
            augment_data: Whether to augment data during loading
            augmentation_factor: How many augmented samples to create per original sample
            feature_set: Type of features to extract
        """
        # Use config defaults
        self.sample_rate = sample_rate or CONFIG.audio.sample_rate
        self.chunk_duration = chunk_duration or CONFIG.audio.chunk_duration
        self.feature_set = feature_set or CONFIG.features.feature_set
        self.augment_data = augment_data if augment_data is not None else CONFIG.augmentation.augment_data
        self.augmentation_factor = augmentation_factor or CONFIG.augmentation.augmentation_factor
        
        self.chunk_size = int(self.chunk_duration * self.sample_rate)
        self.data = []
        self.feature_names = None
        
        # Initialize components
        self.feature_extractor = AudioFeatures(
            sample_rate=self.sample_rate,
            feature_set=self.feature_set
        )
        
        self.embedding_model = AudioEmbedding()
        if not self.embedding_model.load_model():
            raise RuntimeError("Failed to load embedding model for dataset creation")
        
        if self.augment_data:
            self.augmenter = AudioAugmenter(sr=self.sample_rate)
            print(f"Data augmentation enabled with factor {self.augmentation_factor}")
        
        # Load data
        self._load_data_from_sources(csv_path, audio_folder, additional_folders)
        
        # Post-processing validation
        self._validate_and_clean_data()
        
        if not self.data:
            raise ValueError("No valid data loaded. Check paths, files, and extraction logic.")
        
        print(f"Dataset initialization complete. Total samples: {len(self.data)}")
        if self.feature_names:
            print(f"Number of audio features: {len(self.feature_names)}")
    
    def _load_data_from_sources(self, csv_path: str, audio_folder: str, additional_folders: Dict[str, str]):
        """
        Loads data from CSV and additional folder sources.
        
        Args:
            csv_path: Path to CSV file
            audio_folder: Path to audio folder
            additional_folders: Dictionary of additional folders by label
        """
        # Load CSV data
        if csv_path and audio_folder:
            self._load_csv_data(csv_path, audio_folder)
        
        # Load additional folders
        if additional_folders:
            self._load_additional_folders(additional_folders)
    
    def _load_csv_data(self, csv_path: str, audio_folder: str):
        """
        Loads data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            audio_folder: Path to audio folder
        """
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        print("Loading data from CSV...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV files"):
            audio_path = os.path.join(audio_folder, f"{row['filename']}.wav")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found {audio_path}, skipping.")
                continue
            
            try:
                start_time = row['start_time']
                end_time = row['end_time']
                duration = end_time - start_time
                
                if duration <= 0:
                    continue
                
                audio, _ = librosa.load(audio_path, sr=self.sample_rate, 
                                     offset=start_time, duration=duration)
                
                if len(audio) < 100:
                    continue
                
                self._extract_chunks(audio, row['stress'], row['filename'])
                
            except Exception as e:
                print(f"Error processing CSV entry for {audio_path}: {e}, skipping.")
    
    def _load_additional_folders(self, additional_folders: Dict[str, str]):
        """
        Loads data from additional folders.
        
        Args:
            additional_folders: Dictionary mapping labels to folder paths
        """
        print("Loading data from additional folders...")
        
        for label, folder in additional_folders.items():
            if not os.path.exists(folder):
                print(f"Warning: Additional folder not found: {folder}")
                continue
            
            audio_files = glob.glob(os.path.join(folder, '*.wav')) + \
                         glob.glob(os.path.join(folder, '*.mp3'))
            
            if not audio_files:
                print(f"Warning: No audio files found in {folder}")
                continue
            
            for audio_path in tqdm(audio_files, desc=f"Processing Label {label} Folder"):
                try:
                    audio, _ = librosa.load(audio_path, sr=self.sample_rate)
                    if len(audio) < self.chunk_size:
                        continue
                    
                    filename = os.path.basename(audio_path)
                    self._extract_chunks(audio, int(label), filename)
                    
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}, skipping.")
    
    def _extract_chunks(self, audio: np.ndarray, label: int, filename: str):
        """
        Extracts chunks from audio and optionally creates augmented versions.
        
        Args:
            audio: Audio signal
            label: Class label (0 or 1)
            filename: Base filename
        """
        audio = ensure_audio_contiguous(audio)
        hop_size = int(0.5 * self.sample_rate)
        
        for start in range(0, len(audio) - self.chunk_size + 1, hop_size):
            chunk = audio[start:start+self.chunk_size]
            chunk = ensure_audio_contiguous(chunk)
            
            # Skip silent chunks
            if not validate_audio_chunk(chunk):
                continue
            
            # Process original chunk
            self._process_chunk(chunk, label, f"{filename}_{start}")
            
            # Create augmented versions if enabled
            if self.augment_data:
                num_augmentations = max(self.augmentation_factor, 2 if label == 1 else 1)
                
                for i in range(num_augmentations):
                    try:
                        # Apply stress-specific augmentation
                        augmented_chunk = self.augmenter.apply_stress_specific_augmentation(
                            chunk, is_stress=(label == 1)
                        )
                        augmented_chunk = ensure_audio_contiguous(augmented_chunk)
                        
                        self._process_chunk(
                            augmented_chunk, 
                            label, 
                            f"{filename}_{start}_aug{i}"
                        )
                    except Exception as e:
                        print(f"Augmentation failed for {filename}_{start}_aug{i}: {e}")
                        continue
    
    def _process_chunk(self, chunk: np.ndarray, label: int, chunk_id: str):
        """
        Processes a single audio chunk: extracts embeddings and features.
        
        Args:
            chunk: Audio chunk
            label: Class label
            chunk_id: Identifier for this chunk
        """
        try:
            # Extract embedding
            embedding = self.embedding_model.extract_embedding_from_numpy(chunk, self.sample_rate)
            if embedding is None:
                return
            
            # Extract audio features
            raw_audio_features, feature_names = self.feature_extractor.extract_audio_features(
                chunk, self.sample_rate, self.feature_set
            )
            
            if raw_audio_features is None:
                return
            
            # Store feature names on first successful extraction
            if self.feature_names is None:
                self.feature_names = feature_names
            
            # Store the data (embedding, raw_audio_features, chunk_id, label)
            self.data.append((
                embedding.cpu().numpy(),
                raw_audio_features,
                chunk_id,
                label
            ))
            
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
    
    def _validate_and_clean_data(self):
        """
        Validates and cleans the loaded data.
        """
        if not self.feature_names:
            return
        
        first_len = len(self.feature_names)
        inconsistent_count = 0
        filtered_data = []
        
        for item in self.data:
            if len(item[1]) == first_len:  # Index 1 is raw_audio_features
                filtered_data.append(item)
            else:
                inconsistent_count += 1
        
        if inconsistent_count > 0:
            print(f"Warning: Removed {inconsistent_count} samples due to inconsistent feature vector lengths.")
            self.data = filtered_data
    
    def __len__(self) -> int:
        """
        Returns the size of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Gets a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (embedding, audio_features, label)
        """
        embedding, audio_features, chunk_id, label = self.data[idx]
        
        # Convert to tensors
        embedding_tensor = torch.from_numpy(embedding).float()
        audio_features_tensor = torch.from_numpy(audio_features).float()
        
        return embedding_tensor, audio_features_tensor, label
    
    @property
    def groups(self) -> List[str]:
        """
        Returns group identifiers for GroupKFold cross-validation.
        
        Returns:
            List of group identifiers
        """
        return [item[2].split('_')[0] for item in self.data]  # Use filename as group
    
    @property
    def feature_dimensions(self) -> Tuple[int, int]:
        """
        Returns the dimensions of embedding and audio features.
        
        Returns:
            Tuple of (embedding_dim, audio_features_dim)
        """
        if not self.data:
            return 0, 0
        
        embedding_dim = self.data[0][0].shape[0]
        audio_features_dim = self.data[0][1].shape[0]
        
        return embedding_dim, audio_features_dim
    
    def get_raw_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns raw audio features and labels for analysis.
        
        Returns:
            Tuple of (features_array, labels_array)
        """
        if not self.data:
            return np.array([]), np.array([])
        
        features = np.array([item[1] for item in self.data])
        labels = np.array([item[3] for item in self.data])
        
        return features, labels
    
    def class_distribution(self) -> Dict[str, Any]:
        """
        Returns class distribution statistics.
        
        Returns:
            Dictionary with class distribution information
        """
        if not self.data:
            return {}
        
        labels = [item[3] for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        
        return {
            'total_samples': len(labels),
            'class_counts': dict(zip(unique, counts)),
            'class_percentages': dict(zip(unique, counts / len(labels) * 100))
        }


class FoldDataset(Dataset):
    """
    Simple dataset wrapper for cross-validation folds with pre-processed data.
    """
    
    def __init__(self, embeddings: np.ndarray, audio_features: np.ndarray, labels: np.ndarray):
        """
        Initialize fold dataset.
        
        Args:
            embeddings: Embedding features array
            audio_features: Audio features array (already scaled)
            labels: Labels array
        """
        self.embeddings = embeddings
        self.audio_features = audio_features
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.embeddings[idx]).float(),
            torch.from_numpy(self.audio_features[idx]).float(),
            torch.tensor(self.labels[idx]).float()
        )


class StressModelTrainer:
    """
    Main trainer class for the stress detection model.
    """
    
    def __init__(self, config=None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object (uses global CONFIG if None)
        """
        self.config = config or CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer using device: {self.device}")
    
    def train_and_evaluate_model(self, 
                                dataset: StressDataset,
                                epochs: int = None,
                                batch_size: int = None,
                                n_splits: int = None,
                                learning_rate: float = None) -> Optional[EnhancedStressLSTM]:
        """
        Trains and evaluates the stress detection model using cross-validation.
        
        Args:
            dataset: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            n_splits: Number of cross-validation splits
            learning_rate: Learning rate for optimizer
            
        Returns:
            Trained model or None if training fails
        """
        # Use config defaults
        epochs = epochs or self.config.training.epochs
        batch_size = batch_size or self.config.training.batch_size
        n_splits = n_splits or self.config.training.n_splits
        learning_rate = learning_rate or self.config.training.learning_rate
        
        embedding_dim, audio_features_dim = dataset.feature_dimensions
        if audio_features_dim == 0:
            raise ValueError("Audio features dimension is 0.")
        
        # Update config with determined dimensions
        self.config.update_audio_features_dim(audio_features_dim)
        
        indices = np.arange(len(dataset))
        y = np.array([dataset.data[i][3] for i in indices])
        groups = np.array(dataset.groups)
        
        gkf = GroupKFold(n_splits=n_splits)
        fold_metrics = {'acc': [], 'f1': [], 'auc': []}
        best_model_state = None
        best_avg_auc = 0
        
        print(f"Starting {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(indices, y, groups=groups)):
            print(f"\n--- Fold {fold+1}/{n_splits} ---")
            
            # Prepare fold data
            fold_results = self._train_single_fold(
                dataset, train_idx, test_idx, fold,
                epochs, batch_size, learning_rate,
                embedding_dim, audio_features_dim
            )
            
            if fold_results:
                acc, f1, auc, model_state = fold_results
                fold_metrics['acc'].append(acc)
                fold_metrics['f1'].append(f1)
                fold_metrics['auc'].append(auc)
                
                # Track best model
                if auc > best_avg_auc:
                    best_avg_auc = auc
                    best_model_state = model_state
                    print(f"  --> New overall best model found (AUC: {auc:.4f})!")
            else:
                print(f"Warning: Fold {fold+1} failed")
                fold_metrics['acc'].append(0.0)
                fold_metrics['f1'].append(0.0)
                fold_metrics['auc'].append(0.0)
        
        # Print final results
        self._print_cv_results(fold_metrics)
        
        # Save best model and return it
        if best_model_state:
            final_model = self._save_and_return_best_model(
                best_model_state, embedding_dim, audio_features_dim, best_avg_auc
            )
            
            # Save feature names
            if dataset.feature_names:
                feature_extractor = AudioFeatures()
                feature_extractor.save_features_names(dataset.feature_names)
            
            return final_model
        
        return None
    
    def _train_single_fold(self, 
                          dataset: StressDataset,
                          train_idx: np.ndarray,
                          test_idx: np.ndarray,
                          fold: int,
                          epochs: int,
                          batch_size: int,
                          learning_rate: float,
                          embedding_dim: int,
                          audio_features_dim: int) -> Optional[Tuple[float, float, float, dict]]:
        """
        Trains a single fold of cross-validation.
        
        Returns:
            Tuple of (accuracy, f1, auc, best_model_state) or None if failed
        """
        try:
            # Get raw data for this fold
            X_emb_train = np.array([dataset.data[i][0] for i in train_idx])
            X_audio_train_raw = np.array([dataset.data[i][1] for i in train_idx])
            y_train = np.array([dataset.data[i][3] for i in train_idx])
            
            X_emb_test = np.array([dataset.data[i][0] for i in test_idx])
            X_audio_test_raw = np.array([dataset.data[i][1] for i in test_idx])
            y_test = np.array([dataset.data[i][3] for i in test_idx])
            
            # Fit scaler on train data only
            scaler = StandardScaler()
            X_audio_train_scaled = scaler.fit_transform(X_audio_train_raw)
            X_audio_test_scaled = scaler.transform(X_audio_test_raw)
            
            # Create datasets and data loaders
            train_fold_dataset = FoldDataset(X_emb_train, X_audio_train_scaled, y_train)
            test_fold_dataset = FoldDataset(X_emb_test, X_audio_test_scaled, y_test)
            
            train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_fold_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # Initialize model and training components
            model = EnhancedStressLSTM(
                embedding_dim=embedding_dim,
                audio_features_dim=audio_features_dim
            ).to(self.device)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.2, patience=3, verbose=False
            )
            
            # Training loop
            best_fold_auc = 0
            patience = 0
            patience_limit = self.config.training.patience_limit
            best_model_state_fold = None
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for emb_batch, audio_feat_batch, labels_batch in tqdm(
                    train_loader, desc=f"Epoch {epoch+1} Train", leave=False
                ):
                    emb = emb_batch.unsqueeze(1).float().to(self.device)
                    audio_feat = audio_feat_batch.unsqueeze(1).float().to(self.device)
                    labels = labels_batch.float().unsqueeze(1).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(emb, audio_feat)
                    loss = criterion(outputs, labels)
                    
                    if not torch.isfinite(loss):
                        print(f"Warning: Non-finite loss in Fold {fold+1}, Epoch {epoch+1}")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config.training.max_grad_norm)
                    optimizer.step()
                    train_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
                
                # Validation phase
                model.eval()
                fold_preds = []
                fold_labels = []
                
                with torch.no_grad():
                    for emb_batch, audio_feat_batch, labels_batch in test_loader:
                        emb = emb_batch.unsqueeze(1).float().to(self.device)
                        audio_feat = audio_feat_batch.unsqueeze(1).float().to(self.device)
                        outputs = model(emb, audio_feat)
                        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                        fold_preds.extend(probs)
                        fold_labels.extend(labels_batch.numpy())
                
                fold_preds = np.array(fold_preds)
                fold_labels = np.array(fold_labels)
                preds_binary = (fold_preds >= 0.5).astype(int)
                
                # Calculate metrics
                acc = accuracy_score(fold_labels, preds_binary)
                f1 = f1_score(fold_labels, preds_binary)
                try:
                    auc = roc_auc_score(fold_labels, fold_preds)
                except ValueError:
                    auc = 0.5
                
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
                scheduler.step(auc)
                
                # Early stopping and checkpointing
                if auc > best_fold_auc:
                    best_fold_auc = auc
                    patience = 0
                    best_model_state_fold = model.state_dict().copy()
                    print(f"  -> New best AUC for Fold {fold+1}: {best_fold_auc:.4f}")
                else:
                    patience += 1
                    if patience >= patience_limit:
                        print(f"Early stopping triggered at epoch {epoch+1} for Fold {fold+1}")
                        break
            
            # Final evaluation with best model
            if best_model_state_fold is not None:
                model.load_state_dict(best_model_state_fold)
                final_acc, final_f1, final_auc = self._evaluate_model(model, test_loader)
                print(f"Fold {fold+1} Best Val Metrics -> Acc: {final_acc:.4f}, F1: {final_f1:.4f}, AUC: {final_auc:.4f}")
                return final_acc, final_f1, final_auc, best_model_state_fold
            
            return None
            
        except Exception as e:
            print(f"Error in fold {fold+1}: {e}")
            return None
    
    def _evaluate_model(self, model: EnhancedStressLSTM, data_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluates the model on a data loader.
        
        Returns:
            Tuple of (accuracy, f1_score, auc_score)
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for emb_batch, audio_feat_batch, labels_batch in data_loader:
                emb = emb_batch.unsqueeze(1).float().to(self.device)
                audio_feat = audio_feat_batch.unsqueeze(1).float().to(self.device)
                outputs = model(emb, audio_feat)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_preds.extend(probs)
                all_labels.extend(labels_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        preds_binary = (all_preds >= 0.5).astype(int)
        
        acc = accuracy_score(all_labels, preds_binary)
        f1 = f1_score(all_labels, preds_binary)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5
        
        return acc, f1, auc
    
    def _print_cv_results(self, fold_metrics: Dict[str, List[float]]):
        """
        Prints cross-validation results.
        
        Args:
            fold_metrics: Dictionary containing metrics for each fold
        """
        print("\n--- Cross-Validation Summary ---")
        mean_acc, std_acc = np.mean(fold_metrics['acc']), np.std(fold_metrics['acc'])
        mean_f1, std_f1 = np.mean(fold_metrics['f1']), np.std(fold_metrics['f1'])
        mean_auc, std_auc = np.mean(fold_metrics['auc']), np.std(fold_metrics['auc'])
        
        print(f"Average Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"Average AUC:      {mean_auc:.4f} ± {std_auc:.4f}")
    
    def _save_and_return_best_model(self, 
                                   best_model_state: dict,
                                   embedding_dim: int,
                                   audio_features_dim: int,
                                   best_auc: float) -> EnhancedStressLSTM:
        """
        Saves the best model and returns the instantiated model.
        
        Returns:
            The best trained model
        """
        model_save_path = self.config.predictor.model_path
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Save with additional metadata
        save_dict = {
            'model_state_dict': best_model_state,
            'embedding_dim': embedding_dim,
            'audio_features_dim': audio_features_dim,
            'best_auc': best_auc,
            'config': {
                'feature_set': self.config.features.feature_set,
                'sample_rate': self.config.audio.sample_rate,
                'chunk_duration': self.config.audio.chunk_duration
            }
        }
        
        torch.save(save_dict, model_save_path)
        print(f"Best model saved to {model_save_path} (AUC: {best_auc:.4f})")
        
        # Return instantiated model
        final_model = EnhancedStressLSTM(embedding_dim, audio_features_dim).to(self.device)
        final_model.load_state_dict(best_model_state)
        final_model.eval()
        
        return final_model


# Convenience functions
def create_dataset(**kwargs) -> StressDataset:
    """
    Creates a stress dataset with default configuration.
    
    Returns:
        Configured StressDataset instance
    """
    return StressDataset(**kwargs)


def train_model(dataset: StressDataset, **kwargs) -> Optional[EnhancedStressLSTM]:
    """
    Trains a stress detection model.
    
    Args:
        dataset: Training dataset
        **kwargs: Additional training parameters
        
    Returns:
        Trained model or None if training fails
    """
    trainer = StressModelTrainer()
    return trainer.train_and_evaluate_model(dataset, **kwargs) 