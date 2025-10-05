"""
Development PipelineSepsis Precision Management Model based on the bulk RNA-Seq matrix of THREE most likely causal genes identified by genetic association study.
!!!WE ARE DEDICATED TO PROMOTE GLOBAL HEALTH EQUITY BY PROVIDING NOVEL DISEASE RISK PREDICTION PANEL AND THERAPY WHICH EVERYONE ACCESSIBLE!!!
#################################################################################################
######################Interdisciplinary Research For Better Lives################################
#################################################################################################
Developed by <Dr. Adrian Tang & Dr. Jasper Chu's collaborative interdisciplinary research team>
Dr. Adrian would like to acknowledge Mrs. Minjie Huang for her technical support.
If you have any questions, feel free to contact Dr. Adrian at 039319101@njucm.edu.cn
"""
import os
import sys
import warnings
import logging
import math
import random
import itertools
from typing import Dict, List, Tuple
from pathlib import Path
import copy
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, recall_score, precision_score, confusion_matrix,
    roc_curve
)
from imblearn.over_sampling import BorderlineSMOTE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau,
    CosineAnnealingLR
)
import optuna
from datetime import datetime
import gc
from collections import defaultdict
import time

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# ===============================================
# 1. Config
# ===============================================

class Config:
    DATA_PATH = r"<The path which the gene expression data is stored.>"
    OUTPUT_DIR = r"<The path which the result is stored.>"

    SELECTED_GENES = ['NEU1', 'CLIC1', 'FLOT1']

    N_FOLDS = 3
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIXED_PRECISION = True
    RANDOM_SEED = 1234 # 42 in postoperative sepsis risk prediciton; 720214 in sepsis diagnosis model; 650522 in sepsis prognosis model

    MODELS = [
        'Transformer',
        'Tabnet',
        'CNN_RNN_Attention',
        'Self_Supervised_Model',
    ]

    # Feature selection configuration - Phase 1
    NUM_FEATURES_TO_SELECT = 3
    FEATURE_SELECTION_TRIALS = 100  # Trials per combination-model pair in Phase 1

    # Main hyperparameter optimization of models with best panel configuration - Phase 2
    TARGET_AUROC = 0.90
    MAX_OPTIMIZATION_ROUNDS = 100
    INITIAL_N_TRIALS = 100
    MAX_N_TRIALS = 100
    TRIAL_INCREMENT = 0

    OPTUNA_TIMEOUT = 10800
    MAX_EPOCHS = 100
    PATIENCE = 20
    WARMUP_EPOCHS = 50
    MIN_LR = 1e-9
    MAX_LR = 0.2
    AUGMENTATION_FACTOR = 1.2
    NOISE_STD = 0.0003
    QUALITY_THRESHOLD = 0.8
    N_BOOTSTRAP = 1000
    CONFIDENCE_LEVEL = 0.95


def setup_logging():
    log_dir = Path(Config.OUTPUT_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"exhaustive_feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ===============================================
# 2. DataProcessor
# ===============================================

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.selected_features = None
        self.feature_names = None
        self.all_feature_names = None
        self.logger = logging.getLogger(__name__)

    def load_and_validate_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        try:
            df = pd.read_excel(self.config.DATA_PATH)
            self.logger.info(f"Data shape: {df.shape}")
            missing_genes = [gene for gene in self.config.SELECTED_GENES if gene not in df.columns]
            if missing_genes:
                raise ValueError(f"Missing gene: {missing_genes}")
            if 'Group' not in df.columns:
                raise ValueError("'Group' column not found")
            feature_cols = self.config.SELECTED_GENES
            for col in feature_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            X_raw = df[feature_cols].values.astype(np.float64)
            y = (df['Group'] == '').astype(int).values #'Deceased' in prognosis prediction model,'Sepsis' in risk and diagnosis prediction model
            for i in range(X_raw.shape[1]):
                Q1 = np.percentile(X_raw[:, i], 25)
                Q3 = np.percentile(X_raw[:, i], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                X_raw[:, i] = np.clip(X_raw[:, i], lower_bound, upper_bound)

            outlier_mask = np.ones(len(X_raw), dtype=bool)
            self.logger.info(f"Data shape after processing: {X_raw.shape}")
            return df, X_raw, y, outlier_mask
        except Exception as e:
            self.logger.error(f"Error in data loading: {e}")
            raise

    def feature_engineering(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        features = []
        feature_names = []

        features.append(X)
        feature_names.extend([f'original_{gene}' for gene in self.config.SELECTED_GENES])
        gene_names = self.config.SELECTED_GENES

        for i in range(len(gene_names)):
            for j in range(i + 1, len(gene_names)):
                ratio = X[:, i] / (X[:, j] + 1e-8)
                features.append(ratio.reshape(-1, 1))
                feature_names.append(f'ratio_{gene_names[i]}_{gene_names[j]}')

        mean_feat = np.mean(X, axis=1, keepdims=True)
        std_feat = np.std(X, axis=1, keepdims=True)
        features.extend([mean_feat, std_feat])
        feature_names.extend(['mean_expression', 'std_expression'])

        for i, gene in enumerate(gene_names):
            log_feat = np.log1p(np.abs(X[:, i]))
            features.append(log_feat.reshape(-1, 1))
            feature_names.append(f'log_{gene}')

        X_engineered = np.concatenate(features, axis=1)

        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=1e-4)
        X_engineered = var_selector.fit_transform(X_engineered)

        valid_features = var_selector.get_support()
        feature_names = [name for name, valid in zip(feature_names, valid_features) if valid]

        self.logger.info(f"Shape of engineered features: {X_engineered.shape}")
        self.all_feature_names = feature_names
        return X_engineered.astype(np.float32), feature_names

    def apply_feature_selection(self, X: np.ndarray, selected_indices: List[int]) -> np.ndarray:
        return X[:, selected_indices]

    def stratified_split_with_validation(self, X: np.ndarray, y: np.ndarray,
                                         outlier_mask: np.ndarray = None) -> Dict:
        if outlier_mask is not None:
            X = X[outlier_mask]
            y = y[outlier_mask]

        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.config.TEST_RATIO,
            random_state=self.config.RANDOM_SEED
        )
        train_val_idx, test_idx = next(sss1.split(X, y))

        val_size = self.config.VAL_RATIO / (self.config.TRAIN_RATIO + self.config.VAL_RATIO)
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=self.config.RANDOM_SEED
        )
        train_idx, val_idx = next(sss2.split(X[train_val_idx], y[train_val_idx]))

        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_SEED
        )
        cv_folds = list(skf.split(X[train_idx], y[train_idx]))

        splits = {
            'train': (X[train_idx], y[train_idx]),
            'val': (X[val_idx], y[val_idx]),
            'test': (X[test_idx], y[test_idx]),
            'cv_folds': cv_folds,
            'train_idx': train_idx,
            'all_data': (X, y)
        }

        for split_name, (X_split, y_split) in [(k, v) for k, v in splits.items() if
                                               k not in ['cv_folds', 'train_idx', 'all_data']]:
            sepsis_count = np.sum(y_split)
            control_count = len(y_split) - sepsis_count
            self.logger.info(
                f"{split_name.upper()} - Total: {len(y_split)}, Cases: {sepsis_count}, Controls: {control_count}")

        return splits

    def fit_transformers(self, X_train: np.ndarray):
        self.scaler = RobustScaler()
        self.scaler.fit(X_train)
        self.logger.info("Data transformer fitted")

    def transform_data(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Data transformer not fitted")
        X_scaled = self.scaler.transform(X)
        return X_scaled.astype(np.float32)


# ===============================================
# 3. Data Augmenter
# ===============================================

class Augmenter:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def assess_quality(self, X_original: np.ndarray, X_synthetic: np.ndarray,
                       y_original: np.ndarray, y_synthetic: np.ndarray) -> float:
        try:
            quality_scores = []
            for i in range(X_original.shape[1]):
                orig_feature = X_original[:, i]
                synth_feature = X_synthetic[:, i]
                try:
                    ks_stat, _ = stats.ks_2samp(orig_feature, synth_feature)
                    ks_similarity = 1 - ks_stat
                except:
                    ks_similarity = 0.5
                quality_scores.append(ks_similarity)

            original_ratio = np.mean(y_original)
            synthetic_ratio = np.mean(y_synthetic)
            ratio_similarity = 1 - abs(original_ratio - synthetic_ratio)

            feature_quality = np.mean(quality_scores)
            overall_quality = 0.7 * feature_quality + 0.3 * ratio_similarity
            return max(0, min(1, overall_quality))
        except Exception as e:
            self.logger.warning(f"Quality evaluation failed: {e}")
            return 0.5

    def smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(np.unique(y)) < 2:
            return X, y

        minority_class = np.argmin(np.bincount(y.astype(int)))
        minority_count = np.sum(y == minority_class)
        majority_count = np.sum(y != minority_class)

        if minority_count < 3:
            return X, y

        target_samples = min(
            int(minority_count * self.config.AUGMENTATION_FACTOR),
            int(majority_count * 0.5)
        )

        if target_samples <= minority_count:
            return X, y

        try:
            smote = BorderlineSMOTE(
                sampling_strategy={minority_class: target_samples},
                k_neighbors=min(3, minority_count - 1),
                random_state=self.config.RANDOM_SEED,
                kind='borderline-1'
            )

            X_resampled, y_resampled = smote.fit_resample(X, y)
            new_samples_mask = np.arange(len(X_resampled)) >= len(X)
            X_new = X_resampled[new_samples_mask]
            y_new = y_resampled[new_samples_mask]

            quality_score = self.assess_quality(X, X_new, y, y_new)

            if quality_score >= self.config.QUALITY_THRESHOLD:
                self.logger.info(f"SMOTE quality score: {quality_score:.3f}")
                return X_resampled, y_resampled
            else:
                self.logger.warning(f"SMOTE quality low ({quality_score:.3f}), using original data")
                return X, y

        except Exception as e:
            self.logger.warning(f"SMOTE failed: {e}")
            return X, y

    def augment_training_data(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        original_size = len(X_train)

        X_final, y_final = self.smote(X_train, y_train)

        final_size = len(X_final)
        augmentation_ratio = final_size / original_size

        unique, counts = np.unique(y_final.astype(int), return_counts=True)

        return X_final.astype(np.float32), y_final.astype(np.float32)


# ===============================================
# 4. Model architectures
# ===============================================

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.register_buffer('positional_encoding', self._create_positional_encoding(input_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        attention_weights = torch.softmax(self.attention_pool(x), dim=1)
        x = torch.sum(attention_weights * x, dim=1)
        output = self.classifier(x)
        return output.squeeze(-1)


class TabNet(nn.Module):
    def __init__(self, input_dim, n_d=32, n_a=32, n_steps=3, gamma=1.3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU()
        )

        self.feature_transformers = nn.ModuleList()
        self.attention_transformers = nn.ModuleList()

        for step in range(n_steps):
            feat_transformer = nn.Sequential(
                nn.Linear(input_dim, n_d + n_a),
                nn.BatchNorm1d(n_d + n_a),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.feature_transformers.append(feat_transformer)

            att_transformer = nn.Sequential(
                nn.Linear(n_a if step > 0 else input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.attention_transformers.append(att_transformer)

        self.decision_predictors = nn.ModuleList()
        for _ in range(n_steps):
            predictor = nn.Sequential(
                nn.Linear(n_d, n_d),
                nn.BatchNorm1d(n_d),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.decision_predictors.append(predictor)

        self.final_classifier = nn.Sequential(
            nn.Linear(n_d, n_d),
            nn.BatchNorm1d(n_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_d, n_d // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_d // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.initial_bn(x)
        x = self.initial_projection(x)

        decision_out = torch.zeros(x.size(0), self.n_d, device=x.device)
        prior_scales = torch.ones(x.size(0), self.input_dim, device=x.device)

        for step in range(self.n_steps):
            features = self.feature_transformers[step](x * prior_scales)

            d = features[:, :self.n_d]
            a = features[:, self.n_d:]

            if step == 0:
                att_input = x
            else:
                att_input = a

            M = torch.softmax(self.attention_transformers[step](att_input), dim=-1)

            prior_scales = prior_scales * (self.gamma - M)
            prior_scales = torch.clamp(prior_scales, min=0.1)

            decision_step = self.decision_predictors[step](d)
            decision_out = decision_out + decision_step

        output = self.final_classifier(decision_out)
        return output.squeeze(-1)


class CNNRNNAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, num_layers=2, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
            )
        ])

        self.feature_fusion = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.pos_encoding_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        batch_size = x.size(0)
        x_input = x.unsqueeze(1)

        conv_features = []
        for conv_block in self.conv_blocks:
            conv_feat = conv_block(x_input)
            conv_features.append(conv_feat)

        combined_conv = torch.cat(conv_features, dim=1)
        fused_features = self.feature_fusion(combined_conv)
        seq_features = fused_features.transpose(1, 2)

        lstm_out, _ = self.lstm(seq_features)
        gru_out, _ = self.gru(seq_features)
        combined_rnn = torch.cat([lstm_out, gru_out], dim=-1)

        pos_encoding = self.pos_encoding_layer(combined_rnn)
        combined_rnn = combined_rnn + pos_encoding

        attn_out, _ = self.self_attention(combined_rnn, combined_rnn, combined_rnn)
        pooled_output = torch.mean(attn_out, dim=1)
        output = self.classifier(pooled_output)
        return output.squeeze(-1)


class SelfSupervisedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, projection_dim=64):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, projection_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 8, 1)
        )

        self.pretrained = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x, return_features=False, ssl_mode=False):
        features = self.backbone(x)

        if ssl_mode:
            projections = self.projection_head(features)
            predictions = self.prediction_head(projections)
            return features, projections, predictions

        if return_features:
            projections = self.projection_head(features)
            return features, projections

        logits = self.classifier(features)
        return logits.squeeze(-1)

    def ssl_loss(self, z1, z2, p1, p2, temperature=0.5):
        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        p1_norm = F.normalize(p1, dim=1)
        p2_norm = F.normalize(p2, dim=1)

        sim1 = -(p1_norm * z2_norm.detach()).sum(dim=1).mean()
        sim2 = -(p2_norm * z1_norm.detach()).sum(dim=1).mean()

        return (sim1 + sim2) * 0.5


# ===============================================
# 5. Loss Functions and Trainer
# ===============================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


class BalancedLoss(nn.Module):
    def __init__(self, focal_weight=0.6, bce_weight=0.4, label_smoothing=0.1):
        super().__init__()
        self.focal_loss = FocalLoss(label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        targets = targets.float()
        focal = self.focal_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.focal_weight * focal + self.bce_weight * bce


class Trainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.scaler = GradScaler() if config.MIXED_PRECISION else None
        self.criterion = BalancedLoss()
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            target = target.float()

            optimizer.zero_grad()

            if self.config.MIXED_PRECISION and self.scaler:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()

            if scheduler and hasattr(scheduler, 'step') and not isinstance(scheduler, (ReduceLROnPlateau,
                                                                                       CosineAnnealingWarmRestarts)):
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.float()

                if self.config.MIXED_PRECISION:
                    with autocast():
                        output = self.model(data)
                        loss = F.binary_cross_entropy_with_logits(output, target)
                else:
                    output = self.model(data)
                    loss = F.binary_cross_entropy_with_logits(output, target)

                total_loss += loss.item()
                probs = torch.sigmoid(output)
                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        try:
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.5

        return avg_loss, auc


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-6, warmup_epochs=8, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.restore_best = restore_best
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.epoch = 0

    def __call__(self, score, model):
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                if self.restore_best:
                    self.best_weights = copy.deepcopy(model.state_dict())
            return False

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best_weights(self, model):
        if self.best_weights is not None and self.restore_best:
            model.load_state_dict(self.best_weights)


# ===============================================
# 6. Optimizer
# ===============================================

class Optimizer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.optimization_history = defaultdict(list)

    def create_model(self, model_type, input_dim, params):
        if model_type == 'Transformer':
            return Transformer(
                input_dim=input_dim,
                d_model=params.get('d_model', 64),
                num_heads=params.get('num_heads', 2),
                num_layers=params.get('num_layers', 7),
                dropout=params.get('dropout', 0.31)
            )
        elif model_type == 'Tabnet':
            return TabNet(
                input_dim=input_dim,
                n_d=params.get('n_d', 32),
                n_a=params.get('n_a', 32),
                n_steps=params.get('n_steps', 3),
                gamma=params.get('gamma', 1.3),
                dropout=params.get('dropout', 0.1)
            )
        elif model_type == 'CNN_RNN_Attention':
            return CNNRNNAttention(
                input_dim=input_dim,
                hidden_dim=params.get('hidden_dim', 96),
                num_layers=params.get('num_layers', 2),
                dropout=params.get('dropout', 0.15)
            )
        elif model_type == 'Self_Supervised_Model':
            return SelfSupervisedModel(
                input_dim=input_dim,
                hidden_dim=params.get('hidden_dim', 128),
                projection_dim=params.get('projection_dim', 64)
            )
        else:
            raise ValueError(f"Unknown model: {model_type}")

    def adaptive_hyperparameters(self, trial, model_type, round_num):
        base_ranges = self._get_base_ranges(model_type)

        if round_num <= 3:
            return self._expand_ranges(trial, base_ranges, factor=1.5)
        elif round_num <= 8:
            return self._standard_ranges(trial, base_ranges)
        else:
            return self._narrow_ranges(trial, base_ranges, model_type, factor=0.7)

    def _get_base_ranges(self, model_type):
        if model_type == 'Transformer':
            return {
                'batch_size': [32, 48, 96, 128, 256, 512, 1024],
                'optimizer': ['AdamW', 'Adam'],
                'lr': (0.003, 0.008),
                'weight_decay': (5e-5, 2e-4),
                'scheduler': ['cosine', 'plateau'],
                'dropout': (0.25, 0.4),
                'label_smoothing': (0.01, 0.02),
                'num_layers': (6, 25)
            }
        elif model_type == 'Tabnet':
            return {
                'batch_size': [32, 48, 96, 128, 256, 512, 1024],
                'optimizer': ['AdamW', 'Adam'],
                'lr': (0.002, 0.009),
                'weight_decay': (4e-5, 2e-4),
                'scheduler': ['cosine', 'plateau'],
                'dropout': (0.05, 0.2),
                'label_smoothing': (0.005, 0.03),
                'n_d': [20, 24, 32, 40, 48],
                'n_a': [20, 24, 32, 40, 48],
                'n_steps': (3, 5),
                'gamma': (1.1, 1.5)
            }
        elif model_type == 'CNN_RNN_Attention':
            return {
                'batch_size': [32, 48, 96, 128, 256, 512, 1024],
                'optimizer': ['AdamW', 'Adam'],
                'lr': (0.0008, 0.004),
                'weight_decay': (8e-5, 4e-4),
                'scheduler': ['cosine', 'plateau'],
                'dropout': (0.08, 0.25),
                'label_smoothing': (0.008, 0.04),
                'hidden_dim': [80, 96, 128, 160],
                'num_layers': (2, 8)
            }
        elif model_type == 'Self_Supervised_Model':
            return {
                'batch_size': [32, 48, 96, 128, 256, 512, 1024],
                'optimizer': ['AdamW', 'Adam'],
                'lr': (0.0003, 0.003),
                'weight_decay': (1e-4, 6e-4),
                'scheduler': ['cosine', 'plateau'],
                'dropout': (0.08, 0.25),
                'label_smoothing': (0.03, 0.2),
                'projection_dim': [48, 64, 80, 96]
            }
        return {}

    def _safe_int_conversion(self, value, default_val=1):
        try:
            if isinstance(value, (int, float)):
                return max(1, int(round(value)))
            else:
                return default_val
        except (ValueError, TypeError):
            return default_val

    def _expand_ranges(self, trial, base_ranges, factor=1.5):
        params = {}
        for key, value in base_ranges.items():
            if key == 'batch_size':
                params[key] = trial.suggest_categorical(key, value)
            elif key in ['optimizer', 'scheduler']:
                params[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], int):
                low, high = value
                if key in ['lr', 'weight_decay']:
                    params[key] = trial.suggest_float(key, low / factor, high * factor, log=True)
                else:
                    params[key] = trial.suggest_float(key, low / factor, high * factor)
            elif isinstance(value, list):
                params[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int):
                low, high = value
                new_low = max(1, self._safe_int_conversion(low / factor))
                new_high = max(new_low + 1, self._safe_int_conversion(high * factor))
                params[key] = trial.suggest_int(key, new_low, new_high)

        params.setdefault('d_model', 64)
        params.setdefault('num_heads', 2)
        params.setdefault('hidden_dim', 128)

        return params

    def _standard_ranges(self, trial, base_ranges):
        params = {}
        for key, value in base_ranges.items():
            if key == 'batch_size':
                params[key] = trial.suggest_categorical(key, value)
            elif key in ['optimizer', 'scheduler']:
                params[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], int):
                low, high = value
                if key in ['lr', 'weight_decay']:
                    params[key] = trial.suggest_float(key, low, high, log=True)
                else:
                    params[key] = trial.suggest_float(key, low, high)
            elif isinstance(value, list):
                params[key] = trial.suggest_categorical(key, value)
            elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int):
                low, high = value
                low, high = self._safe_int_conversion(low), self._safe_int_conversion(high)
                if high <= low:
                    high = low + 1
                params[key] = trial.suggest_int(key, low, high)

        params.setdefault('d_model', 64)
        params.setdefault('num_heads', 2)
        params.setdefault('hidden_dim', 128)

        return params

    def _narrow_ranges(self, trial, base_ranges, model_type, factor=0.7):
        history = self.optimization_history.get(model_type, [])
        if not history:
            return self._standard_ranges(trial, base_ranges)

        best_result = max(history, key=lambda x: x['score'])
        best_params = best_result['params']

        params = {}
        for key, value in base_ranges.items():
            if key in best_params:
                best_val = best_params[key]

                if key == 'batch_size':
                    options = [32, 48, 96, 128, 256, 512, 1024]
                    try:
                        best_idx = options.index(best_val) if best_val in options else len(options) // 2
                    except (ValueError, TypeError):
                        best_idx = len(options) // 2
                    start_idx = max(0, best_idx - 2)
                    end_idx = min(len(options), best_idx + 3)
                    params[key] = trial.suggest_categorical(key, options[start_idx:end_idx])

                elif isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], int):
                    low, high = value
                    try:
                        best_val_float = float(best_val)
                        if key in ['lr', 'weight_decay']:
                            new_low = max(low, best_val_float * factor)
                            new_high = min(high, best_val_float / factor)
                            if new_low >= new_high:
                                new_low, new_high = low, high
                            params[key] = trial.suggest_float(key, new_low, new_high, log=True)
                        else:
                            range_size = high - low
                            new_low = max(low, best_val_float - range_size * (1 - factor) / 2)
                            new_high = min(high, best_val_float + range_size * (1 - factor) / 2)
                            if new_low >= new_high:
                                new_low, new_high = low, high
                            params[key] = trial.suggest_float(key, new_low, new_high)
                    except (ValueError, TypeError):
                        if key in ['lr', 'weight_decay']:
                            params[key] = trial.suggest_float(key, low, high, log=True)
                        else:
                            params[key] = trial.suggest_float(key, low, high)

                elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int):
                    low, high = value
                    try:
                        best_val_int = self._safe_int_conversion(best_val)
                        range_size = high - low
                        new_low = max(low, best_val_int - self._safe_int_conversion(range_size * (1 - factor) / 2))
                        new_high = min(high, best_val_int + self._safe_int_conversion(range_size * (1 - factor) / 2))

                        new_low = self._safe_int_conversion(new_low)
                        new_high = self._safe_int_conversion(new_high)

                        if new_high <= new_low:
                            new_high = new_low + 1

                        params[key] = trial.suggest_int(key, new_low, new_high)
                    except (ValueError, TypeError):
                        low, high = self._safe_int_conversion(low), self._safe_int_conversion(high)
                        if high <= low:
                            high = low + 1
                        params[key] = trial.suggest_int(key, low, high)

                elif isinstance(value, list):
                    if best_val in value:
                        try:
                            best_idx = value.index(best_val)
                            start_idx = max(0, best_idx - 1)
                            end_idx = min(len(value), best_idx + 2)
                            subset = value[start_idx:end_idx]
                            params[key] = trial.suggest_categorical(key, subset if subset else value)
                        except (ValueError, IndexError):
                            params[key] = trial.suggest_categorical(key, value)
                    else:
                        params[key] = trial.suggest_categorical(key, value)
                else:
                    if isinstance(value, list):
                        params[key] = trial.suggest_categorical(key, value)
                    else:
                        params[key] = best_val
            else:
                if key == 'batch_size':
                    params[key] = trial.suggest_categorical(key, value)
                elif isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], int):
                    low, high = value
                    if key in ['lr', 'weight_decay']:
                        params[key] = trial.suggest_float(key, low, high, log=True)
                    else:
                        params[key] = trial.suggest_float(key, low, high)
                elif isinstance(value, list):
                    params[key] = trial.suggest_categorical(key, value)
                elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int):
                    low, high = value
                    low, high = self._safe_int_conversion(low), self._safe_int_conversion(high)
                    if high <= low:
                        high = low + 1
                    params[key] = trial.suggest_int(key, low, high)

        params.setdefault('d_model', 64)
        params.setdefault('num_heads', 2)
        params.setdefault('hidden_dim', 128)

        return params

    def create_optimizer(self, model, params):
        optimizer_name = params.get('optimizer', 'AdamW')
        lr = params['lr']
        weight_decay = params['weight_decay']

        if optimizer_name == 'AdamW':
            return optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'Adam':
            return optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            return optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )

    def create_scheduler(self, optimizer, params, steps_per_epoch):
        scheduler_name = params.get('scheduler', 'cosine')

        if scheduler_name == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.MAX_EPOCHS,
                eta_min=self.config.MIN_LR
            )
        elif scheduler_name == 'onecycle':
            return OneCycleLR(
                optimizer,
                max_lr=params['lr'],
                epochs=self.config.MAX_EPOCHS,
                steps_per_epoch=steps_per_epoch
            )
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=8,
                min_lr=self.config.MIN_LR
            )
        else:
            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.MAX_EPOCHS,
                eta_min=self.config.MIN_LR
            )

    def objective(self, trial, model_type, X_train, y_train, X_val, y_val, round_num):
        try:
            params = self.adaptive_hyperparameters(trial, model_type, round_num)
            model = self.create_model(model_type, X_train.shape[1], params).to(self.config.DEVICE)

            train_dataset = TensorDataset(
                torch.from_numpy(X_train.astype(np.float32)).float(),
                torch.from_numpy(y_train.astype(np.float32)).float()
            )
            val_dataset = TensorDataset(
                torch.from_numpy(X_val.astype(np.float32)).float(),
                torch.from_numpy(y_val.astype(np.float32)).float()
            )

            batch_size = params.get('batch_size', 48)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            optimizer = self.create_optimizer(model, params)
            scheduler = self.create_scheduler(optimizer, params, len(train_loader))

            trainer = Trainer(model, self.config.DEVICE, self.config)

            patience = max(8, 15 - round_num // 3)
            early_stopping = EarlyStopping(patience=patience, warmup_epochs=3)

            best_auc = 0
            max_epochs = min(self.config.MAX_EPOCHS, 20 + round_num * 5)

            for epoch in range(max_epochs):
                try:
                    train_loss = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
                    val_loss, val_auc = trainer.validate(val_loader)

                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_auc)
                    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                        scheduler.step()

                    if val_auc > best_auc:
                        best_auc = val_auc

                    if early_stopping(val_auc, model):
                        break

                    trial.report(val_auc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                except Exception as e:
                    self.logger.warning(f"Training epoch {epoch} failed: {e}")
                    break

            return max(best_auc, 0.5)

        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return 0.0

    def optimize_round(self, model_type, X_train, y_train, X_val, y_val, round_num, n_trials):
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=max(3, n_trials // 20),
                n_warmup_steps=5,
                interval_steps=1
            ),
            sampler=optuna.samplers.TPESampler(
                multivariate=True,
                n_startup_trials=max(5, n_trials // 15)
            )
        )

        objective = lambda trial: self.objective(trial, model_type, X_train, y_train, X_val, y_val, round_num)

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.OPTUNA_TIMEOUT,
            show_progress_bar=True
        )

        self.optimization_history[model_type].append({
            'round': round_num,
            'score': study.best_value,
            'params': study.best_params,
            'n_trials': n_trials
        })

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }


# ===============================================
# 7. Evaluator
# ===============================================

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def bootstrap_metrics(self, y_true, y_pred, metric_func, n_bootstrap=1000):
        n_samples = len(y_true)
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue

            try:
                score = metric_func(y_true[indices], y_pred[indices])
                bootstrap_scores.append(score)
            except:
                continue

        if not bootstrap_scores:
            return 0.0, 0.0, 0.0

        bootstrap_scores = np.array(bootstrap_scores)
        original_score = metric_func(y_true, y_pred)
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)

        return original_score, ci_lower, ci_upper

    def evaluation(self, model, X_test, y_test, model_type):
        try:
            model.eval()
            with torch.no_grad():
                if self.config.MIXED_PRECISION:
                    with autocast():
                        logits = model(torch.FloatTensor(X_test).to(self.config.DEVICE))
                else:
                    logits = model(torch.FloatTensor(X_test).to(self.config.DEVICE))

                y_pred_proba = torch.sigmoid(logits).cpu().numpy()

            y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)
            y_pred_proba = np.clip(y_pred_proba, 1e-8, 1 - 1e-8)

            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)

            metrics = {}

            auroc, auroc_ci_lower, auroc_ci_upper = self.bootstrap_metrics(
                y_test, y_pred_proba, roc_auc_score
            )
            metrics['AUROC'] = {
                'value': auroc,
                'ci_lower': auroc_ci_lower,
                'ci_upper': auroc_ci_upper
            }

            auprc, auprc_ci_lower, auprc_ci_upper = self.bootstrap_metrics(
                y_test, y_pred_proba, average_precision_score
            )
            metrics['AUPRC'] = {
                'value': auprc,
                'ci_lower': auprc_ci_lower,
                'ci_upper': auprc_ci_upper
            }

            for metric_name, metric_func in [
                ('Accuracy', accuracy_score),
                ('F1-Score', f1_score),
                ('Precision', precision_score),
                ('Recall', recall_score)
            ]:
                score, ci_lower, ci_upper = self.bootstrap_metrics(y_test, y_pred, metric_func)
                metrics[metric_name] = {
                    'value': score,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }

            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            except:
                specificity = 0
                sensitivity = 0

            def specificity_func(y_true, y_pred):
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    return tn / (tn + fp) if (tn + fp) > 0 else 0
                except:
                    return 0

            def sensitivity_func(y_true, y_pred):
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    return tp / (tp + fn) if (tp + fn) > 0 else 0
                except:
                    return 0

            spec_score, spec_ci_lower, spec_ci_upper = self.bootstrap_metrics(y_test, y_pred, specificity_func)
            sens_score, sens_ci_lower, sens_ci_upper = self.bootstrap_metrics(y_test, y_pred, sensitivity_func)

            metrics['Specificity'] = {'value': spec_score, 'ci_lower': spec_ci_lower, 'ci_upper': spec_ci_upper}
            metrics['Sensitivity'] = {'value': sens_score, 'ci_lower': sens_ci_lower, 'ci_upper': sens_ci_upper}

            return {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'optimal_threshold': optimal_threshold,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return None


# ===============================================
# 8. Main Pipeline
# ===============================================

class FeatureSelectionPipeline:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self._create_directories()

        # Phase 1: Feature selection records for ALL combinations
        self.feature_selection_records = []  # Records for each feature combination-model pair

        # Phase 2: Final optimization records
        self.best_features_by_model = {}
        self.global_best_score = 0.0
        self.global_best_model = None
        self.global_best_model_type = None
        self.optimization_results = {}
        self.final_optimization_records = []

    def _create_directories(self):
        directories = [
            Path(self.config.OUTPUT_DIR),
            Path(self.config.OUTPUT_DIR, "models"),
            Path(self.config.OUTPUT_DIR, "results"),
            Path(self.config.OUTPUT_DIR, "logs"),
            Path(self.config.OUTPUT_DIR, "visualizations"),
            Path(self.config.OUTPUT_DIR, "phase1_feature_selection"),
            Path(self.config.OUTPUT_DIR, "phase2_final_optimization"),
            Path(self.config.OUTPUT_DIR, "best_models")
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _save_phase1_results_to_excel(self):

        self.logger.info("Saving Phase 1: Feature Selection Results to Excel...")

        try:
            phase1_data = []
            for record in self.feature_selection_records:
                row = {
                    'Model_Type': record['model_type'],
                    'Combination_ID': record['combination_id'],
                    'Feature_1': record['features'][0] if len(record['features']) > 0 else '',
                    'Feature_2': record['features'][1] if len(record['features']) > 1 else '',
                    'Feature_3': record['features'][2] if len(record['features']) > 2 else '',
                    'Feature_Combination': ', '.join(record['features']),

                    'AUROC': record['metrics']['AUROC']['value'],
                    'AUROC_CI_Lower': record['metrics']['AUROC']['ci_lower'],
                    'AUROC_CI_Upper': record['metrics']['AUROC']['ci_upper'],
                    'AUROC_95%_CI': f"[{record['metrics']['AUROC']['ci_lower']:.4f}, {record['metrics']['AUROC']['ci_upper']:.4f}]",

                    'AUPRC': record['metrics']['AUPRC']['value'],
                    'AUPRC_CI_Lower': record['metrics']['AUPRC']['ci_lower'],
                    'AUPRC_CI_Upper': record['metrics']['AUPRC']['ci_upper'],
                    'AUPRC_95%_CI': f"[{record['metrics']['AUPRC']['ci_lower']:.4f}, {record['metrics']['AUPRC']['ci_upper']:.4f}]",

                    'Accuracy': record['metrics']['Accuracy']['value'],
                    'Accuracy_CI_Lower': record['metrics']['Accuracy']['ci_lower'],
                    'Accuracy_CI_Upper': record['metrics']['Accuracy']['ci_upper'],
                    'Accuracy_95%_CI': f"[{record['metrics']['Accuracy']['ci_lower']:.4f}, {record['metrics']['Accuracy']['ci_upper']:.4f}]",

                    'Precision': record['metrics']['Precision']['value'],
                    'Precision_CI_Lower': record['metrics']['Precision']['ci_lower'],
                    'Precision_CI_Upper': record['metrics']['Precision']['ci_upper'],
                    'Precision_95%_CI': f"[{record['metrics']['Precision']['ci_lower']:.4f}, {record['metrics']['Precision']['ci_upper']:.4f}]",

                    'Recall': record['metrics']['Recall']['value'],
                    'Recall_CI_Lower': record['metrics']['Recall']['ci_lower'],
                    'Recall_CI_Upper': record['metrics']['Recall']['ci_upper'],
                    'Recall_95%_CI': f"[{record['metrics']['Recall']['ci_lower']:.4f}, {record['metrics']['Recall']['ci_upper']:.4f}]",

                    'F1_Score': record['metrics']['F1-Score']['value'],
                    'F1_Score_CI_Lower': record['metrics']['F1-Score']['ci_lower'],
                    'F1_Score_CI_Upper': record['metrics']['F1-Score']['ci_upper'],
                    'F1_Score_95%_CI': f"[{record['metrics']['F1-Score']['ci_lower']:.4f}, {record['metrics']['F1-Score']['ci_upper']:.4f}]",

                    'Sensitivity': record['metrics']['Sensitivity']['value'],
                    'Sensitivity_CI_Lower': record['metrics']['Sensitivity']['ci_lower'],
                    'Sensitivity_CI_Upper': record['metrics']['Sensitivity']['ci_upper'],
                    'Sensitivity_95%_CI': f"[{record['metrics']['Sensitivity']['ci_lower']:.4f}, {record['metrics']['Sensitivity']['ci_upper']:.4f}]",

                    'Specificity': record['metrics']['Specificity']['value'],
                    'Specificity_CI_Lower': record['metrics']['Specificity']['ci_lower'],
                    'Specificity_CI_Upper': record['metrics']['Specificity']['ci_upper'],
                    'Specificity_95%_CI': f"[{record['metrics']['Specificity']['ci_lower']:.4f}, {record['metrics']['Specificity']['ci_upper']:.4f}]",

                    'Optimal_Threshold': record['optimal_threshold'],
                    'Timestamp': record['timestamp']
                }
                phase1_data.append(row)

            df_phase1 = pd.DataFrame(phase1_data)
            df_phase1 = df_phase1.sort_values(['Model_Type', 'AUROC'], ascending=[True, False])

            phase1_dir = Path(self.config.OUTPUT_DIR, "phase1_feature_selection")
            excel_path = phase1_dir / "phase1_all_combinations_results.xlsx"
            csv_path = phase1_dir / "phase1_all_combinations_results.csv"

            df_phase1.to_excel(excel_path, index=False)
            df_phase1.to_csv(csv_path, index=False)

            self.logger.info(f"Phase 1 results saved successfully:")
            self.logger.info(f"  Excel: {excel_path}")
            self.logger.info(f"  CSV: {csv_path}")
            self.logger.info(f"  Total records: {len(df_phase1)}")

            # saving best combinations per model
            best_combinations = []
            for model_type in self.config.MODELS:
                model_records = [r for r in self.feature_selection_records if r['model_type'] == model_type]
                if model_records:
                    best_record = max(model_records, key=lambda x: x['metrics']['AUROC']['value'])
                    best_combinations.append({
                        'Model_Type': model_type,
                        'Best_Features': ', '.join(best_record['features']),
                        'Best_AUROC': best_record['metrics']['AUROC']['value'],
                        'AUROC_95%_CI': f"[{best_record['metrics']['AUROC']['ci_lower']:.4f}, {best_record['metrics']['AUROC']['ci_upper']:.4f}]"
                    })

            df_best = pd.DataFrame(best_combinations)
            best_path = phase1_dir / "phase1_best_combinations_per_model.xlsx"
            df_best.to_excel(best_path, index=False)
            self.logger.info(f"  Best combinations per model: {best_path}")

            return excel_path

        except Exception as e:
            self.logger.error(f"Failed to save Phase 1 results: {e}")
            return None

    def _save_phase2_results_to_excel(self):

        self.logger.info("Saving Phase 2: Final Optimization Results to Excel...")

        try:
            phase2_data = []
            for record in self.final_optimization_records:
                row = {
                    'Model_Type': record['model_type'],
                    'Optimization_Round': record['optimization_round'],
                    'Best_Feature_1': record['best_features'][0] if len(record['best_features']) > 0 else '',
                    'Best_Feature_2': record['best_features'][1] if len(record['best_features']) > 1 else '',
                    'Best_Feature_3': record['best_features'][2] if len(record['best_features']) > 2 else '',
                    'Best_Feature_Combination': ', '.join(record['best_features']),

                    'AUROC': record['metrics']['AUROC']['value'],
                    'AUROC_CI_Lower': record['metrics']['AUROC']['ci_lower'],
                    'AUROC_CI_Upper': record['metrics']['AUROC']['ci_upper'],
                    'AUROC_95%_CI': f"[{record['metrics']['AUROC']['ci_lower']:.4f}, {record['metrics']['AUROC']['ci_upper']:.4f}]",

                    'AUPRC': record['metrics']['AUPRC']['value'],
                    'AUPRC_CI_Lower': record['metrics']['AUPRC']['ci_lower'],
                    'AUPRC_CI_Upper': record['metrics']['AUPRC']['ci_upper'],
                    'AUPRC_95%_CI': f"[{record['metrics']['AUPRC']['ci_lower']:.4f}, {record['metrics']['AUPRC']['ci_upper']:.4f}]",

                    'Accuracy': record['metrics']['Accuracy']['value'],
                    'Accuracy_CI_Lower': record['metrics']['Accuracy']['ci_lower'],
                    'Accuracy_CI_Upper': record['metrics']['Accuracy']['ci_upper'],
                    'Accuracy_95%_CI': f"[{record['metrics']['Accuracy']['ci_lower']:.4f}, {record['metrics']['Accuracy']['ci_upper']:.4f}]",

                    'Precision': record['metrics']['Precision']['value'],
                    'Precision_CI_Lower': record['metrics']['Precision']['ci_lower'],
                    'Precision_CI_Upper': record['metrics']['Precision']['ci_upper'],
                    'Precision_95%_CI': f"[{record['metrics']['Precision']['ci_lower']:.4f}, {record['metrics']['Precision']['ci_upper']:.4f}]",

                    'Recall': record['metrics']['Recall']['value'],
                    'Recall_CI_Lower': record['metrics']['Recall']['ci_lower'],
                    'Recall_CI_Upper': record['metrics']['Recall']['ci_upper'],
                    'Recall_95%_CI': f"[{record['metrics']['Recall']['ci_lower']:.4f}, {record['metrics']['Recall']['ci_upper']:.4f}]",

                    'F1_Score': record['metrics']['F1-Score']['value'],
                    'F1_Score_CI_Lower': record['metrics']['F1-Score']['ci_lower'],
                    'F1_Score_CI_Upper': record['metrics']['F1-Score']['ci_upper'],
                    'F1_Score_95%_CI': f"[{record['metrics']['F1-Score']['ci_lower']:.4f}, {record['metrics']['F1-Score']['ci_upper']:.4f}]",

                    'Sensitivity': record['metrics']['Sensitivity']['value'],
                    'Sensitivity_CI_Lower': record['metrics']['Sensitivity']['ci_lower'],
                    'Sensitivity_CI_Upper': record['metrics']['Sensitivity']['ci_upper'],
                    'Sensitivity_95%_CI': f"[{record['metrics']['Sensitivity']['ci_lower']:.4f}, {record['metrics']['Sensitivity']['ci_upper']:.4f}]",

                    'Specificity': record['metrics']['Specificity']['value'],
                    'Specificity_CI_Lower': record['metrics']['Specificity']['ci_lower'],
                    'Specificity_CI_Upper': record['metrics']['Specificity']['ci_upper'],
                    'Specificity_95%_CI': f"[{record['metrics']['Specificity']['ci_lower']:.4f}, {record['metrics']['Specificity']['ci_upper']:.4f}]",

                    'Optimal_Threshold': record['optimal_threshold'],
                    'Target_AUROC_Achieved': 'Yes' if record['metrics']['AUROC'][
                                                          'value'] >= self.config.TARGET_AUROC else 'No',
                    'Timestamp': record['timestamp']
                }

                for param_name, param_value in record['best_params'].items():
                    row[f'Best_Param_{param_name}'] = param_value

                phase2_data.append(row)

            df_phase2 = pd.DataFrame(phase2_data)
            df_phase2 = df_phase2.sort_values('AUROC', ascending=False)

            phase2_dir = Path(self.config.OUTPUT_DIR, "phase2_final_optimization")
            excel_path = phase2_dir / "phase2_final_optimization_results.xlsx"
            csv_path = phase2_dir / "phase2_final_optimization_results.csv"

            df_phase2.to_excel(excel_path, index=False)
            df_phase2.to_csv(csv_path, index=False)

            self.logger.info(f"Phase 2 results saved successfully:")
            self.logger.info(f"  Excel: {excel_path}")
            self.logger.info(f"  CSV: {csv_path}")
            self.logger.info(f"  Total records: {len(df_phase2)}")

            return excel_path

        except Exception as e:
            self.logger.error(f"Failed to save Phase 2 results: {e}")
            return None

    def run(self):
        self.logger.info("Starting Feature Combination Selection Pipeline")
        self.logger.info(f"Phase 1: Testing all 165 feature combinations")
        self.logger.info(
            f"Phase 2: Final optimization with best features, up to {self.config.MAX_OPTIMIZATION_ROUNDS} rounds")
        start_time = time.time()

        try:
            # ==================== DATA PREPARATION ====================
            processor = DataProcessor(self.config)
            df, X_raw, y, outlier_mask = processor.load_and_validate_data()

            X_engineered, all_feature_names = processor.feature_engineering(X_raw)
            processor.all_feature_names = all_feature_names

            self.logger.info(f"Total engineered features available: {len(all_feature_names)}")

            splits = processor.stratified_split_with_validation(X_engineered, y, outlier_mask)
            X_train, y_train = splits['train']
            X_val, y_val = splits['val']
            X_test, y_test = splits['test']

            processor.fit_transformers(X_train)
            X_train_scaled = processor.transform_data(X_train)
            X_val_scaled = processor.transform_data(X_val)
            X_test_scaled = processor.transform_data(X_test)

            # ==================== PHASE 1: FEATURE SELECTION ====================
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info("PHASE 1: FEATURE COMBINATION SELECTION")
            self.logger.info(f"{'=' * 80}")

            # Generate all possible 3-feature combinations
            all_combinations = list(
                itertools.combinations(range(len(all_feature_names)), self.config.NUM_FEATURES_TO_SELECT))
            total_combinations = len(all_combinations)

            self.logger.info(f"Total feature combinations to evaluate: {total_combinations}")
            self.logger.info(f"Models to test: {len(self.config.MODELS)}")
            self.logger.info(f"Total evaluations in Phase 1: {total_combinations * len(self.config.MODELS)}")

            # Initialize optimizers and evaluators
            optimizer = Optimizer(self.config)
            evaluator = Evaluator(self.config)
            augmenter = Augmenter(self.config)

            # Track best features for each model
            model_best_combinations = {model_type: {'auroc': 0.0} for model_type in self.config.MODELS}

            # Iterate through all combinations
            for comb_idx, feature_indices in enumerate(all_combinations, 1):
                selected_feature_names = [all_feature_names[i] for i in feature_indices]
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(f"Combination {comb_idx}/{total_combinations}: {selected_feature_names}")
                self.logger.info(f"{'=' * 80}")

                # Apply feature selection
                X_train_selected = processor.apply_feature_selection(X_train_scaled, list(feature_indices))
                X_val_selected = processor.apply_feature_selection(X_val_scaled, list(feature_indices))
                X_test_selected = processor.apply_feature_selection(X_test_scaled, list(feature_indices))

                # Data augmentation
                X_train_aug, y_train_aug = augmenter.augment_training_data(X_train_selected, y_train)
                X_train_aug = X_train_aug.astype(np.float32)
                y_train_aug = y_train_aug.astype(np.float32)
                X_val_selected = X_val_selected.astype(np.float32)
                X_test_selected = X_test_selected.astype(np.float32)

                # Evaluate each model with this combination
                for model_type in self.config.MODELS:
                    self.logger.info(f"\nEvaluating {model_type} with combination {comb_idx}...")

                    try:
                        # Hyperparameter optimization for this combination-model pair
                        opt_result = optimizer.optimize_round(
                            model_type, X_train_aug, y_train_aug, X_val_selected, y_val,
                            round_num=1, n_trials=self.config.FEATURE_SELECTION_TRIALS
                        )
                        best_params = opt_result['best_params']
                        best_val_auc = opt_result['best_value']

                        self.logger.info(f"  Best validation AUC: {best_val_auc:.4f}")

                        # Train final model with best parameters
                        model = optimizer.create_model(
                            model_type, X_train_aug.shape[1], best_params
                        ).to(self.config.DEVICE)

                        if model_type == 'Self_Supervised_Model':
                            self._train_self_supervised_model(
                                model, X_train_aug, y_train_aug, X_val_selected, y_val, best_params
                            )
                        else:
                            self._train_model(
                                model, X_train_aug, y_train_aug, X_val_selected, y_val, best_params
                            )

                        # Evaluate on test set
                        evaluation = evaluator.evaluation(model, X_test_selected, y_test, model_type)

                        if evaluation:
                            test_auc = evaluation['metrics']['AUROC']['value']
                            self.logger.info(f"  Test AUROC: {test_auc:.4f}")

                            # Record for Phase 1
                            phase1_record = {
                                'model_type': model_type,
                                'combination_id': comb_idx,
                                'features': selected_feature_names,
                                'indices': list(feature_indices),
                                'metrics': evaluation['metrics'],
                                'optimal_threshold': evaluation['optimal_threshold'],
                                'timestamp': datetime.now().isoformat()
                            }
                            self.feature_selection_records.append(phase1_record)

                            # Updating the best combination for this model
                            if test_auc > model_best_combinations[model_type]['auroc']:
                                model_best_combinations[model_type] = {
                                    'features': selected_feature_names,
                                    'indices': list(feature_indices),
                                    'auroc': test_auc,
                                    'evaluation': evaluation,
                                    'params': best_params,
                                    'combination_id': comb_idx
                                }
                                self.logger.info(f"  >>> New best for {model_type}: {test_auc:.4f}")

                    except Exception as e:
                        self.logger.error(f"  Evaluation failed for {model_type}: {e}")
                        continue

                # Memory cleanup
                if comb_idx % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                elapsed_time = time.time() - start_time
                self.logger.info(f"\nProgress: {comb_idx}/{total_combinations} combinations completed")
                self.logger.info(f"Time elapsed: {elapsed_time / 3600:.2f} hours")

            # Storing the best features for each model
            self.best_features_by_model = model_best_combinations

            # Logging the best combinations found
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info("PHASE 1 COMPLETE - BEST COMBINATIONS PER MODEL")
            self.logger.info(f"{'=' * 80}")
            for model_type, info in model_best_combinations.items():
                self.logger.info(f"{model_type}:")
                self.logger.info(f"  Best Features: {info['features']}")
                self.logger.info(f"  Best Test AUROC: {info['auroc']:.4f}")
                self.logger.info(f"  Combination ID: {info['combination_id']}")

            # Save Phase 1 results
            phase1_excel_path = self._save_phase1_results_to_excel()

            # ==================== PHASE 2: FINAL OPTIMIZATION WITH BEST FEATURES ====================
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info("PHASE 2: FINAL OPTIMIZATION WITH BEST FEATURES")
            self.logger.info(f"{'=' * 80}")

            for round_num in range(1, self.config.MAX_OPTIMIZATION_ROUNDS + 1):
                self.logger.info(f"\nOptimization Round {round_num}/{self.config.MAX_OPTIMIZATION_ROUNDS}")
                self.logger.info(f"Current global best AUROC: {self.global_best_score:.4f}")
                self.logger.info(f"Target AUROC: {self.config.TARGET_AUROC:.4f}")

                round_achieved_target = False
                current_n_trials = min(
                    self.config.INITIAL_N_TRIALS + (round_num - 1) * self.config.TRIAL_INCREMENT,
                    self.config.MAX_N_TRIALS
                )

                for model_type in self.config.MODELS:
                    self.logger.info(f"\nOptimizing {model_type} (Round {round_num})...")

                    if model_type not in self.best_features_by_model:
                        self.logger.warning(f"No best features found for {model_type}, skipping")
                        continue

                    best_feature_info = self.best_features_by_model[model_type]
                    selected_indices = best_feature_info['indices']
                    selected_features = best_feature_info['features']

                    self.logger.info(f"Using best features: {selected_features}")

                    try:
                        # Apply feature selection
                        X_train_model = processor.apply_feature_selection(X_train_scaled, selected_indices)
                        X_val_model = processor.apply_feature_selection(X_val_scaled, selected_indices)
                        X_test_model = processor.apply_feature_selection(X_test_scaled, selected_indices)

                        # Augment training data
                        X_train_aug, y_train_aug = augmenter.augment_training_data(X_train_model, y_train)
                        X_train_aug = X_train_aug.astype(np.float32)
                        y_train_aug = y_train_aug.astype(np.float32)

                        # Hyperparameter optimization
                        opt_result = optimizer.optimize_round(
                            model_type, X_train_aug, y_train_aug, X_val_model, y_val,
                            round_num, current_n_trials
                        )
                        best_params = opt_result['best_params']
                        best_val_auc = opt_result['best_value']

                        self.logger.info(f"{model_type} best validation AUC: {best_val_auc:.4f}")

                        # Train final model
                        final_model = optimizer.create_model(
                            model_type, X_train_aug.shape[1], best_params
                        ).to(self.config.DEVICE)

                        if model_type == 'Self_Supervised_Model':
                            final_auc = self._train_self_supervised_model(
                                final_model, X_train_aug, y_train_aug, X_val_model, y_val, best_params
                            )
                        else:
                            final_auc = self._train_model(
                                final_model, X_train_aug, y_train_aug, X_val_model, y_val, best_params
                            )

                        # Evaluate on test set
                        evaluation = evaluator.evaluation(
                            final_model, X_test_model, y_test, model_type
                        )

                        if evaluation:
                            test_auc = evaluation['metrics']['AUROC']['value']
                            self.logger.info(f"{model_type} test AUROC: {test_auc:.4f}")

                            # Record for Phase 2
                            phase2_record = {
                                'model_type': model_type,
                                'optimization_round': round_num,
                                'best_features': selected_features,
                                'metrics': evaluation['metrics'],
                                'optimal_threshold': evaluation['optimal_threshold'],
                                'best_params': best_params,
                                'timestamp': datetime.now().isoformat()
                            }
                            self.final_optimization_records.append(phase2_record)

                            # Updating the global best
                            if test_auc > self.global_best_score:
                                self.global_best_score = test_auc
                                self.global_best_model = final_model
                                self.global_best_model_type = model_type

                                torch.save(
                                    final_model.state_dict(),
                                    Path(self.config.OUTPUT_DIR, "models",
                                         f"best_model_{model_type}_round{round_num}.pth")
                                )

                                self.logger.info(f"New global best: {model_type}, AUROC: {test_auc:.4f}")

                            # Check if target achieved
                            if test_auc >= self.config.TARGET_AUROC:
                                self.logger.info(
                                    f"Target achieved! {model_type} AUROC: {test_auc:.4f} >= {self.config.TARGET_AUROC}")
                                round_achieved_target = True

                                self.optimization_results[model_type] = {
                                    'round_achieved': round_num,
                                    'final_auc': test_auc,
                                    'best_params': best_params,
                                    'evaluation': evaluation,
                                    'selected_features': selected_features
                                }
                    except Exception as e:
                        self.logger.error(f"{model_type} optimization failed: {e}")
                        continue

                # Check if target achieved in this round
                if round_achieved_target:
                    self.logger.info(f"\nOptimization completed successfully!")
                    self.logger.info(f"Final best AUROC: {self.global_best_score:.4f}")
                    self.logger.info(f"Best model: {self.global_best_model_type}")
                    self.logger.info(f"Total optimization rounds: {round_num}")
                    break

                elapsed_time = time.time() - start_time
                self.logger.info(f"\nRound {round_num} completed")
                self.logger.info(f"Time elapsed: {elapsed_time/3600:.2f} hours")
                self.logger.info(f"Current best AUROC: {self.global_best_score:.4f}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            else:
                self.logger.warning(f"Reached maximum optimization rounds ({self.config.MAX_OPTIMIZATION_ROUNDS})")
                self.logger.info(f"Final best AUROC: {self.global_best_score:.4f}")
                self.logger.info(f"Best model: {self.global_best_model_type}")

            # Save Phase 2 results
            phase2_excel_path = self._save_phase2_results_to_excel()

            # Generate final report
            self._generate_final_report(phase1_excel_path, phase2_excel_path)

            total_time = time.time() - start_time
            self.logger.info(f"\nTotal pipeline time: {total_time/3600:.2f} hours")

            return self.optimization_results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train_model(self, model, X_train, y_train, X_val, y_val, params):
        """Train regular model"""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        batch_size = params.get('batch_size', 48)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer_instance = Optimizer(self.config)
        optimizer_model = optimizer_instance.create_optimizer(model, params)
        scheduler = optimizer_instance.create_scheduler(optimizer_model, params, len(train_loader))

        trainer = Trainer(model, self.config.DEVICE, self.config)
        early_stopping = EarlyStopping(
            patience=self.config.PATIENCE,
            warmup_epochs=self.config.WARMUP_EPOCHS
        )

        best_val_auc = 0
        for epoch in range(self.config.MAX_EPOCHS):
            try:
                train_loss = trainer.train_epoch(train_loader, optimizer_model, scheduler, epoch)
                val_loss, val_auc = trainer.validate(val_loader)

                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_auc)
                elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step()

                if val_auc > best_val_auc:
                    best_val_auc = val_auc

                if epoch % 20 == 0:
                    self.logger.info(f"    Epoch {epoch}: Train Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")

                if early_stopping(val_auc, model):
                    self.logger.info(f"    Early stopping at epoch {epoch}")
                    break

            except Exception as e:
                self.logger.warning(f"    Training epoch {epoch} failed: {e}")
                break

        early_stopping.restore_best_weights(model)
        return best_val_auc

    def _train_self_supervised_model(self, model, X_train, y_train, X_val, y_val, params):
        """Train self-supervised model"""
        self.logger.info("    Starting self-supervised model training...")

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        batch_size = params.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer_instance = Optimizer(self.config)
        optimizer_model = optimizer_instance.create_optimizer(model, params)
        scheduler = optimizer_instance.create_scheduler(optimizer_model, params, len(train_loader))

        # Self-supervised pretraining
        pretrain_epochs = min(40, self.config.MAX_EPOCHS // 3)
        for epoch in range(pretrain_epochs):
            model.train()
            total_ssl_loss = 0

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.config.DEVICE)

                noise1 = torch.randn_like(data) * 0.01
                noise2 = torch.randn_like(data) * 0.01
                data1 = data + noise1
                data2 = data + noise2

                optimizer_model.zero_grad()

                _, z1, p1 = model(data1, ssl_mode=True)
                _, z2, p2 = model(data2, ssl_mode=True)

                ssl_loss = model.ssl_loss(z1, z2, p1, p2)

                ssl_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_model.step()

                total_ssl_loss += ssl_loss.item()

            if epoch % 15 == 0:
                avg_ssl_loss = total_ssl_loss / len(train_loader)
                self.logger.info(f"    Pretraining Epoch {epoch}: SSL Loss={avg_ssl_loss:.4f}")

        # Fine-tuning phase
        self.logger.info("    Starting fine-tuning phase...")
        for param_group in optimizer_model.param_groups:
            param_group['lr'] *= 0.1

        early_stopping = EarlyStopping(patience=20, warmup_epochs=5)
        best_val_auc = 0

        for epoch in range(self.config.MAX_EPOCHS - pretrain_epochs):
            model.train()
            total_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                target = target.float()

                optimizer_model.zero_grad()

                logits = model(data)
                loss = F.binary_cross_entropy_with_logits(logits, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer_model.step()

                total_loss += loss.item()

            # Validation
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                    logits = model(data)
                    probs = torch.sigmoid(logits)

                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())

            try:
                val_auc = roc_auc_score(val_targets, val_preds)
            except:
                val_auc = 0.5

            if val_auc > best_val_auc:
                best_val_auc = val_auc

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_auc)

            if epoch % 15 == 0:
                avg_loss = total_loss / len(train_loader)
                self.logger.info(f"    Fine-tuning Epoch {epoch}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")

            if early_stopping(val_auc, model):
                self.logger.info(f"    Fine-tuning early stopping at epoch {epoch}")
                break

        early_stopping.restore_best_weights(model)
        return best_val_auc

    def _generate_final_report(self, phase1_path, phase2_path):
        """Generating comprehensive final report"""
        print("\n" + "=" * 120)
        print("EXHAUSTIVE FEATURE COMBINATION SELECTION PIPELINE - FINAL REPORT")
        print("=" * 120)

        # Phase 1 Summary
        print(f"\nPHASE 1: EXHAUSTIVE FEATURE COMBINATION EVALUATION")
        print(f"-" * 120)
        print(f"   Total Feature Combinations: 165 (C(11,3))")
        print(f"   Models Evaluated: {len(self.config.MODELS)}")
        print(f"   Total Evaluations: {165 * len(self.config.MODELS)} = 660")
        if phase1_path:
            print(f"   Results Saved: {phase1_path}")

        if self.best_features_by_model:
            print(f"\n   Best Features Selected Per Model:")
            for model_type, info in self.best_features_by_model.items():
                print(f"      {model_type}:")
                print(f"         Features: {info['features']}")
                print(f"         Test AUROC: {info['auroc']:.4f}")
                print(f"         Combination ID: {info['combination_id']}/165")

        # Phase 2 Summary
        print(f"\nPHASE 2: FINAL OPTIMIZATION WITH BEST FEATURES")
        print(f"-" * 120)
        print(f"   Maximum Optimization Rounds: {self.config.MAX_OPTIMIZATION_ROUNDS}")
        print(f"   Target AUROC: {self.config.TARGET_AUROC:.4f}")
        print(f"   Global Best AUROC: {self.global_best_score:.4f}")
        print(f"   Global Best Model: {self.global_best_model_type or 'None'}")
        if phase2_path:
            print(f"   Results Saved: {phase2_path}")

        if self.final_optimization_records:
            print(f"\n   Phase 2 Best Performance by Model:")
            best_by_model = {}
            for record in self.final_optimization_records:
                model_type = record['model_type']
                auroc = record['metrics']['AUROC']['value']
                if model_type not in best_by_model or auroc > best_by_model[model_type]['auroc']:
                    best_by_model[model_type] = {
                        'auroc': auroc,
                        'record': record
                    }

            for model_type, info in best_by_model.items():
                record = info['record']
                print(f"      {model_type}:")
                print(f"         Best Features: {record['best_features']}")
                print(f"         Best AUROC: {record['metrics']['AUROC']['value']:.4f} "
                      f"(95% CI: [{record['metrics']['AUROC']['ci_lower']:.4f}, {record['metrics']['AUROC']['ci_upper']:.4f}])")
                print(f"         AUPRC: {record['metrics']['AUPRC']['value']:.4f} "
                      f"(95% CI: [{record['metrics']['AUPRC']['ci_lower']:.4f}, {record['metrics']['AUPRC']['ci_upper']:.4f}])")
                print(f"         Accuracy: {record['metrics']['Accuracy']['value']:.4f} "
                      f"(95% CI: [{record['metrics']['Accuracy']['ci_lower']:.4f}, {record['metrics']['Accuracy']['ci_upper']:.4f}])")
                print(f"         Sensitivity: {record['metrics']['Sensitivity']['value']:.4f} "
                      f"(95% CI: [{record['metrics']['Sensitivity']['ci_lower']:.4f}, {record['metrics']['Sensitivity']['ci_upper']:.4f}])")
                print(f"         Specificity: {record['metrics']['Specificity']['value']:.4f} "
                      f"(95% CI: [{record['metrics']['Specificity']['ci_lower']:.4f}, {record['metrics']['Specificity']['ci_upper']:.4f}])")
                print(f"         Optimization Round: {record['optimization_round']}")
                print(f"         Target Achieved: {'Yes' if record['metrics']['AUROC']['value'] >= self.config.TARGET_AUROC else 'No'}")

        # Successful models
        if self.optimization_results:
            print(f"\nMODELS ACHIEVING TARGET AUROC:")
            print(f"-" * 120)
            for model_type, result in self.optimization_results.items():
                print(f"   {model_type}:")
                print(f"      Achievement Round: {result['round_achieved']}")
                print(f"      Test AUROC: {result['final_auc']:.4f}")
                print(f"      Features Used: {result['selected_features']}")
        else:
            print(f"\nNo models achieved target AUROC {self.config.TARGET_AUROC:.4f}")

        # File locations
        print(f"\nOUTPUT FILES:")
        print(f"-" * 120)
        print(f"   Phase 1 Results: {Path(self.config.OUTPUT_DIR) / 'phase1_feature_selection'}")
        print(f"   Phase 2 Results: {Path(self.config.OUTPUT_DIR) / 'phase2_final_optimization'}")
        print(f"   Model Weights: {Path(self.config.OUTPUT_DIR) / 'models'}")
        print(f"   Logs: {Path(self.config.OUTPUT_DIR) / 'logs'}")

        print(f"\nAll results saved to: {self.config.OUTPUT_DIR}")
        print("=" * 120)

# ===============================================
# 9. Main Function
# ===============================================

def main():
    """Main function"""
    setup_logging()
    set_seeds(Config.RANDOM_SEED)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA Available: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("Warning: GPU acceleration recommended")

    # Create pipeline
    pipeline = FeatureSelectionPipeline()

    try:
        print(f"\nStarting Feature Combination Selection Pipeline")
        print(f"Phase 1: Testing all 165 feature combinations (C(11,3))")
        print(f"  - {Config.FEATURE_SELECTION_TRIALS} trials per combination-model pair")
        print(f"  - Total evaluations: 165 combinations × {len(Config.MODELS)} models = 660")
        print(f"Phase 2: Final optimization (up to {Config.MAX_OPTIMIZATION_ROUNDS} rounds)")
        print(f"  - {Config.INITIAL_N_TRIALS} trials per round")
        print(f"Target AUROC: {Config.TARGET_AUROC:.2f}")
        print(f"Models: {', '.join(Config.MODELS)}")
        print("\n" + "="*80)

        results = pipeline.run()

        if results:
            print(f"\nOptimization completed successfully!")
            print(f"{len(results)} models achieved target AUROC!")
        else:
            print(f"\nOptimization completed without achieving target AUROC {Config.TARGET_AUROC:.2f}")
            print(f"Best achieved AUROC: {pipeline.global_best_score:.4f}")

    except KeyboardInterrupt:
        print("\nUser interrupted optimization...")
        return None

    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nMemory cleaned!")

if __name__ == "__main__":
    main()