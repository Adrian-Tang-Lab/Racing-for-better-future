"""
==========================
 Interdisciplinary Research for Better Lives
==========================

Developed by:
    Dr. Adrian Tang and Dr. Jasper Chu's Collaborative Research Networks

Project:
    Binary Classification Based on Gene Expression Matrix and Classification Probabilities Predicted by Five Machine Learning Models 

Acknowledgments:
    We thank all investigators who participated in this large-scale translation study for their devotion.

Contact Information:
    Dr. Adrian Tang: 039319101@njucm.edu.cn
"""

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
import optuna
import matplotlib.pyplot as plt
import copy
import joblib
import os
import logging
import warnings
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier

def set_seed(seed=970404):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(970404)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(file_path, variance_threshold=0.1, k_best=10, scaler_path='models/scaler.pkl',
                             selector_path='models/feature_selector.pkl', selectk_path='models/select_k_best.pkl'):

    os.makedirs('models', exist_ok=True)

    data = pd.read_excel(file_path)
    samples = data.iloc[:, 0]
    labels = data.iloc[:, 1]
    features = data.iloc[:, 2:]

    selector = VarianceThreshold(threshold=variance_threshold)
    features_selected = selector.fit_transform(features)
    selected_feature_names = features.columns[selector.get_support(indices=True)]
    logging.info(f"Selected {features_selected.shape[1]} features out of {features.shape[1]} using VarianceThreshold")

    select_k_best = SelectKBest(score_func=f_classif, k=k_best)
    features_selected = select_k_best.fit_transform(features_selected, labels)
    final_selected_features = selected_feature_names[select_k_best.get_support(indices=True)]
    logging.info(f"Further selected {features_selected.shape[1]} features using SelectKBest")

    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_selected)

    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_resampled, y_resampled = smote.fit_resample(features_scaled, labels)

    counter_resampled = Counter(y_resampled)
    logging.info(f"Resampled class distribution: {counter_resampled}")

    y_resampled = np.array(y_resampled)

    joblib.dump(scaler, scaler_path)
    joblib.dump(selector, selector_path)
    joblib.dump(select_k_best, selectk_path)

    return X_resampled, y_resampled, final_selected_features

file_path = r"10_gene_exp_group.xlsx"

X_resampled, y_resampled, selected_features = load_and_preprocess_data(file_path)

def split_data(X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
        stratify=y_train_val, random_state=random_state
    )
    logging.info(f"Data split into Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_resampled, y_resampled)

def get_oof_predictions(model, X, y, n_splits=5, random_state=42):

    oof_pred = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        model_clone = clone(model)
        model_clone.fit(X_train_fold, y_train_fold)
        oof_pred[valid_idx] = model_clone.predict_proba(X_valid_fold)[:, 1]
        logging.info(f"{model.__class__.__name__} - Fold {fold + 1} completed.")

    return oof_pred

def train_and_optimize_base_model(model_name, model, X_train, y_train, n_splits=10, n_trials=100):

    def get_model_params(trial):
        param_grid = {
            'LightGBM': {
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            },
            'CatBoost': {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 0.0, 1.0)
            },
            'GBM': {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            },
            'XGBoost': {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0)
            },
            'RandomForest': {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        }
        return param_grid.get(model_name)

    def create_classifier(params):
        classifiers = {
            'LightGBM': LGBMClassifier,
            'CatBoost': CatBoostClassifier,
            'GBM': GradientBoostingClassifier,
            'XGBoost': XGBClassifier,
            'RandomForest': RandomForestClassifier
        }
        return classifiers[model_name](**params, random_state=42, verbose=0 if model_name == 'CatBoost' else None)

    def objective(trial):
        params = get_model_params(trial)
        if params is None:
            raise ValueError("Unsupported model name")
        
        clf = create_classifier(params)
        oof_preds = get_oof_predictions(clf, X_train, y_train, n_splits=n_splits)
        return roc_auc_score(y_train, oof_preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=3600)

    logging.info(f"Best trial for {model_name}: {study.best_trial.params} with AUC: {study.best_trial.value}")

    best_clf = create_classifier(study.best_trial.params)
    oof_pred_final = get_oof_predictions(best_clf, X_train, y_train, n_splits=n_splits)

    return best_clf, oof_pred_final, study.best_trial.params

def train_base_models_with_optuna(X_train, y_train, models_dict, n_splits=5, n_trials=50):
    base_model_features = pd.DataFrame()
    best_models, best_params = {}, {}

    for name, model in models_dict.items():
        logging.info(f"Optimizing and training base model: {name}")
        best_clf, oof_pred, params = train_and_optimize_base_model(name, model, X_train, y_train, n_splits, n_trials)
        
        base_model_features[name] = oof_pred
        best_models[name], best_params[name] = best_clf, params
        
        joblib.dump(best_clf, f'models/best_{name}.pkl')
        logging.info(f"Best parameters for {name}: {params}")

    joblib.dump(best_params, 'models/best_base_models_params.pkl')
    return base_model_features, best_models, best_params

base_models = {
    'LightGBM': LGBMClassifier(random_state=42, n_estimators=100, verbose=-1),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0, n_estimators=100),
    'GBM': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=100),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100)
}

base_model_features_train, best_base_models, best_base_params = train_base_models_with_optuna(
    X_train, y_train, base_models, n_splits=10, n_trials=100
)


def get_test_predictions_with_best_models(best_models, X_train, y_train, X_test):
    test_predictions = {
        name: model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        for name, model in best_models.items()
    }
    
    for name in best_models.keys():
        logging.info(f"Training best base model on full training data: {name}")

    return pd.DataFrame(test_predictions)

def get_validation_predictions_with_best_models(best_models, X_train, y_train, X_val):
    val_predictions = {
        name: model.fit(X_train, y_train).predict_proba(X_val)[:, 1]
        for name, model in best_models.items()
    }

    for name in best_models.keys():
        logging.info(f"Training best base model on full training data: {name}")

    return pd.DataFrame(val_predictions)


base_model_features_val = get_validation_predictions_with_best_models(best_base_models, X_train, y_train, X_val)

base_model_features_test = get_test_predictions_with_best_models(best_base_models, X_train, y_train, X_test)

def combine_features(original_X, base_model_features):

    combined_X = np.hstack((original_X, base_model_features.values))
    return combined_X

X_train_combined = combine_features(X_train, base_model_features_train)
X_val_combined = combine_features(X_val, base_model_features_val)
X_test_combined = combine_features(X_test, base_model_features_test)

logging.info(f"Combined Train features shape: {X_train_combined.shape}")
logging.info(f"Combined Validation features shape: {X_val_combined.shape}")
logging.info(f"Combined Test features shape: {X_test_combined.shape}")

class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads=8, n_layers=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (batch_size, 1, 128)
        x = self.transformer_encoder(x)      # (batch_size, 1, 128)
        x = self.dropout(x.squeeze(1))       # (batch_size, 128)
        return self.fc(x)

class TabNetClassifierWrapper:
    def __init__(self, input_dim, params=None):
        params = params or {}
        self.model = TabNetClassifier(
            input_dim=input_dim,
            output_dim=2,
            n_d=params.get('n_d', 128),
            n_a=params.get('n_a', 128),
            n_steps=params.get('n_steps', 8),
            gamma=params.get('gamma', 1.3),
            lambda_sparse=params.get('lambda_sparse', 1e-5),
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={'lr': params.get('lr', 0.0003)},
            scheduler_params={'step_size': 10, 'gamma': 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=params.get('mask_type', 'entmax')
        )

    def train(self, X_train, y_train, X_valid, y_valid, max_epochs=300):
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=['valid'],
            eval_metric=['auc'],
            max_epochs=max_epochs,
            patience=30,
            batch_size=64,
            virtual_batch_size=32,
            num_workers=0
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def evaluate_model_with_ci(y_true, y_probs, n_bootstrap=1000, random_state=42):
    y_pred = (y_probs >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1-score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs),
    }

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    metrics["pr_auc"] = auc(recall_curve, precision_curve)

    rng = np.random.default_rng(random_state)
    roc_auc_scores, pr_auc_scores = [], []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        roc_auc_scores.append(roc_auc_score(y_true[indices], y_probs[indices]))
        precision_bs, recall_bs, _ = precision_recall_curve(y_true[indices], y_probs[indices])
        pr_auc_scores.append(auc(recall_bs, precision_bs))

    metrics["roc_auc_ci"] = np.percentile(roc_auc_scores, [2.5, 97.5]) if roc_auc_scores else [0, 0]
    metrics["pr_auc_ci"] = np.percentile(pr_auc_scores, [2.5, 97.5]) if pr_auc_scores else [0, 0]

    return metrics

def print_metrics(metrics, dataset_type="Test"):
    print(f"\n{dataset_type} Metrics:")
    logging.info(f"\n{dataset_type} Metrics:")
    
    for key, value in metrics.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            ci_str = f"{value[0]:.4f} - {value[1]:.4f} (95% CI)"
        else:
            ci_str = f"{value:.4f}"
        
        print(f"{key}: {ci_str}")
        logging.info(f"{key}: {ci_str}")

def objective_transformer(trial, X, y):
    # 超参数建议
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    n_layers = trial.suggest_int('n_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    patience = trial.suggest_int('patience', 10, 30)

    # 模型和设备设置
    model = TransformerModel(input_dim=X.shape[1], n_heads=n_heads, n_layers=n_layers, dropout=dropout).to(
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # K折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                 torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                               torch.tensor(y_val, dtype=torch.long)), batch_size=64, shuffle=False)

        best_val_auc, epochs_no_improve = 0.0, 0

        for epoch in range(100):
            model.train()
            epoch_loss = sum(train_step(X_batch, y_batch, model, criterion, optimizer) for X_batch, y_batch in train_loader)
            scheduler.step()

            val_auc = evaluate_model(model, val_loader)
            aucs.append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc, epochs_no_improve = val_auc, 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    return np.mean(aucs)

def train_step(X_batch, y_batch, model, criterion, optimizer):
    X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
    optimizer.zero_grad()
    loss = criterion(model(X_batch), y_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def evaluate_model(model, val_loader):
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
            probabilities = torch.softmax(model(X_batch), dim=1)[:, 1]
            val_probs.extend(probabilities.cpu().numpy())
            val_true.extend(y_batch.cpu().numpy())
    return roc_auc_score(val_true, val_probs)

def objective_tabnet(trial, X_train, y_train, X_val, y_val):
    # 超参数建议
    params = {
        'n_d': trial.suggest_int('n_d', 16, 64),
        'n_a': trial.suggest_int('n_a', 16, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'mask_type': "entmax"
    }

    model = TabNetClassifierWrapper(input_dim=X_train.shape[1], params=params)
    model.train(X_train, y_train, X_val, y_val, max_epochs=300)

    # 计算验证集AUC
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    return val_auc

def optimize_transformer(X, y, n_trials=100):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_transformer(trial, X, y), n_trials=n_trials, timeout=3600)
    
    best_params = study.best_trial.params
    best_value = study.best_trial.value
    logging.info(f"Best Transformer trial: {best_params} with AUC: {best_value}")
    
    return best_params

def optimize_tabnet(X_train, y_train, X_val, y_val, n_trials=100):

    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective_tabnet(trial, X_train, y_train, X_val, y_val)
    study.optimize(func, n_trials=n_trials, timeout=3600)
    logging.info(f"Best TabNet trial: {study.best_trial.params} with AUC: {study.best_trial.value}")
    return study.best_trial.params

def train_transformer(X_train, y_train, X_val, y_val, best_params):
    # 初始化模型、设备、损失函数和优化器
    model = TransformerModel(
        input_dim=X_train.shape[1],
        n_heads=best_params['n_heads'],
        n_layers=best_params['n_layers'],
        dropout=best_params['dropout']
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 数据转换为张量
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=64, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ), batch_size=64, shuffle=False)

    # 训练过程
    best_val_auc, epochs_no_improve = 0.0, 0
    patience, max_epochs = best_params.get('patience', 20), 150
    best_model_state, train_loss_history, valid_loss_history = None, [], []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = sum(train_step(model, criterion, optimizer, X_batch, y_batch) for X_batch, y_batch in train_loader)

        val_auc, valid_loss = validate_model(model, criterion, val_loader)
        train_loss_history.append(epoch_loss / len(train_loader))
        valid_loss_history.append(valid_loss / len(val_loader))

        logging.info(f"Transformer Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader):.4f}, "
                     f"Validation Loss: {valid_loss / len(val_loader):.4f}, Validation AUROC: {val_auc:.4f}")

        # 检查是否改进
        if val_auc > best_val_auc:
            best_val_auc, epochs_no_improve = val_auc, 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_loss_history, valid_loss_history

def train_step(model, criterion, optimizer, X_batch, y_batch):
    X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def validate_model(model, criterion, val_loader):
    model.eval()
    val_probs, val_true, valid_loss = [], [], 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(model.device), y_batch.to(model.device)
            outputs = model(X_batch)
            valid_loss += criterion(outputs, y_batch).item()
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            val_probs.extend(probabilities.cpu().numpy())
            val_true.extend(y_batch.cpu().numpy())

    return roc_auc_score(val_true, val_probs), valid_loss

def train_tabnet_model(X_train, y_train, X_val, y_val, best_params):

    model = TabNetClassifierWrapper(input_dim=X_train.shape[1], params=best_params)
    model.train(X_train, y_train, X_val, y_val, max_epochs=300)
    return model

def get_model_predictions_transformer(model, X_test):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    return probabilities

def get_model_predictions_tabnet(model, X_test):

    return model.predict_proba(X_test)[:, 1]

def soft_voting(preds1, preds2):

    return (preds1 + preds2) / 2

def plot_loss_curve(train_loss, valid_loss, save_path):

    plt.figure(figsize=(7.09, 4))  # 180 mm width ≈ 7.09 inches
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=7, fontname='Arial')
    plt.ylabel('Loss', fontsize=7, fontname='Arial')
    plt.legend(fontsize=6)
    plt.title('Transformer Model: Train and Validation Loss', fontsize=8, fontname='Arial')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Fig3_Loss_Curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Loss curves saved as Fig3.")

def save_evaluation_metrics(metrics_dict, save_path):

    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.to_csv(os.path.join(save_path, 'evaluation_metrics.csv'), index=True)
    logging.info("Evaluation metrics saved as evaluation_metrics.csv")

def save_model(model, model_name, save_path):

    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}.pth'))
    elif isinstance(model, TabNetClassifierWrapper):
        model.model.save_model(os.path.join(save_path, f'{model_name}.zip'))
    else:
        joblib.dump(model, os.path.join(save_path, f'{model_name}.pkl'))
    logging.info(f"Model {model_name} saved.")

def main():

    save_path = r"risk_prediction_model\prediction model performance"
    os.makedirs(save_path, exist_ok=True)

    logging.info("Starting hyperparameter optimization for Transformer model.")
    best_params_transformer = optimize_transformer(X_train_combined, y_train, n_trials=100)

    logging.info("Starting hyperparameter optimization for TabNet model.")
    best_params_tabnet = optimize_tabnet(X_train_combined, y_train, X_val_combined, y_val, n_trials=100)

    joblib.dump(best_params_transformer, os.path.join(save_path, 'best_params_transformer.pkl'))
    joblib.dump(best_params_tabnet, os.path.join(save_path, 'best_params_tabnet.pkl'))
    logging.info("Best hyperparameters saved.")

    logging.info("Training Transformer model with best hyperparameters.")
    transformer_model, transformer_train_loss, transformer_valid_loss = train_transformer(
        X_train_combined, y_train, X_val_combined, y_val, best_params_transformer
    )

    logging.info("Training TabNet model with best hyperparameters.")
    tabnet_model = train_tabnet_model(
        X_train_combined, y_train, X_val_combined, y_val, best_params_tabnet
    )

    save_model(transformer_model, 'transformer_final', save_path)
    save_model(tabnet_model, 'tabnet_final', save_path)

    logging.info("Generating predictions on the test set.")
    transformer_test_probs = get_model_predictions_transformer(transformer_model, X_test_combined)
    tabnet_test_probs = get_model_predictions_tabnet(tabnet_model, X_test_combined)

    logging.info("Performing soft voting ensemble.")
    soft_voting_probs = soft_voting(transformer_test_probs, tabnet_test_probs)

    logging.info("Evaluating Transformer model.")
    transformer_metrics = evaluate_model_with_ci(y_test, transformer_test_probs)
    logging.info("Evaluating TabNet model.")
    tabnet_metrics = evaluate_model_with_ci(y_test, tabnet_test_probs)
    logging.info("Evaluating Final Soft Voting model.")
    final_metrics = evaluate_model_with_ci(y_test, soft_voting_probs)

    print_metrics(transformer_metrics, "Transformer Test")
    print_metrics(tabnet_metrics, "TabNet Test")
    print_metrics(final_metrics, "Final Soft Voting Test")

    save_evaluation_metrics({
        'Transformer': transformer_metrics,
        'TabNet': tabnet_metrics,
        'Final_Soft_Voting': final_metrics
    }, save_path)

    plot_loss_curve(transformer_train_loss, transformer_valid_loss, save_path)

    logging.info("Model training, evaluation, and saving completed successfully.")

if __name__ == "__main__":
    main()
