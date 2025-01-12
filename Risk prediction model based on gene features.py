"""
==========================
 Interdisciplinary Research for Better Lives
==========================

Developed by:
    Dr. Adrian Tang and Dr. Jasper Chu's Collaborative Research Networks

Project:
    Binary Classification Based on Gene Expression Matrix

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
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import copy
import optuna
import logging
import joblib
import os
import warnings
from collections import Counter


warnings.filterwarnings("ignore")


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


file_path = r"10_gene_exp_group.xlsx"
data = pd.read_excel(file_path)
samples = data.iloc[:, 0]
labels = data.iloc[:, 1]
features = data.iloc[:, 2:]


selector = VarianceThreshold(threshold=0.1)
features_selected = selector.fit_transform(features)
selected_features = features.columns[selector.get_support(indices=True)]
logging.info(f"Selected {features_selected.shape[1]} features out of {features.shape[1]}")


scaler = RobustScaler()
features_scaled = scaler.fit_transform(features_selected)


smote = SMOTE(random_state=42, sampling_strategy='minority')
X_resampled, y_resampled = smote.fit_resample(features_scaled, labels)


counter_resampled = Counter(y_resampled)
logging.info(f"Resampled class distribution: {counter_resampled}")


y_resampled = np.array(y_resampled)


os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(selector, 'models/feature_selector.pkl')


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
)  # 0.25 x 0.8 = 0.2

logging.info(f"Data split into Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")


class TransformerModel(nn.Module):
    def __init__(self, input_dim, n_heads=8, n_layers=3, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.fc = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        x = self.dropout(x)
        return self.fc(x)


class TabNetClassifierWrapper:
    def __init__(self, input_dim, params=None):
        if params is None:
            params = {}
        self.model = TabNetClassifier(
            input_dim=input_dim,
            output_dim=2,
            n_d=params.get('n_d', 128),
            n_a=params.get('n_a', 128),
            n_steps=params.get('n_steps', 8),
            gamma=params.get('gamma', 1.3),
            lambda_sparse=params.get('lambda_sparse', 1e-5),
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=params.get('lr', 0.0003)),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=params.get('mask_type', "entmax"),
        )

    def train(self, X_train, y_train, X_valid, y_valid, max_epochs=300):
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["valid"],
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=30,
            batch_size=64,
            virtual_batch_size=32,
            num_workers=0,
        )

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def evaluate_model_with_ci(y_true, y_probs, n_bootstrap=1000, random_state=42):

    y_pred = (y_probs >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)


    roc_auc = roc_auc_score(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)


    rng = np.random.default_rng(random_state)
    roc_auc_scores = []
    pr_auc_scores = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        roc_auc_scores.append(roc_auc_score(y_true[indices], y_probs[indices]))
        precision_bs, recall_bs, _ = precision_recall_curve(y_true[indices], y_probs[indices])
        pr_auc_scores.append(auc(recall_bs, precision_bs))

    roc_auc_ci = np.percentile(roc_auc_scores, [2.5, 97.5]) if len(roc_auc_scores) > 0 else [0, 0]
    pr_auc_ci = np.percentile(pr_auc_scores, [2.5, 97.5]) if len(pr_auc_scores) > 0 else [0, 0]

    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1-score": report["1"]["f1-score"],
        "roc_auc": roc_auc,
        "roc_auc_ci": roc_auc_ci,
        "pr_auc": pr_auc,
        "pr_auc_ci": pr_auc_ci,
    }


def print_metrics(metrics, dataset_type="Test"):
    print(f"\n{dataset_type} Metrics:")
    logging.info(f"\n{dataset_type} Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            print(f"{key}: {value[0]:.4f} - {value[1]:.4f} (95% CI)")
            logging.info(f"{key}: {value[0]:.4f} - {value[1]:.4f} (95% CI)")
        else:
            print(f"{key}: {value:.4f}")
            logging.info(f"{key}: {value:.4f}")


def objective_transformer(trial, X, y):

    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    n_layers = trial.suggest_int('n_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.3, 0.7)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    patience = trial.suggest_int('patience', 10, 30)


    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]


        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


        model = TransformerModel(input_dim=X.shape[1], n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


        best_val_auc = 0.0
        epochs_no_improve = 0
        max_epochs = 100
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()


            model.eval()
            val_probs = []
            val_true = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    probabilities = torch.softmax(outputs, dim=1)[:, 1]
                    val_probs.extend(probabilities.cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())
            val_auc = roc_auc_score(val_true, val_probs)
            aucs.append(val_auc)


            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    return np.mean(aucs)


def objective_tabnet(trial, X_train, y_train, X_val, y_val):

    n_d = trial.suggest_int('n_d', 16, 64)
    n_a = trial.suggest_int('n_a', 16, 64)
    n_steps = trial.suggest_int('n_steps', 3, 10)
    gamma = trial.suggest_float('gamma', 1.0, 2.0)
    lambda_sparse = trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)


    params = {
        'n_d': n_d,
        'n_a': n_a,
        'n_steps': n_steps,
        'gamma': gamma,
        'lambda_sparse': lambda_sparse,
        'lr': lr,
        'mask_type': "entmax"
    }
    model = TabNetClassifierWrapper(input_dim=X_train.shape[1], params=params)


    model.train(X_train, y_train, X_val, y_val, max_epochs=300)


    y_val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_probs)

    return val_auc


def optimize_model_transformer(X, y, n_trials=100):

    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective_transformer(trial, X, y)
    study.optimize(func, n_trials=n_trials, timeout=3600)
    logging.info(f"Best Transformer trial: {study.best_trial.params} with AUC: {study.best_trial.value}")
    return study.best_trial.params


def optimize_model_tabnet(X_train, y_train, X_val, y_val, n_trials=100):

    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective_tabnet(trial, X_train, y_train, X_val, y_val)
    study.optimize(func, n_trials=n_trials, timeout=3600)
    logging.info(f"Best TabNet trial: {study.best_trial.params} with AUC: {study.best_trial.value}")
    return study.best_trial.params


def train_final_transformer(X_train, y_train, X_val, y_val, X_test, y_test, best_params):

    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))


    X_combined_tensor = torch.tensor(X_combined, dtype=torch.float32)
    y_combined_tensor = torch.tensor(y_combined, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_combined_tensor, y_combined_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = TransformerModel(
        input_dim=X_combined.shape[1],
        n_heads=best_params['n_heads'],
        n_layers=best_params['n_layers'],
        dropout=best_params['dropout']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


    best_val_auc = 0.0
    epochs_no_improve = 0
    patience = best_params.get('patience', 20)
    max_epochs = 150
    best_model_state = None
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()


        model.eval()
        test_probs = []
        test_true = []
        valid_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                test_probs.extend(probabilities.cpu().numpy())
                test_true.extend(y_batch.cpu().numpy())

        test_auc = roc_auc_score(test_true, test_probs)
        train_loss_history.append(epoch_loss / len(train_loader))
        valid_loss_history.append(valid_loss / len(test_loader))

        logging.info(
            f"Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader):.4f}, "
            f"Test Loss: {valid_loss / len(test_loader):.4f}, Test AUROC: {test_auc:.4f}"
        )


        if test_auc > best_val_auc:
            best_val_auc = test_auc
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_train_loss = train_loss_history.copy()
            best_valid_loss = valid_loss_history.copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break


    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    model.eval()
    test_probs = []
    test_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probabilities.cpu().numpy())
            test_true.extend(y_batch.cpu().numpy())

    test_metrics = evaluate_model_with_ci(np.array(test_true), np.array(test_probs))


    save_path = r"D:\Sepsis\NM version\risk_prediction_model\prediction model performance"
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'transformer_final.pth'))

    return model, test_metrics, best_train_loss, best_valid_loss


def train_final_tabnet(X_train, y_train, X_val, y_val, X_test, y_test, best_params):

    X_combined = np.vstack((X_train, X_val))
    y_combined = np.hstack((y_train, y_val))

    model = TabNetClassifierWrapper(input_dim=X_combined.shape[1], params=best_params)
    model.train(X_combined, y_combined, X_test, y_test, max_epochs=300)

    y_test_probs = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_probs)
    precision_val, recall_val, _ = precision_recall_curve(y_test, y_test_probs)
    pr_auc = auc(recall_val, precision_val)

    y_test_pred = (y_test_probs >= 0.5).astype(int)
    report = classification_report(y_test, y_test_pred, output_dict=True)

    rng = np.random.default_rng(42)
    roc_auc_scores = []
    pr_auc_scores = []

    for _ in range(1000):
        indices = rng.integers(0, len(y_test), len(y_test))
        if len(np.unique(y_test[indices])) < 2:
            continue
        roc_auc_scores.append(roc_auc_score(y_test[indices], y_test_probs[indices]))
        precision_bs, recall_bs, _ = precision_recall_curve(y_test[indices], y_test_probs[indices])
        pr_auc_scores.append(auc(recall_bs, precision_bs))

    roc_auc_ci = np.percentile(roc_auc_scores, [2.5, 97.5]) if len(roc_auc_scores) > 0 else [0, 0]
    pr_auc_ci = np.percentile(pr_auc_scores, [2.5, 97.5]) if len(pr_auc_scores) > 0 else [0, 0]

    test_metrics = {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1-score": report["1"]["f1-score"],
        "roc_auc": test_auc,
        "roc_auc_ci": roc_auc_ci,
        "pr_auc": pr_auc,
        "pr_auc_ci": pr_auc_ci,
    }

    save_path = r"D:\Sepsis\NM version\risk_prediction_model\prediction model performance"
    os.makedirs(save_path, exist_ok=True)
    model.model.save_model(os.path.join(save_path, 'tabnet_final.zip'))

    return model, test_metrics

def plot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(7.09, 4))  # 180 mm width ≈ 7.09 inches
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Test Loss')
    plt.xlabel('Epoch', fontsize=7, fontname='Arial')
    plt.ylabel('Loss', fontsize=7, fontname='Arial')
    plt.legend(fontsize=6)
    plt.title('Transformer Model: Train and Test Loss', fontsize=8, fontname='Arial')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Fig3_Loss_Curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():

    save_path = r"D:\Sepsis\NM version\risk_prediction_model\prediction model performance"
    os.makedirs(save_path, exist_ok=True)

    logging.info("Starting hyperparameter optimization for Transformer model.")
    best_params_transformer = optimize_model_transformer(X_train, y_train, n_trials=100)

    logging.info("Starting hyperparameter optimization for TabNet model.")
    best_params_tabnet = optimize_model_tabnet(X_train, y_train, X_val, y_val, n_trials=100)

    joblib.dump(best_params_transformer, os.path.join(save_path, 'best_params_transformer.pkl'))
    joblib.dump(best_params_tabnet, os.path.join(save_path, 'best_params_tabnet.pkl'))
    logging.info("Best hyperparameters saved.")

    logging.info("Training and evaluating the final Transformer model.")
    transformer_model, transformer_test_metrics, transformer_train_loss, transformer_valid_loss = train_final_transformer(
        X_train, y_train, X_val, y_val, X_test, y_test, best_params_transformer
    )

    logging.info("Training and evaluating the final TabNet model.")
    tabnet_model, tabnet_test_metrics = train_final_tabnet(
        X_train, y_train, X_val, y_val, X_test, y_test, best_params_tabnet
    )

    print_metrics(transformer_test_metrics, "Transformer Test")
    print_metrics(tabnet_test_metrics, "TabNet Test")

    plot_loss(transformer_train_loss, transformer_valid_loss, save_path)
    logging.info("Loss curves saved as Fig3.")

    evaluation_metrics = {
        'Transformer': transformer_test_metrics,
        'TabNet': tabnet_test_metrics
    }
    joblib.dump(evaluation_metrics, os.path.join(save_path, 'evaluation_metrics.pkl'))
    logging.info("Evaluation metrics saved.")

    logging.info("Model training, evaluation, and saving completed successfully.")

if __name__ == "__main__":
    main()
