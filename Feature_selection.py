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

import os
import random
import warnings
import joblib
import logging
import pandas as pd
import numpy as np
from functools import partial
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import multiprocessing
from itertools import combinations

random.seed(1019)
np.random.seed(1019)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

def process_model_feature_subset(model_name, feature_subset, X, y, param_dist, cv, auc_scorer):
    try:
        X_subset = X[:, feature_subset]
        
        model_mapping = {
            'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42, tree_method='hist', use_label_encoder=False),
            'LightGBM': lgb.LGBMClassifier(random_state=42, device='cpu', verbose=-1),
            'CatBoost': CatBoostClassifier(verbose=0, random_state=42, task_type='CPU'),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
            'LogisticRegression_L1': LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=42),
            'LogisticRegression_ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000, random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'GBM': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42)
        }
        
        model = next((v for k, v in model_mapping.items() if model_name.startswith(k)), None)
        if model is None:
            raise ValueError(f"未识别的模型名称: {model_name}")

        if param_dist:
            randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, scoring=auc_scorer, cv=cv, verbose=0, random_state=42, n_jobs=1)
            randomized_search.fit(X_subset, y)
            best_auc, best_model = randomized_search.best_score_, randomized_search.best_estimator_
        else:
            model.fit(X_subset, y)
            best_auc = cross_val_score(model, X_subset, y, cv=cv, scoring='roc_auc_ovr').mean()
            best_model = model

        return {'model_name': model_name, 'feature_subset': feature_subset, 'AUC': best_auc, 'model': best_model}

    except Exception as e:
        logging.error(f"处理模型 {model_name} 在特征组合 {feature_subset} 时出错：{e}")
        return None

def worker(comb, models, param_distributions, X, y, cv, auc_scorer, return_list):
    local_results = [
        process_model_feature_subset(model_name, comb, X, y, param_distributions.get(model_name), cv, auc_scorer)
        for model_name in models.keys()
    ]
    return_list.extend(filter(None, local_results))

if __name__ == '__main__':

    data_path = r'15geneexp_group.xlsx'
    output_path = r'risk_prediction_model'
    os.makedirs(output_path, exist_ok=True)

    logging.info("加载数据...")
    data = pd.read_excel(data_path)
    logging.info(f"数据预览：\n{data.head()}")

    target_column = 'Group'
    feature_columns = data.columns[2:]
    X, y = data[feature_columns].values, data[target_column].values

    logging.info("编码目标变量...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logging.info(f"类别标签映射：{dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    logging.info("进行数据增强（SMOTE）...")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y_encoded)
    logging.info(f"增强后的数据样本数：{X_resampled.shape[0]}")

    logging.info("标准化特征...")
    X_scaled = StandardScaler().fit_transform(X_resampled)

    models = {
        # 'LogisticRegression_L1': LogisticRegression(penalty='l1', solver='saga', max_iter=10000, random_state=42),
        # 'LogisticRegression_ElasticNet': LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=10000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GBM': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42, tree_method='hist', use_label_encoder=False),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42, task_type='CPU'),
        'LightGBM': lgb.LGBMClassifier(random_state=42, device='cpu', verbose=-1),
        # 'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
        # 'SVM': SVC(kernel='linear', probability=True, random_state=42),
        # 'DecisionTree': DecisionTreeClassifier(random_state=42)
    }

    param_distributions = {
        # 'LogisticRegression_L1': {'C': np.logspace(-4, 4, 20), 'solver': ['saga'], 'penalty': ['l1']},
        # 'LogisticRegression_ElasticNet': {'C': np.logspace(-4, 4, 20), 'l1_ratio': np.linspace(0.1, 0.9, 9), 'solver': ['saga'], 'penalty': ['elasticnet']},
        # 'RandomForest': {'n_estimators': [100, 200, 300, 400, 500], 'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'bootstrap': [True, False]},
        'RandomForest': {'n_estimators': [500], 'max_depth': [50], 'min_samples_split': [2], 'min_samples_leaf': [1], 'bootstrap': [True, False]},
        # 'GBM': {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'subsample': [0.6, 0.8, 1.0], 'min_samples_split': [2, 5, 10]},
        'GBM': {'n_estimators': [500], 'learning_rate': [0.1], 'max_depth': [7], 'subsample': [0.8], 'min_samples_split': [5]},
        # 'XGBoost': {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'gamma': [0, 0.1, 0.2, 0.3]},
        'XGBoost': {'n_estimators': [500], 'learning_rate': [0.1], 'max_depth': [9]},
        # 'CatBoost': {'iterations': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'depth': [3, 5, 7, 9], 'l2_leaf_reg': [1, 3, 5, 7, 9]},
        'CatBoost': {'iterations': [500], 'learning_rate': [0.1], 'depth': [9], 'l2_leaf_reg': [5]},
        # 'LightGBM': {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'num_leaves': [31, 50, 70, 90], 'max_depth': [-1, 10, 20, 30], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0]},
        'LightGBM': {'n_estimators': [500], 'learning_rate': [0.1], 'num_leaves': [90], 'max_depth': [30], 'subsample': [0.6], 'colsample_bytree': [0.6]},
        # 'AdaBoost': {'n_estimators': [50, 100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2]},
        # 'SVM': {'C': np.logspace(-4, 4, 20), 'gamma': ['scale', 'auto']},
        # 'DecisionTree': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    feature_indices = list(range(X_scaled.shape[1]))
    all_combinations = list(combinations(feature_indices, 15))
    logging.info(f"总特征组合数量（15个特征）：{len(all_combinations)}")

    manager = multiprocessing.Manager()
    results = manager.list()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    for comb in all_combinations:
        pool.apply_async(worker, args=(comb, models, param_distributions, X_scaled, y_resampled, cv, make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr'), results))

    pool.close()
    pool.join()

    results_df = pd.DataFrame([{'model_name': res['model_name'], 'feature_subset': res['feature_subset'], 'AUC': res['AUC'], 'best_params': res.get('best_params', None)} for res in list(results)])
    print('results_df', results_df)
    # best_combinations = results_df.loc[results_df.groupby('model_name')['AUC'].idxmax()]

    # permutation_importance_results = {}
    # for index, row in best_combinations.iterrows():
    #     model_name, feature_subset, best_params = row['model_name'], row['feature_subset'], row.get('best_params', '无')
    #     logging.info(f"\n模型 {model_name} 的最佳 AUC: {row['AUC']:.4f}，最佳特征子集索引: {feature_subset}，最佳参数: {best_params}")
    
    #     try:
    #         X_best = X_scaled[:, feature_subset]
    #         best_model = models.get(model_name, None)
    #         if best_model is None:
    #             raise ValueError(f"未识别的模型名称: {model_name}")

    #         best_model.fit(X_best, y_resampled)
    #         perm_importance = permutation_importance(best_model, X_best, y_resampled, n_repeats=30, random_state=42, n_jobs=-1)
    #         importance_df = pd.DataFrame({'permutation_importance': perm_importance.importances_mean}, index=[feature_columns[i] for i in feature_subset]).sort_values(by='permutation_importance', ascending=False)

    #         output_file = os.path.join(output_path, f"{model_name}_best_permutation_importance.csv")
    #         importance_df.to_csv(output_file)
    #         logging.info(f"    已保存 {model_name} 的 Permutation Importance 至 {output_file}")
    #         permutation_importance_results[model_name] = importance_df

    #     except Exception as e:
    #         logging.error(f" 计算模型 {model_name} 的 Permutation Importance 时出错：{e}")

    # summary_file = os.path.join(output_path, "All_Models_Best_Permutation_Importance.xlsx")
    # with pd.ExcelWriter(summary_file) as writer:
    #     for model_name, df in permutation_importance_results.items():
    #         df.to_excel(writer, sheet_name=model_name)
    # logging.info(f"已保存所有模型的 Permutation Importance 汇总至 {summary_file}")
    # logging.info("所有步骤完成。")
