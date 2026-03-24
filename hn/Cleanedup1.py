#%% ── Imports ─────────────────────────────────────────────────────────────────
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, validation_curve, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from xgboost import XGBClassifier
from mrmr import mrmr_classif
from matplotlib.patches import Patch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load preprocessed data
data = joblib.load('preprocessed_data.pkl')

X_train = data['X_train']
X_test  = data['X_test']
Y_train = data['Y_train']
Y_test  = data['Y_test']
features = data['features']

print("Data loaded successfully.")


#%% ── Custom mRMR transformer ─────────────────────────────────────────────────
class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=20):
        self.n_features_to_select = n_features_to_select
        self.selected_features_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.selected_features_ = mrmr_classif(
            X=X_df, y=y, K=self.n_features_to_select
        )
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).reset_index(drop=True)
        return X_df[self.selected_features_].values


#%% ── Helper function for validation curve plots ─────────────────────────────
def plot_validation_curve(ax, param_range, train_scores, val_scores, xlabel, title, x_labels=None):
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.plot(param_range, train_mean, label='Train AUC', color='#1f77b4')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#1f77b4')

    ax.plot(param_range, val_mean, label='CV AUC', color='#ff7f0e')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#ff7f0e')

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels([str(x) for x in x_labels])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('AUC')
    ax.set_title(title)
    ax.legend()


#%% ── Step 1: OOB curve + RFE fit ────────────────────────────────────────────
oob_scores = []
n_estimators_range = range(10, 300, 10)

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train, Y_train)
    oob_scores.append(rf.oob_score_)

optimal_n = n_estimators_range[oob_scores.index(max(oob_scores))]
print(f"Optimal n_estimators (OOB): {optimal_n}")

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, oob_scores)
plt.axvline(x=optimal_n, color='r', linestyle='--',
            label=f'Optimal: {optimal_n} trees')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Score')
plt.title('OOB Score vs Number of Trees (RFE features)')
plt.legend()
plt.show()

rfe = RFE(estimator=RandomForestClassifier(random_state=42),
          n_features_to_select=20, step=1)
rfe.fit(X_train, Y_train)
X_train_rfe = rfe.transform(X_train)


#%% ── Step 2: mRMR fit ────────────────────────────────────────────────────────
Y_train_reset = pd.Series(Y_train).reset_index(drop=True)

mrmr_selector = MRMRSelector(n_features_to_select=45)
mrmr_selector.fit(X_train, Y_train_reset)
X_train_mrmr = mrmr_selector.transform(X_train)

oob_scores_mrmr = []
for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X_train_mrmr, Y_train_reset)
    oob_scores_mrmr.append(rf.oob_score_)

optimal_n_mrmr = 150
print(f"Optimal n_estimators mRMR (OOB): {optimal_n_mrmr}")

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, oob_scores_mrmr)
plt.axvline(x=optimal_n_mrmr, color='r', linestyle='--',
            label=f'Optimal: {optimal_n_mrmr} trees')
plt.xlabel('Number of Trees')
plt.ylabel('OOB Score')
plt.title('OOB Score vs Number of Trees (mRMR features)')
plt.legend()
plt.show()


#%% ── Step 3: LASSO fits ──────────────────────────────────────────────────────
lasso_lr = SelectFromModel(Lasso(max_iter=10000, random_state=42, alpha=0.019),
                           max_features=10, threshold='mean')
lasso_lr.fit(X_train, Y_train)
X_train_lasso_lr = lasso_lr.transform(X_train)

lasso_svm = SelectFromModel(Lasso(max_iter=10000, random_state=42, alpha=0.019),
                            max_features=15, threshold='mean')
lasso_svm.fit(X_train, Y_train)
X_train_lasso_svm = lasso_svm.transform(X_train)

lasso_rf = SelectFromModel(Lasso(max_iter=10000, random_state=42, alpha=0.019),
                           max_features=10, threshold='mean')
lasso_rf.fit(X_train, Y_train)
X_train_lasso_rf = lasso_rf.transform(X_train)

lasso_xgb = SelectFromModel(Lasso(max_iter=10000, random_state=42, alpha=0.019),
                            max_features=15, threshold='mean')
lasso_xgb.fit(X_train, Y_train)
X_train_lasso_xgb = lasso_xgb.transform(X_train)


#%% ── Random Forest Validation Curves (RFE features) ─────────────────────────
depths = [1, 3, 5, 7, 10, 15, 20, None]
splits = [1, 5, 10, 15, 20, 30, 40, 50]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=optimal_n, random_state=42),
    X_train_rfe, Y_train,
    param_name='max_depth', param_range=depths,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], range(len(depths)), train_scores, val_scores,
                      'Max Depth', 'Validation Curve - Max Depth (RF)', x_labels=depths)

train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=optimal_n, random_state=42),
    X_train_rfe, Y_train,
    param_name='min_samples_split', param_range=splits,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1], splits, train_scores, val_scores,
                      'Min Samples Split', 'Validation Curve - Min Samples Split (RF)')

plt.suptitle('Random Forest Hyperparameter Validation Curves (RFE features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── Random Forest Validation Curves (mRMR features) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=optimal_n_mrmr, random_state=42),
    X_train_mrmr, Y_train_reset,
    param_name='max_depth', param_range=depths,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], range(len(depths)), train_scores, val_scores,
                      'Max Depth', 'Validation Curve - Max Depth (mRMR)', x_labels=depths)

train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=optimal_n_mrmr, random_state=42),
    X_train_mrmr, Y_train_reset,
    param_name='min_samples_split', param_range=splits,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1], splits, train_scores, val_scores,
                      'Min Samples Split', 'Validation Curve - Min Samples Split (mRMR)')

plt.suptitle('Random Forest Hyperparameter Validation Curves (mRMR features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── XGBoost Validation Curves (RFE features) ────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

n_estimators_range_xgb = [50, 100, 150, 200, 300]
train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', learning_rate=0.1, max_depth=3),
    X_train_rfe, Y_train,
    param_name='n_estimators', param_range=n_estimators_range_xgb,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0, 0], n_estimators_range_xgb, train_scores, val_scores,
                      'n_estimators', 'Validation Curve - n_estimators (XGBoost)')

depths_xgb = [1, 2, 3, 4, 5, 6, 7]
train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=150, learning_rate=0.1),
    X_train_rfe, Y_train,
    param_name='max_depth', param_range=depths_xgb,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0, 1], depths_xgb, train_scores, val_scores,
                      'max_depth', 'Validation Curve - max_depth (XGBoost)')

learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=150, max_depth=3),
    X_train_rfe, Y_train,
    param_name='learning_rate', param_range=learning_rates,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0, 2], learning_rates, train_scores, val_scores,
                      'learning_rate', 'Validation Curve - learning_rate (XGBoost)')

subsample_range = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=150, max_depth=3, learning_rate=0.1),
    X_train_rfe, Y_train,
    param_name='subsample', param_range=subsample_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1, 0], subsample_range, train_scores, val_scores,
                      'subsample', 'Validation Curve - subsample (XGBoost)')

colsample_range = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=150, max_depth=3, learning_rate=0.1),
    X_train_rfe, Y_train,
    param_name='colsample_bytree', param_range=colsample_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1, 1], colsample_range, train_scores, val_scores,
                      'colsample_bytree', 'Validation Curve - colsample_bytree (XGBoost)')

min_child_range = [1, 2, 3, 5, 7, 10]
train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=150, max_depth=3, learning_rate=0.1),
    X_train_rfe, Y_train,
    param_name='min_child_weight', param_range=min_child_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1, 2], min_child_range, train_scores, val_scores,
                      'min_child_weight', 'Validation Curve - min_child_weight (XGBoost)')

plt.suptitle('XGBoost Hyperparameter Validation Curves (RFE features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── XGBoost Validation Curves (mRMR features) ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', learning_rate=0.1, max_depth=3),
    X_train_mrmr, Y_train_reset,
    param_name='n_estimators', param_range=n_estimators_range_xgb,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0, 0], n_estimators_range_xgb, train_scores, val_scores,
                      'n_estimators', 'Validation Curve - n_estimators (XGBoost)')

train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=optimal_n_mrmr, learning_rate=0.1),
    X_train_mrmr, Y_train_reset,
    param_name='max_depth', param_range=depths_xgb,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0, 1], depths_xgb, train_scores, val_scores,
                      'max_depth', 'Validation Curve - max_depth (XGBoost)')

train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=optimal_n_mrmr, max_depth=3),
    X_train_mrmr, Y_train_reset,
    param_name='learning_rate', param_range=learning_rates,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0, 2], learning_rates, train_scores, val_scores,
                      'learning_rate', 'Validation Curve - learning_rate (XGBoost)')

train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=optimal_n_mrmr, max_depth=3, learning_rate=0.1),
    X_train_mrmr, Y_train_reset,
    param_name='subsample', param_range=subsample_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1, 0], subsample_range, train_scores, val_scores,
                      'subsample', 'Validation Curve - subsample (XGBoost)')

train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=optimal_n_mrmr, max_depth=3, learning_rate=0.1),
    X_train_mrmr, Y_train_reset,
    param_name='colsample_bytree', param_range=colsample_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1, 1], colsample_range, train_scores, val_scores,
                      'colsample_bytree', 'Validation Curve - colsample_bytree (XGBoost)')

train_scores, val_scores = validation_curve(
    XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=optimal_n_mrmr, max_depth=3, learning_rate=0.1),
    X_train_mrmr, Y_train_reset,
    param_name='min_child_weight', param_range=min_child_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1, 2], min_child_range, train_scores, val_scores,
                      'min_child_weight', 'Validation Curve - min_child_weight (XGBoost)')

plt.suptitle('XGBoost Hyperparameter Validation Curves (mRMR features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── Logistic Regression Validation Curves (RFE features) ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

C_range_lr = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores, val_scores = validation_curve(
    LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
    X_train_rfe, Y_train,
    param_name='C', param_range=C_range_lr,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], C_range_lr, train_scores, val_scores,
                      'C', 'Validation Curve - C (Logistic Regression)')
axes[0].set_xscale('log')

penalties = ['l1', 'l2']
train_scores, val_scores = validation_curve(
    LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
    X_train_rfe, Y_train,
    param_name='penalty', param_range=penalties,
    cv=5, scoring='roc_auc'
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)
x = np.arange(len(penalties))
axes[1].bar(x - 0.2, train_mean, 0.4, label='Train AUC',
            color='#1f77b4', alpha=0.8, yerr=train_std, capsize=5)
axes[1].bar(x + 0.2, val_mean, 0.4, label='CV AUC',
            color='#ff7f0e', alpha=0.8, yerr=val_std, capsize=5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(penalties)
axes[1].set_ylabel('AUC')
axes[1].set_title('Validation Curve - Penalty (Logistic Regression)')
axes[1].legend()

plt.suptitle('Logistic Regression Hyperparameter Validation Curves (RFE features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── Logistic Regression Validation Curves (mRMR features) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_scores, val_scores = validation_curve(
    LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
    X_train_mrmr, Y_train_reset,
    param_name='C', param_range=C_range_lr,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], C_range_lr, train_scores, val_scores,
                      'C', 'Validation Curve - C (Logistic Regression)')
axes[0].set_xscale('log')

train_scores, val_scores = validation_curve(
    LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
    X_train_mrmr, Y_train_reset,
    param_name='penalty', param_range=penalties,
    cv=5, scoring='roc_auc'
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)
x = np.arange(len(penalties))
axes[1].bar(x - 0.2, train_mean, 0.4, label='Train AUC',
            color='#1f77b4', alpha=0.8, yerr=train_std, capsize=5)
axes[1].bar(x + 0.2, val_mean, 0.4, label='CV AUC',
            color='#ff7f0e', alpha=0.8, yerr=val_std, capsize=5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(penalties)
axes[1].set_ylabel('AUC')
axes[1].set_title('Validation Curve - Penalty (Logistic Regression)')
axes[1].legend()

plt.suptitle('Logistic Regression Hyperparameter Validation Curves (mRMR features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── SVM Validation Curves (RFE features) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_features_range = [15, 20, 30, 40, 50, 60, 70]
train_scores, val_scores = validation_curve(
    Pipeline([
        ('rfe', RFE(estimator=RandomForestClassifier(random_state=42), step=1)),
        ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42))
    ]),
    X_train, Y_train,
    param_name='rfe__n_features_to_select', param_range=n_features_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], n_features_range, train_scores, val_scores,
                      'n_features_to_select', 'Validation Curve - n_features_to_select (RFE + SVM)')

rfe_svm = RFE(estimator=RandomForestClassifier(random_state=42),
              n_features_to_select=35, step=1)
rfe_svm.fit(X_train, Y_train)
X_train_rfe_35 = rfe_svm.transform(X_train)

C_range_svm = [0.01, 0.1, 1, 5, 10, 50, 100]
train_scores, val_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
    X_train_rfe_35, Y_train,
    param_name='C', param_range=C_range_svm,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1], C_range_svm, train_scores, val_scores,
                      'C', 'Validation Curve - C (SVM on RFE-35 features)')
axes[1].set_xscale('log')

plt.suptitle('SVM Hyperparameter Validation Curves (RFE features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── SVM Validation Curves (mRMR features) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_features_range_mrmr = [15, 20, 30, 40, 50, 60, 70]
train_scores, val_scores = validation_curve(
    Pipeline([
        ('mrmr', MRMRSelector()),
        ('svm', SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42))
    ]),
    X_train, Y_train_reset,
    param_name='mrmr__n_features_to_select', param_range=n_features_range_mrmr,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], n_features_range_mrmr, train_scores, val_scores,
                      'n_features_to_select', 'Validation Curve - n_features_to_select (mRMR + SVM)')

train_scores, val_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
    X_train_mrmr, Y_train_reset,
    param_name='C', param_range=C_range_svm,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1], C_range_svm, train_scores, val_scores,
                      'C', 'Validation Curve - C (SVM op mRMR-45 features)')
axes[1].set_xscale('log')

plt.suptitle('SVM Hyperparameter Validation Curves (mRMR features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── LASSO + Logistic Regression Validation Curves ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

C_range_lasso_lr = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
train_scores, val_scores = validation_curve(
    LogisticRegression(max_iter=10000, penalty='l2', solver='liblinear'),
    X_train_lasso_lr, Y_train,
    param_name='C', param_range=C_range_lasso_lr,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], C_range_lasso_lr, train_scores, val_scores,
                      'C', 'Validation Curve - C (Logistic Regression)')
axes[0].set_xscale('log')

train_scores, val_scores = validation_curve(
    LogisticRegression(max_iter=10000, solver='liblinear'),
    X_train_lasso_lr, Y_train,
    param_name='penalty', param_range=penalties,
    cv=5, scoring='roc_auc'
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)
x = np.arange(len(penalties))
axes[1].bar(x - 0.2, train_mean, 0.4, label='Train AUC',
            color='#1f77b4', alpha=0.8, yerr=train_std, capsize=5)
axes[1].bar(x + 0.2, val_mean, 0.4, label='CV AUC',
            color='#ff7f0e', alpha=0.8, yerr=val_std, capsize=5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(penalties)
axes[1].set_ylabel('AUC')
axes[1].set_title('Validation Curve - Penalty (Logistic Regression)')
axes[1].legend()

alpha_range_lr = [0.001, 0.005, 0.01, 0.05, 0.1]
train_scores, val_scores = validation_curve(
    Pipeline([
        ('lasso', SelectFromModel(Lasso(max_iter=10000, random_state=42),
                                  max_features=10, threshold='mean')),
        ('logistic', LogisticRegression(max_iter=10000, penalty='l2',
                                        solver='liblinear', C=0.1))
    ]),
    X_train, Y_train,
    param_name='lasso__estimator__alpha', param_range=alpha_range_lr,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[2], alpha_range_lr, train_scores, val_scores,
                      'alpha', 'Validation Curve - alpha (LASSO + LR)')
axes[2].set_xscale('log')

plt.suptitle('LASSO + Logistic Regression Hyperparameter Validation Curves',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── LASSO + SVM Validation Curves ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

C_range_lasso_svm = [0.01, 0.05, 0.1, 0.5, 1.0]
train_scores, val_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale', probability=True, random_state=42),
    X_train_lasso_svm, Y_train,
    param_name='C', param_range=C_range_lasso_svm,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], C_range_lasso_svm, train_scores, val_scores,
                      'C', 'Validation Curve - C (SVM)')
axes[0].set_xscale('log')

gamma_range = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
train_scores, val_scores = validation_curve(
    SVC(kernel='rbf', probability=True, random_state=42, C=0.5),
    X_train_lasso_svm, Y_train,
    param_name='gamma', param_range=gamma_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1], gamma_range, train_scores, val_scores,
                      'gamma', 'Validation Curve - gamma (SVM)')
axes[1].set_xscale('log')

alpha_range_svm = [0.001, 0.005, 0.01, 0.05, 0.1]
train_scores, val_scores = validation_curve(
    Pipeline([
        ('lasso', SelectFromModel(Lasso(max_iter=10000, random_state=42),
                                  max_features=15, threshold='mean')),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42, C=0.5))
    ]),
    X_train, Y_train,
    param_name='lasso__estimator__alpha', param_range=alpha_range_svm,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[2], alpha_range_svm, train_scores, val_scores,
                      'alpha', 'Validation Curve - alpha (LASSO + SVM)')
axes[2].set_xscale('log')

plt.suptitle('LASSO + SVM Hyperparameter Validation Curves',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── LASSO + Random Forest Validation Curves ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

split_range = [5, 10, 15, 20]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=150, random_state=42),
    X_train_lasso_rf, Y_train,
    param_name='min_samples_split', param_range=split_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], split_range, train_scores, val_scores,
                      'min_samples_split', 'Validation Curve - min_samples_split (RF)')

depth_range = [1, 3, 5, 8, 10, 15, 20]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=150, random_state=42),
    X_train_lasso_rf, Y_train,
    param_name='max_depth', param_range=depth_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1], depth_range, train_scores, val_scores,
                      'max_depth', 'Validation Curve - max_depth (RF)')

alpha_range_rf = [0.001, 0.005, 0.01, 0.05, 0.1]
train_scores, val_scores = validation_curve(
    Pipeline([
        ('lasso', SelectFromModel(Lasso(max_iter=10000, random_state=42),
                                  max_features=10, threshold='mean')),
        ('rf', RandomForestClassifier(n_estimators=150, random_state=42))
    ]),
    X_train, Y_train,
    param_name='lasso__estimator__alpha', param_range=alpha_range_rf,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[2], alpha_range_rf, train_scores, val_scores,
                      'alpha', 'Validation Curve - alpha (LASSO + RF)')
axes[2].set_xscale('log')

plt.suptitle('LASSO + Random Forest Hyperparameter Validation Curves',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── LASSO + XGBoost Validation Curves ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

depth_range_xgb = [1, 2, 3]
train_scores, val_scores = validation_curve(
    XGBClassifier(learning_rate=0.1, subsample=0.7, colsample_bytree=0.6,
                  random_state=42, eval_metric='logloss'),
    X_train_lasso_xgb, Y_train,
    param_name='max_depth', param_range=depth_range_xgb,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[0], depth_range_xgb, train_scores, val_scores,
                      'max_depth', 'Validation Curve - max_depth (XGBoost)')

lr_range = [0.05, 0.1, 0.2]
train_scores, val_scores = validation_curve(
    XGBClassifier(max_depth=2, subsample=0.7, colsample_bytree=0.6,
                  random_state=42, eval_metric='logloss'),
    X_train_lasso_xgb, Y_train,
    param_name='learning_rate', param_range=lr_range,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[1], lr_range, train_scores, val_scores,
                      'learning_rate', 'Validation Curve - learning_rate (XGBoost)')

alpha_range_xgb = [0.001, 0.005, 0.01, 0.05, 0.1]
train_scores, val_scores = validation_curve(
    Pipeline([
        ('lasso', SelectFromModel(Lasso(max_iter=10000, random_state=42),
                                  max_features=15, threshold='mean')),
        ('xgb', XGBClassifier(max_depth=2, learning_rate=0.1, subsample=0.7,
                               colsample_bytree=0.6, random_state=42,
                               eval_metric='logloss'))
    ]),
    X_train, Y_train,
    param_name='lasso__estimator__alpha', param_range=alpha_range_xgb,
    cv=5, scoring='roc_auc'
)
plot_validation_curve(axes[2], alpha_range_xgb, train_scores, val_scores,
                      'alpha', 'Validation Curve - alpha (LASSO + XGBoost)')
axes[2].set_xscale('log')

plt.suptitle('LASSO + XGBoost Hyperparameter Validation Curves',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


#%% ── Pipeline en param grid ──────────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('variance', VarianceThreshold(threshold=0.1)),
    ('selector', RFE(estimator=RandomForestClassifier(random_state=42))),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = [
    # ── RFE + RF ───────────────────────────────────────────────────────
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [18, 20, 22],
        'selector__step': [1],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [100, 110, 120],
        'clf__max_depth': [3, 4, 5],
        'clf__min_samples_split': [35, 40, 45],
    },
    # ── RFE + SVM ──────────────────────────────────────────────────────
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [25, 30, 35],
        'selector__step': [1],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [0.5, 0.75, 1.0],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # ── RFE + XGBoost ──────────────────────────────────────────────────
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [15, 20, 25],
        'selector__step': [1],
        'clf': [XGBClassifier(random_state=42, eval_metric='logloss')],
        'clf__n_estimators': [150],
        'clf__max_depth': [1, 2],
        'clf__learning_rate': [0.05, 0.1, 0.15],
        'clf__subsample': [0.7],
        'clf__colsample_bytree': [0.6],
        'clf__min_child_weight': [1, 2, 3],
    },
    # ── RFE + Logistic Regression ──────────────────────────────────────
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [10, 20, 30],
        'selector__step': [1],
        'clf': [LogisticRegression(random_state=42, max_iter=1000)],
        'clf__C': [0.01, 0.05, 0.1],
        'clf__penalty': ['l1'],
        'clf__solver': ['liblinear'],
    },
    # ── mRMR + RF ──────────────────────────────────────────────────────
    {
        'selector': [MRMRSelector()],
        'selector__n_features_to_select': [30, 35, 40],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [150],
        'clf__max_depth': [1, 2, 3],
        'clf__min_samples_split': [25, 30, 35],
    },
    # ── mRMR + SVM ─────────────────────────────────────────────────────
    {
        'selector': [MRMRSelector()],
        'selector__n_features_to_select': [30, 35, 40],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [0.01, 0.1, 1],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # ── LASSO + SVM ────────────────────────────────────────────────────
    {
        'selector': [SelectFromModel(Lasso(max_iter=10000))],
        'selector__estimator__alpha': [0.001, 0.005, 0.01],
        'selector__max_features': [12, 15, 18],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [0.3, 0.5, 1.0],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # ── LASSO + Logistic Regression ────────────────────────────────────
    {
        'selector': [SelectFromModel(Lasso(max_iter=10000))],
        'selector__estimator__alpha': [0.001, 0.005, 0.01, 0.05, 0.1],
        'selector__max_features': [10, 15, 20],
        'clf': [LogisticRegression(random_state=42, max_iter=10000)],
        'clf__C': [0.01, 0.05, 0.1, 0.5, 1.0, 5, 10],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear'],
    },
    # ── Passthrough + LASSO als classifier ─────────────────────────────
    {
        'selector': ['passthrough'],
        'clf': [LogisticRegression(random_state=42, max_iter=10000,
                                   penalty='l1', solver='liblinear')],
        'clf__C': [0.001, 0.005, 0.01, 0.05, 0.1],
    },
]


#%% ── Randomized Search ───────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=150,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, Y_train)

print("Best parameters:", random_search.best_params_)
print("Best CV AUC:", random_search.best_score_)


#%% ── Finale GridSearchCV ─────────────────────────────────────────────────────
param_grid_final = [
    # ── RFE + SVM — beste en stabielste model ─────────────────────────
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [25, 30, 35],
        'selector__step': [1],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [0.3, 0.5, 0.75, 1.0, 2.0, 5.0],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # ── RFE + RF — tweede beste ────────────────────────────────────────
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [18, 20, 22, 25],
        'selector__step': [1],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [110, 120, 150],
        'clf__max_depth': [3, 4, 5],
        'clf__min_samples_split': [35, 40, 45],
    },
    # ── mRMR + RF — consistent in top ─────────────────────────────────
    {
        'selector': [MRMRSelector()],
        'selector__n_features_to_select': [30, 35, 40],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [150],
        'clf__max_depth': [1, 2, 3],
        'clf__min_samples_split': [25, 30, 35],
    },
    # ── LASSO + SVM — literatuurconforme aanpak ────────────────────────
    {
        'selector': [SelectFromModel(Lasso(max_iter=10000))],
        'selector__estimator__alpha': [0.001, 0.005, 0.01],
        'selector__max_features': [12, 15, 18],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [0.3, 0.5, 1.0, 2.0, 5.0],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
]

grid_search = GridSearchCV(
    pipeline,
    param_grid_final,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, Y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV AUC:", grid_search.best_score_)


#%% ── Step 11: Extract results and best model info ────────────────────────────
results = pd.DataFrame(grid_search.cv_results_)

best_pipeline = grid_search.best_estimator_
best_clf = best_pipeline.named_steps['clf']
best_selector = best_pipeline.named_steps['selector']

variance_step = best_pipeline.named_steps['variance']
features_after_variance = features[variance_step.get_support()]

if hasattr(best_selector, 'support_'):
    selected_feature_names = features_after_variance[best_selector.support_]
elif hasattr(best_selector, 'selected_features_'):
    selected_feature_names = pd.Index(best_selector.selected_features_)
else:
    selected_feature_names = features_after_variance

if hasattr(best_clf, 'feature_importances_'):
    importances = best_clf.feature_importances_
    indices = np.argsort(importances)[::-1]

def get_selector_name(params):
    selector = params.get('selector', 'passthrough')
    if selector == 'passthrough':
        return 'None'
    elif isinstance(selector, RFE):
        return 'RFE'
    elif isinstance(selector, MRMRSelector):
        return 'mRMR'
    elif hasattr(selector, 'estimator') and isinstance(selector.estimator, Lasso):
        return 'LASSO'
    return 'Other'

def get_clf_name(params):
    clf = params.get('clf')
    return type(clf).__name__

results['selector_name'] = results['params'].apply(get_selector_name)
results['clf_name'] = results['params'].apply(get_clf_name)


#%% ── Step 12: Report-quality plots ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

selectors = results['selector_name'].unique()
classifiers = results['clf_name'].unique()

x = np.arange(len(selectors))
width = 0.2
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

for i, clf_name in enumerate(classifiers):
    means = []
    for sel in selectors:
        subset = results[(results['selector_name'] == sel) &
                         (results['clf_name'] == clf_name)]['mean_test_score']
        means.append(subset.mean() if len(subset) > 0 else 0)
    axes[0].bar(x + i * width, means, width, label=clf_name, color=colors[i], alpha=0.8)

axes[0].set_xticks(x + width * (len(classifiers) - 1) / 2)
axes[0].set_xticklabels(selectors, fontsize=10)
axes[0].set_ylabel('Mean CV AUC')
axes[0].set_title('CV AUC by Selector and Classifier')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.5, 1.0)

y_prob_cv = cross_val_predict(
    best_pipeline, X_train, Y_train,
    cv=cv, method='predict_proba'
)[:, 1]

fpr, tpr, _ = roc_curve(Y_train, y_prob_cv)
roc_auc_cv = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='#2196F3', lw=2,
             label=f'Best model (AUC = {roc_auc_cv:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve - Best Model (CV)')
axes[1].legend(fontsize=9)

n_top = min(15, len(results))
results_sorted = results.sort_values('mean_test_score', ascending=True).tail(n_top)
colors_bar = ['#2196F3' if s == 'RFE' else
               '#4CAF50' if s == 'mRMR' else
               '#FF9800' if s == 'LASSO' else '#9E9E9E'
               for s in results_sorted['selector_name']]

axes[2].barh(range(n_top), results_sorted['mean_test_score'],
             xerr=results_sorted['std_test_score'],
             color=colors_bar, alpha=0.8)
axes[2].set_yticks(range(n_top))
axes[2].set_yticklabels([f"{row['selector_name']} + {row['clf_name']}"
                          for _, row in results_sorted.iterrows()], fontsize=9)
axes[2].set_xlabel('Mean CV AUC')
axes[2].set_title(f'Top {n_top} Combinations')
axes[2].set_xlim(0.5, 1.0)

legend_elements = [Patch(facecolor='#2196F3', label='RFE'),
                   Patch(facecolor='#4CAF50', label='mRMR'),
                   Patch(facecolor='#FF9800', label='LASSO'),
                   Patch(facecolor='#9E9E9E', label='None')]
axes[2].legend(handles=legend_elements, fontsize=8)

plt.suptitle('Pipeline Comparison - Feature Selection vs Classifier',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nTop 5 combinations:")
print(results[['selector_name', 'clf_name', 'mean_test_score', 'std_test_score']]
      .sort_values('mean_test_score', ascending=False).head(5))


#%% ── Top N modellen selecteren ───────────────────────────────────────────────
N = 4

top_results = (results
               .sort_values('mean_test_score', ascending=False)
               .drop_duplicates(subset=['selector_name', 'clf_name'])
               .head(N))

top_pipelines = []
for _, row in top_results.iterrows():
    params = row['params']
    pipeline_copy = clone(best_pipeline)
    pipeline_copy.set_params(**params)
    pipeline_copy.fit(X_train, Y_train)
    top_pipelines.append({
        'pipeline': pipeline_copy,
        'selector': row['selector_name'],
        'clf': row['clf_name'],
        'cv_auc': row['mean_test_score']
    })

print("\nTop N modellen geselecteerd:")
print("-" * 60)
for i, model in enumerate(top_pipelines):
    print(f"{i+1}. {model['selector']} + {model['clf']}")
    print(f"   CV AUC: {model['cv_auc']:.4f}")
    params = top_results.iloc[i]['params']
    clf_params = {k: v for k, v in params.items() if k.startswith('clf__')}
    sel_params = {k: v for k, v in params.items()
                  if k.startswith('selector__') and k != 'selector'}
    print(f"   Selector params: {sel_params}")
    print(f"   Classifier params: {clf_params}")
    print()
