#%% Imports
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.base import BaseEstimator, TransformerMixin
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

#%% Custom mRMR transformer to use in pipeline
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

#%% Build pipeline
pipeline = Pipeline([
    ('selector', RFE(estimator=RandomForestClassifier(random_state=42))),
    ('clf', RandomForestClassifier(random_state=42))
])

#%% Param grid
param_grid = [
    # ── RFE (narrow - already validated) ──────────────────────────────
    # RFE + RF
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [10, 15, 20],
        'selector__step': [1],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [110],
        'clf__max_depth': [4, 5, 6],
        'clf__min_samples_split': [35, 40, 45],
    },
    # RFE + SVM (best so far - keep narrow)
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [25, 30, 35],
        'selector__step': [1],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [10],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # RFE + XGBoost (new - broader)
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [10, 20, 30],
        'selector__step': [1],
        'clf': [XGBClassifier(random_state=42, eval_metric='logloss')],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.3],
    },
    # RFE + Logistic Regression (new - broader)
    {
        'selector': [RFE(estimator=RandomForestClassifier(random_state=42))],
        'selector__n_features_to_select': [10, 20, 30],
        'selector__step': [1],
        'clf': [LogisticRegression(random_state=42, max_iter=1000)],
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear'],
    },

    # ── mRMR (narrow - already validated) ─────────────────────────────
    # mRMR + RF
    {
        'selector': [MRMRSelector()],
        'selector__n_features_to_select': [25, 30, 35],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [150, 200, 250],
        'clf__max_depth': [8, 10, 12],
        'clf__min_samples_split': [35, 40, 45],
    },
    # mRMR + SVM
    {
        'selector': [MRMRSelector()],
        'selector__n_features_to_select': [20, 30, 40],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [10],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # mRMR + XGBoost (new - broader)
    {
        'selector': [MRMRSelector()],
        'selector__n_features_to_select': [10, 20, 30, 40],
        'clf': [XGBClassifier(random_state=42, eval_metric='logloss')],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.3],
    },
    # mRMR + Logistic Regression (new - broader)
    {
        'selector': [MRMRSelector()],
        'selector__n_features_to_select': [10, 20, 30, 40],
        'clf': [LogisticRegression(random_state=42, max_iter=1000)],
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear'],
    },

    # ── LASSO (new - broader) ──────────────────────────────────────────
    # LASSO + RF
    {
        'selector': [SelectFromModel(Lasso(max_iter=10000))],
        'selector__estimator__alpha': [0.001, 0.01, 0.1],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [5, 8, 10],
        'clf__min_samples_split': [20, 40],
    },
    # LASSO + SVM
    {
        'selector': [SelectFromModel(Lasso(max_iter=10000))],
        'selector__estimator__alpha': [0.001, 0.01, 0.1],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [10],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # LASSO + XGBoost
    {
        'selector': [SelectFromModel(Lasso(max_iter=10000))],
        'selector__estimator__alpha': [0.001, 0.01, 0.1],
        'clf': [XGBClassifier(random_state=42, eval_metric='logloss')],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.3],
    },
    # LASSO + Logistic Regression
    {
        'selector': [SelectFromModel(Lasso(max_iter=10000))],
        'selector__estimator__alpha': [0.001, 0.01, 0.1],
        'clf': [LogisticRegression(random_state=42, max_iter=1000)],
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear'],
    },

    # ── Passthrough (narrow - performed poorly) ────────────────────────
    # Passthrough + RF
    {
        'selector': ['passthrough'],
        'clf': [RandomForestClassifier(random_state=42)],
        'clf__n_estimators': [200],
        'clf__max_depth': [10],
        'clf__min_samples_split': [40],
    },
    # Passthrough + SVM
    {
        'selector': ['passthrough'],
        'clf': [SVC(probability=True, random_state=42)],
        'clf__C': [10],
        'clf__kernel': ['rbf'],
        'clf__gamma': ['scale'],
    },
    # Passthrough + XGBoost
    {
        'selector': ['passthrough'],
        'clf': [XGBClassifier(random_state=42, eval_metric='logloss')],
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.1, 0.3],
    },
    # Passthrough + Logistic Regression
    {
        'selector': ['passthrough'],
        'clf': [LogisticRegression(random_state=42, max_iter=1000)],
        'clf__C': [0.1, 1],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear'],
    },
]

#%% Cross-validated randomized search
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

#%% Extract results and best model info
results = pd.DataFrame(random_search.cv_results_)

best_pipeline = random_search.best_estimator_
best_clf = best_pipeline.named_steps['clf']
best_selector = best_pipeline.named_steps['selector']

# Get selected feature names depending on selector type
if hasattr(best_selector, 'support_'):
    selected_feature_names = features[best_selector.support_]
elif hasattr(best_selector, 'selected_features_'):
    selected_feature_names = pd.Index(best_selector.selected_features_)
else:
    selected_feature_names = features  # passthrough case

#%% Helper functions to extract selector and classifier names
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

#%% Plots
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# ── Plot 1: Grouped bar chart of CV AUC by selector and classifier ─────
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

axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(selectors, fontsize=10)
axes[0].set_ylabel('Mean CV AUC')
axes[0].set_title('CV AUC by Selector and Classifier')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.5, 1.0)

# ── Plot 2: Cross-validated ROC curve of best model ────────────────────
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

# ── Plot 3: Top 15 combinations colored by selector ────────────────────
results_sorted = results.sort_values('mean_test_score', ascending=True).tail(15)
colors_bar = ['#2196F3' if s == 'RFE' else
               '#4CAF50' if s == 'mRMR' else
               '#FF9800' if s == 'LASSO' else '#9E9E9E'
               for s in results_sorted['selector_name']]

axes[2].barh(range(15), results_sorted['mean_test_score'],
             xerr=results_sorted['std_test_score'],
             color=colors_bar, alpha=0.8)
axes[2].set_yticks(range(15))
axes[2].set_yticklabels([f"{row['selector_name']} + {row['clf_name']}"
                          for _, row in results_sorted.iterrows()], fontsize=9)
axes[2].set_xlabel('Mean CV AUC')
axes[2].set_title('Top 15 Combinations')
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

# Print top 5 combinations
print("\nTop 5 combinations:")
print(results[['selector_name', 'clf_name', 'mean_test_score', 'std_test_score']]
      .sort_values('mean_test_score', ascending=False).head(5))