#%% Imports
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve


# Load preprocessed data
data = joblib.load('preprocessed_data.pkl')

X_train = data['X_train']
X_test  = data['X_test']
Y_train = data['Y_train']
Y_test  = data['Y_test']
features = data['features']

print("Data loaded successfully.")


#%% Build pipeline

pipeline = Pipeline([
    ('rfe', RFE(estimator=RandomForestClassifier(random_state=42))),
    ('clf', RandomForestClassifier(random_state=42))
])

#%% Define hyperparameters to search
param_grid = {
    'rfe__n_features_to_select': [10, 15, 20],   # how many features RFE keeps
    'rfe__step': [1],                          # how many features removed per iteration
    'clf__n_estimators': [110],              # number of trees
    'clf__max_depth': [4, 5, 6],              # max tree depth
    'clf__min_samples_split': [15, 20, 30, 40],             # min samples to split a node
}

#%% OOB score for quick evaluation of hyperparameters (not used in final grid search)

# oob_scores = []
# n_estimators_range = range(10, 300, 10)

# for n in n_estimators_range:
#     rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
#     rf.fit(X_train, Y_train)
#     oob_scores.append(rf.oob_score_)

# plt.figure(figsize=(10, 5))
# plt.plot(n_estimators_range, oob_scores)
# plt.xlabel('Number of trees')
# plt.ylabel('OOB Score')
# plt.title('OOB Score vs Number of Trees')
# plt.axvline(x=n_estimators_range[oob_scores.index(max(oob_scores))], 
#             color='r', linestyle='--', label=f'Best: {n_estimators_range[oob_scores.index(max(oob_scores))]} trees')
# plt.legend()
# plt.show()

# print(f"Optimal n_estimators: {n_estimators_range[oob_scores.index(max(oob_scores))]}")

#%% learning curve


#%% Fit RFE first
# rfe = RFE(estimator=RandomForestClassifier(random_state=42), 
#           n_features_to_select=15, step=1)
# rfe.fit(X_train, Y_train)

# X_train_rfe = rfe.transform(X_train)

#%% learning curve for min samples split

# splits = [15, 20, 30, 40, 50]
# train_scores, val_scores = validation_curve(
#     RandomForestClassifier(n_estimators=100, random_state=42),
#     X_train_rfe, Y_train,
#     param_name='min_samples_split',
#     param_range=splits,
#     cv=5,
#     scoring='roc_auc'
# )

# plt.plot(splits, train_scores.mean(axis=1), label='Train AUC')
# plt.plot(splits, val_scores.mean(axis=1), label='CV AUC')
# plt.xlabel('Min Samples Split')
# plt.ylabel('AUC')
# plt.title('Validation Curve - Min Samples Split')
# plt.legend()
# plt.show()

#%% Then run validation curve on reduced features
# depths = [3, 5, 7, 10, 15, 20, None]
# train_scores, val_scores = validation_curve(
#     RandomForestClassifier(n_estimators=110, random_state=42),
#     X_train_rfe, Y_train,      # use reduced features here
#     param_name='max_depth',
#     param_range=depths,
#     cv=5,
#     scoring='roc_auc'
# )

# plt.plot(range(len(depths)), train_scores.mean(axis=1), label='Train AUC')
# plt.plot(range(len(depths)), val_scores.mean(axis=1), label='CV AUC')
# plt.xticks(range(len(depths)), [str(d) for d in depths])
# plt.xlabel('Max Depth')
# plt.ylabel('AUC')
# plt.title('Validation Curve - Max Depth')
# plt.legend()
# plt.show()

#%% Cross-validated grid search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, Y_train)

#%% Evaluate best model
print("Best parameters:", grid_search.best_params_)
print("Best CV AUC:", grid_search.best_score_)


#%% Plot CV results and feature importances
results = pd.DataFrame(grid_search.cv_results_)
results_sorted = results.sort_values('mean_test_score', ascending=True)

best_pipeline = grid_search.best_estimator_
selected_mask = best_pipeline.named_steps['rfe'].support_
selected_feature_names = features[selected_mask]
importances = best_pipeline.named_steps['clf'].feature_importances_
indices = np.argsort(importances)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# CV AUC for all combinations
axes[0].barh(range(len(results_sorted)), results_sorted['mean_test_score'],
             xerr=results_sorted['std_test_score'])
axes[0].set_yticks(range(len(results_sorted)))
axes[0].set_yticklabels([str(p) for p in results_sorted['params']], fontsize=6)
axes[0].set_xlabel('Mean CV AUC')
axes[0].set_title('CV AUC for all hyperparameter combinations')

# Feature importances
axes[1].barh(range(len(importances)), importances[indices][::-1])
axes[1].set_yticks(range(len(importances)))
axes[1].set_yticklabels(selected_feature_names[indices][::-1], fontsize=8)
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importances of selected features')

plt.tight_layout()
plt.show()

# Print top 5 combinations
print("\nTop 5 combinations:")
print(results_sorted[['params', 'mean_test_score', 'std_test_score']].tail(5))


# y_pred = grid_search.predict(X_test)
# y_prob = grid_search.predict_proba(X_test)[:, 1]

# print("\nClassification Report:")
# print(classification_report(Y_test, y_pred, target_names=['T12', 'T34']))
# print("Test AUC:", roc_auc_score(Y_test, y_prob))
# print("Confusion Matrix:")
# print(confusion_matrix(Y_test, y_pred))