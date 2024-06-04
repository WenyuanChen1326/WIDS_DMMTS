import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.metrics import roc_curve, auc
from fairlearn import metrics
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, EqualizedOdds

import warnings
warnings.filterwarnings('ignore')

truth_col = "Y_truth_%s"
pred_col = [
    "y_pred_%d_lgb", 
    "y_pred_%d_rf", 
    "y_pred_%d_log_reg", 
    "y_pred_%d_l1"
    ]

def calAuc(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    return auc(fpr, tpr)
    
def read_csv(sep, test=True):
    df = pd.read_csv(f"./Output/{'test' if test else 'train'}_impute_{sep}.csv")
    for k in df:
        if str(df[k].dtype) == 'object':
            df[k] = df[k].astype('category')
    y_true = df[truth_col % sep]
    y_pred = {}
    if test:
        for col in pred_col:
            y = df[col % sep]
            fpr, tpr, thresholds = roc_curve(y_true, y)
            threshold = thresholds[np.argmax(tpr-fpr)]
            y_pred[col[10:]] = np.array(y > threshold, dtype=float)
    df.drop(truth_col % sep, axis=1, inplace=True)
    if test:
        df.drop([col % sep for col in pred_col], axis=1, inplace=True)
    
    return df, y_true, y_pred

def discretizate(y, y_pred):
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return np.array(y_pred > best_threshold, dtype=float)

metricsDi = {
    "demographic_parity": metrics.selection_rate, 
    'equalized_odds(FPR)': metrics.false_positive_rate, 
    'equalized_odds(TPR)': metrics.true_positive_rate, 
    }
metricsFunc = [
    metrics.demographic_parity_difference, 
    metrics.demographic_parity_ratio, 
    metrics.equalized_odds_difference, 
    metrics.equalized_odds_ratio, 
    ]

def cal(y_test, y_test_pred, f):
    # print(calAuc(y_test, y_test_pred))
    # y_test_pred = discretizate(y_test, y_test_pred)
    tpr_summary = metrics.MetricFrame(
        metrics=metricsDi,
        y_true=y_test,
        y_pred=y_test_pred,
        sensitive_features=f)
    
    return tpr_summary.overall, \
           tpr_summary.by_group, \
           {func.__name__:func(y_true=y_test,
                               y_pred=y_test_pred,
                               sensitive_features=f) 
            for func in metricsFunc}
           
LGB_PARAS = {
    'objective': 'binary',
    'metric': 'auc', 
    'is_unbalance': True,
    'boosting_type': 'dart', 
    'n_estimators': 50,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'subsample_freq': 10,
    'verbose': -1,
}

# LGB_PARAS = {
#     'bagging_fraction': 0.8, 
#     'bagging_freq': 5, 
#     'boosting_type': 'gbdt', 
#     'feature_fraction': 0.9, 
#     'is_unbalanced': True, 
#     'learning_rate': 0.1, 
#     'metric': 'binary_logloss', 
#     'n_estimators': 50, 
#     'num_leaves': 31, 
#     'objective': 'binary', 
#     'verbose': -1, 
#     'verbosity': -1
#     }

def mitigate(X_train, y_train, X_test, featureName, constraints):
    model = lgb.LGBMClassifier(**LGB_PARAS)
    mitigator = ExponentiatedGradient(model, constraints=(constraints))
    mitigator.fit(X_train, y_train, sensitive_features=X_train[featureName])
    return mitigator.predict(X_test)

def write(true_data, pred_data, feature, excelName):
    di1 = {}
    di2 = {}
    di3 = {}
    for k, v in pred_data.items():
        overall, by_group, diff_ration = cal(true_data, v, feature)
        di1[k] = overall
        di2[k] = by_group
        di3[k] = pd.Series(diff_ration)
    overall = pd.DataFrame(di1)
    by_group = pd.concat(di2, axis=1)
    diff_ration = pd.DataFrame(di3)
    
    with pd.ExcelWriter(excelName) as writer:
        overall.to_excel(writer, sheet_name='overall')
        by_group.to_excel(writer, sheet_name='by_group')
        diff_ration.to_excel(writer, sheet_name='diff_ration')
    
for sep in [30, 60, 90]:
    X_test, true_data, pred_data = read_csv(sep)
    X_train, y_train, _ = read_csv(sep, False)
    
    payer_type = list(str(x) for x in X_test['payer_type'])
    patient_race = list(str(x) for x in X_test['patient_race'])
    for featureName, f in {'payer_type':payer_type, 'patient_race':patient_race}.items():
        pred_data['lgb_mitigated_dp'] = mitigate(X_train, y_train, X_test, featureName, DemographicParity(difference_bound=0.02))
        pred_data['lgb_mitigated_eo'] = mitigate(X_train, y_train, X_test, featureName, EqualizedOdds(difference_bound=0.02))
        
        write(true_data, pred_data, f, f'output_{sep}_{featureName}.xlsx')
        








