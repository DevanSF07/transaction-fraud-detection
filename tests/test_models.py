import numpy as np
import joblib
from fraudlab.data import INPUTS
from fraudlab.features import stream_features
from fraudlab.model import resample,prepare_folds,fit_pair,probabilities,tune
from fraudlab.metrics import select_threshold,evaluate

def test_resampling_balance(transactions,config):
    x=transactions[INPUTS].to_numpy();y=transactions.Class.to_numpy()
    a,b,report=resample(x,y,config)
    assert sum(b==0)==sum(b==1)
    assert report["after_rus"]["fraud"]==int(y.sum())
    assert report["after_smote"]["fraud"]>=int(y.sum())

def test_threshold_meets_validation_recall():
    y=np.array([0,0,1,0,1,1]);s=np.array([.1,.6,.9,.4,.5,.3])
    threshold=select_threshold(y,s,2/3)
    assert threshold==.5
    metrics=evaluate(y,s,threshold)
    assert metrics["confusion_matrix"]==dict(tn=2,fp=1,fn=1,tp=2)
    assert metrics["recall"]==2/3

def test_fold_training_and_serialization(transactions,config,tmp_path):
    X,_=stream_features(transactions[INPUTS]);y=transactions.Class.to_numpy()
    folds,metadata=prepare_folds(X,y,transactions.Time.to_numpy(),config)
    assert all(m["train_end_time"]<m["validation_start_time"] for m in metadata)
    params,trials=tune(folds,config)
    a,b,c,d=folds[-1];pair=fit_pair(a,b,params,config)
    before=probabilities(pair,c,params["xgb_weight"])
    path=tmp_path/"models.joblib";joblib.dump(pair,path)
    np.testing.assert_allclose(before,probabilities(joblib.load(path),c,params["xgb_weight"]))

def test_weighted_tree_shap_matches_score(transactions,config):
    from fraudlab.explain import tree_shap
    x=transactions[INPUTS].to_numpy();y=transactions.Class.to_numpy()
    params=dict(rf_depth=3,rf_leaf=1,rf_features="sqrt",xgb_depth=2,learning_rate=.1,
        min_child_weight=1,subsample=1.,colsample=1.,reg_lambda=1.)
    pair=fit_pair(x,y,params,config)
    values,base,residual=tree_shap(pair,x[:3],x[:30],.6)
    assert residual.max()<1e-4
    np.testing.assert_allclose(values.sum(axis=1)+base,probabilities(pair,x[:3],.6),atol=1e-4)
