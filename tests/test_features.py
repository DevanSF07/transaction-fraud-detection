import numpy as np
import pandas as pd
import pytest
from fraudlab.data import INPUTS
from fraudlab.features import stream_features,CollinearitySelector,make_preprocessor

def test_causality_and_batch_equivalence(transactions):
    raw=transactions[INPUTS]
    full,state=stream_features(raw)
    first,s=stream_features(raw.iloc[:60]);second,t=stream_features(raw.iloc[60:],s)
    pd.testing.assert_frame_equal(full,pd.concat([first,second]))
    assert state==t
    modified=raw.copy();modified.loc[60:,"Amount"]=999999
    other,_=stream_features(modified)
    pd.testing.assert_frame_equal(full.iloc[:60],other.iloc[:60])

def test_current_amount_excluded_from_history(transactions):
    x,_=stream_features(transactions[INPUTS])
    assert np.isnan(x.stream_mean_prior10.iloc[0])
    assert x.stream_amount_lag1.iloc[1]==transactions.Amount.iloc[0]
    assert x.stream_ema_prior.iloc[1]==transactions.Amount.iloc[0]

def test_reject_past_transaction(transactions):
    _,state=stream_features(transactions[INPUTS].iloc[:10])
    with pytest.raises(ValueError,match="order"):
        stream_features(transactions[INPUTS].iloc[:1],state)

def test_selector_prunes_linear_redundancy():
    rng=np.random.default_rng(1);a=rng.normal(size=(200,3))
    x=np.column_stack([a,a[:,0]+a[:,1],np.ones(200)])
    selector=CollinearitySelector(correlation_threshold=.99,vif_threshold=10).fit(x)
    assert len(selector.indices_)==3
    assert max(selector.final_vifs_)<=10

def test_preprocessor_does_not_learn_from_validation(transactions,config):
    raw=transactions[INPUTS].copy();raw.loc[1:5,"V1"]=np.nan
    x,_=stream_features(raw)
    prep=make_preprocessor(config);prep.fit(x.iloc[:80])
    imputer=prep.named_steps["mice"]
    initial=imputer.initial_imputer_.statistics_.copy()
    extreme=x.iloc[80:].copy();extreme["V1"]=1e9
    result=prep.transform(extreme)
    np.testing.assert_array_equal(initial,imputer.initial_imputer_.statistics_)
    assert np.isfinite(result).all()
