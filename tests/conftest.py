import numpy as np
import pandas as pd
import pytest
from fraudlab.data import INPUTS

@pytest.fixture
def transactions():
    # Toy data is solely for unit tests, never a runtime fallback or benchmark.
    rng=np.random.default_rng(1)
    x=pd.DataFrame(rng.normal(size=(120,30)),columns=INPUTS)
    x["Time"]=np.arange(len(x),dtype=float)
    x["Amount"]=rng.uniform(1,100,len(x))
    x["Class"]=0;x.loc[np.arange(3,120,10),"Class"]=1
    x["source_row"]=np.arange(len(x))
    return x

@pytest.fixture
def config():
    return dict(seed=42,threads=1,cv_folds=2,optuna_trials=1,rf_trees=8,xgb_trees=8,
        imputer_iterations=2,imputer_neighbors=4,correlation_threshold=.85,vif_threshold=10.,
        selector_sample_size=100,rus_ratio=.2,smote_ratio=1.,smote_neighbors=3)
