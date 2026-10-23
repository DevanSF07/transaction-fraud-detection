import numpy as np
import pytest
from fraudlab.data import INPUTS,validate,load,chronological_parts,inject_missing

def test_no_synthetic_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):load(tmp_path/"missing.csv")

def test_schema_and_labels(transactions):
    with pytest.raises(ValueError):validate(transactions)
    x=transactions.drop(columns="source_row");validate(x)
    x.loc[0,"Class"]=np.nan
    with pytest.raises(ValueError):validate(x)

def test_exact_mask_count_and_target_preservation(transactions):
    parts=chronological_parts(transactions)
    masked,masks=inject_missing(parts,137)
    assert sum(len(m) for m in masks)==137
    assert sum(p[INPUTS].isna().sum().sum() for p in masked)==137
    for original,new in zip(parts,masked):
        assert original.Class.equals(new.Class)
        assert original.Time.equals(new.Time)
    again,_=inject_missing(parts,137)
    for a,b in zip(masked,again):assert a.equals(b)

def test_tied_time_not_split(transactions):
    transactions.loc[70:76,"Time"]=70
    a,b,c=chronological_parts(transactions)
    assert a.Time.max()<b.Time.min()<c.Time.min()
    assert b.Time.max()<c.Time.min()

def test_load_deduplicates_and_detects_conflicts(transactions,tmp_path):
    import pandas as pd
    x=transactions.drop(columns="source_row")
    path=tmp_path/"data.csv"
    pd.concat([x,x.iloc[:1]]).to_csv(path,index=False)
    data,audit=load(path)
    assert len(data)==120 and audit["exact_duplicate_rows"]==1
    conflict=x.iloc[:1].copy();conflict["Class"]=1
    pd.concat([x,conflict]).to_csv(path,index=False)
    with pytest.raises(ValueError,match="conflicting"):load(path)
