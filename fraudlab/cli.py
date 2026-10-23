"""CLI for downloading, auditing, training, and state-aware stream scoring."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import joblib
import pandas as pd
import yaml
from .data import download, load, validate, write_json
from .features import StreamState, stream_features
from .model import probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("download");d.add_argument("--output",default="data/creditcard.csv")
    a=sub.add_parser("audit");a.add_argument("--data",required=True);a.add_argument("--output",default="runs/audit.json")
    t=sub.add_parser("train");t.add_argument("--data",required=True);t.add_argument("--output",required=True)
    t.add_argument("--config",default="configs/default.yaml")
    s=sub.add_parser("score");s.add_argument("--model",required=True);s.add_argument("--input",required=True)
    s.add_argument("--output",required=True);s.add_argument("--state-in");s.add_argument("--state-out",required=True)
    s.add_argument("--start-empty",action="store_true",help="Explicitly discard historical context")
    e=sub.add_parser("explain");e.add_argument("--model",required=True);e.add_argument("--input",required=True)
    e.add_argument("--output",required=True);e.add_argument("--state-in")
    e.add_argument("--start-empty",action="store_true");e.add_argument("--limit",type=int,default=3)
    args=p.parse_args()
    if args.command=="download": print(json.dumps(download(args.output),indent=2));return
    if args.command=="audit":
        _,report=load(args.data);write_json(args.output,report);print(json.dumps(report,indent=2));return
    if args.command=="train":
        from .run import run
        config=yaml.safe_load(Path(args.config).read_text())
        run(args.data,args.output,config);return
    artifact=joblib.load(args.model)  # Load only your own trusted artifact.
    raw=validate(pd.read_csv(args.input),labeled=False)
    if args.start_empty and args.state_in:p.error("Choose --start-empty or --state-in")
    state=None if args.start_empty else (StreamState(**json.loads(Path(args.state_in).read_text()))
        if args.state_in else artifact["state_before_test"])
    config=artifact["config"]
    features,state=stream_features(raw,state,config["rolling_window"],config["ema_alpha"])
    features=features[artifact["feature_columns"]]
    transformed=artifact["preprocessor"].transform(features)
    if args.command=="explain":
        if args.limit<1:p.error("--limit must be positive")
        from .explain import explain_rows
        explain_rows(artifact,transformed[:args.limit],args.output,
                     config["lime_samples"],config["seed"])
        print(f"Wrote explanations for {min(args.limit,len(raw))} rows to {args.output}");return
    score=probabilities(artifact["pair"],transformed,artifact["params"]["xgb_weight"])
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"Time":raw.Time,"fraud_score":score,
        "flag":(score>=artifact["threshold"]).astype(int),"threshold":artifact["threshold"]}).to_csv(destination,index=False)
    # Nonstandard JSON NaN is never emitted for a missing history observation.
    saved=asdict(state);saved["amounts"]=[x if pd.notna(x) else None for x in saved["amounts"]]
    write_json(args.state_out,saved)
    print(f"Scored {len(raw)} chronological transactions; state written to {args.state_out}")

if __name__=="__main__":main()
