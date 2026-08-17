"""
upworthy_h1.py — Causal attention-premium analysis on the Upworthy Research
Archive (randomized headline A/B tests).

Data: download the EXPLORATORY packages CSV from https://osf.io/jd64p/
      and pass its path with --csv.

Identification: every comparison is WITHIN a randomized test (same article,
same period, randomly assigned viewers) — outcome and features are centered
within test, so any between-article confounding drops out. The model is a
Bayesian regression of within-test-centered empirical log-odds of clicking
on within-test-centered style features.

Run:  python upworthy_h1.py --csv upworthy-archive-exploratory-packages.csv
"""
import argparse
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from features import headline_features

STYLE = [
    "sensational_ratio", "urgency_ratio", "caps_ratio", "all_caps_words",
    "exclamations", "questions", "clickbait_opener", "second_person",
    "has_number", "vader_pos", "vader_neg", "emotional_intensity",
    "title_len_words",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-tests", type=int, default=8000,
                    help="subsample tests for speed; raise for final run")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    need = {"clickability_test_id", "headline", "impressions", "clicks"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing expected columns: {missing} — "
                         f"columns found: {list(df.columns)[:20]}")

    df = df.dropna(subset=["headline"])
    df = df[df["impressions"] > 0]
    # Keep tests with >= 2 distinct headline arms (within-test contrast exists)
    sizes = df.groupby("clickability_test_id")["headline"].nunique()
    keep = sizes[sizes >= 2].index
    df = df[df["clickability_test_id"].isin(keep)]
    if df["clickability_test_id"].nunique() > args.max_tests:
        chosen = (pd.Series(keep).sample(args.max_tests, random_state=args.seed))
        df = df[df["clickability_test_id"].isin(chosen)]
    print(f"{df['clickability_test_id'].nunique()} tests, {len(df)} arms")

    # Outcome: empirical log-odds of click-through
    df["logit_ctr"] = np.log((df["clicks"] + 0.5) /
                             (df["impressions"] - df["clicks"] + 0.5))
    feats = pd.DataFrame([headline_features(t) for t in df["headline"]],
                         index=df.index)[STYLE]

    # Center outcome and features WITHIN test (fixed-effects equivalent)
    g = df.groupby("clickability_test_id")
    y = (df["logit_ctr"] - g["logit_ctr"].transform("mean")).values
    Xc = feats - feats.groupby(df["clickability_test_id"]).transform("mean")
    sd = Xc.std().replace(0, 1)
    X = (Xc / sd).values
    # Weight arms by impressions (precision of the empirical logit)
    w = np.sqrt(df["impressions"].values / df["impressions"].mean())

    with pm.Model(coords={"feature": STYLE}) as m:
        beta = pm.Normal("beta", 0, 0.5, dims="feature")
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y", X @ beta, sigma / w, observed=y)
        tr = pm.sample(draws=args.draws, tune=args.draws, chains=4, cores=1,
                       random_seed=args.seed, progressbar=False)

    s = az.summary(tr, var_names=["beta"]).round(4)
    s.index = [i.replace("beta[", "").rstrip("]") for i in s.index]
    print("\nCausal effect of style on within-test log-odds of clicking")
    print("(standardized; positive = that feature attracts clicks):")
    print(s.sort_values("mean", key=abs, ascending=False).to_string())
    tr.to_netcdf("trace_upworthy.nc")
    print("\nSaved trace_upworthy.nc")

if __name__ == "__main__":
    main()
