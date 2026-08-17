"""
models.py — Attention score + the two central Bayesian analyses.

Design (per instructor feedback, with a circularity guard):
  Split A: train a small supervised model (language features -> engagement)
           whose output is the ATTENTION SCORE.
  Split B (held out): test the two hypotheses with Bayesian models:
    H1  log-engagement ~ attention_score + controls   (Bayesian linear reg.)
    H2  P(fake)        ~ attention_score + controls   (Bayesian logistic reg.)
  Training the score on split A and testing on split B avoids the circular
  "score trained on engagement predicts engagement" trap.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "sensational_ratio", "urgency_ratio", "caps_ratio", "all_caps_words",
    "exclamations", "questions", "clickbait_opener", "second_person",
    "has_number", "vader_pos", "vader_neg", "emotional_intensity",
]  # NOTE: title_len_words deliberately excluded from the SCORE (it is a length
   # artifact, not attention *language*); it enters the Bayesian models as a control.

def main(seed: int = 42, draws: int = 1000, tune: int = 1000):
    df = pd.read_parquet("features.parquet")
    df["log_engagement"] = np.log1p(df["n_tweets"])

    # ---- Split A/B --------------------------------------------------------
    A, B = train_test_split(df, test_size=0.5, random_state=seed,
                            stratify=df[["source", "fake"]].astype(str).agg("_".join, axis=1))

    # ---- Attention score: language features -> engagement (trained on A) --
    scaler = StandardScaler().fit(A[FEATURES])
    score_model = Ridge(alpha=1.0).fit(scaler.transform(A[FEATURES]), A["log_engagement"])
    # Score = percentile rank of predicted engagement from language alone
    raw_B = score_model.predict(scaler.transform(B[FEATURES]))
    B = B.assign(attention_score=pd.Series(raw_B, index=B.index).rank(pct=True))

    coef = pd.Series(score_model.coef_, index=FEATURES).sort_values(key=abs, ascending=False)
    print("Attention-score model: top language features (ridge coefs, std'ized):")
    print(coef.round(3).head(8).to_string(), "\n")

    # ---- Bayesian models on held-out B ------------------------------------
    a = (B["attention_score"] - B["attention_score"].mean()).values
    gossip = (B["source"] == "gossipcop").astype(float).values  # crude topic-domain control
    tlen = ((B["title_len_words"] - B["title_len_words"].mean()) / B["title_len_words"].std()).values
    y_eng = B["log_engagement"].values
    y_fake = B["fake"].values

    with pm.Model() as h1:
        alpha = pm.Normal("alpha", 0, 5)
        b_att = pm.Normal("b_attention", 0, 2)
        b_src = pm.Normal("b_gossipcop", 0, 2)
        b_len = pm.Normal("b_title_len", 0, 2)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y", alpha + b_att * a + b_src * gossip + b_len * tlen, sigma, observed=y_eng)
        tr1 = pm.sample(draws=draws, tune=tune, chains=2, cores=2,
                        random_seed=seed, progressbar=False)

    with pm.Model() as h2:
        alpha = pm.Normal("alpha", 0, 2.5)
        b_att = pm.Normal("b_attention", 0, 2)
        b_src = pm.Normal("b_gossipcop", 0, 2)
        b_len = pm.Normal("b_title_len", 0, 2)
        pm.Bernoulli("y", logit_p=alpha + b_att * a + b_src * gossip + b_len * tlen, observed=y_fake)
        tr2 = pm.sample(draws=draws, tune=tune, chains=2, cores=2,
                        random_seed=seed, progressbar=False)

    for name, tr in (("H1: log-engagement ~ attention", tr1),
                     ("H2: P(fake) ~ attention", tr2)):
        s = az.summary(tr, var_names=["b_attention", "b_gossipcop", "b_title_len"])
        print(f"\n{name}")
        print(s.round(3).to_string())

    tr1.to_netcdf("trace_engagement.nc")
    tr2.to_netcdf("trace_misinfo.nc")
    B.to_parquet("holdout_scored.parquet")

    # Persist score pipeline for the dashboard
    import joblib
    joblib.dump({"scaler": scaler, "model": score_model, "features": FEATURES,
                 "rank_reference": np.sort(raw_B)}, "attention_score.joblib")
    print("\nSaved: traces, holdout_scored.parquet, attention_score.joblib")

if __name__ == "__main__":
    main()
