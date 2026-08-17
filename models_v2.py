"""
models_v2.py — Upgraded central analyses ("The Attention Premium" framing).

H1 (engagement): HIERARCHICAL Bayesian regression with partial pooling of the
    attention-score slope across the four collection strata (source x label).
    Motivation: tweet-count medians differ wildly across strata (collection
    artifacts), so we estimate the style->engagement slope WITHIN strata and
    pool partially — Week 4 hierarchical-models material, and robust to the
    cross-strata proxy problem.

H2 (misinformation, PRIMARY spec): Bayesian logistic of fake ~ RAW style
    features. Interpretable signs per feature, full posteriors.
    The engagement-trained score -> fake spec (models.py) is retained as an
    appendix analysis: its sign flip vs. this model is itself evidence about
    what the engagement proxy measures.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

STYLE = [
    "sensational_ratio", "urgency_ratio", "caps_ratio", "all_caps_words",
    "exclamations", "questions", "clickbait_opener", "second_person",
    "has_number", "vader_pos", "vader_neg", "emotional_intensity",
]

def main(seed=42, draws=2000, tune=2000):
    df = pd.read_parquet("holdout_scored.parquet")  # held-out split B only
    df["log_eng"] = np.log1p(df["n_tweets"])
    df["stratum"] = df["source"] + "_" + df["fake"].map({0: "real", 1: "fake"})
    strata = sorted(df["stratum"].unique())
    sidx = df["stratum"].map({s: i for i, s in enumerate(strata)}).values
    a = (df["attention_score"] - df["attention_score"].mean()).values
    tlen = ((df["title_len_words"] - df["title_len_words"].mean())
            / df["title_len_words"].std()).values

    # ---- H1: hierarchical engagement model ---------------------------------
    with pm.Model(coords={"stratum": strata}) as h1:
        mu_a = pm.Normal("mu_alpha", 0, 5)
        sd_a = pm.HalfNormal("sd_alpha", 2)
        z_a = pm.Normal("z_alpha", 0, 1, dims="stratum")
        alpha_s = pm.Deterministic("alpha_s", mu_a + sd_a * z_a, dims="stratum")
        mu_b = pm.Normal("mu_b_attention", 0, 1)
        sd_b = pm.HalfNormal("sd_b_attention", 1)
        z_b = pm.Normal("z_b", 0, 1, dims="stratum")
        b_s = pm.Deterministic("b_attention_s", mu_b + sd_b * z_b, dims="stratum")
        b_len = pm.Normal("b_title_len", 0, 2)
        sigma = pm.HalfNormal("sigma", 2)
        pm.Normal("y", alpha_s[sidx] + b_s[sidx] * a + b_len * tlen,
                  sigma, observed=df["log_eng"].values)
        tr1 = pm.sample(draws=draws, tune=tune, chains=4, cores=1,
                        random_seed=seed, progressbar=False, target_accept=0.95)

    print("H1 hierarchical: attention-score slope, pooled + per stratum")
    print(az.summary(tr1, var_names=["mu_b_attention", "b_attention_s", "b_title_len"])
          .round(3).to_string())

    # ---- H2 primary: raw style features -> fake ----------------------------
    X = ((df[STYLE] - df[STYLE].mean()) / df[STYLE].std()).values
    gossip = (df["source"] == "gossipcop").astype(float).values
    with pm.Model(coords={"feature": STYLE}) as h2:
        alpha = pm.Normal("alpha", 0, 2.5)
        beta = pm.Normal("beta", 0, 1, dims="feature")
        b_src = pm.Normal("b_gossipcop", 0, 2)
        b_len2 = pm.Normal("b_title_len", 0, 1)
        pm.Bernoulli("y", logit_p=alpha + X @ beta + b_src * gossip + b_len2 * tlen,
                     observed=df["fake"].values)
        tr2 = pm.sample(draws=draws, tune=tune, chains=4, cores=1,
                        random_seed=seed, progressbar=False)

    s2 = az.summary(tr2, var_names=["beta"]).round(3)
    s2.index = [i.replace("beta[", "").rstrip("]") for i in s2.index]
    print("\nH2 primary: fake ~ raw style features (standardized log-odds)")
    print(s2.sort_values("mean", key=abs, ascending=False).to_string())

    tr1.to_netcdf("trace_engagement_hier.nc")
    tr2.to_netcdf("trace_style_misinfo.nc")
    print("\nSaved: trace_engagement_hier.nc, trace_style_misinfo.nc")

if __name__ == "__main__":
    main()
