"""
pairs.py — Generative–discriminative loop (Week 7 material).

For each input headline, GENERATE a style-matched counterfactual pair
(neutral vs sensationalized, same content), then DISCRIMINATE: score both
with the attention-language pipeline and test whether the model separates
them, reporting the posterior of the within-pair score gap.

Generator modes:
  --mode rule       offline rule-based sensationalizer. SANITY CHECK ONLY:
                    it inserts terms from the same lexicon the features
                    count, so near-perfect discrimination is expected and
                    NOT evidence. Use to verify plumbing.
  --mode anthropic  LLM generator (run locally with ANTHROPIC_API_KEY set).
                    This is the real experiment: the generator's mechanism
                    is independent of the feature lexicon, so discrimination
                    performance is meaningful.

Synthetic pairs are for VALIDATION AND DEMO ONLY — never train the
misinformation classifier on them (you'd learn to detect generator style,
not misinformation).
"""
import argparse
import random
import numpy as np
import pandas as pd
import joblib
from features import headline_features

PREFIXES = ["SHOCKING: ", "BREAKING: ", "You Won't Believe: ", "EXPOSED: "]
INTENSIFIERS = ["absolutely ", "totally ", "completely "]

def sensationalize_rule(title: str, rng: random.Random) -> str:
    t = title.rstrip(".!? ")
    words = t.split()
    if len(words) > 3:
        i = rng.randrange(1, len(words) - 1)
        words[i] = words[i].upper()
    t = " ".join(words)
    return rng.choice(PREFIXES) + t + "!"

def sensationalize_llm(titles, model="claude-sonnet-4-6"):
    """LLM generator — run locally. Rewrites each headline in maximally
    attention-grabbing style WITHOUT changing its factual claims."""
    import os, anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    out = []
    for t in titles:
        msg = client.messages.create(
            model=model, max_tokens=100,
            messages=[{"role": "user", "content":
                "Rewrite this headline in maximally sensational, attention-grabbing "
                "style. Keep every factual claim identical — change only style/tone. "
                "Reply with the rewritten headline only.\n\nHeadline: " + t}])
        out.append(msg.content[0].text.strip())
    return out

def attention_scores(titles, bundle):
    f = pd.DataFrame([headline_features(t) for t in titles])[bundle["features"]]
    raw = bundle["model"].predict(bundle["scaler"].transform(f))
    return np.searchsorted(bundle["rank_reference"], raw) / len(bundle["rank_reference"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rule", "anthropic"], default="rule")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bundle = joblib.load("attention_score.joblib")
    holdout = pd.read_parquet("holdout_scored.parquet")
    # Use REAL (non-fake) headlines as the neutral member of each pair
    neutral = (holdout.loc[holdout["fake"] == 0, "title"]
               .sample(args.n, random_state=args.seed).tolist())

    if args.mode == "rule":
        sensational = [sensationalize_rule(t, rng) for t in neutral]
        print("[mode=rule] Sanity check only — generator shares the feature "
              "lexicon; near-perfect separation is expected, not evidence.\n")
    else:
        sensational = sensationalize_llm(neutral)

    s_neu = attention_scores(neutral, bundle)
    s_sen = attention_scores(sensational, bundle)
    gap = s_sen - s_neu

    # Style discriminator: standardized sum of sensational-direction features
    STYLE_UP = ["sensational_ratio", "caps_ratio", "all_caps_words",
                "exclamations", "vader_neg", "urgency_ratio"]
    fn = pd.DataFrame([headline_features(t) for t in neutral])
    fs = pd.DataFrame([headline_features(t) for t in sensational])
    allf = pd.concat([fn, fs]); mu, sig = allf[STYLE_UP].mean(), allf[STYLE_UP].std() + 1e-9
    style_gap = (((fs[STYLE_UP] - mu) / sig).sum(1).values
                 - ((fn[STYLE_UP] - mu) / sig).sum(1).values)

    # Bayesian posterior for the mean within-pair gap (conjugate normal,
    # weakly-informative prior) + pairwise discrimination accuracy
    n = len(gap); m = gap.mean(); sd = gap.std(ddof=1)
    post_sd = sd / np.sqrt(n)
    draws = np.random.default_rng(args.seed).normal(m, post_sd, 20000)
    acc = (gap > 0).mean()

    print(f"n pairs: {n}")
    print("STYLE discriminator (which version is sensationalized?):")
    print(f"  accuracy: {(style_gap > 0).mean():.1%}   mean gap {style_gap.mean():+.2f}")
    print("ENGAGEMENT-trained attention score (does sensationalizing raise predicted engagement?):")
    print(f"  sensational scored higher in {acc:.1%} of pairs")
    print(f"  mean within-pair score gap: {m:+.3f}  "
          f"[89% interval {np.percentile(draws,5.5):+.3f}, {np.percentile(draws,94.5):+.3f}]")
    pd.DataFrame({"neutral": neutral, "sensational": sensational,
                  "score_neutral": s_neu, "score_sensational": s_sen,
                  "gap": gap, "style_gap": style_gap}).to_parquet(f"pairs_{args.mode}.parquet")
    print(f"Saved pairs_{args.mode}.parquet")

if __name__ == "__main__":
    main()
