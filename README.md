# AttentionLens — The Attention Premium: Truth, Style, and Engagement in the News Pyramid

*Bayesian Machine Learning with Generative AI Applications* (ADSP 32014),
University of Chicago. Connected to a Forum for Free Inquiry fellowship on how the news
pyramid has changed in the attention economy.

**Central question:** Do articles that use more attention-grabbing language actually get
more engagement, and are they more likely to be misinformation?

**Headline finding:** Misinformation has a distinctive stylistic signature — more
interrogative, negative, capitalized, and sensational — but of those markers only
negativity (and, marginally, sensational vocabulary) is *causally* rewarded with clicks
in randomized headline experiments; the strongest misinformation marker (question-style
headlines) actually reduces them. The "clickbait wins the internet" story is not
supported: the attention economy rewards negativity specifically, not misinformation
style generally.

## Pipeline

```
features.py     headline -> attention-language features (lexicons, VADER sentiment)
models.py       appendix: engagement-trained attention score (split A) + score-based
                models on held-out split B, incl. the proxy-contamination sign flip
models_v2.py    PRIMARY analyses:
                  H1  hierarchical Bayesian regression — attention slope partially
                      pooled across the 4 collection strata (non-centered)
                  H2  Bayesian logistic regression — fake ~ raw style features
upworthy_h1.py  CAUSAL analysis — within-test Bayesian regression on randomized
                Upworthy headline A/B tests
pairs.py        generative–discriminative pair experiment (rule + LLM modes)
app.py          AttentionLens dashboard (Streamlit): paste a headline, see the
                original vs de-sensationalized vs sensationalized versions scored
                with 94% credible-interval bands
final_project_bayes_run.ipynb   end-to-end reproducibility notebook
```

## Data (not redistributed here — download links)

- **FakeNewsNet** (Shu et al.): the four released CSVs from
  https://github.com/KaiDMML/FakeNewsNet (`dataset/*.csv`) go in `data/`.
  21,724 deduplicated headlines with fact-checker labels; engagement proxy =
  tweet-ID counts in the release.
- **Upworthy Research Archive** (Matias et al.): the *exploratory packages* CSV from
  https://osf.io/jd64p/ — 2,607 qualifying randomized headline tests (12,010 arms)
  after filtering to tests with ≥2 distinct headlines.

## Reproduce

```
pip install -r requirements.txt
python features.py
python models.py
python models_v2.py
python upworthy_h1.py --csv <path-to-upworthy-exploratory-packages.csv>
python pairs.py --mode rule --n 300
streamlit run app.py
```

macOS note: `pm.sample(..., cores=1)` is set in the scripts because parallel chains
crash under macOS/Jupyter; on Linux you can restore `cores=4`. Fitted artifacts
(`*.joblib`, `*.parquet`, `*.nc`) are committed, so `streamlit run app.py` works
immediately after cloning without refitting.

## Final results (4 chains × 2,000 draws, target_accept = 0.95; 94% HDIs)

**Misinformation style (H2, Bayesian logistic, held-out split).** Fake-labeled headlines
are credibly more interrogative (questions +0.29 [0.25, 0.33] standardized log-odds),
more negative (+0.19 [0.13, 0.25]), higher-caps (+0.14 [0.08, 0.21]), and more
sensational (+0.09 [0.05, 0.13]); celebrity-clickbait markers (second person −0.18,
clickbait openers −0.09, emotional intensity −0.13) skew *real* because GossipCop's
real class is entertainment news. Misinformation style is negative/alarmist, not
engagement-bait.

**Engagement association (H1, hierarchical, within collection strata).** Small at best:
pooled mu = 0.15 [−0.24, 0.51]; largest stratum slope (gossipcop_fake) 0.29
[0.10, 0.48]; headline length −0.19 [−0.22, −0.16]. Cross-strata engagement
comparisons are unreliable — tweet-count medians range from 6 (politifact_real) to 79
(politifact_fake) for collection-methodology reasons, which motivates the hierarchical
design.

**Causal effects (2,607 randomized Upworthy tests).** Negative sentiment increases
clicks (+0.017 [0.011, 0.023]); sensational vocabulary adds a marginal premium
(+0.010 [0.005, 0.015]); question marks (−0.031 [−0.036, −0.026]), exclamations
(−0.014), and capitalization (−0.009) reduce them.

**Appendix sign flip.** Regressing the fake label on the engagement-trained score gives
a large *negative* coefficient — opposite to the raw-style model — evidence that the
engagement proxy partially encodes FakeNewsNet's collection process rather than organic
virality. Measuring "virality" is itself hard; that is a finding.

**Diagnostics.** R-hat = 1.00 for all reported parameters; the hierarchical model
(non-centered) recorded 29 divergent transitions out of 8,000 post-warmup draws (0.4%)
with estimates stable across runs; all other models sampled without divergences.

## Design decisions

1. **Circularity guard** — the attention score is trained on split A; every hypothesis
   test runs on held-out split B.
2. **Headline length excluded from the score** (length artifact, not attention
   *language*); it enters all models as a control.
3. **Hierarchical partial pooling across source × label strata** instead of trusting
   pooled comparisons on a proxy whose scale differs across differently-collected
   subsets.
4. **Raw-feature H2 as primary; score-based H2 as appendix** — the sign divergence
   between them is evidence about what the proxy measures.
5. **Synthetic pairs are validation/demo only** — never classifier training data.

## Limitations

Headlines only (no article bodies); engagement-proxy validity (collection
methodology); Upworthy external validity (one progressive publisher, 2013–2015,
clicks not shares); AllSides-style outlet-level bias labels inherit to all of an
outlet's articles. GDELT-based longitudinal trend analysis is ongoing fellowship work
beyond this repository.
