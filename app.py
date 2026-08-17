"""
app.py — AttentionLens v2 (run: streamlit run app.py)

Paste a headline or article. The app:
  1. Scores the ORIGINAL: attention score + P(misinformation) with an 89%
     credible interval from the fitted Bayesian logistic posterior.
  2. GENERATES a de-sensationalized and a sensationalized version of the
     same content (LLM if ANTHROPIC_API_KEY is set; transparent rule-based
     fallback otherwise).
  3. Scores all three side by side — the pair experiment, live.

Notes:
  - Models are trained on HEADLINES; for longer text the first line is
    treated as the headline for scoring.
  - This is not a fact-checker. Scores are corpus associations, not
    verdicts about specific claims.
"""
import os
import re
import joblib
import numpy as np
import pandas as pd
import arviz as az
import streamlit as st
from features import headline_features, SENSATIONAL

st.set_page_config(page_title="AttentionLens", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------- artifacts
@st.cache_resource
def load_artifacts():
    bundle = joblib.load("attention_score.joblib")
    post = az.from_netcdf("trace_misinfo.nc").posterior
    holdout = pd.read_parquet("holdout_scored.parquet")
    return bundle, post, holdout

bundle, post, holdout = load_artifacts()
MEAN_SCORE = holdout["attention_score"].mean()

# ------------------------------------------------------------------ scoring
def attention_score(title: str) -> float:
    f = pd.DataFrame([headline_features(title)])[bundle["features"]]
    raw = bundle["model"].predict(bundle["scaler"].transform(f))[0]
    return float(np.searchsorted(bundle["rank_reference"], raw)
                 / len(bundle["rank_reference"]))

def p_fake(score: float):
    a = post["alpha"].values.ravel()
    b = post["b_attention"].values.ravel()
    p = 1 / (1 + np.exp(-(a + b * (score - MEAN_SCORE))))
    return p.mean(), np.percentile(p, 3), np.percentile(p, 97)

# --------------------------------------------------------------- generation
NEUTRALIZE = {w: "" for w in SENSATIONAL} | {
    "slams": "criticizes", "destroys": "rebuts", "blasts": "criticizes",
    "exposed": "reported", "busted": "found",
}
PREFIXES = ["SHOCKING: ", "BREAKING: ", "You Won't Believe: ", "EXPOSED: ",
            "JUST IN: ", "ALERT: "]

def desensationalize_rule(text: str) -> str:
    out = []
    for w in re.findall(r"\S+", text):
        key = re.sub(r"\W", "", w).lower()
        if key in NEUTRALIZE:
            if NEUTRALIZE[key]:
                out.append(NEUTRALIZE[key])
        elif len(key) > 2 and w.isupper():
            out.append(w.capitalize())
        else:
            out.append(w)
    return " ".join(out).replace("!", ".").strip()

def sensationalize_rule(text: str) -> str:
    for p in PREFIXES:  # don't double-prefix already-sensational text
        if text.upper().startswith(p.upper()):
            text = text[len(p):]
    words = text.rstrip(".!? ").split()
    h = sum(ord(c) for c in text)  # deterministic per-headline variety
    if len(words) > 3:
        # upcase the longest content word rather than always the middle one
        idx = max(range(1, len(words)), key=lambda i: (len(words[i]), -i))
        words[idx] = words[idx].upper()
        if h % 3 == 0 and len(words) > 5:
            j = max((i for i in range(1, len(words)) if i != idx),
                    key=lambda i: (len(words[i]), -i))
            words[j] = words[j].upper()
    bang = "!" * (1 + h % 2)
    return PREFIXES[h % len(PREFIXES)] + " ".join(words) + bang

def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))

def generate_llm(text: str, direction: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    if direction == "down":
        instr = ("Rewrite this news text in neutral, classic inverted-pyramid "
                 "style: factual headline, no sensational or emotionally loaded "
                 "language. Keep every factual claim identical.")
    else:
        instr = ("Rewrite this news text in maximally sensational, "
                 "attention-grabbing style. Keep every factual claim identical — "
                 "change only style and tone.")
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=300,
        messages=[{"role": "user",
                   "content": f"{instr}\n\nReply with the rewrite only.\n\nText: {text}"}])
    return msg.content[0].text.strip()

def generate(text: str, direction: str) -> str:
    if llm_available():
        try:
            return generate_llm(text, direction)
        except Exception as e:
            st.warning(f"LLM generation failed ({e}); using rule-based fallback.")
    return (desensationalize_rule(text) if direction == "down"
            else sensationalize_rule(text))

# ----------------------------------------------------------------------- UI
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #F6F7F4; color: #1A2330; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 1150px; }

.al-masthead { border-bottom: 2px solid #1A2330; padding-bottom: .6rem; margin-bottom: .2rem;
  display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; }
.al-title { font-weight: 700; font-size: 1.9rem; letter-spacing: -0.02em; }
.al-title span { color: #2F55D4; }
.al-mode { font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: #5C6672;
  border: 1px solid #C9CEC6; border-radius: 3px; padding: .15rem .5rem; background: #fff; }
.al-tag { font-size: .85rem; color: #5C6672; margin: .4rem 0 1.2rem; }

.stTextArea textarea { background: #FFFFFF; border: 1.5px solid #C9CEC6; border-radius: 6px;
  font-family: 'Source Serif 4', serif; font-size: 1.05rem; color: #1A2330; }
.stTextArea textarea:focus { border-color: #2F55D4; box-shadow: none; }

.al-card { background: #FFFFFF; border: 1px solid #E3E6E0; border-radius: 8px;
  padding: 1.1rem 1.2rem 1.2rem; height: 100%; }
.al-card.original { border-top: 3px solid #1A2330; }
.al-card.variant  { border-top: 3px solid #C9CEC6; }
.al-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; letter-spacing: .12em;
  text-transform: uppercase; color: #5C6672; margin-bottom: .55rem; }
.al-headline { font-family: 'Source Serif 4', serif; font-size: 1.06rem; line-height: 1.45;
  min-height: 4.2rem; margin-bottom: 1rem; }

.al-metric { margin-top: .8rem; }
.al-label { font-size: .74rem; color: #5C6672; display: flex; justify-content: space-between; }
.al-read { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 500; }
.al-read.attn { color: #2F55D4; } .al-read.misinfo { color: #B3452F; }
.al-delta { font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: #5C6672; }

.al-track { position: relative; height: 10px; background: #ECEEE9; border-radius: 5px;
  margin-top: .35rem; overflow: hidden; }
.al-fill { position: absolute; left: 0; top: 0; bottom: 0; background: #2F55D4;
  border-radius: 5px; transition: width .4s ease; }
.al-band { position: absolute; top: 0; bottom: 0; background: rgba(179,69,47,.28);
  border-left: 1px solid #B3452F; border-right: 1px solid #B3452F; }
.al-pt { position: absolute; top: -2px; width: 3px; height: 14px; background: #B3452F; }
.al-ci { font-family: 'IBM Plex Mono', monospace; font-size: .68rem; color: #5C6672; margin-top: .25rem; }

.al-foot { border-top: 1px solid #E3E6E0; margin-top: 1.6rem; padding-top: .9rem;
  font-size: .8rem; color: #5C6672; line-height: 1.55; }
@media (prefers-reduced-motion: reduce) { .al-fill { transition: none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

mode = "LLM rewriter" if llm_available() else "Deterministic rewriter · free, offline"
st.markdown(f"""
<div class="al-masthead">
  <div class="al-title">Attention<span>Lens</span></div>
  <div class="al-mode">{mode}</div>
</div>
<div class="al-tag">Bayesian attention-language measurement with visible uncertainty.
Not a fact-checker: a neutral restatement of a false claim is still false.</div>
""", unsafe_allow_html=True)

text = st.text_area("Paste a headline or article", height=120,
                    placeholder="Paste a headline to measure it...",
                    label_visibility="collapsed")

def card(name: str, v: str, s: float, pm_: float, lo: float, hi: float,
         s_orig: float, original: bool, note: str = "") -> str:
    delta = ("" if original else
             f'<span class="al-delta">{s - s_orig:+.0%} vs original</span>')
    note_html = (f'<div class="al-ci" style="margin:-.5rem 0 .8rem">{note}</div>'
                 if note else "")
    return f"""
<div class="al-card {'original' if original else 'variant'}">
  <div class="al-eyebrow">{name}</div>
  <div class="al-headline">{v}</div>
  {note_html}
  <div class="al-metric">
    <div class="al-label"><span>Predicted-engagement percentile</span>{delta}</div>
    <div class="al-read attn">{s:.0%}</div>
    <div class="al-track"><div class="al-fill" style="width:{s*100:.0f}%"></div></div>
  </div>
  <div class="al-metric">
    <div class="al-label"><span>P(misinformation)</span></div>
    <div class="al-read misinfo">{pm_:.0%}</div>
    <div class="al-track">
      <div class="al-band" style="left:{lo*100:.1f}%; width:{max(hi-lo,0.005)*100:.1f}%"></div>
      <div class="al-pt" style="left:{pm_*100:.1f}%"></div>
    </div>
    <div class="al-ci">94% credible interval {lo:.0%} – {hi:.0%}</div>
  </div>
</div>"""

if text.strip():
    headline = next(l for l in text.splitlines() if l.strip()).strip()
    if headline != text.strip():
        st.caption("Scoring uses the first line as the headline (models are headline-trained).")

    with st.spinner("Generating counterfactual versions..."):
        down = generate(headline, "down")
        up = generate(headline, "up")

    down_note = ("No sensational markers to remove — the original already "
                 "reads as neutral." if down == headline else "")
    versions = [("De-sensationalized" + (" · unchanged" if down_note else ""),
                 down, False, down_note),
                ("Original", headline, True, ""),
                ("Sensationalized", up, False, "")]
    scored = []
    for n, v, o, note in versions:
        s = attention_score(v)
        scored.append((n, v, s, *p_fake(s), o, note))
    s_orig = scored[1][2]

    for col, (n, v, s, pm_, lo, hi, o, note) in zip(st.columns(3, gap="medium"), scored):
        with col:
            st.markdown(card(n, v, s, pm_, lo, hi, s_orig, o, note),
                        unsafe_allow_html=True)

    st.markdown("""
<div class="al-foot">
The engagement percentile is predicted from language features alone (trained on FakeNewsNet,
validated causally against 2,607 randomized Upworthy headline tests). P(misinformation) is a
corpus-level association from a Bayesian logistic regression — the shaded band is the 94%
credible interval, the model saying how sure it is. Counterfactual versions preserve the
original's factual claims and are for style comparison only.
</div>""", unsafe_allow_html=True)
else:
    st.caption("Try a real headline — then compare how its counterfactual versions measure.")
