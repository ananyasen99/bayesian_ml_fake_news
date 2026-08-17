"""
features.py — Attention-language feature extraction from headlines.

Offline-friendly: uses VADER sentiment + handcrafted clickbait/sensationalism
features. Swap in transformer emotion scores later via add_transformer_emotions()
(requires internet access to Hugging Face; run locally, not required for v1).
"""
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SENSATIONAL = {
    "shocking", "breaking", "bombshell", "explosive", "devastating", "insane",
    "unbelievable", "outrageous", "stunning", "horrifying", "terrifying",
    "slams", "destroys", "obliterates", "exposed", "busted", "scandal",
    "secret", "banned", "shameful", "disgusting", "epic", "viral", "chaos",
    "meltdown", "furious", "rage", "erupts", "blasts", "humiliates",
}
URGENCY = {
    "now", "just", "urgent", "alert", "warning", "must", "before", "hurry",
    "immediately", "tonight", "today", "finally", "revealed",
}
CLICKBAIT_OPENERS = re.compile(
    r"^(this|these|here('s| is)|what|why|how|you won'?t|the real reason|watch)\b", re.I
)

_vader = SentimentIntensityAnalyzer()


def headline_features(title: str) -> dict:
    t = str(title)
    words = re.findall(r"[A-Za-z']+", t)
    lw = [w.lower() for w in words]
    n = max(len(words), 1)
    letters = [c for c in t if c.isalpha()]
    vs = _vader.polarity_scores(t)
    return {
        "sensational_ratio": sum(w in SENSATIONAL for w in lw) / n,
        "urgency_ratio": sum(w in URGENCY for w in lw) / n,
        "caps_ratio": (sum(c.isupper() for c in letters) / max(len(letters), 1)),
        "all_caps_words": sum(1 for w in words if len(w) > 2 and w.isupper()) / n,
        "exclamations": t.count("!"),
        "questions": t.count("?"),
        "clickbait_opener": int(bool(CLICKBAIT_OPENERS.search(t.strip()))),
        "second_person": sum(w in {"you", "your", "you're", "yours"} for w in lw) / n,
        "has_number": int(bool(re.search(r"\d", t))),
        "vader_pos": vs["pos"],
        "vader_neg": vs["neg"],
        "emotional_intensity": abs(vs["compound"]),
        "title_len_words": len(words),
    }


def load_fakenewsnet(data_dir: str = "data") -> pd.DataFrame:
    frames = []
    for source in ("politifact", "gossipcop"):
        for label_name, label in (("fake", 1), ("real", 0)):
            df = pd.read_csv(f"{data_dir}/{source}_{label_name}.csv")
            df["source"] = source
            df["fake"] = label
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["title"]).drop_duplicates(subset=["title"])
    # Engagement proxy: number of tweets sharing the article
    df["n_tweets"] = df["tweet_ids"].fillna("").astype(str).str.split("\t").map(
        lambda x: len([t for t in x if t.strip()])
    )
    # Outlet domain (for AllSides mapping / outlet-level effects later)
    df["domain"] = (
        df["news_url"].fillna("").astype(str)
        .str.replace(r"^https?://", "", regex=True)
        .str.replace(r"^www\.", "", regex=True)
        .str.split("/").str[0]
    )
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame([headline_features(t) for t in df["title"]], index=df.index)
    return pd.concat([df[["title", "source", "fake", "n_tweets", "domain"]], feats], axis=1)


if __name__ == "__main__":
    df = load_fakenewsnet()
    table = build_feature_table(df)
    table.to_parquet("features.parquet")
    print(f"{len(table)} articles, {table['fake'].mean():.1%} fake")
    print(table.filter(like="ratio").describe().round(4))
