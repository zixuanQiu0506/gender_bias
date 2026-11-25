"""
CS 341NL Final Project
Predicting Gender Bias in Machine Translation
Zixuan Qiu, Spring 2026
"""

import csv
import os
import re
import time
import json


try:
    from deep_translator import GoogleTranslator
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report, precision_score,
                                 recall_score)
    import matplotlib
    matplotlib.use("Agg")          # works without a display window
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"\nMissing library: {e}")
    print("Please run:  pip install scikit-learn numpy matplotlib deep-translator")
    exit(1)


DATA_DIR        = "data"
RESULTS_DIR     = "results"
WINOMT_FILE     = os.path.join(DATA_DIR, "winomt.tsv")
TRANSLATED_FILE = os.path.join(DATA_DIR, "translated_es.csv")   
LABELED_FILE    = os.path.join(DATA_DIR, "labeled.csv")
FEATURES_FILE   = os.path.join(DATA_DIR, "features.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# STEP 1: Load WinoMT

def load_winomt(filepath):
    """
    Read winomt.tsv.
    Format (tab-separated): gold_gender | entity_idx | sentence | entity
    The pronoun is found by scanning the sentence itself.
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            sentences.append({
                "gold_gender": parts[0],
                "entity_idx":  parts[1],
                "sentence":    parts[2],
                "entity":      parts[3],
            })
    return sentences



# STEP 2: Translate English to Spanish

def translate_all(sentences, cache_path):
    """
    Translate every sentence to Spanish using Google Translate (free).
    Saves results to a cache file so we only do this once.
    """
    # If cache already exists, just load it
    if os.path.exists(cache_path):
        print(f"  Found cached translations at {cache_path}. Skipping API calls.")
        translations = []
        with open(cache_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                translations.append(row["translation"])
        return translations

    print(f"  Translating {len(sentences)} sentences to Spanish...")
    print("  This will take about 15 minutes. Please wait.")
    translator = GoogleTranslator(source="en", target="es")
    translations = []

    for i, item in enumerate(sentences):
        if i % 200 == 0:
            print(f"    Progress: {i}/{len(sentences)}")
        try:
            t = translator.translate(item["sentence"])
            translations.append(t if t else "")
        except Exception:
            translations.append("")
        time.sleep(0.05)   # avoid rate limit

    # Save to cache
    with open(cache_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sentence", "translation"])
        writer.writeheader()
        for item, t in zip(sentences, translations):
            writer.writerow({"sentence": item["sentence"], "translation": t})

    print(f"  Saved translations to {cache_path}")
    return translations



# STEP 3: Label each sentence as biased (1) or not biased (0)

def extract_predicted_gender(translation):
    """
    Look at the Spanish translation and guess what gender was assigned.
    We check for Spanish gendered articles: la/una = female, el/un = male.
    If no articles found, we look at word endings (-a = female, -o = male).
    Returns 'female', 'male', or 'unknown'.
    """
    t = translation.lower()

    # Method 1: look for gendered article followed by a word
    matches = re.findall(r'\b(la|el|una|un)\s+\w+', t)
    if matches:
        article = matches[0].split()[0]
        if article in ("la", "una"):
            return "female"
        if article in ("el", "un"):
            return "male"

    # Method 2: look at word endings
    for word in t.split():
        word = word.strip(".,;:?!\"'")
        if len(word) < 3:
            continue
        if word.endswith(("ora", "era", "esa", "ista")):
            return "female"
        if word.endswith(("or", "er")):
            return "male"
        if word.endswith("a"):
            return "female"
        if word.endswith("o"):
            return "male"

    return "unknown"


def label_sentences(sentences, translations):
    """
    For each sentence, compare predicted gender to gold gender.
    bias_label = 0 if translation was correct
    bias_label = 1 if translation was biased (wrong gender)
    """
    labeled = []
    skipped = 0
    for item, translation in zip(sentences, translations):
        if not translation:
            skipped += 1
            continue
        predicted = extract_predicted_gender(translation)
        if predicted == "unknown":
            skipped += 1
            continue
        item = dict(item)  
        item["translation"]      = translation
        item["predicted_gender"] = predicted
        item["bias_label"]       = 0 if predicted == item["gold_gender"] else 1
        labeled.append(item)

    print(f"  Labeled {len(labeled)} sentences. Skipped {skipped} (unknown gender).")
    if not labeled:
        print("ERROR: No sentences were labeled. Check that your winomt.tsv file is correct.")
        exit(1)
    biased = sum(r["bias_label"] for r in labeled)
    print(f"  Biased: {biased} ({100*biased/len(labeled):.1f}%), "
          f"Not biased: {len(labeled)-biased} ({100*(len(labeled)-biased)/len(labeled):.1f}%)")
    return labeled



# STEP 4: Extract features from the English source sentences


# Jobs typically associated with women (source: US Bureau of Labor Statistics,
# same list used by Stanovsky et al. 2019)
FEMALE_TYPICAL = {
    "nurse", "secretary", "receptionist", "housekeeper", "cook",
    "attendant", "cashier", "teacher", "librarian", "hairdresser",
    "cleaner", "maid", "nanny", "socialworker", "social worker",
    "dietitian", "hygienist", "sitter", "counselor", "assistant",
    "clerk", "designer", "editor", "tailor"
}

# Jobs typically associated with men
MALE_TYPICAL = {
    "ceo", "programmer", "engineer", "driver", "lawyer", "physician",
    "carpenter", "plumber", "firefighter", "soldier", "guard",
    "detective", "sheriff", "analyst", "developer", "farmer",
    "electrician", "pilot", "mechanic", "janitor", "manager",
    "supervisor", "technician", "constructor", "laborer", "chief",
    "accountant", "auditor", "architect", "contractor"
}

FEMALE_PRONOUNS = {"she", "her", "hers"}
MALE_PRONOUNS   = {"he",  "him", "his"}


def get_entity_type(entity_str):
    """Classify profession as female-typical, male-typical, or neutral."""
    e = entity_str.lower().replace("the ", "").strip()
    for job in FEMALE_TYPICAL:
        if job in e:
            return "female_typical"
    for job in MALE_TYPICAL:
        if job in e:
            return "male_typical"
    return "neutral"


def extract_pronoun_gender(sentence):
    """Find the pronoun in the sentence and return its gender."""
    words = sentence.lower().split()
    for w in words:
        w = w.strip(".,;:?!\"'")
        if w in FEMALE_PRONOUNS:
            return "female"
        if w in MALE_PRONOUNS:
            return "male"
    return "unknown"


def extract_features(row):
    """
    Convert one sentence into 6 binary features (all 0 or 1).
    """
    entity_type    = get_entity_type(row["entity"])
    gold           = row["gold_gender"]
    pronoun_gender = extract_pronoun_gender(row["sentence"])

    is_stereotypical = int(
        (entity_type == "female_typical" and gold == "female") or
        (entity_type == "male_typical"   and gold == "male")
    )

    return {
        "sentence":                 row["sentence"],
        "entity":                   row["entity"],
        "gold_gender":              gold,
        "bias_label":               int(row["bias_label"]),
        "is_stereotypical":         is_stereotypical,
        "pronoun_is_female":        int(pronoun_gender == "female"),
        "pronoun_is_male":          int(pronoun_gender == "male"),
        "entity_is_female_typical": int(entity_type == "female_typical"),
        "entity_is_male_typical":   int(entity_type == "male_typical"),
        "gold_gender_is_female":    int(gold == "female"),
    }


FEATURE_COLS = [
    "is_stereotypical",
    "pronoun_is_female",
    "pronoun_is_male",
    "entity_is_female_typical",
    "entity_is_male_typical",
    "gold_gender_is_female",
]



# STEP 5: Train classifiers and evaluate

def train_and_evaluate(feature_rows):
    """
    Train Logistic Regression and evaluate on held-out test set.
    Uses an 80/20 train/test split.
    """
    X = np.array([[r[col] for col in FEATURE_COLS] for r in feature_rows])
    y = np.array([r["bias_label"] for r in feature_rows])

    # 80% train, 20% test, stratified so class ratios are balanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)} samples | Test: {len(y_test)} samples")

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_lr = lr.predict(X_test)
    results = {
        "accuracy":  round(accuracy_score(y_test, y_lr) * 100, 1),
        "precision": round(precision_score(y_test, y_lr, average="weighted", zero_division=0) * 100, 1),
        "recall":    round(recall_score(y_test, y_lr, average="weighted", zero_division=0) * 100, 1),
        "f1":        round(f1_score(y_test, y_lr, average="weighted", zero_division=0) * 100, 1),
    }

    # Print detailed report
    print("\n  Logistic Regression — detailed report:")
    print(classification_report(y_test, y_lr,
                                target_names=["Not Biased (0)", "Biased (1)"]))

    # Print feature weights
    print("  Feature weights (Logistic Regression):")
    for name, weight in zip(FEATURE_COLS, lr.coef_[0]):
        bar = "+" * int(abs(weight) * 10) if weight > 0 else "-" * int(abs(weight) * 10)
        print(f"    {name:<32} {weight:+.4f}  {bar}")

    return results, y_test, y_lr, lr, X_train, y_train


# STEP 6: Analysis — bias rate by group

def analyze_bias_rates(feature_rows):
    """
    Calculate what percentage of sentences are biased in each group.
    This answers: which kinds of sentences are most likely to be biased?
    """
    def bias_rate(subset):
        if not subset:
            return 0
        return 100 * sum(r["bias_label"] for r in subset) / len(subset)

    stereo = [r for r in feature_rows if r["is_stereotypical"] == 1]
    anti   = [r for r in feature_rows if r["is_stereotypical"] == 0]
    female = [r for r in feature_rows if r["gold_gender_is_female"] == 1]
    male   = [r for r in feature_rows if r["gold_gender_is_female"] == 0]

    rates = {
        "Stereotypical":      round(bias_rate(stereo), 1),
        "Anti-stereotypical": round(bias_rate(anti), 1),
        "Gold = Female":      round(bias_rate(female), 1),
        "Gold = Male":        round(bias_rate(male), 1),
    }
    return rates



# STEP 7: Save charts

def save_charts(results, bias_rates, feature_rows, lr):
    """Save two charts to the results/ folder."""

    groups = list(bias_rates.keys())
    rates  = list(bias_rates.values())
    colors = ["#4472C4", "#E94560", "#4472C4", "#E94560"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(groups, rates, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Bias Rate (%)")
    ax.set_title("Bias Rate by Sentence Type")
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, label="50% line")
    ax.legend()
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path1 = os.path.join(RESULTS_DIR, "chart1_bias_by_group.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  Saved: {path1}")

    weights = lr.coef_[0]
    colors  = ["#E94560" if w < 0 else "#4472C4" for w in weights]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(FEATURE_COLS, weights, color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Weight")
    ax.set_title("Logistic Regression Feature Weights\n"
                 "(negative = predicts NOT biased, positive = predicts biased)")
    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, "chart2_feature_weights.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved: {path2}")



# STEP 8: Save results to a text file and CSV
def save_results(results, bias_rates, feature_rows):
    # Save results as CSV
    table_path = os.path.join(RESULTS_DIR, "results_table.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Accuracy", "Precision", "Recall", "F1"])
        writer.writeheader()
        writer.writerow({
            "Accuracy":  results["accuracy"],
            "Precision": results["precision"],
            "Recall":    results["recall"],
            "F1":        results["f1"],
        })
    print(f"  Saved: {table_path}")

    # Save summary text
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("CS 341NL Final Project — Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total sentences: {len(feature_rows)}\n")
        biased = sum(r["bias_label"] for r in feature_rows)
        f.write(f"Biased: {biased} ({100*biased/len(feature_rows):.1f}%)\n")
        f.write(f"Not biased: {len(feature_rows)-biased} "
                f"({100*(len(feature_rows)-biased)/len(feature_rows):.1f}%)\n\n")

        f.write("Logistic Regression Results:\n")
        f.write(f"  Accuracy:  {results['accuracy']}%\n")
        f.write(f"  Precision: {results['precision']}%\n")
        f.write(f"  Recall:    {results['recall']}%\n")
        f.write(f"  F1 Score:  {results['f1']}%\n")

        f.write("\nBias Rate by Group:\n")
        for group, rate in bias_rates.items():
            f.write(f"  {group:<25} {rate}%\n")

    print(f"  Saved: {summary_path}")


def main():
    print("\n" + "=" * 60)
    print("CS 341NL Final Project — Starting Pipeline")
    print("=" * 60 + "\n")

    # Check that winomt.tsv exists
    if not os.path.exists(WINOMT_FILE):
        print(f"ERROR: Cannot find {WINOMT_FILE}")
        print("Please make sure you copied winomt.tsv into the data/ folder.")
        print("See README.md for instructions.")
        exit(1)

    # STEP 1
    print("Step 1: Loading WinoMT data...")
    sentences = load_winomt(WINOMT_FILE)
    print(f"  Loaded {len(sentences)} sentences.")

    # STEP 2
    print("\nStep 2: Translating to Spanish...")
    translations = translate_all(sentences, TRANSLATED_FILE)

    # STEP 3
    print("\nStep 3: Labeling sentences as biased or not biased...")
    labeled = label_sentences(sentences, translations)

    # STEP 4
    print("\nStep 4: Extracting features from English sentences...")
    feature_rows = [extract_features(r) for r in labeled]
    stereo_count = sum(r["is_stereotypical"] for r in feature_rows)
    print(f"  Extracted features for {len(feature_rows)} sentences.")
    print(f"  Stereotypical: {stereo_count} | Anti-stereotypical: {len(feature_rows)-stereo_count}")

    # Save labeled + features to CSV
    fieldnames = list(feature_rows[0].keys())
    with open(FEATURES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)
    print(f"  Saved features to {FEATURES_FILE}")

    # STEP 5
    print("\nStep 5: Training and evaluating classifiers...")
    results, y_test, y_lr, lr, X_train, y_train = train_and_evaluate(feature_rows)

    # Print results table
    print("\n  ── Results ──")
    print(f"  Accuracy:  {results['accuracy']}%")
    print(f"  Precision: {results['precision']}%")
    print(f"  Recall:    {results['recall']}%")
    print(f"  F1 Score:  {results['f1']}%")

    # STEP 6
    print("\nStep 6: Analyzing bias rates by group...")
    bias_rates = analyze_bias_rates(feature_rows)
    for group, rate in bias_rates.items():
        print(f"  {group:<25} {rate}%")

    # STEP 7
    print("\nStep 7: Saving charts...")
    save_charts(results, bias_rates, feature_rows, lr)

    # STEP 8
    print("\nStep 8: Saving results...")
    save_results(results, bias_rates, feature_rows)

    print("\n" + "=" * 60)
    print("DONE! All results saved to the results/ folder.")
    print("Files created:")
    print("  results/chart1_bias_by_group.png")
    print("  results/chart2_feature_weights.png")
    print("  results/results_table.csv")
    print("  results/summary.txt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()