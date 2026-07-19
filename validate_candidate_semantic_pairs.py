from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


CANDIDATE_PATH = Path(
    "outputs/candidate-cleaned-kaggle-2020/"
    "predictions.parquet"
)

PRODUCTION_PATH = Path(
    "outputs/backtest/predictions.parquet"
)

OUTPUT_DIR = Path(
    "outputs/candidate-cleaned-kaggle-2020/"
    "validation/semantic_pairing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def normalize(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_tournament(value: object) -> str:
    text = normalize(value)

    text = re.sub(
        r"\b(?:atp|wta|mens|women|singles)\b",
        " ",
        text,
    )

    text = re.sub(
        r"\b(?:19|20)\d{2}\b",
        " ",
        text,
    )

    return " ".join(text.split())


def normalize_round(value: object) -> str:
    text = normalize(value)

    patterns = [
        (r"\bfinal\b|^f$", "F"),
        (r"semi|^sf$", "SF"),
        (r"quarter|^qf$", "QF"),
        (r"round of 16|last 16|^r16$", "R16"),
        (r"round of 32|^r32$", "R32"),
        (r"round of 64|^r64$", "R64"),
        (r"round of 128|^r128$", "R128"),
        (r"qualifying 1|qualifier 1|^q1$", "Q1"),
        (r"qualifying 2|qualifier 2|^q2$", "Q2"),
        (r"qualifying 3|qualifier 3|^q3$", "Q3"),
        (r"round robin|group", "RR"),
    ]

    for pattern, label in patterns:
        if re.search(pattern, text):
            return label

    return text.upper()


def name_signatures(value: object) -> list[tuple[str, str]]:
    raw = "" if pd.isna(value) else str(value).strip()
    normal = normalize(raw)
    tokens = normal.split()

    if not tokens:
        return []

    signatures: set[tuple[str, str]] = set()

    raw_tokens = raw.split()

    # Handles source format such as:
    # Swiatek I.
    # Haddad Maia B.
    # Wang Xin.
    if (
        len(tokens) >= 2
        and raw_tokens
        and "." in raw_tokens[-1]
    ):
        surname = " ".join(tokens[:-1])
        given_marker = tokens[-1]

        signatures.add(
            (surname, given_marker)
        )

    # Handles normal full-name formats such as:
    # Iga Swiatek
    # Beatriz Haddad Maia
    # Novak Djokovic
    for surname_length in range(
        1,
        min(3, len(tokens) - 1) + 1,
    ):
        surname = " ".join(
            tokens[-surname_length:]
        )

        given_tokens = tokens[:-surname_length]

        if not given_tokens:
            continue

        full_given = "".join(given_tokens)
        initials = "".join(
            token[0]
            for token in given_tokens
            if token
        )

        signatures.add(
            (surname, full_given)
        )
        signatures.add(
            (surname, initials)
        )

    return sorted(signatures)


def markers_compatible(
    first: str,
    second: str,
) -> bool:
    if not first or not second:
        return False

    return (
        first == second
        or first.startswith(second)
        or second.startswith(first)
    )


def name_similarity(
    first: object,
    second: object,
) -> float:
    first_normalized = normalize(first)
    second_normalized = normalize(second)

    if not first_normalized or not second_normalized:
        return 0.0

    if first_normalized == second_normalized:
        return 100.0

    base = max(
        fuzz.WRatio(
            first_normalized,
            second_normalized,
        ),
        fuzz.token_set_ratio(
            first_normalized,
            second_normalized,
        ),
    )

    signature_score = 0.0

    for first_surname, first_marker in name_signatures(
        first
    ):
        for second_surname, second_marker in name_signatures(
            second
        ):
            surname_score = fuzz.ratio(
                first_surname,
                second_surname,
            )

            if (
                first_surname == second_surname
                and markers_compatible(
                    first_marker,
                    second_marker,
                )
            ):
                signature_score = max(
                    signature_score,
                    99.0,
                )

            elif (
                surname_score >= 92
                and markers_compatible(
                    first_marker,
                    second_marker,
                )
            ):
                signature_score = max(
                    signature_score,
                    95.0,
                )

    return float(
        max(base, signature_score)
    )


def select_boosted(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if "model" not in frame.columns:
        return frame.copy()

    model = (
        frame["model"]
        .fillna("")
        .astype(str)
        .str.casefold()
    )

    exact = model.eq("boosted")

    if exact.any():
        return frame.loc[exact].copy()

    contains = model.str.contains(
        "boost",
        regex=False,
        na=False,
    )

    return frame.loc[contains].copy()


def prepare(
    frame: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    frame = select_boosted(frame)

    required = [
        "date",
        "tour",
        "player1",
        "player2",
        "actual",
        "probability",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"{source} is missing columns: {missing}"
        )

    output = frame.copy()

    output["_date"] = pd.to_datetime(
        output["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    output["_tour"] = (
        output["tour"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    output["_actual"] = pd.to_numeric(
        output["actual"],
        errors="coerce",
    )

    output["_probability"] = pd.to_numeric(
        output["probability"],
        errors="coerce",
    )

    output = output[
        output["_date"].notna()
        & output["_actual"].notna()
        & output["_probability"].notna()
    ].copy()

    output["_actual"] = output[
        "_actual"
    ].astype(int)

    output["_probability"] = output[
        "_probability"
    ].clip(
        1e-12,
        1 - 1e-12,
    )

    if "competition_segment" not in output.columns:
        output["competition_segment"] = ""

    if "tournament" not in output.columns:
        output["tournament"] = ""

    if "round" not in output.columns:
        output["round"] = ""

    return output.reset_index(drop=True)


def metrics(
    actual: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    actual = np.asarray(
        actual,
        dtype=int,
    )

    probability = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        1e-12,
        1 - 1e-12,
    )

    predicted = (
        probability >= 0.5
    ).astype(int)

    return {
        "matches": int(len(actual)),
        "accuracy": float(
            np.mean(predicted == actual)
        ),
        "brier": float(
            np.mean(
                (probability - actual) ** 2
            )
        ),
        "log_loss": float(
            -np.mean(
                actual * np.log(probability)
                + (1 - actual)
                * np.log(1 - probability)
            )
        ),
    }


def bootstrap(
    frame: pd.DataFrame,
    iterations: int = 1500,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(20260718)

    actual = frame["actual"].to_numpy(
        dtype=int
    )

    candidate_probability = frame[
        "candidate_probability"
    ].to_numpy(
        dtype=float
    )

    production_probability = frame[
        "production_probability"
    ].to_numpy(
        dtype=float
    )

    count = len(frame)

    differences = {
        "accuracy": [],
        "brier": [],
        "log_loss": [],
    }

    for _ in range(iterations):
        indexes = rng.integers(
            0,
            count,
            size=count,
        )

        candidate_metrics = metrics(
            actual[indexes],
            candidate_probability[indexes],
        )

        production_metrics = metrics(
            actual[indexes],
            production_probability[indexes],
        )

        for metric in differences:
            differences[metric].append(
                candidate_metrics[metric]
                - production_metrics[metric]
            )

    return {
        metric: [
            float(
                np.quantile(values, 0.025)
            ),
            float(
                np.quantile(values, 0.975)
            ),
        ]
        for metric, values in differences.items()
    }


candidate = prepare(
    pd.read_parquet(CANDIDATE_PATH),
    "candidate",
)

production = prepare(
    pd.read_parquet(PRODUCTION_PATH),
    "production",
)

candidate = candidate[
    candidate["competition_segment"].isin(
        [
            "ATP_MAIN",
            "WTA_MAIN",
        ]
    )
].copy()

candidate["_row_id"] = np.arange(
    len(candidate)
)

production["_row_id"] = np.arange(
    len(production)
)

print("=== INPUT INVENTORY ===")
print()
print("Candidate boosted rows:", f"{len(candidate):,}")
print(
    candidate.groupby(
        [
            "_tour",
            "competition_segment",
        ]
    ).size().to_string()
)

print()
print("Production boosted rows:", f"{len(production):,}")
print(
    production["_tour"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print(
    "Candidate date range:",
    candidate["_date"].min(),
    "through",
    candidate["_date"].max(),
)

print(
    "Production date range:",
    production["_date"].min(),
    "through",
    production["_date"].max(),
)

candidate_groups = {
    key: frame
    for key, frame in candidate.groupby(
        ["_date", "_tour"],
        sort=False,
    )
}

production_groups = {
    key: frame
    for key, frame in production.groupby(
        ["_date", "_tour"],
        sort=False,
    )
}

matched_rows: list[dict[str, object]] = []

for group_key in sorted(
    set(candidate_groups)
    & set(production_groups)
):
    candidate_group = candidate_groups[group_key]
    production_group = production_groups[group_key]

    possible_matches = []

    for candidate_index, candidate_row in (
        candidate_group.iterrows()
    ):
        for production_index, production_row in (
            production_group.iterrows()
        ):
            direct_scores = [
                name_similarity(
                    candidate_row["player1"],
                    production_row["player1"],
                ),
                name_similarity(
                    candidate_row["player2"],
                    production_row["player2"],
                ),
            ]

            swapped_scores = [
                name_similarity(
                    candidate_row["player1"],
                    production_row["player2"],
                ),
                name_similarity(
                    candidate_row["player2"],
                    production_row["player1"],
                ),
            ]

            direct_average = float(
                np.mean(direct_scores)
            )

            swapped_average = float(
                np.mean(swapped_scores)
            )

            if swapped_average > direct_average:
                player_scores = swapped_scores
                production_swapped = True
            else:
                player_scores = direct_scores
                production_swapped = False

            player_minimum = float(
                min(player_scores)
            )

            player_average = float(
                np.mean(player_scores)
            )

            tournament_score = float(
                fuzz.token_set_ratio(
                    clean_tournament(
                        candidate_row["tournament"]
                    ),
                    clean_tournament(
                        production_row["tournament"]
                    ),
                )
            )

            candidate_round = normalize_round(
                candidate_row["round"]
            )

            production_round = normalize_round(
                production_row["round"]
            )

            if (
                candidate_round
                and production_round
                and candidate_round
                == production_round
            ):
                round_score = 100.0
            elif (
                not candidate_round
                or not production_round
            ):
                round_score = 70.0
            else:
                round_score = 0.0

            combined_score = (
                0.80 * player_average
                + 0.15 * tournament_score
                + 0.05 * round_score
            )

            high_confidence = (
                player_minimum >= 88
                and player_average >= 92
                and combined_score >= 85
                and (
                    tournament_score >= 50
                    or player_minimum >= 98
                )
            )

            if high_confidence:
                possible_matches.append(
                    {
                        "candidate_index":
                            candidate_index,
                        "production_index":
                            production_index,
                        "production_swapped":
                            production_swapped,
                        "player_minimum":
                            player_minimum,
                        "player_average":
                            player_average,
                        "tournament_score":
                            tournament_score,
                        "round_score":
                            round_score,
                        "combined_score":
                            combined_score,
                    }
                )

    possible_matches.sort(
        key=lambda row: (
            row["combined_score"],
            row["player_minimum"],
            row["tournament_score"],
        ),
        reverse=True,
    )

    used_candidate = set()
    used_production = set()

    for match in possible_matches:
        candidate_index = match[
            "candidate_index"
        ]
        production_index = match[
            "production_index"
        ]

        if (
            candidate_index in used_candidate
            or production_index in used_production
        ):
            continue

        candidate_row = candidate.loc[
            candidate_index
        ]

        production_row = production.loc[
            production_index
        ]

        production_swapped = bool(
            match["production_swapped"]
        )

        production_probability = float(
            production_row["_probability"]
        )

        production_actual = int(
            production_row["_actual"]
        )

        if production_swapped:
            production_probability = (
                1 - production_probability
            )
            production_actual = (
                1 - production_actual
            )

        candidate_actual = int(
            candidate_row["_actual"]
        )

        outcome_agrees = (
            candidate_actual
            == production_actual
        )

        matched_rows.append(
            {
                "date": candidate_row["_date"],
                "tour": candidate_row["_tour"],
                "segment":
                    candidate_row[
                        "competition_segment"
                    ],
                "candidate_player1":
                    candidate_row["player1"],
                "candidate_player2":
                    candidate_row["player2"],
                "production_player1":
                    production_row["player1"],
                "production_player2":
                    production_row["player2"],
                "candidate_tournament":
                    candidate_row["tournament"],
                "production_tournament":
                    production_row["tournament"],
                "candidate_round":
                    candidate_row["round"],
                "production_round":
                    production_row["round"],
                "production_swapped":
                    production_swapped,
                "player_minimum":
                    match["player_minimum"],
                "player_average":
                    match["player_average"],
                "tournament_score":
                    match["tournament_score"],
                "round_score":
                    match["round_score"],
                "combined_score":
                    match["combined_score"],
                "outcome_agrees":
                    outcome_agrees,
                "actual":
                    candidate_actual,
                "candidate_probability":
                    float(
                        candidate_row[
                            "_probability"
                        ]
                    ),
                "production_probability":
                    production_probability,
            }
        )

        used_candidate.add(candidate_index)
        used_production.add(production_index)


paired = pd.DataFrame(matched_rows)

if paired.empty:
    raise SystemExit(
        "No high-confidence semantic pairs were found."
    )

disagreements = paired[
    ~paired["outcome_agrees"]
].copy()

paired = paired[
    paired["outcome_agrees"]
].copy()

paired = paired.reset_index(drop=True)

paired_path = (
    OUTPUT_DIR
    / "high_confidence_pairs.parquet"
)

paired.to_parquet(
    paired_path,
    index=False,
)

paired.to_csv(
    OUTPUT_DIR
    / "high_confidence_pairs.csv",
    index=False,
)

disagreements.to_csv(
    OUTPUT_DIR
    / "outcome_disagreements.csv",
    index=False,
)

print()
print("=== PAIR QUALITY ===")
print("Accepted pairs:", f"{len(paired):,}")
print(
    "Outcome disagreements excluded:",
    f"{len(disagreements):,}",
)

print()
print("Accepted pairs by tour:")
print(
    paired["tour"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print("Accepted pairs by segment:")
print(
    paired["segment"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print("Name confidence:")
print(
    paired[
        [
            "player_minimum",
            "player_average",
            "combined_score",
        ]
    ]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .to_string()
)

reports = []

for label, frame in [
    ("ALL", paired),
    *[
        (tour, group)
        for tour, group in paired.groupby(
            "tour"
        )
    ],
]:
    candidate_metrics = metrics(
        frame["actual"].to_numpy(),
        frame[
            "candidate_probability"
        ].to_numpy(),
    )

    production_metrics = metrics(
        frame["actual"].to_numpy(),
        frame[
            "production_probability"
        ].to_numpy(),
    )

    intervals = bootstrap(frame)

    reports.append(
        {
            "group": label,
            "matches": len(frame),
            "candidate_accuracy":
                candidate_metrics["accuracy"],
            "production_accuracy":
                production_metrics["accuracy"],
            "accuracy_difference":
                candidate_metrics["accuracy"]
                - production_metrics["accuracy"],
            "accuracy_ci_low":
                intervals["accuracy"][0],
            "accuracy_ci_high":
                intervals["accuracy"][1],
            "candidate_brier":
                candidate_metrics["brier"],
            "production_brier":
                production_metrics["brier"],
            "brier_difference":
                candidate_metrics["brier"]
                - production_metrics["brier"],
            "brier_ci_low":
                intervals["brier"][0],
            "brier_ci_high":
                intervals["brier"][1],
            "candidate_log_loss":
                candidate_metrics["log_loss"],
            "production_log_loss":
                production_metrics["log_loss"],
            "log_loss_difference":
                candidate_metrics["log_loss"]
                - production_metrics["log_loss"],
            "log_loss_ci_low":
                intervals["log_loss"][0],
            "log_loss_ci_high":
                intervals["log_loss"][1],
        }
    )

report = pd.DataFrame(reports)

report_path = (
    OUTPUT_DIR
    / "semantic_comparison.csv"
)

report.to_csv(
    report_path,
    index=False,
)

print()
print("=== HIGH-CONFIDENCE SAME-MATCH RESULTS ===")
print(
    report.to_string(
        index=False,
        formatters={
            "candidate_accuracy":
                "{:.2%}".format,
            "production_accuracy":
                "{:.2%}".format,
            "accuracy_difference":
                "{:+.2%}".format,
            "accuracy_ci_low":
                "{:+.2%}".format,
            "accuracy_ci_high":
                "{:+.2%}".format,
            "candidate_brier":
                "{:.6f}".format,
            "production_brier":
                "{:.6f}".format,
            "brier_difference":
                "{:+.6f}".format,
            "brier_ci_low":
                "{:+.6f}".format,
            "brier_ci_high":
                "{:+.6f}".format,
            "candidate_log_loss":
                "{:.6f}".format,
            "production_log_loss":
                "{:.6f}".format,
            "log_loss_difference":
                "{:+.6f}".format,
            "log_loss_ci_low":
                "{:+.6f}".format,
            "log_loss_ci_high":
                "{:+.6f}".format,
        },
    )
)

summary = {
    "accepted_pairs": int(len(paired)),
    "outcome_disagreements": int(
        len(disagreements)
    ),
    "pairs_by_tour": {
        str(key): int(value)
        for key, value in (
            paired["tour"]
            .value_counts()
            .to_dict()
            .items()
        )
    },
    "results": report.to_dict(
        orient="records"
    ),
    "decision": (
        "NO PROMOTION. Review semantic pairing results "
        "and then begin shadow paper-trading only when "
        "ATP and WTA both have adequate validated overlap."
    ),
}

summary_path = (
    OUTPUT_DIR
    / "semantic_comparison.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print("Saved pairs:", paired_path)
print("Saved results:", report_path)
print("Saved summary:", summary_path)

print()
print("=== CURRENT DECISION ===")
print(summary["decision"])
