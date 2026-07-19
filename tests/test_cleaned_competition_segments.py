import pandas as pd

from tennis_predictor.data.match_consolidation import (
    derive_competition_segment,
)


def test_preserves_explicit_cleaned_segments():
    frame = pd.DataFrame(
        {
            "tour": [
                "ATP",
                "ATP",
                "ATP",
                "WTA",
                "ATP",
            ],
            "tier": [
                "ATP 250",
                "ITF M15",
                "Challenger",
                "WTA Tour",
                "Davis Cup",
            ],
            "competition_segment": [
                "ATP_MAIN",
                "ITF_MEN",
                "ATP_CHALLENGER",
                "WTA_MAIN",
                "TEAM_EVENT",
            ],
        }
    )

    assert derive_competition_segment(
        frame
    ).tolist() == [
        "ATP_MAIN",
        "ITF_MEN",
        "ATP_CHALLENGER",
        "WTA_MAIN",
        "TEAM_EVENT",
    ]


def test_classifies_itf_tier_variants_without_explicit_segment():
    frame = pd.DataFrame(
        {
            "tour": [
                "ATP",
                "ATP",
                "WTA",
                "WTA",
            ],
            "tier": [
                "ITF M15",
                "ITF M25",
                "ITF W25",
                "W60",
            ],
        }
    )

    assert derive_competition_segment(
        frame
    ).tolist() == [
        "ITF_MEN",
        "ITF_MEN",
        "ITF_WOMEN",
        "ITF_WOMEN",
    ]
