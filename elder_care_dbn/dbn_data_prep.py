"""Reshape the raw visit log into (t, t+1) transition pairs matching the 2TBN's
observed variables: diagnosis, living, treatment. `sickness` and `anchor` are
hidden and are not produced here.
"""

import numpy as np
import pandas as pd

_HOME_CARE_VARIANTS = {"Unit.HOME_CARE_NEW_AIDS", "Unit.HOME_CARE_OLD_AIDS"}


def _clean_living(value: str) -> str:
    """Normalize a raw "Unit.X" living-location string to the model's
    living0/livingt categories, collapsing HOME_CARE_NEW_AIDS/OLD_AIDS into a
    single HOME_CARE.

    Args:
        value: raw "Place of origin"/"Further care" string from the visit log.
    """
    if value in _HOME_CARE_VARIANTS:
        value = "Unit.HOME_CARE"
    return value.removeprefix("Unit.")


def _clean_diagnosis(value: str) -> str:
    """Strip the "Sickness." prefix from a raw diagnosis string.

    Args:
        value: raw "Diagnosis" string from the visit log.
    """
    return value.removeprefix("Sickness.")


def _clean_treatment(value: str) -> str:
    """Strip the "Unit." prefix from a raw "Place of care" string.

    Args:
        value: raw "Place of care" string from the visit log.
    """
    return value.removeprefix("Unit.")


def _explode_visit_spans(visits: pd.DataFrame) -> pd.DataFrame:
    """One row per (Elder ID, tick) for every tick in [Start Time, End Time) of a
    visit: the ticks during which the elder is actively at `treatment`, still
    living at `living` (the transition to `Further care` lands on End Time,
    handled separately in `_endpoint_events`). Zero-length visits (Start == End)
    contribute no rows here; they're fully handled by the endpoint event.
    """
    span_len = (visits["End Time"] - visits["Start Time"]).clip(lower=0)
    repeated = visits.loc[visits.index.repeat(span_len)].copy()
    offset = repeated.groupby(level=0).cumcount()
    tick = repeated["Start Time"].to_numpy() + offset.to_numpy()
    return pd.DataFrame(
        {
            "Elder ID": repeated["Elder ID"].to_numpy(),
            "tick": tick,
            "diagnosis": repeated["Diagnosis"].map(_clean_diagnosis).to_numpy(),
            "living": repeated["Place of living"].map(_clean_living).to_numpy(),
            "treatment": repeated["Place of care"].map(_clean_treatment).to_numpy(),
        }
    )


def _endpoint_events(visits: pd.DataFrame) -> pd.DataFrame:
    """One row per visit at tick == End Time: living transitions to `Further care`
    ("Unit.ED" is a logging glitch -> treated as "no change"). Zero-length visits
    (Start == End) also carry their own treatment here, since `_explode_visit_spans`
    skips them.
    """
    origin = visits["Place of living"].map(_clean_living)
    further_raw = visits["Further care"]
    further = np.where((further_raw == "Unit.ED") | (further_raw == "Unit.HOSPITAL"), origin, further_raw.map(_clean_living))

    zero_length = visits["End Time"] == visits["Start Time"]
    treatment = np.where(zero_length, visits["Place of care"].map(_clean_treatment), None)

    return pd.DataFrame(
        {
            "Elder ID": visits["Elder ID"].to_numpy(),
            "tick": visits["End Time"].to_numpy(),
            "diagnosis": visits["Diagnosis"].map(_clean_diagnosis).to_numpy(),
            "living": further,
            "treatment": treatment,
        }
    )


def _reconstruct_elder_timelines(visits: pd.DataFrame) -> pd.DataFrame:
    """Combine `_explode_visit_spans` and `_endpoint_events` into one per-tick
    timeline per elder, covering every tick from their first event to their
    last: gaps between visits are forward-filled (diagnosis/living carry over,
    treatment defaults to "NONE"), and the timeline is truncated right after
    living first reaches "DEAD" (elders' visits stop there in the raw data,
    so there's nothing real to fill in beyond that point).

    Args:
        visits: raw visit log, as read from the CSV (one row per visit).
    """
    events = pd.concat([_endpoint_events(visits), _explode_visit_spans(visits)], ignore_index=True)
    # last (span) wins on a shared tick, e.g. back-to-back visits or a zero-length visit
    events = events.drop_duplicates(subset=["Elder ID", "tick"], keep="last")
    events = events.sort_values(["Elder ID", "tick"])

    timelines = []
    for elder_id, group in events.groupby("Elder ID", sort=False):
        group = group.set_index("tick").sort_index()
        full_range = np.arange(group.index.min(), group.index.max() + 1)
        g = group.reindex(full_range)
        g["diagnosis"] = g["diagnosis"].ffill()
        g["living"] = g["living"].ffill()
        g["treatment"] = g["treatment"].fillna("NONE")

        dead_ticks = g.index[g["living"] == "DEAD"]
        if len(dead_ticks) > 0:
            g = g.loc[: dead_ticks[0]]

        g["Elder ID"] = elder_id
        timelines.append(g)

    return pd.concat(timelines).rename_axis("tick").reset_index()


def build_transition_table(csv_path: str) -> pd.DataFrame:
    """Return one row per (Elder ID, t, t+1) with columns diagnosis0/diagnosist,
    living0/livingt, treatment0/treatmentt -- ready for gum.BNLearner.learnParameters
    against the twodbn structure (sickness/anchor are hidden and excluded)."""
    visits = pd.read_csv(csv_path, index_col=0)
    timeline = _reconstruct_elder_timelines(visits).sort_values(["Elder ID", "tick"])
    nxt = timeline.groupby("Elder ID")[["diagnosis", "living", "treatment"]].shift(-1)

    pairs = pd.DataFrame(
        {
            "Elder ID": timeline["Elder ID"],
            "diagnosis0": timeline["diagnosis"],
            "diagnosist": nxt["diagnosis"],
            "living0": timeline["living"],
            "livingt": nxt["living"],
            "treatment0": timeline["treatment"],
            "treatmentt": nxt["treatment"],
        }
    )
    return pairs.dropna(subset=["diagnosist"]).reset_index(drop=True)


if __name__ == "__main__":
    table = build_transition_table("data/visits_hc_450_ed_50_hosp_25.csv")
    print(table.shape)
    print(table.head(20))
    table.to_csv("data/dbn_transitions.csv", index=False)
