"""
scenario_generator.py
Generates (persona, preference_vector, city_set) tuples used as input
to the synthetic data generation pipeline.

Each scenario becomes one training conversation.  City sampling is done
so that each scenario includes a mix of good, medium and poor matches,
teaching the model to discriminate rather than just recite facts.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from fine_tuning.configs.generation_config import PERSONAS, PREFERENCE_DIMENSIONS

# How many cities to inject per scenario.
# Kept deliberately small: 6 cities × multiple 400-char section previews
# pushed the generated JSON past Gemini Flash's 8192-token output limit.
CITIES_PER_SCENARIO = 3
GOOD_MATCH_COUNT = 2   # cities that fit well
MEDIUM_MATCH_COUNT = 1 # partial fit
POOR_MATCH_COUNT = 0   # model still learns discrimination from good vs medium


# ---------------------------------------------------------------------------
# Scenario ID
# ---------------------------------------------------------------------------

def make_scenario_id(persona_type: str, preference_vector: dict, run_id: str) -> str:
    pref_hash = hashlib.md5(
        json.dumps(preference_vector, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return f"{persona_type}::{pref_hash}::{run_id}"


# ---------------------------------------------------------------------------
# City sampling
# ---------------------------------------------------------------------------

def _population(city: dict) -> int:
    return city.get("population_2021") or city.get("population_2011") or 0


def _score_city_for_persona(city: dict, pref_vector: dict) -> int:
    """
    Rough match score between a city and a preference vector.
    Higher = better fit.  Used to bucket cities into good/medium/poor.
    """
    score = 0
    pop = _population(city)

    # Demografie preference
    dem = pref_vector.get("demografie", "")
    if "mare" in dem and pop > 100_000:
        score += 2
    if "mic" in dem and pop < 50_000:
        score += 2
    if "mediu" in dem and 50_000 <= pop <= 200_000:
        score += 2

    # Transport preference
    transp = pref_vector.get("transport", "")
    if "feroviară" in transp or "tren" in transp:
        if city.get("has_railway"):
            score += 2
    if "autostradă" in transp:
        if city.get("motorways"):
            score += 1

    # Sanatate
    san = pref_vector.get("sanatate", "")
    if ("spital" in san or "sănătate" in san) and city.get("has_hospital"):
        score += 1

    # Educatie
    edu = pref_vector.get("educatie", "")
    if ("universitat" in edu or "facultate" in edu) and city.get("education_institutions"):
        score += 1

    return score


def sample_cities_for_scenario(
    city_index: dict[str, dict],
    pref_vector: dict,
    rng: random.Random,
) -> list[dict]:
    """
    Return a list of CITIES_PER_SCENARIO city dicts: a mix of good, medium,
    poor matches relative to pref_vector.
    """
    scored = sorted(
        city_index.values(),
        key=lambda c: _score_city_for_persona(c, pref_vector),
        reverse=True,
    )

    # Buckets (avoid overlap)
    total = len(scored)
    top_n = max(GOOD_MATCH_COUNT * 3, 10)
    mid_n = max(MEDIUM_MATCH_COUNT * 3, 8)

    good_pool = scored[:top_n]
    mid_pool = scored[top_n: top_n + mid_n]
    poor_pool = scored[top_n + mid_n:]

    def safe_sample(pool, k):
        return rng.sample(pool, min(k, len(pool))) if pool else []

    cities = (
        safe_sample(good_pool, GOOD_MATCH_COUNT)
        + safe_sample(mid_pool, MEDIUM_MATCH_COUNT)
        + safe_sample(poor_pool, POOR_MATCH_COUNT)
    )

    # If we couldn't fill all slots, pad from the full index
    while len(cities) < CITIES_PER_SCENARIO:
        extra = rng.choice(list(city_index.values()))
        if extra not in cities:
            cities.append(extra)

    rng.shuffle(cities)
    return cities[:CITIES_PER_SCENARIO]


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def generate_scenarios(
    city_index: dict[str, dict],
    run_id: str,
    variants_per_persona: int = 6,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Return a flat list of scenario dicts, one per (persona, variant) combination.

    Each scenario dict contains:
        scenario_id, persona_type, persona_description, preference_vector,
        cities (list of city dicts), run_id
    """
    rng = random.Random(seed)
    scenarios: list[dict[str, Any]] = []

    for persona_type, persona_data in PERSONAS.items():
        base_pref = persona_data["preference_vector"]

        for variant_idx in range(variants_per_persona):
            # Add small variation to phrasing seed so each variant sounds different
            pref_variant = dict(base_pref)
            pref_variant["_variant"] = variant_idx  # included in hash → unique ID

            scenario_id = make_scenario_id(persona_type, pref_variant, run_id)
            cities = sample_cities_for_scenario(city_index, base_pref, rng)

            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "persona_type": persona_type,
                    "persona_description": persona_data["description"],
                    "preference_vector": pref_variant,
                    "cities": cities,
                    "run_id": run_id,
                    "variant_idx": variant_idx,
                }
            )

    rng.shuffle(scenarios)
    return scenarios
