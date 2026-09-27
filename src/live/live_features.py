"""
Live Match State and Feature Engineering Module.

Normalizes raw Cricket Data API match responses into structured representations
and transforms live match states into the exact 8 features required by the
existing win-probability model.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "target_score",
    "current_score",
    "wickets_lost",
    "runs_remaining",
    "balls_remaining",
    "overs_completed",
    "current_run_rate",
    "required_run_rate",
]


@dataclass
class NormalizedInnings:
    team: str
    innings_number: int
    runs: int
    wickets: int
    overs_str: str
    total_legal_balls: int
    overs_completed: float


@dataclass
class NormalizedMatch:
    match_id: str
    name: str
    match_type: str
    status: str
    venue: str
    date: str
    teams: List[str]
    match_started: bool
    match_ended: bool
    innings: List[NormalizedInnings] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveMatchFeatures:
    target_score: float
    current_score: float
    wickets_lost: int
    runs_remaining: float
    balls_remaining: int
    overs_completed: float
    current_run_rate: float
    required_run_rate: float
    chasing_team: str
    defending_team: str
    overs_bowled_str: str = ""
    is_terminal: bool = False
    terminal_state: Optional[str] = None
    terminal_probability: Optional[float] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a 1-row DataFrame containing the 8 model features."""
        return pd.DataFrame(
            [
                {
                    "target_score": self.target_score,
                    "current_score": self.current_score,
                    "wickets_lost": self.wickets_lost,
                    "runs_remaining": self.runs_remaining,
                    "balls_remaining": self.balls_remaining,
                    "overs_completed": self.overs_completed,
                    "current_run_rate": self.current_run_rate,
                    "required_run_rate": self.required_run_rate,
                }
            ]
        )[FEATURE_COLS]


def parse_cricket_overs(overs_val: Union[float, int, str]) -> Tuple[int, int, int, float]:
    """
    Parses cricket overs notation into completed overs, balls in over, total legal balls,
    and true fractional overs.

    In cricket notation:
    18.4 means 18 completed overs and 4 legal deliveries:
    - completed_overs = 18
    - balls_in_over = 4
    - total_legal_balls = 18 * 6 + 4 = 112
    - true fractional overs = 112 / 6.0 = 18.6667

    Returns:
        (completed_overs, balls_in_over, total_legal_balls, overs_completed_fraction)
    """
    if overs_val is None or (isinstance(overs_val, float) and math.isnan(overs_val)):
        return 0, 0, 0, 0.0

    s = str(overs_val).strip()
    if not s:
        return 0, 0, 0, 0.0

    if "." in s:
        parts = s.split(".")
        completed_overs = int(parts[0]) if parts[0].isdigit() else 0
        ball_str = parts[1]
        balls_in_over = int(ball_str[0]) if ball_str and ball_str[0].isdigit() else 0
    else:
        completed_overs = int(s) if s.isdigit() else 0
        balls_in_over = 0

    # Safety clamp: legal deliveries in an over cannot exceed 6
    if balls_in_over >= 6:
        completed_overs += balls_in_over // 6
        balls_in_over = balls_in_over % 6

    total_legal_balls = completed_overs * 6 + balls_in_over
    overs_completed_fraction = round(total_legal_balls / 6.0, 4)

    return completed_overs, balls_in_over, total_legal_balls, overs_completed_fraction


def normalize_match_dict(raw: Dict[str, Any]) -> NormalizedMatch:
    """
    Converts raw Cricket Data API dictionary into a NormalizedMatch object.
    Safely handles missing keys, inconsistent formats, and null values.
    """
    match_id = str(raw.get("id") or "")
    name = str(raw.get("name") or "Unknown Match")
    match_type = str(raw.get("matchType") or "").lower().strip()
    status = str(raw.get("status") or "").strip()
    venue = str(raw.get("venue") or "Unknown Venue").strip()
    date = str(raw.get("date") or "").strip()
    teams = [str(t).strip() for t in raw.get("teams", []) if t]
    match_started = bool(raw.get("matchStarted", False))
    match_ended = bool(raw.get("matchEnded", False))

    innings_list: List[NormalizedInnings] = []
    raw_scores = raw.get("score", [])
    if isinstance(raw_scores, list):
        for idx, sc in enumerate(raw_scores):
            if not isinstance(sc, dict):
                continue
            r = int(sc.get("r", 0) or 0)
            w = int(sc.get("w", 0) or 0)
            o_raw = sc.get("o", 0)
            inning_label = str(sc.get("inning", f"Inning {idx+1}")).strip()

            comp_o, b_in_o, total_b, o_frac = parse_cricket_overs(o_raw)

            # Match team name from label
            team_name = ""
            for t in teams:
                if t.lower() in inning_label.lower():
                    team_name = t
                    break
            if not team_name and teams:
                team_name = teams[idx % len(teams)]

            innings_list.append(
                NormalizedInnings(
                    team=team_name,
                    innings_number=idx + 1,
                    runs=r,
                    wickets=w,
                    overs_str=str(o_raw),
                    total_legal_balls=total_b,
                    overs_completed=o_frac,
                )
            )

    return NormalizedMatch(
        match_id=match_id,
        name=name,
        match_type=match_type,
        status=status,
        venue=venue,
        date=date,
        teams=teams,
        match_started=match_started,
        match_ended=match_ended,
        innings=innings_list,
        raw_data=raw,
    )


def is_t20_format(format_str: Optional[str]) -> bool:
    """
    Checks whether a format string represents a standard T20 or T20I match.
    Case-insensitive. Strictly rejects Test, ODI, First-class, List A, etc.
    """
    if not format_str or not isinstance(format_str, str):
        return False
    fmt = format_str.strip().lower()
    return fmt in ["t20", "t20i"]


def is_t20_match(match: Union[NormalizedMatch, Dict[str, Any], str, None]) -> bool:
    """
    Checks whether a match (NormalizedMatch, raw dict, or format string) is a T20/T20I match.
    Strictly filters out non-T20 formats (Test, ODI, First-Class, List A, etc.).
    """
    if match is None:
        return False
    if isinstance(match, str):
        return is_t20_format(match)
    if isinstance(match, NormalizedMatch):
        return is_t20_format(match.match_type)
    if isinstance(match, dict):
        fmt = match.get("matchType") or match.get("match_type")
        return is_t20_format(fmt)
    return False


def filter_t20_matches(
    matches: List[Union[NormalizedMatch, Dict[str, Any]]],
) -> List[Union[NormalizedMatch, Dict[str, Any]]]:
    """Filters a list of matches to retain ONLY T20/T20I fixtures."""
    return [m for m in matches if is_t20_match(m)]


def extract_match_state_features(
    norm_match: NormalizedMatch,
) -> Tuple[Optional[LiveMatchFeatures], Optional[str]]:
    """
    Validates the normalized match state and converts it into LiveMatchFeatures.

    Returns:
        Tuple of (LiveMatchFeatures or None, status_message or None).
        If features cannot be extracted, features is None and status_message explains why.
    """
    status_lower = norm_match.status.lower()

    # 1. Abandoned / No Result Check
    if any(kw in status_lower for kw in ["abandoned", "no result", "cancelled", "match drawn", "match tied without super over"]):
        if not norm_match.innings or len(norm_match.innings) < 2:
            return None, "Prediction unavailable: match has no valid result."

    # 2. Match Format Check (Only standard T20 / T20I matches supported)
    if not is_t20_match(norm_match):
        fmt = norm_match.match_type or "unknown"
        return (
            None,
            f"Live prediction unavailable: match format is '{fmt}', only T20 matches are supported.",
        )

    # 3. Super Over Check
    if "super over" in status_lower or len(norm_match.innings) > 2:
        return (
            None,
            "Live prediction unavailable: Super Over match states are not supported by the standard 120-ball model.",
        )

    # 4. Teams Validation
    if len(norm_match.teams) < 2:
        return None, "Live prediction unavailable: required match-state data is missing (need two teams)."

    # 5. Innings Check
    if not norm_match.innings or len(norm_match.innings) == 0:
        return None, "Live prediction unavailable: required match-state data is missing (no innings data)."

    # 6. First Innings Check
    if len(norm_match.innings) == 1:
        return None, "Prediction unavailable: second innings has not started."

    # Exactly 2 innings available (Standard T20 chase)
    inn1 = norm_match.innings[0]
    inn2 = norm_match.innings[1]

    # Target calculation
    # Check if target is explicitly mentioned in status (e.g. DLS target)
    target_score = float(inn1.runs + 1)
    dls_match = re.search(r"target\s*(\d+)", status_lower)
    if dls_match:
        target_score = float(dls_match.group(1))

    # Identify chasing team and defending team
    chasing_team = inn2.team
    defending_team = inn1.team
    if not chasing_team or chasing_team == defending_team:
        # Resolve from teams list
        if inn1.team == norm_match.teams[0]:
            chasing_team = norm_match.teams[1]
            defending_team = norm_match.teams[0]
        else:
            chasing_team = norm_match.teams[0]
            defending_team = norm_match.teams[1]

    current_score = float(inn2.runs)
    wickets_lost = min(10, max(0, int(inn2.wickets)))
    legal_balls_completed = min(120, max(0, int(inn2.total_legal_balls)))
    balls_remaining = max(0, 120 - legal_balls_completed)
    overs_completed = round(legal_balls_completed / 6.0, 4)
    runs_remaining = max(0.0, target_score - current_score)

    # Rates
    current_run_rate = (
        round(current_score / overs_completed, 4) if overs_completed > 0 else 0.0
    )
    required_run_rate = (
        round(runs_remaining / (balls_remaining / 6.0), 4)
        if balls_remaining > 0 and runs_remaining > 0
        else 0.0
    )

    # 7. Finite numbers and sanity bounds validation
    values_to_check = [
        target_score,
        current_score,
        float(wickets_lost),
        runs_remaining,
        float(balls_remaining),
        overs_completed,
        current_run_rate,
        required_run_rate,
    ]
    for v in values_to_check:
        if v is None or math.isnan(v) or math.isinf(v):
            return None, "Live prediction unavailable: required match-state data is missing (invalid feature value)."

    # 8. Terminal States Handling
    is_terminal = False
    terminal_state = None
    terminal_prob = None

    if current_score >= target_score:
        is_terminal = True
        terminal_state = "Target Reached (1.0)"
        terminal_prob = 1.0
    elif (wickets_lost >= 10 or balls_remaining <= 0 or norm_match.match_ended) and current_score < target_score:
        # If match ended or balls/wickets exhausted before reaching target
        is_terminal = True
        if wickets_lost >= 10:
            terminal_state = "All Out (0.0)"
        elif balls_remaining <= 0:
            terminal_state = "Balls Exhausted (0.0)"
        else:
            terminal_state = "Match Ended (0.0)"
        terminal_prob = 0.0

    overs_bowled_str = f"{legal_balls_completed // 6}.{legal_balls_completed % 6}"

    features = LiveMatchFeatures(
        target_score=target_score,
        current_score=current_score,
        wickets_lost=wickets_lost,
        runs_remaining=runs_remaining,
        balls_remaining=balls_remaining,
        overs_completed=overs_completed,
        current_run_rate=current_run_rate,
        required_run_rate=required_run_rate,
        chasing_team=chasing_team,
        defending_team=defending_team,
        overs_bowled_str=overs_bowled_str,
        is_terminal=is_terminal,
        terminal_state=terminal_state,
        terminal_probability=terminal_prob,
    )

    return features, None
