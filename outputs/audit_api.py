"""
Comprehensive Audit Script for Cricket Data API Live Coverage (Phase 9.1).
Safely audits raw API endpoints, tests pagination, inspects match formats and status,
and determines exact coverage of the Nigeria vs Sierra Leone T20 fixture.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.live.cricket_api import CricketApiClient, mask_sensitive_url
from src.live.live_features import is_t20_match, normalize_match_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_api")


def sanitize_payload(obj: Any) -> Any:
    """Recursively removes or masks any API key in dictionaries or strings."""
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if "apikey" in k.lower() or "api_key" in k.lower():
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = sanitize_payload(v)
        return sanitized
    elif isinstance(obj, list):
        return [sanitize_payload(item) for item in obj]
    elif isinstance(obj, str):
        return mask_sensitive_url(obj)
    return obj


def is_target_fixture(m: Dict[str, Any]) -> bool:
    """Checks if a match dict is the Nigeria vs Sierra Leone fixture or involves these teams."""
    text_corpus = " ".join([
        str(m.get("name", "")),
        str(m.get("status", "")),
        str(m.get("venue", "")),
        str(m.get("matchType", "")),
        " ".join(str(t) for t in m.get("teams", []))
    ]).lower()
    return "nigeria" in text_corpus or "sierra leone" in text_corpus


def search_for_target(matches: List[Dict[str, Any]], keywords: List[str] = None) -> List[Dict[str, Any]]:
    """Searches a list of match dicts specifically for Nigeria or Sierra Leone."""
    return [m for m in matches if is_target_fixture(m)]


def run_audit():
    logger.info("Initializing CricketApiClient...")
    client = CricketApiClient()

    audit_timestamp = datetime.now(timezone.utc).isoformat()
    keywords = ["nigeria", "sierra leone", "ngr", "srl"]

    audit_report = {
        "audit_timestamp": audit_timestamp,
        "target_fixture": "Nigeria vs Sierra Leone (Quadrangular T20I Series in Nigeria 2026, 27 Sept 2026)",
        "target_keywords": keywords,
        "endpoints_queried": {},
        "discovered_target_matches": [],
        "case_analysis": {},
        "raw_responses_summary": {},
        "series_coverage_audit": {},
        "final_conclusion": None,
    }

    # 1. Query /currentMatches?offset=0
    logger.info("1. Querying /currentMatches (offset=0)...")
    try:
        raw_cur_0 = client._request("currentMatches", params={"offset": 0})
        data_cur_0 = raw_cur_0.get("data", [])
        info_cur_0 = raw_cur_0.get("info", {})
        audit_report["endpoints_queried"]["currentMatches_offset_0"] = {
            "status": raw_cur_0.get("status"),
            "matches_count": len(data_cur_0),
            "info": info_cur_0,
            "match_types_present": list({str(m.get("matchType", "")).lower() for m in data_cur_0}),
            "match_names": [m.get("name") for m in data_cur_0],
        }
        audit_report["raw_responses_summary"]["currentMatches_offset_0"] = sanitize_payload(data_cur_0)

        # Search for target
        found_cur_0 = search_for_target(data_cur_0, keywords)
        for m in found_cur_0:
            audit_report["discovered_target_matches"].append({
                "endpoint": "/currentMatches",
                "offset": 0,
                "match_id": m.get("id"),
                "name": m.get("name"),
                "matchType": m.get("matchType"),
                "status": m.get("status"),
                "matchStarted": m.get("matchStarted"),
                "matchEnded": m.get("matchEnded"),
                "date": m.get("date"),
                "dateTimeGMT": m.get("dateTimeGMT"),
                "teams": m.get("teams"),
                "raw_match": sanitize_payload(m),
            })
    except Exception as e:
        logger.error(f"Error querying /currentMatches offset=0: {e}")
        audit_report["endpoints_queried"]["currentMatches_offset_0"] = {"error": str(e)}

    # 2. Query /currentMatches?offset=25 if hits allow or totalRows > 25
    total_cur = info_cur_0.get("totalRows", 0) if "info_cur_0" in locals() and isinstance(info_cur_0, dict) else 0
    if total_cur > 25:
        logger.info(f"Querying /currentMatches (offset=25) because totalRows={total_cur}...")
        try:
            raw_cur_25 = client._request("currentMatches", params={"offset": 25})
            data_cur_25 = raw_cur_25.get("data", [])
            audit_report["endpoints_queried"]["currentMatches_offset_25"] = {
                "status": raw_cur_25.get("status"),
                "matches_count": len(data_cur_25),
                "info": raw_cur_25.get("info", {}),
                "match_types_present": list({str(m.get("matchType", "")).lower() for m in data_cur_25}),
                "match_names": [m.get("name") for m in data_cur_25],
            }
            audit_report["raw_responses_summary"]["currentMatches_offset_25"] = sanitize_payload(data_cur_25)
            found_cur_25 = search_for_target(data_cur_25, keywords)
            for m in found_cur_25:
                audit_report["discovered_target_matches"].append({
                    "endpoint": "/currentMatches",
                    "offset": 25,
                    "match_id": m.get("id"),
                    "name": m.get("name"),
                    "matchType": m.get("matchType"),
                    "status": m.get("status"),
                    "matchStarted": m.get("matchStarted"),
                    "matchEnded": m.get("matchEnded"),
                    "date": m.get("date"),
                    "dateTimeGMT": m.get("dateTimeGMT"),
                    "teams": m.get("teams"),
                    "raw_match": sanitize_payload(m),
                })
        except Exception as e:
            audit_report["endpoints_queried"]["currentMatches_offset_25"] = {"error": str(e)}

    # 3. Query /matches?offset=0
    logger.info("2. Querying /matches (offset=0)...")
    try:
        raw_m_0 = client._request("matches", params={"offset": 0})
        data_m_0 = raw_m_0.get("data", [])
        info_m_0 = raw_m_0.get("info", {})
        audit_report["endpoints_queried"]["matches_offset_0"] = {
            "status": raw_m_0.get("status"),
            "matches_count": len(data_m_0),
            "info": info_m_0,
            "match_types_present": list({str(m.get("matchType", "")).lower() for m in data_m_0}),
            "match_names": [m.get("name") for m in data_m_0],
        }
        audit_report["raw_responses_summary"]["matches_offset_0"] = sanitize_payload(data_m_0)

        found_m_0 = search_for_target(data_m_0, keywords)
        for m in found_m_0:
            audit_report["discovered_target_matches"].append({
                "endpoint": "/matches",
                "offset": 0,
                "match_id": m.get("id"),
                "name": m.get("name"),
                "matchType": m.get("matchType"),
                "status": m.get("status"),
                "matchStarted": m.get("matchStarted"),
                "matchEnded": m.get("matchEnded"),
                "date": m.get("date"),
                "dateTimeGMT": m.get("dateTimeGMT"),
                "teams": m.get("teams"),
                "raw_match": sanitize_payload(m),
            })
    except Exception as e:
        logger.error(f"Error querying /matches offset=0: {e}")
        audit_report["endpoints_queried"]["matches_offset_0"] = {"error": str(e)}

    # 4. Query /matches?offset=25
    logger.info("3. Querying /matches (offset=25)...")
    try:
        raw_m_25 = client._request("matches", params={"offset": 25})
        data_m_25 = raw_m_25.get("data", [])
        info_m_25 = raw_m_25.get("info", {})
        audit_report["endpoints_queried"]["matches_offset_25"] = {
            "status": raw_m_25.get("status"),
            "matches_count": len(data_m_25),
            "info": info_m_25,
            "match_types_present": list({str(m.get("matchType", "")).lower() for m in data_m_25}),
            "match_names": [m.get("name") for m in data_m_25],
        }
        audit_report["raw_responses_summary"]["matches_offset_25"] = sanitize_payload(data_m_25)

        found_m_25 = search_for_target(data_m_25, keywords)
        for m in found_m_25:
            audit_report["discovered_target_matches"].append({
                "endpoint": "/matches",
                "offset": 25,
                "match_id": m.get("id"),
                "name": m.get("name"),
                "matchType": m.get("matchType"),
                "status": m.get("status"),
                "matchStarted": m.get("matchStarted"),
                "matchEnded": m.get("matchEnded"),
                "date": m.get("date"),
                "dateTimeGMT": m.get("dateTimeGMT"),
                "teams": m.get("teams"),
                "raw_match": sanitize_payload(m),
            })
    except Exception as e:
        logger.error(f"Error querying /matches offset=25: {e}")
        audit_report["endpoints_queried"]["matches_offset_25"] = {"error": str(e)}

    # 5. Query /series?search=Nigeria or series catalogue to check tournament coverage
    logger.info("4. Checking /series endpoint for series coverage...")
    try:
        raw_series = client._request("series", params={"search": "Nigeria", "offset": 0})
        audit_report["endpoints_queried"]["series_search_nigeria"] = {
            "status": raw_series.get("status"),
            "data": sanitize_payload(raw_series.get("data", [])),
            "info": raw_series.get("info", {}),
        }
    except Exception as e:
        audit_report["endpoints_queried"]["series_search_nigeria"] = {"note": str(e)}

    # Also check /series with search "Quadrangular"
    try:
        raw_series_quad = client._request("series", params={"search": "Quadrangular", "offset": 0})
        audit_report["endpoints_queried"]["series_search_quadrangular"] = {
            "status": raw_series_quad.get("status"),
            "data": sanitize_payload(raw_series_quad.get("data", [])),
            "info": raw_series_quad.get("info", {}),
        }
    except Exception as e:
        audit_report["endpoints_queried"]["series_search_quadrangular"] = {"note": str(e)}

    # 5. Query /series_info for the Nigeria Quadrangular tournament
    logger.info("5. Querying /series_info for Nigeria tournament...")
    nigeria_series_id = "c1447761-1361-426a-8735-d7adaa73f407"
    try:
        raw_series_info = client._request("series_info", params={"id": nigeria_series_id})
        series_match_list = raw_series_info.get("data", {}).get("matchList", [])
        audit_report["series_coverage_audit"] = {
            "series_id": nigeria_series_id,
            "status": raw_series_info.get("status"),
            "info": raw_series_info.get("data", {}).get("info", {}),
            "matchList": sanitize_payload(series_match_list),
            "matchList_count": len(series_match_list),
        }
        for m in series_match_list:
            if is_target_fixture(m):
                audit_report["discovered_target_matches"].append({
                    "endpoint": "/series_info",
                    "offset": 0,
                    "match_id": m.get("id"),
                    "name": m.get("name"),
                    "matchType": m.get("matchType"),
                    "status": m.get("status"),
                    "matchStarted": m.get("matchStarted"),
                    "matchEnded": m.get("matchEnded"),
                    "date": m.get("date"),
                    "dateTimeGMT": m.get("dateTimeGMT"),
                    "teams": m.get("teams"),
                    "raw_match": sanitize_payload(m),
                })
    except Exception as e:
        audit_report["series_coverage_audit"] = {"error": str(e)}

    # If any target match was discovered, test /match_info
    target_match_details = []
    for disc in audit_report["discovered_target_matches"]:
        mid = disc["match_id"]
        logger.info(f"Target match discovered! Fetching /match_info for id={mid}...")
        try:
            m_info = client.get_match_info(mid)
            target_match_details.append({
                "match_id": mid,
                "info": sanitize_payload(m_info),
            })
        except Exception as e:
            target_match_details.append({"match_id": mid, "error": str(e)})

    audit_report["target_match_info_details"] = target_match_details

    # Evaluate the 6 Distinction Criteria
    target_found = len(audit_report["discovered_target_matches"]) > 0

    # 1. Match absent from raw API response
    case_1_absent_raw = not target_found
    # 2. Match present but rejected by project filtering
    case_2_filter_rejected = False
    if target_found:
        for disc in audit_report["discovered_target_matches"]:
            raw_m = disc["raw_match"]
            if not is_t20_match(raw_m):
                case_2_filter_rejected = True
    # 3. Match present but status interpreted incorrectly
    case_3_status_misinterpreted = False
    if target_found:
        for disc in audit_report["discovered_target_matches"]:
            norm_m = normalize_match_dict(disc["raw_match"])
            # check if status shows active but normalized flags don't
            if norm_m.status and not norm_m.match_started:
                case_3_status_misinterpreted = True
    # 4. Match present only at another pagination offset
    case_4_another_offset = False
    if target_found:
        offsets = {disc["offset"] for disc in audit_report["discovered_target_matches"]}
        if 0 not in offsets:
            case_4_another_offset = True
    # 5. Match present in /matches but absent from /currentMatches
    case_5_in_matches_only = False
    if target_found:
        endpoints = {disc["endpoint"] for disc in audit_report["discovered_target_matches"]}
        if "/matches" in endpoints and "/currentMatches" not in endpoints:
            case_5_in_matches_only = True
    # 6. Match present in /match_info but not discoverable through listing endpoints
    series_match_list = audit_report.get("series_coverage_audit", {}).get("matchList", [])
    case_6_direct_only = any(is_target_fixture(m) for m in series_match_list)

    audit_report["case_analysis"] = {
        "case_1_absent_from_raw_api": {
            "evaluated": True,
            "result": case_1_absent_raw,
            "description": "Match absent from all queried listing endpoints (/currentMatches, /matches offsets 0, 25)."
        },
        "case_2_rejected_by_filtering": {
            "evaluated": True,
            "result": case_2_filter_rejected,
            "description": "Match present in raw data but rejected by is_t20_match() or format filter."
        },
        "case_3_status_misinterpreted": {
            "evaluated": True,
            "result": case_3_status_misinterpreted,
            "description": "Match present in raw data but live status, innings, or matchStarted flag parsed incorrectly."
        },
        "case_4_pagination_offset": {
            "evaluated": True,
            "result": case_4_another_offset,
            "description": "Match absent from offset 0 but present at higher pagination offset (e.g. offset 25)."
        },
        "case_5_in_matches_not_currentMatches": {
            "evaluated": True,
            "result": case_5_in_matches_only,
            "description": "Match present in general /matches listing but missing from /currentMatches endpoint."
        },
        "case_6_in_match_info_only": {
            "evaluated": True,
            "result": case_6_direct_only,
            "description": "Match present only via direct ID lookup or series indexing (verified via /series_info)."
        }
    }

    if target_found:
        if case_5_in_matches_only:
            audit_report["final_conclusion"] = "LIVE_T20_PRESENT_IN_MATCHES_ENDPOINT"
        elif case_2_filter_rejected:
            audit_report["final_conclusion"] = "LIVE_T20_FORMAT_FILTER_ISSUE"
        elif case_3_status_misinterpreted:
            audit_report["final_conclusion"] = "LIVE_T20_STATUS_PARSING_ISSUE"
        elif case_4_another_offset:
            audit_report["final_conclusion"] = "LIVE_T20_PAGINATION_OFFSET_ISSUE"
        else:
            audit_report["final_conclusion"] = "LIVE_T20_FOUND_AND_AVAILABLE"
    else:
        audit_report["final_conclusion"] = "LIVE_PROVIDER_COVERAGE_LIMITATION"

    # Save to outputs/live_api_coverage_audit.json
    out_path = PROJECT_ROOT / "outputs" / "live_api_coverage_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    logger.info(f"Audit completed. Output written to {out_path}")
    logger.info(f"Final conclusion: {audit_report['final_conclusion']}")
    return audit_report


if __name__ == "__main__":
    run_audit()
