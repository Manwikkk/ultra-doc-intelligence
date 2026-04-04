"""
extractor.py — Structured shipment data extraction from document text.

Hybrid approach:
  - Ultraship TMS-aware regex for all fields
  - LLM fallback for anything regex misses
  - Pure regex runs first and wins for deterministic fields

Handles the exact Ultraship TMS PDF layout:
  - "Reference ID LD53657"  → shipment_id
  - First name after "Pickup\n" → shipper
  - First name after "Drop\n"  → consignee
  - First token on carrier row  → carrier_name
  - "Shipping Date 02-08-2026 Shipping Time 09:00 - 17:00" → pickup
  - "Delivery Date 02-08-2026 Delivery Time 09:00 - 17:00" → delivery
  - "Flatbed:$ 1000.00 USD" → rate + equipment
  - "$1000.00" / "Agreed Amount" → rate
  - "56000.00 lbs" → weight
  - "FTL" / "LTL" → mode
  - "USD" / "CAD" → currency
"""
from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Any, Optional

from pipeline.vector_store import load_index
from pipeline.llm_router import route_llm_raw
from models import ShipmentData, LLMResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Ultraship TMS-specific extractors (deterministic, layout-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ultraship(text: str) -> dict[str, Any]:
    """
    Layout-aware extraction for Ultraship TMS Rate Confirmation documents.
    Handles the exact multi-line PDF text structure produced by pypdf.
    """
    results: dict[str, Any] = {}

    # ── Reference / Shipment ID ─────────────────────────────────────────────
    # "Reference ID LD53657"
    m = re.search(r"Reference\s+ID\s+([A-Z0-9\-]+)", text, re.IGNORECASE)
    if m:
        results["shipment_id"] = m.group(1).strip()

    # Also try BOL, PRO, Load patterns as fallback
    if not results.get("shipment_id"):
        m = re.search(r"\b(LD\d{4,12}|BOL\d{4,12})\b", text)
        if m:
            results["shipment_id"] = m.group(1)

    # ── Shipper — first company/name line after "Pickup\n" ──────────────────
    # Pattern: "Pickup\nAAA\nLos Angeles..."  OR  "Pickup\nCompany Name\n..."
    m = re.search(
        r"Pickup\s*\n\s*([A-Za-z0-9][^\n]{1,60}?)\s*\n",
        text, re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip()
        # Skip if it looks like an address word
        if not re.match(r"^\d+\s", candidate) and len(candidate) > 1:
            results["shipper"] = candidate

    # ── Consignee — first company/name line after "Drop\n" ──────────────────
    m = re.search(
        r"Drop\s*\n\s*([A-Za-z0-9][^\n]{1,60}?)\s*\n",
        text, re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip()
        if not re.match(r"^\d+\s", candidate) and len(candidate) > 1:
            results["consignee"] = candidate

    # ── Carrier Name ─────────────────────────────────────────────────────────
    # "Carrier Carrier MC Phone Equipment ...\nSWIFT SHIFT LOGISTICS LLC MC..."
    m = re.search(
        r"Carrier\s+Carrier\s+MC\s+Phone[^\n]*\n\s*([A-Z][A-Z\s&,.']+?(?:LLC|INC|CORP|CO\.|LTD|TRANSPORT|LOGISTICS|FREIGHT|TRUCKING|CARRIERS?)?)\s+(?:MC|DOT|\d)",
        text, re.IGNORECASE
    )
    if m:
        results["carrier_name"] = m.group(1).strip().rstrip(",").strip()

    # Fallback: look for known carrier line patterns
    if not results.get("carrier_name"):
        m = re.search(
            r"\n([A-Z][A-Z\s]{3,50}(?:LLC|INC|CORP|LTD|TRANSPORT|LOGISTICS|FREIGHT|TRUCKING))\s+MC\d+",
            text, re.IGNORECASE
        )
        if m:
            results["carrier_name"] = m.group(1).strip()

    # ── Dispatcher ──────────────────────────────────────────────────────────
    # "Dispatcher Zukhruf Rukha"
    # (Not a standard ShipmentData field but useful to know it's parsed)

    # ── Pickup Date/Time ─────────────────────────────────────────────────────
    # "Shipping Date 02-08-2026 Booking Date..." + "Shipping Time 09:00 - 17:00"
    # OR "Shipping Date\n02-08-2026\nShipping Time\n09:00 - 17:00"
    date_m = re.search(
        r"Shipping\s+Date\s*[:\n\s]+(\d{2}[-/]\d{2}[-/]\d{4})",
        text, re.IGNORECASE
    )
    time_m = re.search(
        r"Shipping\s+Time\s*[:\n\s]+([\d:]+\s*[-–]\s*[\d:]+)",
        text, re.IGNORECASE
    )
    if date_m:
        dt = date_m.group(1).strip()
        if time_m:
            dt += f" {time_m.group(1).strip()}"
        results["pickup_datetime"] = dt

    # ── Delivery Date/Time ───────────────────────────────────────────────────
    # "Delivery Date 02-08-2026 Delivery Time 09:00 - 17:00"
    del_date_m = re.search(
        r"Delivery\s+Date\s*[:\n\s]+(\d{2}[-/]\d{2}[-/]\d{4})",
        text, re.IGNORECASE
    )
    del_time_m = re.search(
        r"Delivery\s+Time\s*[:\n\s]+([\d:]+\s*[-–]\s*[\d:]+)",
        text, re.IGNORECASE
    )
    if del_date_m:
        dt = del_date_m.group(1).strip()
        if del_time_m:
            dt += f" {del_time_m.group(1).strip()}"
        results["delivery_datetime"] = dt

    # ── Rate ─────────────────────────────────────────────────────────────────
    # "Agreed Amount (USD) ... $1000.00"
    # OR "Flatbed:$ 1000.00 USD"
    # OR customer rate from rate breakdown total

    # Try "Agreed Amount" pattern first (most reliable for shipper RC)
    m = re.search(
        r"Agreed\s+Amount\s*\([A-Z]+\)\s*\n?.*?\$\s*([\d,]+\.?\d{0,2})",
        text, re.IGNORECASE | re.DOTALL
    )
    if m:
        try:
            results["rate"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # Try Rate Breakdown total
    if not results.get("rate"):
        m = re.search(
            r"(?:Rate Breakdown Total|Carrier Pay Total|Total)\s*\n?.*?([\d,]+\.\d{2})\s*USD",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if val > 0:
                    results["rate"] = val
            except ValueError:
                pass

    # Try inline "$1000.00" after customer/carrier row
    if not results.get("rate"):
        # "Test ABC rukhazukhruf@gmail.com +1 234 567 8900 $1000.00"
        m = re.search(
            r"(?:rukhazukhruf|@gmail|@ultraship)[^\n]*\$\s*([\d,]+\.?\d{0,2})",
            text, re.IGNORECASE
        )
        if m:
            try:
                results["rate"] = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    # Fallback: any dollar amount
    if not results.get("rate"):
        m = re.search(r"\$\s*([\d,]+\.\d{2})", text)
        if m:
            try:
                val = float(m.group(1).replace(",", ""))
                if val > 0:
                    results["rate"] = val
            except ValueError:
                pass

    # ── Currency ─────────────────────────────────────────────────────────────
    m = re.search(r"\b(USD|CAD|EUR|GBP|MXN)\b", text)
    if m:
        results["currency"] = m.group(1)
    elif "$" in text:
        results["currency"] = "USD"

    # ── Weight ───────────────────────────────────────────────────────────────
    # "56000.00 lbs" or "Weight 56000.00 lbs"
    m = re.search(r"([\d,]+\.?\d*)\s*(lbs?|pounds?|kg|kilograms?|cwt)\b", text, re.IGNORECASE)
    if m:
        num  = m.group(1).strip()
        unit = m.group(2).strip()
        results["weight"] = f"{num} {unit}"

    # ── Mode ─────────────────────────────────────────────────────────────────
    m = re.search(
        r"\b(FTL|LTL|FCL|LCL|Full\s+Truck\s+Load|Less\s+Than\s+Truck(?:load)?|Air\s+Freight|Ocean\s+Freight|Rail|Intermodal|Drayage)\b",
        text, re.IGNORECASE
    )
    if m:
        raw = m.group(1).upper().strip()
        # Normalize verbose names
        norm = {
            "FULL TRUCK LOAD": "FTL",
            "LESS THAN TRUCK LOAD": "LTL",
            "LESS THAN TRUCKLOAD": "LTL",
            "AIR FREIGHT": "Air",
            "OCEAN FREIGHT": "Ocean",
        }
        results["mode"] = norm.get(raw, raw)

    # ── Equipment Type ────────────────────────────────────────────────────────
    # "Flatbed:$ 1000.00" or "Equipment ... Flatbed"
    # Pattern 1: equipment column in carrier row
    m = re.search(
        r"(?:Equipment|Trailer\s+Type|Truck\s+Type)\s*[:\n\s]*([A-Za-z\s'\"0-9]{3,30}?)(?:\s+\$|\s+MC|\s+\d|\n|,|$)",
        text, re.IGNORECASE
    )
    if m:
        eq = m.group(1).strip()
        if eq and eq.lower() not in ("agreed amount", "size", "phone", "carrier"):
            results["equipment_type"] = eq

    # Pattern 2: "Flatbed:" prefix in rate breakdown
    if not results.get("equipment_type"):
        m = re.search(
            r"(Dry\s*Van|Flatbed|Reefer|Refrigerated|Tanker|Open\s*Top|Step\s*Deck|RGN|Lowboy|Box\s*Truck|Sprinter|53'?\s*(?:Dry\s*Van|Trailer)?)\s*[:\$]",
            text, re.IGNORECASE
        )
        if m:
            results["equipment_type"] = m.group(1).strip()

    # Pattern 3: standalone keyword
    if not results.get("equipment_type"):
        m = re.search(
            r"\b(Dry\s*Van|Flatbed|Reefer|Refrigerated\s*Trailer|Tanker|Open\s*Top|Step\s*Deck|RGN|Lowboy|Box\s*Truck|Sprinter)\b",
            text, re.IGNORECASE
        )
        if m:
            results["equipment_type"] = m.group(1).strip()

    logger.info("Ultraship regex extracted: %s", {k: v for k, v in results.items() if v is not None})
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Generic regex fallback (non-Ultraship documents)
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_PATTERNS = {
    "shipment_id": [
        r"\bBOL\s*[#:]?\s*([A-Z0-9\-]{4,20})\b",
        r"\b(LD\d{4,12})\b",
        r"(?:Pro|Order|Shipment|Reference|Ref|PO)\s*[#:No.]*\s*([A-Z0-9\-]{4,20})",
        r"(?:bill\s+of\s+lading|B/L)\s*[#:No.]*\s*([A-Z0-9\-]{3,20})",
        r"\b([A-Z]{2,4}\d{5,12})\b",
    ],
    "rate": [
        r"(?:total\s+charge|freight\s+charge|total\s+amount|rate|all[\s\-]?in)[:\s]*\$?\s*([\d,]+\.?\d{0,2})\b",
        r"\$\s*([\d,]+\.?\d{0,2})\s*(?:USD|CAD|total|flat)?",
        r"([\d,]+\.\d{2})\s*(?:USD|CAD|dollars?)",
    ],
    "currency": [
        r"\b(USD|CAD|EUR|GBP|MXN)\b",
    ],
    "weight": [
        r"([\d,]+\.?\d*)\s*(lbs?|pounds?|kg|kilograms?|cwt)\b",
        r"(?:weight|wt\.?)[:\s]*([\d,]+\.?\d*)\s*(lbs?|pounds?|kg)?",
    ],
    "mode": [
        r"\b(FTL|LTL|FCL|LCL|Intermodal|Drayage)\b",
    ],
    "equipment_type": [
        r"\b(Dry\s*Van|Flatbed|Reefer|Refrigerated|Tanker|Open\s*Top|Step\s*Deck|RGN|Lowboy|Box\s*Truck|Sprinter)\b",
    ],
    "pickup_datetime": [
        r"(?:pick[\s\-]?up|ship[\s\-]?date)[:\s]+([A-Za-z0-9,\s/:\-]+?(?:\d{4}|\d{2}:\d{2}))",
    ],
    "delivery_datetime": [
        r"(?:deliver[y]?|drop[\s\-]?off|destination[\s\-]?date)[:\s]+([A-Za-z0-9,\s/:\-]+?(?:\d{4}|\d{2}:\d{2}))",
    ],
}


def _generic_regex_extract(text: str) -> dict[str, Any]:
    results: dict[str, Any] = {}
    normalized = " ".join(text.split())
    for field, patterns in _GENERIC_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, normalized, re.IGNORECASE)
            if m:
                if field == "weight" and m.lastindex and m.lastindex >= 2:
                    num  = m.group(1).strip()
                    unit = m.group(2).strip() if m.group(2) else ""
                    raw  = f"{num} {unit}".strip()
                elif m.lastindex and m.lastindex >= 1:
                    raw = m.group(1).strip()
                else:
                    raw = m.group(0).strip()

                if not raw:
                    continue

                if field == "rate":
                    try:
                        val = float(raw.replace(",", "").strip())
                        if val > 0:
                            results[field] = val
                    except ValueError:
                        pass
                else:
                    results[field] = raw
                break
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  LLM extraction
# ─────────────────────────────────────────────────────────────────────────────

def _build_extraction_prompt(text_snippet: str) -> str:
    return f"""You are an expert logistics document parser for Ultraship TMS.
Extract the following fields from this Rate Confirmation / Bill of Lading document.

KEY EXTRACTION RULES:
- shipment_id: Look for "Reference ID" followed by an ID like LD53657, or BOL numbers.
- shipper: The company/person name that appears directly after the word "Pickup" on its own line (e.g., "AAA").
- consignee: The company/person name that appears directly after the word "Drop" on its own line (e.g., "xyz").
- carrier_name: The carrier company name (e.g., "SWIFT SHIFT LOGISTICS LLC") found in Carrier Details table.
- pickup_datetime: Combine "Shipping Date" and "Shipping Time" values (e.g., "02-08-2026 09:00 - 17:00").
- delivery_datetime: Combine "Delivery Date" and "Delivery Time" values.
- equipment_type: Look for Flatbed, Dry Van, Reefer, etc. in Equipment column or rate breakdown line.
- mode: Look for FTL, LTL, FCL, LCL in "Load Type" field.
- rate: The numeric dollar amount from "Agreed Amount" or rate breakdown total. Numbers only.
- currency: Currency code (USD, CAD, etc.).
- weight: Weight with unit (e.g., "56000.00 lbs").

DOCUMENT TEXT:
{text_snippet[:5000]}

Return ONLY a valid JSON object. Use null for fields not found. No markdown, no explanation:
{{
  "shipment_id": null,
  "shipper": null,
  "consignee": null,
  "pickup_datetime": null,
  "delivery_datetime": null,
  "equipment_type": null,
  "mode": null,
  "rate": null,
  "currency": null,
  "weight": null,
  "carrier_name": null
}}"""


def _parse_llm_json(response: str) -> dict:
    response = re.sub(r"```(?:json)?", "", response).strip().strip("`").strip()
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _merge(ultraship: dict, generic: dict, llm: dict) -> dict:
    """
    Priority: ultraship regex > generic regex > llm
    Numeric fields (rate) always prefer regex when found.
    Semantic fields (shipper, consignee, carrier) prefer llm over generic regex.
    """
    REGEX_PRIORITY = {"rate", "currency", "weight", "shipment_id", "mode", "pickup_datetime", "delivery_datetime"}
    SEMANTIC       = {"shipper", "consignee", "carrier_name", "equipment_type"}

    all_fields = {
        "shipment_id", "shipper", "consignee", "pickup_datetime",
        "delivery_datetime", "equipment_type", "mode", "rate",
        "currency", "weight", "carrier_name",
    }

    NULL_STRINGS = {"null", "none", "n/a", "unknown", "", "not found", "not specified"}

    def clean(v):
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in NULL_STRINGS:
            return None
        return v

    merged = {}
    for field in all_fields:
        us_val  = clean(ultraship.get(field))
        gen_val = clean(generic.get(field))
        llm_val = clean(llm.get(field))

        if us_val is not None:
            merged[field] = us_val
        elif field in REGEX_PRIORITY and gen_val is not None:
            merged[field] = gen_val
        elif field in SEMANTIC and llm_val is not None:
            merged[field] = llm_val
        elif gen_val is not None:
            merged[field] = gen_val
        elif llm_val is not None:
            merged[field] = llm_val
        else:
            merged[field] = None

    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_structured_data(
    storage_path: Path,
    doc_id: str,
) -> tuple[ShipmentData, LLMResult]:
    """
    Extract structured shipment data using three layers:
      1. Ultraship TMS layout-aware regex (highest priority)
      2. Generic regex patterns (fallback)
      3. LLM extraction (fills remaining gaps)
    """
    _, chunks_meta = load_index(storage_path, doc_id)
    full_text = "\n\n".join(c["text"] for c in chunks_meta)

    logger.info("Extraction: %d chars from %d chunks", len(full_text), len(chunks_meta))

    # Layer 1: Ultraship-specific regex
    ultraship_results = _extract_ultraship(full_text)

    # Layer 2: Generic regex
    generic_results = _generic_regex_extract(full_text)

    # Layer 3: LLM
    prompt = _build_extraction_prompt(full_text)
    try:
        llm_result = route_llm_raw(prompt)
        llm_json   = _parse_llm_json(llm_result.response)
        logger.info("LLM extracted: %s", {k: v for k, v in llm_json.items() if v})
    except Exception as e:
        logger.error("LLM extraction failed: %s", e)
        llm_json   = {}
        llm_result = LLMResult(
            provider="None",
            response="",
            logs=[f"LLM extraction failed: {e}"],
        )

    # Merge all three layers
    merged = _merge(ultraship_results, generic_results, llm_json)

    # Normalise rate to float
    if merged.get("rate") and not isinstance(merged["rate"], float):
        try:
            merged["rate"] = float(str(merged["rate"]).replace(",", "").replace("$", "").strip())
        except (ValueError, TypeError):
            merged["rate"] = None

    shipment_data = ShipmentData(**merged)
    return shipment_data, llm_result
