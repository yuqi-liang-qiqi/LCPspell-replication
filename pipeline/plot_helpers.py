"""Shared styling helpers for sensitivity profile figures."""

from __future__ import annotations

METHOD_FAMILIES = {
    "LCP baselines": ["LCP", "RLCP", "LCPmst", "RLCPmst"],
    "LCPspell": [
        "LCPspell(expcost=0.0)",
        "LCPspell(expcost=0.00)",
        "LCPspell(expcost=0.10)",
        "LCPspell(expcost=0.25)",
        "LCPspell(expcost=0.5)",
        "LCPspell(expcost=0.50)",
        "LCPspell(expcost=0.75)",
        "LCPspell(expcost=1.0)",
        "LCPspell(expcost=1.00)",
        "LCPspell(expcost=1.50)",
        "LCPspell(expcost=2.00)",
    ],
    "RLCPspell": [
        "RLCPspell(expcost=0.0)",
        "RLCPspell(expcost=0.00)",
        "RLCPspell(expcost=0.10)",
        "RLCPspell(expcost=0.25)",
        "RLCPspell(expcost=0.5)",
        "RLCPspell(expcost=0.50)",
        "RLCPspell(expcost=0.75)",
        "RLCPspell(expcost=1.0)",
        "RLCPspell(expcost=1.00)",
        "RLCPspell(expcost=1.50)",
        "RLCPspell(expcost=2.00)",
    ],
    "OM-based": [
        "OM",
        "OMspell(expcost=0)",
        "OMspell(expcost=0.00)",
        "OMspell(expcost=0.10)",
        "OMspell(expcost=0.25)",
        "OMspell(expcost=0.5)",
        "OMspell(expcost=0.50)",
        "OMspell(expcost=0.75)",
        "OMspell(expcost=1.0)",
        "OMspell(expcost=1.00)",
        "OMspell(expcost=1.50)",
        "OMspell(expcost=2.00)",
        "OMspellRS(expcost=0)",
        "OMspellRS(expcost=0.00)",
        "OMspellRS(expcost=0.10)",
        "OMspellRS(expcost=0.25)",
        "OMspellRS(expcost=0.5)",
        "OMspellRS(expcost=0.50)",
        "OMspellRS(expcost=0.75)",
        "OMspellRS(expcost=1.0)",
        "OMspellRS(expcost=1.00)",
        "OMspellRS(expcost=1.50)",
        "OMspellRS(expcost=2.00)",
    ],
    "Others": ["Hamming"],
}

METHOD_TO_FAMILY = {
    method: family for family, methods in METHOD_FAMILIES.items() for method in methods
}

FAMILY_ORDER = ["LCP baselines", "LCPspell", "RLCPspell", "OM-based", "Others"]

COLORS = {
    "LCP baselines": "#3B82F6",
    "LCPspell": "#10B981",
    "RLCPspell": "#8B5CF6",
    "OM-based": "#F59E0B",
    "Others": "#EF4444",
}

LEGEND_LABELS = {"Others": "Hamming"}


def get_family(method_name: str) -> str:
    if method_name in METHOD_TO_FAMILY:
        return METHOD_TO_FAMILY[method_name]
    if "RLCPspell" in method_name:
        return "RLCPspell"
    if "LCPspell" in method_name:
        return "LCPspell"
    if "OMspellRS" in method_name:
        return "OM-based"
    if "OMspell" in method_name or method_name == "OM":
        return "OM-based"
    if method_name in {"LCP", "RLCP", "LCPmst", "RLCPmst"}:
        return "LCP baselines"
    return "Others"


def short_method_label(method: str) -> str:
    if method in {"Hamming", "HAM"}:
        return "HAM"
    if method in {"LCP", "RLCP", "LCPmst", "RLCPmst", "OM"}:
        return method
    if method.startswith("RLCPspell("):
        return (
            method.replace("RLCPspell(", "RLCPs ")
            .replace(")", "")
            .replace("expcost=", "")
        )
    if method.startswith("LCPspell("):
        return (
            method.replace("LCPspell(", "LCPs ")
            .replace(")", "")
            .replace("expcost=", "")
        )
    if method.startswith("OMspellRS("):
        return (
            method.replace("OMspellRS(", "OMsRS ")
            .replace(")", "")
            .replace("expcost=", "")
        )
    if method.startswith("OMspell("):
        return (
            method.replace("OMspell(", "OMs ")
            .replace(")", "")
            .replace("expcost=", "")
        )
    if "RLCPspell_expcost_" in method:
        return "RLCPs " + method.split("_expcost_")[1]
    if "LCPspell_expcost_" in method:
        return "LCPs " + method.split("_expcost_")[1]
    if "OMspellRS_expcost_" in method:
        return "OMsRS " + method.split("_expcost_")[1]
    if "OMspell_expcost_" in method:
        return "OMs " + method.split("_expcost_")[1]
    return method
