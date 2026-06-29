"""Tests for shared figure styling helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_BUNDLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BUNDLE / "pipeline"))

from plot_helpers import get_family, short_method_label


def test_lcpmst_classified_as_lcp_baselines():
    assert get_family("LCPmst") == "LCP baselines"
    assert get_family("RLCPmst") == "LCP baselines"


def test_short_method_label_handles_parenthesis_form():
    assert short_method_label("OMspellRS(0.50)") == "OMsRS 0.50"
    assert short_method_label("OMspell(1.00)") == "OMs 1.00"
    assert short_method_label("LCPspell(expcost=0.50)") == "LCPs 0.50"
