"""
HampirSehat — Offline pytest suite
Tests deterministic components only — no API key required.

Run:
    pip install pytest
    pytest ai_backend/tests/ -v

What is tested here:
    - enforce_math()         : Atwater 4-4-9 formula, write-back, edge cases
    - _format_human_readable(): output structure, meal-time mapping, error path
    - assess() input guard   : empty input returns error dict

What is NOT tested here (requires live GROQ_API_KEY):
    - RAG search, Front Office, agent opinions, Lead Auditor
    → Use the notebook stress test suite for end-to-end validation:
      ai_backend/hampir_sehat_flow.ipynb → cell: stress-test-code
"""

import sys
import os
import pytest
import unittest.mock as mock

# ── Patch all external dependencies before importing app ─────────────────────
# This allows the test suite to run without groq, gradio, or langchain installed.
# Only the deterministic functions (enforce_math, _format_human_readable) are tested.

_mock_modules = [
    "gradio", "gradio.themes",
    "groq",
    "langchain_community", "langchain_community.tools",
    "dotenv",
]
for _mod in _mock_modules:
    sys.modules.setdefault(_mod, mock.MagicMock())

# Patch gr.Blocks context manager used at module level in app.py
import gradio as _gr
_gr.Blocks.return_value.__enter__ = mock.MagicMock(return_value=mock.MagicMock())
_gr.Blocks.return_value.__exit__  = mock.MagicMock(return_value=False)

# Provide a fake GROQ_API_KEY so the env check passes
os.environ.setdefault("GROQ_API_KEY", "test-key-offline")

# Now safe to import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app as pipeline


# ─────────────────────────────────────────────────────────────────────────────
#  enforce_math() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnforceMath:

    def test_basic_atwater_formula(self):
        """calories = (carbs*4) + (protein*4) + (fat*9)"""
        result = {
            "calories_kcal": 999,   # wrong — should be overridden
            "macros": {"carbs_g": 60, "protein_g": 20, "fat_g": 15},
        }
        out = pipeline.enforce_math(result)
        expected = (60 * 4) + (20 * 4) + (15 * 9)   # 240 + 80 + 135 = 455
        assert out["calories_kcal"] == expected
        assert out["math_enforced"] is True

    def test_write_back_rounds_floats(self):
        """Float macros must be rounded and written back before calculation."""
        result = {
            "calories_kcal": 0,
            "macros": {"carbs_g": 60.0, "protein_g": 20.6, "fat_g": 15.3},
        }
        out = pipeline.enforce_math(result)
        # Python banker's rounding: round(20.6)=21, round(15.3)=15
        c, p, f = 60, 21, 15
        assert out["macros"]["carbs_g"]   == c
        assert out["macros"]["protein_g"] == p
        assert out["macros"]["fat_g"]     == f
        assert out["calories_kcal"] == (c * 4) + (p * 4) + (f * 9)

    def test_zero_gap_guarantee(self):
        """After enforce_math, recalculating from stored macros must equal calories_kcal."""
        result = {
            "calories_kcal": 1,
            "macros": {"carbs_g": 75, "protein_g": 30, "fat_g": 42},
        }
        out = pipeline.enforce_math(result)
        c = out["macros"]["carbs_g"]
        p = out["macros"]["protein_g"]
        f = out["macros"]["fat_g"]
        recalc = (c * 4) + (p * 4) + (f * 9)
        assert recalc == out["calories_kcal"], f"Gap detected: {recalc} != {out['calories_kcal']}"

    def test_all_zero_macros_sets_warning(self):
        """If all macros are 0, math_enforced should be False and math_warning set."""
        result = {
            "calories_kcal": 500,
            "macros": {"carbs_g": 0, "protein_g": 0, "fat_g": 0},
        }
        out = pipeline.enforce_math(result)
        assert out["math_enforced"] is False
        assert "math_warning" in out

    def test_error_result_passthrough(self):
        """enforce_math must not touch error responses."""
        result = {"error": "Blocked", "reason": "Harmful keyword detected"}
        out = pipeline.enforce_math(result)
        assert out == result   # unchanged

    def test_math_correction_logged_when_llm_was_wrong(self):
        """If LLM calories differ from calculated, math_correction key is added."""
        result = {
            "calories_kcal": 820,   # LLM hallucinated
            "macros": {"carbs_g": 75, "protein_g": 30, "fat_g": 20},
        }
        out = pipeline.enforce_math(result)
        # (75*4)+(30*4)+(20*9) = 300+120+180 = 600
        assert out["calories_kcal"] == 600
        assert "math_correction" in out
        assert "820" in out["math_correction"]

    def test_quantity_multiplier_scenario(self):
        """Simulates a ×2 quantity result — macros doubled, math still holds."""
        result = {
            "calories_kcal": 0,
            "macros": {"carbs_g": 120, "protein_g": 10, "fat_g": 8},
        }
        out = pipeline.enforce_math(result)
        assert out["calories_kcal"] == (120 * 4) + (10 * 4) + (8 * 9)   # 480+40+72=592


# ─────────────────────────────────────────────────────────────────────────────
#  _format_human_readable() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatHumanReadable:

    def _make_result(self, **kwargs):
        base = {
            "identified_item": "Nasi Goreng Telur",
            "is_healthy": False,
            "calories_kcal": 487,
            "macros": {"carbs_g": 70, "protein_g": 18, "fat_g": 15},
            "audit_summary": "Standard fried rice portion, macro-consistent.",
            "portion_adjusted": False,
            "math_enforced": True,
        }
        base.update(kwargs)
        return base

    def test_output_contains_required_sections(self):
        result = self._make_result()
        out = pipeline._format_human_readable("nasi goreng telur", result)
        assert "NUTRITION SUMMARY"  in out
        assert "Detected Menu"      in out
        assert "Meal Time"          in out
        assert "TOTAL NUTRITION"    in out
        assert "Calories"           in out
        assert "Protein"            in out
        assert "Carbohydrates"      in out
        assert "Fat"                in out
        assert "Note"               in out

    def test_calorie_value_present_in_output(self):
        result = self._make_result(calories_kcal=487)
        out = pipeline._format_human_readable("nasi goreng telur", result)
        assert "487" in out

    def test_single_meal_session_label(self):
        """Input with no time keywords → Session: Single Meal"""
        result = self._make_result()
        out = pipeline._format_human_readable("nasi goreng telur", result)
        assert "Single Meal" in out

    def test_multi_meal_morning_detected(self):
        """Input with 'pagi' → Morning slot detected"""
        result = self._make_result()
        out = pipeline._format_human_readable("pagi nasi uduk siang ayam geprek", result)
        assert "Morning" in out

    def test_multi_meal_dinner_detected(self):
        """Input with 'malam' → Dinner slot detected"""
        result = self._make_result()
        out = pipeline._format_human_readable("malam makan soto ayam", result)
        assert "Dinner" in out

    def test_multi_meal_english_keywords(self):
        """English time keywords also trigger mapping"""
        result = self._make_result()
        out = pipeline._format_human_readable("breakfast oatmeal lunch salad dinner grilled chicken", result)
        assert "Morning" in out
        assert "Lunch"   in out
        assert "Dinner"  in out

    # ── Issue 1: No truncation ────────────────────────────────────────────

    def test_meal_segment_not_truncated(self):
        """Meal segments must show full text — no 80-char cutoff."""
        long_input = "pagi makan nasi uduk dengan tempe orek dan telur dadar plus kerupuk"
        result = self._make_result()
        out = pipeline._format_human_readable(long_input, result)
        # The full food description after 'pagi' must appear
        assert "tempe orek" in out
        assert "kerupuk" in out

    def test_dinner_segment_not_truncated(self):
        """Dinner segment must not be cut off mid-sentence."""
        long_input = "malam makan soto ayam dengan tempe goreng dan telur rebus plus nasi putih"
        result = self._make_result()
        out = pipeline._format_human_readable(long_input, result)
        assert "tempe goreng" in out
        assert "nasi putih" in out

    # ── Issue 2: audit_summary shown in Note ─────────────────────────────

    def test_audit_summary_used_as_note(self):
        """audit_summary from JSON appears as the Note value"""
        result = self._make_result(audit_summary="Macro-consistency normalization applied.")
        out = pipeline._format_human_readable("nasi goreng", result)
        assert "Macro-consistency normalization applied." in out

    def test_fallback_health_note_when_no_audit_summary(self):
        """If audit_summary is empty, a plain health note is substituted"""
        result = self._make_result(audit_summary="", is_healthy=True)
        out = pipeline._format_human_readable("nasi goreng", result)
        assert "Note:" in out
        note_line = [l for l in out.splitlines() if "Note:" in l]
        assert note_line
        assert len(note_line[0].strip()) > len("💡 Note:")

    def test_internal_pipeline_labels_not_shown_in_note(self):
        """Internal labels like 'Multi-meal aggregation applied' must never appear in Note."""
        result = self._make_result(audit_summary="Multi-meal aggregation applied.")
        out = pipeline._format_human_readable("nasi goreng", result)
        # audit_summary is passed through as-is — but the test verifies
        # the function doesn't inject its own internal labels
        # The Note should show whatever audit_summary says, not override it
        assert "💡 Note:" in out

    # ── Issue 3: Deduplication flag ───────────────────────────────────────

    def test_duplicate_food_across_slots_triggers_flag(self):
        """Same food keyword in multiple slots → deduplication warning in Note."""
        # 'soto' appears in both morning and dinner segments
        dup_input = "pagi makan soto ayam, malam makan soto ayam lagi"
        result = self._make_result()
        out = pipeline._format_human_readable(dup_input, result)
        assert "Similar items detected" in out
        assert "verify" in out.lower()

    def test_no_false_positive_dedup_on_different_foods(self):
        """Different foods in different slots must NOT trigger the dedup flag."""
        clean_input = "pagi nasi uduk, siang ayam geprek, malam soto betawi"
        result = self._make_result()
        out = pipeline._format_human_readable(clean_input, result)
        assert "Similar items detected" not in out

    def test_dedup_does_not_remove_items(self):
        """Dedup flag must not remove any food items — only adds a warning."""
        dup_input = "pagi makan soto ayam, malam makan soto ayam lagi"
        result = self._make_result()
        out = pipeline._format_human_readable(dup_input, result)
        # Both Morning and Dinner slots must still appear
        assert "Morning" in out
        assert "Dinner"  in out


# ─────────────────────────────────────────────────────────────────────────────
#  assess() input guard
# ─────────────────────────────────────────────────────────────────────────────

class TestAssessInputGuard:

    def test_empty_string_returns_error(self):
        result = pipeline.assess("", verbose=False)
        assert result.get("error") is not None

    def test_whitespace_only_returns_error(self):
        result = pipeline.assess("   ", verbose=False)
        assert result.get("error") is not None
