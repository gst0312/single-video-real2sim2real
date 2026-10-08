"""The physics gate decides which replayed episodes become training data. It is checked
against the three real replay reports shipped in results/ and against synthetic failures."""
import copy
import json
import pathlib

import pytest

from polaris_lfhv.physics_gate import (CUP_MAX, GAP_MAX, LIFT_MIN, passes_physics_gate,
                                       physics_gate_reasons)

REPORTS = sorted((pathlib.Path(__file__).resolve().parents[1] / "results/phase3_rollouts").glob("cond000_ep*.json"))


def test_thresholds_match_the_documented_gate():
    assert (LIFT_MIN, GAP_MAX, CUP_MAX) == (0.06, 0.05, 0.02)


@pytest.mark.parametrize("path", REPORTS, ids=lambda p: p.stem)
def test_shipped_phase3_reports_pass(path):
    report = json.loads(path.read_text())
    assert physics_gate_reasons(report) == []
    assert passes_physics_gate(report)
    # and they are genuine successes: the rubric fired too
    assert report["success"] is True and report["progress_max"] == 1.0


def test_three_reports_are_shipped():
    assert len(REPORTS) == 3


@pytest.fixture
def good():
    return json.loads(REPORTS[0].read_text())


def test_missed_grasp_is_rejected_for_lift(good):
    r = copy.deepcopy(good)
    r["bottle_lift"] = 0.01
    why = physics_gate_reasons(r)
    assert len(why) == 1 and why[0].startswith("lift 1.0 cm")


def test_bottle_knocked_askew_is_rejected_for_drift(good):
    r = copy.deepcopy(good)
    r["bottle_vs_reference"]["gap_max_m"] = 0.082
    assert physics_gate_reasons(r) == ["drifted 8.2 cm from the reference"]


def test_cup_collision_is_rejected(good):
    r = copy.deepcopy(good)
    r["cup_moved_m"] = 0.03
    assert physics_gate_reasons(r) == ["cup moved 3.0 cm"]


def test_replay_that_never_reached_the_grasp_is_rejected(good):
    r = copy.deepcopy(good)
    r["bottle_vs_reference"]["gap_max_m"] = None
    assert "never reached the grasp step" in physics_gate_reasons(r)


def test_every_reason_is_reported_not_just_the_first(good):
    r = copy.deepcopy(good)
    r["bottle_lift"], r["cup_moved_m"] = 0.0, 0.1
    assert len(physics_gate_reasons(r)) == 2


def test_boundaries_are_inclusive_as_in_both_scripts(good):
    r = copy.deepcopy(good)
    r["bottle_lift"], r["cup_moved_m"] = LIFT_MIN, CUP_MAX
    r["bottle_vs_reference"]["gap_max_m"] = GAP_MAX
    assert passes_physics_gate(r)


def test_pour_accuracy_is_deliberately_not_gated(good):
    # an episode that grasped and carried but poured beside the cup still passes the gate;
    # the rubric, not the gate, judges the pour (docs/design.md)
    r = copy.deepcopy(good)
    r["success"], r["progress_max"] = False, 0.667
    assert passes_physics_gate(r)
