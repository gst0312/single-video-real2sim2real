"""The evaluation helpers that run without a simulator: condition sharding and the
outcome-by-layout analysis."""
import csv
import json

import numpy as np

import analyse_eval
import shard_conditions


def write_conditions(path, n):
    poses = [{"mustard": [0.45 + 0.01 * i, -0.10, -0.019, 1, 0, 0, 0],
              "blue_cup": [0.47, 0.15 + 0.005 * i, -0.019, 1, 0, 0, 0]} for i in range(n)]
    path.write_text(json.dumps({"instruction": "pour the mustard into the blue cup", "poses": poses}))
    return poses


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "episode_length", "success", "progress"])
        w.writeheader()
        w.writerows(rows)


def test_split_is_contiguous_and_keeps_the_instruction(tmp_path):
    conditions = tmp_path / "eval_conditions.json"
    poses = write_conditions(conditions, 7)
    per = shard_conditions.split(conditions, tmp_path / "shards", shards=3)
    assert per == 3
    shards = [json.loads((tmp_path / "shards" / f"shard{i}.json").read_text()) for i in range(3)]
    assert [len(s["poses"]) for s in shards] == [3, 3, 1]
    assert all(s["instruction"] == "pour the mustard into the blue cup" for s in shards)
    assert shards[1]["poses"][0] == poses[3]       # shard i starts at original index i*per


def test_merge_restores_original_condition_indices(tmp_path, capsys):
    out = tmp_path / "eval_2000"
    write_results(out / "shard0" / "eval_results.csv",
                  [{"episode": 0, "episode_length": 1050, "success": True, "progress": 1.0},
                   {"episode": 1, "episode_length": 1050, "success": False, "progress": 0.667},
                   {"episode": 2, "episode_length": 1050, "success": True, "progress": 1.0}])
    write_results(out / "shard1" / "eval_results.csv",
                  [{"episode": 0, "episode_length": 1050, "success": False, "progress": 0.333}])
    shard_conditions.merge(out, shards=2, per=3)
    rows = list(csv.DictReader((out / "eval_results.csv").open()))
    assert [r["episode"] for r in rows] == ["0", "1", "2", "3"]
    assert "2/4 success (50%)" in capsys.readouterr().out


def test_analyse_eval_joins_outcomes_with_layouts(tmp_path, capsys):
    conditions = tmp_path / "eval_conditions.json"
    write_conditions(conditions, 4)
    results = tmp_path / "eval_results.csv"
    write_results(results, [{"episode": 0, "episode_length": 1050, "success": True, "progress": 1.0},
                            {"episode": 1, "episode_length": 1050, "success": False, "progress": 0.667},
                            {"episode": 2, "episode_length": 1050, "success": True, "progress": 1.0},
                            {"episode": 9, "episode_length": 1050, "success": True, "progress": 1.0}])
    rows = analyse_eval.load(results, conditions)
    assert len(rows) == 3                       # the out-of-range episode is skipped
    assert [r["ok"] for r in rows] == [True, False, True]
    assert np.allclose(rows[1]["bottle"], [0.46, -0.10])
    analyse_eval.report(rows)
    out = capsys.readouterr().out
    assert "2/3 succeeded (67%)" in out
    assert "1 grasped but did not pour" in out
