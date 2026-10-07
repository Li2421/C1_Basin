"""Expert pilot gate: 30 independent states, both circulation hypotheses."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .environment import RingExchangeEnv, sample_initial_state
from .expert import CentralizedExpert, all_circulation_hypotheses


def run_pilot(count: int = 30, split: str = "development") -> dict:
    planner = CentralizedExpert()
    results = []
    for seed in range(count):
        for hypothesis in all_circulation_hypotheses():
            env = RingExchangeEnv(instance=sample_initial_state(split, seed))
            plan = planner.plan_hypothesis(env, hypothesis)
            results.append({"seed": seed, "mode": hypothesis.direction, "success": plan.success,
                            "collision": plan.collision, "terminal": plan.terminal_reason,
                            "steps": plan.episode_steps, "min_pair": plan.min_inter_agent_surface_distance,
                            "min_obstacle": plan.min_obstacle_clearance})
    successful = [r for r in results if r["success"]]
    return {"protocol": "30 independent development initial states x CW/CCW expert hypotheses",
            "states": count, "candidate_rollouts": len(results), "successes": len(successful),
            "success_rate": len(successful) / len(results), "collisions": sum(r["collision"] for r in results),
            "modes_with_success": sorted({r["mode"] for r in successful}), "records": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--count", type=int, default=30); parser.add_argument("--output")
    arguments = parser.parse_args(); report = run_pilot(arguments.count)
    text = json.dumps(report, indent=2)
    if arguments.output: Path(arguments.output).write_text(text + "\n", encoding="utf-8")
    print(text)
