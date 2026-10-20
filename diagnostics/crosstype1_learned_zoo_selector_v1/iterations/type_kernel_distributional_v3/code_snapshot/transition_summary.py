"""Descriptive nominal-collision to safety transitions; no deadlock relabeling."""
from collections import Counter


def summarize(rows):
    output={}
    for group in [*sorted({r['type'] for r in rows}),'combined']:
        subset=[r for r in rows if group=='combined' or r['type']==group]
        collision=[r for r in subset if r['nominal_outcome']=='collision']
        counts=Counter(r['safety_outcome'] for r in collision)
        output[group]=dict(total_episodes=len(subset),nominal_collision_episodes=len(collision),
            safety_counts={k:counts[k] for k in ('success','timeout','collision','unknown','other')},
            safety_percent_given_nominal_collision={k:100*counts[k]/len(collision) if collision else None
                for k in ('success','timeout','collision','unknown','other')},
            timeout_not_assumed_deadlock=True)
    return output
