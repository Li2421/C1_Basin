"""Mutually exclusive paper outcomes; independent of controller and termination."""
OUTCOMES = ('success', 'collision_failure', 'safe_deadlock', 'other_timeout')
PROTOCOL = 'exclusive_episode_outcomes_v1'

FIRST_EVENT_OUTCOMES = ('success', 'wall_collision', 'agent_collision', 'safe_deadlock', 'other_timeout')
FIRST_EVENT_PROTOCOL = 'exclusive_first_terminal_event_v2'


def first_event(info):
    """Classify the terminal step, never a solver error or a running episode."""
    if info['agent_collision']:
        return 'agent_collision'
    if info['wall_collision']:
        return 'wall_collision'
    if info['task_success']:
        return 'success'
    if info['deadlock']:
        return 'safe_deadlock'
    if info['termination'] == 'timeout':
        return 'other_timeout'
    raise ValueError('Episode has no terminal event')


def first_event_aggregate(summaries):
    if not summaries or any(s['outcome'] not in FIRST_EVENT_OUTCOMES for s in summaries):
        raise ValueError('Expected completed first-event episodes')
    total_steps = sum(s['episode_steps'] for s in summaries)
    times = [s['completion_time'] for s in summaries if s['outcome'] == 'success']
    return dict(**{name+'_rate': sum(s['outcome'] == name for s in summaries)/len(summaries)
                   for name in FIRST_EVENT_OUTCOMES},
                cbf_intervention_rate=sum(s['intervened_steps'] for s in summaries)/total_steps,
                mean_completion_time=sum(times)/len(times) if times else None)


def classify(summary):
    # Any collision in the observed episode overrides task completion/deadlock.
    if summary['wall_collision'] or summary['agent_collision']:
        return 'collision_failure'
    if summary.get('task_completed', summary['success']):
        return 'success'
    if summary['deadlock']:
        return 'safe_deadlock'
    return 'other_timeout'


def annotate(summary):
    result = dict(summary)
    result.setdefault('task_completed', summary['success'])
    result['outcome'] = classify(result)
    result['success'] = result['outcome'] == 'success'
    result['collision_free_success'] = result['success']
    return result


def aggregate_outcomes(summaries):
    if not summaries:
        raise ValueError('At least one episode is required')
    labels = [classify(s) for s in summaries]
    counts = {name: labels.count(name) for name in OUTCOMES}
    return dict(outcome_protocol=PROTOCOL, outcome_counts=counts,
                **{name + '_rate': count / len(labels) for name, count in counts.items()})
