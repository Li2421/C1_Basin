"""Predeclared controller families; all16 known types are DEVELOPMENT."""
LEARNED=('Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae','ring_ippo','ring_ippo_conservative')
OPTIMIZATION=('distributed_shooting','centralized_sampling','cross_entropy','distributed_cross_entropy','centralized_mppi')
SEARCH=('prioritized_astar',)
UNSEEN=('transformer_bc','gradient_mpc','conflict_based_search')
def family_groups(types):
    assert set(types)==set(LEARNED+OPTIMIZATION+SEARCH)
    return dict(learned=LEARNED,optimization=OPTIMIZATION,search=SEARCH)
def family(kind):
    if kind in LEARNED or kind=='transformer_bc':return 'learned'
    if kind in OPTIMIZATION or kind=='gradient_mpc':return 'optimization'
    assert kind in SEARCH or kind=='conflict_based_search';return 'search'
