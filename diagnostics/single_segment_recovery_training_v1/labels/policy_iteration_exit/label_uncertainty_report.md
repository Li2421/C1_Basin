# Label uncertainty report: policy_iteration_exit

- Complete branch rows: 384
- Matched decision inputs: 6
- Futures per input: 32
- Train / validation inputs: 6 / 0
- Unresolved inputs: 4
- Success-equivalence supported within epsilon_Q=0.02: 0

## Status counts

- ACTION0_SUCCESS_FAVORED: 2
- UNRESOLVED_NO_DISCORDANCE: 4

## Interpretation guardrail

Thirty-two matched futures with no discordant successes remain unresolved under the frozen conservative paired interval. 
They are not evidence of epsilon_Q=0.02 equivalence, and therefore do not activate the deformation tie-break.
