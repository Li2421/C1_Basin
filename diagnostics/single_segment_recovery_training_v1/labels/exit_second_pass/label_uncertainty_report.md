# Label uncertainty report: exit_second_pass

- Complete branch rows: 768
- Matched decision inputs: 12
- Futures per input: 32
- Train / validation inputs: 8 / 4
- Unresolved inputs: 11
- Success-equivalence supported within epsilon_Q=0.02: 0

## Status counts

- ACTION0_SUCCESS_FAVORED: 1
- UNRESOLVED_NO_DISCORDANCE: 8
- UNRESOLVED_PAIRED_SUCCESS_DIFFERENCE: 3

## Interpretation guardrail

Thirty-two matched futures with no discordant successes remain unresolved under the frozen conservative paired interval. 
They are not evidence of epsilon_Q=0.02 equivalence, and therefore do not activate the deformation tie-break.
