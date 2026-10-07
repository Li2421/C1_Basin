# OrthoFlow3 basin-margin learning v1

This controlled experiment compares two identical `214 -> 128 -> 128 -> 3` SiLU actors. `G_CENTER` uses normalized center MSE. `G_MARGIN` uses only `[max(0, ||eta-c||/(r+eps)-0.80)]^2`. The prior low-J actor is frozen and read-only. No Q or J network is trained or used for gradients.

Verified sphere labels use the single continuous `E_bridge`, never the rejected ellipsoids or historical disconnected domain. Anchor families are indivisible across the frozen Q-v2 TRAIN/VAL/TEST split. The four new source states are positions 7--10 of the pre-outcome frozen conservative-ball state permutation: two TRAIN and two VAL groups. No outcome was inspected when selecting them.

The expected label-build increment is 15,025 continuations and 3.48M steps. The complete experiment preflight is below 25,000 continuations, 8M steps, and two hours of projected critical-path rollout time. If verified labels do not reach TRAIN/VAL/TEST >=24/8/8, training stops.
