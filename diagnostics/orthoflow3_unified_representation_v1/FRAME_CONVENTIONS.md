# Frames

For agent i, e_i=(g_i-p_i)/||g_i-p_i|| and t_i=(-e_iy,e_ix).
Every observer-local vector is (e_i·v,t_i·v). Own velocity, raw/bounded Flow,
pair displacements/relative velocities/goals, obstacle endpoints/normals and
policy-frame fields use this same transform.

At near-zero goal distance (1e-10), use the first nonzero current velocity,
current Flow, or displacement from policy origin; only at total degeneracy
use the policy reference axis. All candidates transform with a passive chart
change. No future trajectory or mode determines the frame.

Translation/rotation tests transform geometry, positions/goals and the policy
origin together; rotate all vector fields and policy axes. Ring active proper
rotation may also be tested because no world-fixed policy axis is used.
Four active rotation is a different upstream continuation problem, so exact
invariance is explicitly not required. No reflection-invariance claim.
