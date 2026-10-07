# Common physical schema

Raw scene: agent arrays p/v/g/current Flow, radii and goal tolerances; unordered
obstacle primitives; current policy reference chart; physical execution/time
and monitor configuration. Native parsers contain no learned parameters.

Agents (15 channels): goal distance; local velocity (2); local raw Flow (2);
local speed-bounded Flow (2); radius; tolerance; current goal occupancy;
policy-axis applicability; local displacement to policy origin (2); local
policy-axis unit vector (2). Occupancy is not an absorbing finished-agent flag.

Ordered pairs (9 channels): local relative position (2), distance, relative
velocity (2), other goal relative to observer (2), surface clearance, summed
radii. Shared weights, no pair or agent index. Self pairs are masked.

Agent–obstacle relations (13 channels): closest-point displacement (2), surface
clearance, normal (2), signed curvature, primitive radius/thickness, extent,
two primitive endpoints/center locations (4), curved-primitive flag. Line
endpoints are unordered after observer-local ordering. Circle curvature sign
describes free exterior/interior, not Ring identity. This represents both
annular boundaries and actual polygon/segment boundaries without wall names.

Globals (15): remaining fraction/seconds, horizon seconds, dt, speed bound,
first-action commitment, wall/agent collision margins, active monitor,
progress-window/hold duration, progress/speed thresholds, monitor elapsed
time and empty-history flag. All current Double states are true t0.

Fixed physical scaling: length 4 m; velocity 0.82 m/s; time 60 s; radius
0.2 m; tolerance 0.1 m; margins 0.01 m. No slot-wise fitted normalization.
Train/validation statistics are not needed by the encoder.

Masks carry only entity existence. Counts are log1p(N), log1p(M), physical
cardinality rather than scenario identities. Neural h is not a DB state key.
