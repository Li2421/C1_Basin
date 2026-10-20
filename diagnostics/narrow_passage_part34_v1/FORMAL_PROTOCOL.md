# Formal A1–A5 follow-up protocol

The user's formal assumptions supersede the previous missing-definition status.
This follow-up reads only existing TT traces, metadata and static calculations.

Use physical radius r=0.16, W=0.4, wall center safety distance
c=r+wall_radius+wall_collision_margin+CBF_buffer=0.1717666666666667.
Set D_s to the actual frozen pair-CBF safety distance 0.3201, rather than
the evaluator collision distance 0.32. Fix an additional geometric margin
delta=0.01 m, so the formal progress threshold is D=0.3301. This margin
is chosen as a round positive geometric tolerance, independently of labels.
Do not change the previously saved boundary-level D=0.3201 statistics.

Use a geometry-defined rectangular family, parameterized by wall-CBF margin m:
h=b_plus=-b_minus=W/2-c-m;
a_plus=-a_minus=1.3-c-m;
H=0.68-c-m.
The primary m is 0.001 m. Fixed sensitivity margins are
0.005, 0.001, 0.0001, 0.00001, 0.000001, 0.00000001 m.
These are geometric coverage diagnostics, not outcome-based region selection.
The last margins approach numerical resolution and are not robust certificates
for deployment; record their magnitude explicitly.

Role R: formal robot 1=physical agent 0, robot 2=physical agent 1;
x'=x, y'=y. Role L: formal robot 1=physical agent 1, robot 2=physical
agent 0; x'=-x, y'=y. The second mapping reflects the symmetric Toy walls
and goals. Both preserve s=p0x-p1x; z is respectively p1y-p0y and p0y-p1y.
Evaluate both mappings as fixed choices, without assigning roles by success.

A3 requires membership in C/Y, s<0, 0<=z<D, actual pair-CBF-safe geometry,
and a saved action certificate satisfying the frozen tolerances. Record true-t0
and local continuation states separately. The local continuation retains the
actual archived future noise, eta and monitor history; no reset or rollout.

Use formal D for yield/pass progression and lateral-clearance preservation.
A1: 2r<W<4r. A2: wall-safe rectangles, strict ordering and H>h+D.
A4: a_plus>b_plus+D, opposite goal sides, explicit geometric continuation.
A5: no third robot exists in Toy; its external-robot conditions are vacuous.
Theoretical reference CBF feasibility, actual fixed-eta execution and the
deadlock/horizon conditions remain separate claims.
