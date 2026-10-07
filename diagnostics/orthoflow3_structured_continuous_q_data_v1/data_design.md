# Structured continuous-Q data design

State panel is a deterministic outcome-blind subset of the prior source-isolated Toy inventory: 48 TRAIN, 12 VAL, 16 TEST. Source-group overlap is zero. TEST states were frozen before eta design.

The fixed eta panel has 40 TRAIN, 12 VAL, and 32 TEST probes. Thirty-two TEST probes are required by the predeclared K=32 finite-candidate evaluation. TRAIN probes comprise 18 global E_bridge Sobol scaffold, 14 TRAIN-only state-dependent transition eta, and 8 fixed local boundary probes. No probe moves per state.

Eta split is spatial: minimum normalized TEST-to-TRAIN distance is 0.084608. Transition candidates use only TRAIN outcomes from the fixed-eta audit; scaffold is outcome-blind; boundary offsets are fixed geometry around TRAIN-evidence transition eta.

Requested matrix partitions: TRAIN×TRAIN, VAL×VAL, unseen-state×seen-eta, seen-state×unseen-eta, and unseen-state×unseen-eta. Requested seed records before cache reuse: 56064. TRAIN global pairs request Q4, TRAIN transition/boundary Q8, VAL Q8, and every TEST-bearing pair Q16.
