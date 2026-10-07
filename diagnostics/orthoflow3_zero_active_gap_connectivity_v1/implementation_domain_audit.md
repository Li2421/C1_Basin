# Intermediate eta implementation-domain audit

Authoritative basis source: `/home/zhihan/research/Basin_C1/diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py` (SHA256 `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`).

`BasisFields.correction` in `shared_control/basis_families.py` converts eta to float64 and requires only a finite length-three vector. It applies no lower/upper eta bound, clipping, rejection, or reinterpretation. The authoritative fixed-eta runner passes this correction directly to `u_safe + g`, then to the unchanged second `project_velocity_with_retry` projection. Neither projection receives eta or consults eta bounds.

The values eta1=[0.5,1.25], eta2=[-0.5,0.5], eta3=[0,0.75] occur in migration `orthoflow3_search_config.json`, `prepare_candidate_screen.py`, and related oracle-plan generation as `LOW/HIGH`; they are not controller constraints. No safety theorem or projection assumption in the inspected implementation depends on eta1>=0.5. Intermediate finite bridge eta are implementation-valid under the unchanged control stack.
