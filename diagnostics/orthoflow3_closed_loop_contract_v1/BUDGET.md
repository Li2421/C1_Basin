# Pre-registered continuation budget

Initial targeted audit: 256 unique Q16 continuations, plus 15 identical
numerical retry attempts. All were database-persisted. Five canonical seed
positions remained numerical-uncertified after the allowed retries.

The exact remaining-time alias is now confirmed (WITNESS.md). Authorized
extension, recorded before repair validation: total cap **6,000 new
continuations**, within the allowed 12,000 ceiling. Purpose is solely causal
repair validation, not an eta sweep or further model optimization.

Plan: 12 matched validation states × 17 new frozen proposals × 16 seeds =
3,264 maximum new development slots; eta-zero and pre-repair outcomes reuse
compatible database evidence. Then one fresh cohort of 2 states per primary
scenario, at most 19 tuples × 16 seeds per state = 1,216 slots. Total maximum
4,736 slots before reuse. Retries count separately and remain below cap.
No adaptation follows fresh confirmation. Only one repair iteration is used.
