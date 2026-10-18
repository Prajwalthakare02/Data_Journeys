**How to read this output:**

- `groups seen as DEAD / ALIVE` — train tells us the fate of ~230 travelling groups (~130 alive, ~100 dead). That's the knowledge we transfer onto test passengers **who are in the same group as a known passenger** — e.g., the wife who is in test but whose husband died in train.
- **Rule accuracy on train = 0.8934** — circular (train scored with rules derived from train); the real test is the leaderboard. **Measured public LB of the v1 rule: 0.80382** (reference points: all-women baseline = 0.76589, our ML ensemble = 0.76794 — so the group evidence added ~+0.04 over both).
- **Test women predicted DEAD:** a small list of surnames — artifact of the sinking: these are 3rd-class mothers in family groups which train shows were wiped out. The naive "all women survive" submission gets these wrong. The rule catches them.
- **Test boys upgraded to SURVIVED:** the mirror case — boys from families whose women/children demonstrably survived in train. The naive "all men die" gets these wrong, and there aren't many, so each one is ~0.24% of public-LB accuracy.
- **Survival total** — expect **~148 predicted survivors (~35%)** for test (152 women − ~12 flagged-dead + ~8 upgraded boys); historically ~159 test passengers actually survived, so this is in the right zone.

**Why doesn't this hit the mythical ~0.97 public scores?** Those submissions add ~66 hand-curated male-survivor overrides that don't reproduce from any measurable train pattern — measured here: train men with surviving-women/children family evidence survive at 0.198 vs 0.160 baseline (+4pp — no decision boundary). Our rule adds nothing we can't measure, so expect **public LB ≈ 0.81 ± a couple of points** for the 14 + 8 evidence-backed flips over the 0.76794 female-default baseline. Group-structure is where point-by-point gains live because those rows' fate is *determined* by train evidence, not modeled.

---
