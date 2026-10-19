# Follow-up Research: What Is Worth Running After the Core Release?

## Evidence from this project

Outcome-only E2 had only 102/256 mixed-reward groups, while E3/E4 dense rewards produced 239/256 and
235/256. Nevertheless, E2 reached 42.35% strict validation accuracy, ahead of E3 at 41.69% and E4 at
41.44%. The most important lesson is not that partial feedback is useless; it is that assigning one
handcrafted structural scalar to an entire completion can create more gradients without assigning
credit to the tokens that caused the SQL error.

## Relevant recent work

### 1. EXPO-SQL: clause-level execution credit

[`EXPO-SQL`](https://aclanthology.org/2026.findings-acl.1107/) identifies erroneous SQL clauses using engine errors and incremental clause execution, then
assigns advantages at clause/token level instead of broadcasting a query-level reward uniformly. It
reports 71.3% BIRD-Dev execution accuracy for Qwen2.5-Coder-7B, 2.4 points above the paper's
Arctic-SQL-R1 7B baseline, with a particularly large gain on challenging queries. This is the closest
match to VeriReason's observed failure mode.

### 2. DAPO: informative-group dynamic sampling

[`DAPO`](https://arxiv.org/abs/2503.14476) over-samples prompts and filters all-zero/all-one groups so each optimizer batch contains a
stable number of nonzero-advantage groups. This directly targets E2's 154/256 zero-variance groups.
It is cheap and likely to improve rollout efficiency, but it changes which prompts contribute and
does not itself provide better credit within a mixed group. It should be a secondary ablation, not
the headline intervention.

### 3. Multi-turn execution correction

[`Progress-SQL`](https://arxiv.org/abs/2606.06825),
[`SQL-ASTRA`](https://aclanthology.org/2026.findings-acl.920/), and
[`ReEx-SQL`](https://aclanthology.org/2026.acl-long.35/) make execution feedback part of a multi-turn correction
trajectory. Their rewards measure progress between SQL states, recovery from invalid execution, or
partial column/result agreement. This is a stronger agentic story but requires a new trajectory
format, environment loop, and evaluation surface. It is a good v2 milestone, not a quick extra cell.

### 4. Data verification and inference-time selection

[`ReViSQL`](https://arxiv.org/abs/2603.20004) reports that human verification/correction of BIRD training data materially improves RLVR,
while [`R3-SQL`](https://aclanthology.org/2026.findings-acl.2146/) improves generate-then-rank systems by grouping candidates by execution result and
selectively resampling weak pools. These may improve end metrics, but the former is annotation-heavy
and the latter is primarily inference/ranking rather than a better policy-training reward.

## Recommended E5 experiment

**E5: terminal-anchored clause-localized execution credit.**

Keep E2's immutable E1 parent, 256 prompts, four rollouts, optimizer, seed, and 1,209-example exam.
Change only credit assignment:

1. Terminal strict correctness remains the anchor and final objective.
2. Parse each completion into SQL clause spans.
3. For invalid SQL, combine SQLite error localization with safe prefix/clause tracing.
4. For executable-but-wrong SQL, incrementally compare clause-induced results and identify the first
   divergent clause conservatively.
5. Apply negative/zero advantage to implicated clause tokens and preserve positive terminal credit
   for fully correct completions.
6. Fail closed to terminal-only reward whenever localization is ambiguous.

This is materially different from E3/E4: the intervention is **where the policy gradient is
applied**, not another scalar weight. It is also resume-relevant because it demonstrates token-level
RL credit assignment, execution-engine integration, and a clean causal ablation.

## Proposed gates and cost

- **Local feasibility first:** positive, negative, equivalent, ambiguous, nested-query, alias,
  unsafe, and adversarial tests; then an offline audit on training-only examples.
- **8-prompt GPU smoke:** accept only if localized masks align to tokens, adapter changes, reload
  succeeds, and no unsafe execution occurs.
- **Formal run:** exactly E2's 256-prompt/1,024-rollout budget.
- **Decision:** one unchanged full1209 evaluation; no reward-weight search and no Mini-Dev access.

On the previously measured 48-GB topology, training should be in the same rough one-hour class as
E2/E3/E4, plus a roughly two-hour validation pass. Engineering—not GPU memory—is the main cost.

## Decision

Preserve the cloud disk until E1/E3/E4 adapters and private manifests are copied and E5's local
feasibility gate is decided. Do not keep it merely to tune E3/E4 weights. If E5 localization cannot
pass conservative local tests, publish v1 and delete the retained disk; the existing project is
already a complete and defensible result.
