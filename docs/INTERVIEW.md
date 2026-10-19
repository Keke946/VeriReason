# Interview and Learning Log

This file records explanations the user has personally reviewed and can defend. Do not treat generated answers as mastered until a teach-back has occurred.

Interview readiness is intentionally independent from the completed M0–M11 technical release.
The unchecked items below remain honest learning debt and will be handled in a dedicated teaching
task; technical closure does not imply that the user has mastered them.

## Status rules

- `not_started`: neither explained nor reviewed together.
- `explained`: Codex has explained it or the user has asked about it; this is not mastery.
- `teach_back_due`: enough material exists for review, but the user has not yet explained it back.
- `mastered`: the user has independently explained the concept, design choice, evidence, and one
  likely failure mode. Only the user/control task may mark this status.

## Learning receipt

| Area | What has already been discussed | Current status | Required evidence to mark mastered |
|---|---|---|---|
| Project thesis | Why Text-to-SQL, whether it is “water”, use of upstream code, and why SFT plus verifiable post-training is meaningful | teach_back_due | Explain the research question, E0–E4 ladder, and what is original versus reused |
| M1 environment | Local versus cloud roles, why scripts were written before SSH, 4B/1.7B smoke, and why 1×48 GB was selected | teach_back_due | Explain environment pinning, model/topology gate, and what the four-case smoke does not prove |
| M2 data/licensing | BIRD sources, meaning of a missing repository license, cloud materialization, database-level split, evaluator safety, leakage, manifests, and operational eligibility | teach_back_due | Explain the data path from source receipt to frozen manifests and why heldout stays private |
| M3 vertical slice | Tiny end-to-end QLoRA proof and why it precedes formal SFT | teach_back_due | Walk through data → tokenization → masking → LoRA update → reload → generation → execution evaluation |
| M4 formal SFT | Relation to the earlier LLMRec LoRA task, one epoch versus two, full5364 training, full1209 same-exam evaluation, and interpretation of 40.69% | teach_back_due | Explain QLoRA, loss masking/GAS, fairness of E0/E1, result meaning, and why epoch 2 is deferred |
| M5 error analysis | First explanation of correct/wrong-result/execution-error buckets and why error analysis precedes more training | explained | Explain the dominant failure, limitations of the structural proxy, and how taxonomy changes data-v2/RL |
| M6 reward design | Terminal/dense rewards, AST features, F1, reliability gating, utility, normalization, and reward-hacking concerns have been explained; teach-back is still due | teach_back_due | Explain terminal reward, dense components, gating, normalization, and reward-hacking tests |
| M7+ RL/serving/release | Formal E2/E3/E4 training and common validation completed; E2 is best, M8 is a controlled negative result, and M9 serving plus M10 clean-room reproduction are complete | teach_back_due | Explain the E2–E4 conclusion, serving latency/throughput metrics, and what M10 reproduces without rerunning GPU experiments |

The recommended consolidation point is after the M5 private taxonomy and before any new paid
training. At that point M1–M5 form one coherent story, and the user should complete a structured
teach-back before M6/M7 execution begins. Short questions asked during development count as
discussion, not as a passed teach-back.

## M1/M2 — Data and evaluation

- [ ] Why must the test split be frozen before training?
- [ ] What forms can LLM benchmark contamination take?
- [ ] Why is executable correctness stronger than string matching for the selected task?

## M4 — Supervised fine-tuning

- [ ] What changes during LoRA/QLoRA training?
- [ ] How are prompt and assistant tokens masked in the loss?
- [ ] Why can an incorrect chat template or EOS token damage training?
- [ ] Which SFT failures are data problems versus optimization problems?

Day 4A implementation and the real M3 run were delegated while the user was unavailable.
`docs/M3_LEARNING.md` now preserves both the concepts and measured vertical-slice evidence, but none
are marked mastered. M3 is technically complete; complete this teach-back before calling the SFT
work interview-ready.

## M6/M7 — Verifiable-reward training

- [ ] What is GRPO optimizing?
- [ ] How are group-relative advantages formed?
- [ ] What is the role of KL regularization or reference behavior?
- [ ] How can the model exploit the verifier or reward?
- [ ] Why can reward increase while real task quality falls?

## M8 — Experimental evidence

- [ ] What are the main baselines and why are they fair?
- [ ] What does each ablation isolate?
- [ ] How are uncertainty and stochastic decoding handled?
- [ ] What evidence rules out leakage or formatting tricks?

## M9 — Serving

- [ ] What are prefill and decode?
- [ ] What is stored in the KV cache?
- [ ] How do quantization, batch size, and context length affect memory and latency?
- [ ] What is the difference between TTFT, inter-token latency, and throughput?

## Failure stories

Add one concise entry per material failure: symptom, hypothesis, evidence, fix, and lesson.

- E3 smoke import failure: TRL failed before model loading because the installed `xgrammar 0.2.3`
  metadata listed `libxgrammar_bindings.so` but the file was absent. Reinstalling the exact package
  without dependency changes restored the library; direct `GRPOTrainer` import and `pip check`
  passed before rerun. Lesson: package-version checks alone cannot detect missing binary payloads;
  the smallest relevant import is the useful preflight for native extensions.
- E4 formal pre-model failures: attempt 001 supplied the wrong database-root level and attempt 002
  found a dangling provider-managed `xgrammar` library symlink. Correcting the root and loading the
  native library extracted from the exact locked wheel allowed attempt 003 to complete without any
  scientific-contract change. Lesson: distinguish invocation/environment failures before model
  load from failed learning experiments, and test the exact native import before paid training.
- M9 serving startup failures: attempt 001 exposed vLLM 0.25.1's default prefix cache, contradicting
  the benchmark contract, and attempt 002 exposed a provider-restored non-executable CUDA `cicc`
  when FlashInfer attempted first-use JIT. Explicitly disabling prefix caching and selecting vLLM's
  supported Torch sampler preserved greedy decoding and allowed attempt 003 to complete. Lesson:
  validate effective engine configuration from startup logs, and avoid optional JIT paths that add
  no value to the measured workload.
