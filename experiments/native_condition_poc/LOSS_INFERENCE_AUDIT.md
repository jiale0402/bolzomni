# Loss and final capability: static audit, 2026-09-28

Scope: existing source and saved results only. No training, inference, evaluation,
checkpoint execution, or cluster job was launched for this audit.

## What the measurements establish

The fixed validation probe compares the same 16 held-out chains, fixed seeds,
coordinate sigma 8, and chemical masking level 0.5 at updates 0 and 400.
Evidence: `train.py:121-125`; `runs/native-1729/metrics.jsonl`, lines 2 and 406.

| Quantity (lower is better) | Update 0 | Update 400 |
|---|---:|---:|
| Design identity cross-entropy | 3.1687347442 | 2.8125740886 |
| Design coordinate loss | 0.0272013559 | 0.0272448243 |
| Design smooth-lDDT loss | 0.3684404679 | 0.3682031110 |
| Folding coordinate loss | 0.0140601371 | 0.0138529596 |
| Folding smooth-lDDT loss | 0.2929151729 | 0.2922533080 |

The conspicuous reduction is in the newly initialized identity head's conditional
prediction objective. It is not evidence that native BoltzGen was surpassed in
folding or scaffold design. No sequence top-1 accuracy or recovery measurement is
available from these probe records. Design coordinate loss did not improve.
Random-batch training-window means are not paired evaluations of the same samples.

## Source-confirmed coverage differences

1. In `native_flow.py:114-118`, unmasked, unobserved training identities come from
   ground-truth labels. In `native_flow.py:70-80`, revealed inference identities
   come from this model's sampled predictions and remain fixed once revealed.
   Training does not explicitly unroll those sampled-feedback states. This is a
   factual train/inference difference, not proof of a bug: clean unmasked tokens
   are also used in ordinary masked generative training. Its causal contribution
   to the measured result is unverified.
2. The teacher MSE in `native_flow.py:126-133` uses `obs.identity` with all
   unobserved identities masked and chemical time 1. It is not a preservation
   constraint evaluated on intermediate, model-generated identity feedback.
   Consequently, it does not establish preservation along the full joint sampler.
3. `cache_training.py` fixes one 16-residue contiguous motif per training protein;
   the design panel contains 1-4 motif segments. Data origin alone does not make
   these the same task distribution.
4. `refold.py` uses ProteinMPNN to redesign sequence and ESMFold to measure
   scaffold self-consistency. It measures the generated scaffold's designability,
   not the added head's final sequence quality. The head can still affect this
   metric indirectly through feedback into coordinate generation.

## Unresolved attribution

Final native/Omni differences compare different model architectures and weights,
and reused A100 generation against H100 generation. There is no full initial-Omni
benchmark in this run to isolate training's effect from all integration effects.
Exact initial reproduction of native outputs is not a user requirement and must
not be reinstated as a gate. Do not claim the comparison isolates training alone.

No task has demonstrated reliable overall improvement. The design success point
estimate rises while its mean best-of-8 scRMSD worsens; all reported paired
intervals include zero. The cause of adverse metric changes is not established.
Neither loss reduction nor this static audit proves the implementation is fully
correct, a specific cause is confirmed, or a proposed repair would improve results.
