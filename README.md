# FedC2

This repository contains a minimal reference implementation of **FedC2**, a
hierarchical cross-client collaboration method for federated learning from
label proportions.

## Scope

The release implements the method-specific path described in the paper:

1. compatibility from bag-proportion geometry;
2. top-`H` collaborator selection and sample-count-aware base aggregation;
3. target-relative class coverage and collaborator-local reliability;
4. conservative class-wise classifier-head refinement; and
5. sample-count-weighted consolidation of the final client models.

It also includes the weighted DLLP local objective and the ResNet-18 model used
in the experiments. Baseline implementations, ablations, private experiment
orchestration, checkpoints, datasets, and internal result files are outside the
scope of this minimal release.

## Paper configuration

The main configuration is stored in `configs/paper_default.json`. The central
settings are 50 clients, 50 communication rounds, 5 local epochs, 8 bags per
optimization step, target bag size 32, `H=25`, complementarity strength 0.05,
and complementarity cap 0.25.

Following the paper, each client estimates its class-wise reliability after
local training in every communication round. This implementation uses at most
16 source-owned training bags per client as the local bag subset permitted by
Eqs. (8)-(9). Constructed bag proportions and bag weights are stored at six
decimal places.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Minimal example

```bash
PYTHONPATH=. python examples/minimal_fedc2.py
```

The example uses synthetic client models and proportions to demonstrate one
server aggregation step. It does not reproduce a paper result.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

The tests check the equations and invariants of the released core, including
the zero-evidence fallback and classifier-only refinement.

## Integrating the core into a training loop

At each communication round:

1. optimize every client model with the DLLP objective;
2. compute source-local reliability after local training in every round;
3. combine coverage and reliability into directional class scores;
4. call `aggregate_fedc2_models` to produce the next client-specific states;
5. after the final round, call `consolidate_global_model` for evaluation.

The caller must use the same client partition, LLP bags, preprocessing, model
selection rule, and random seed when reproducing reported results.

## Data and privacy

No datasets, model checkpoints, personal information, machine-specific paths,
server addresses, credentials, or experiment logs are included.

## License

No reuse license is included in this review-stage release.
