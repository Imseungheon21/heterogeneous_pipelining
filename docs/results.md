# Results

## Experimental scope

| Item | Value |
|---|---:|
| Workload | TinyLlama-1.1B |
| Transformer layers | 22 |
| GPUs | 8 |
| Pipeline stages | 2 |
| TP/DP strategy pairs | 9 |
| Layer partitions | 21 |
| Evaluated configurations | 189 |

The retained artifacts identify the GPU count but not the GPU model, so no accelerator model or memory capacity is claimed here.

## Estimator error

Mean absolute prediction error across all 189 recorded configurations was **5.85%**, using measured net time as the denominator.

| Stage 0 | Stage 1 | Mean | Minimum | Maximum |
|---|---|---:|---:|---:|
| TP2_DP2 | TP2_DP2 | 5.13% | 2.71% (17/5) | 8.74% (9/13) |
| TP4_DP1 | TP4_DP1 | 5.37% | 2.63% (19/3) | 9.99% (11/11) |
| TP4_DP1 | TP2_DP2 | 5.42% | 2.62% (21/1) | 9.74% (9/13) |
| TP4_DP1 | TP1_DP4 | 5.52% | 2.65% (20/2) | 10.88% (9/13) |
| TP2_DP2 | TP1_DP4 | 5.75% | 2.77% (20/2) | 9.89% (7/15) |
| TP2_DP2 | TP4_DP1 | 5.78% | 2.84% (21/1) | 9.54% (10/12) |
| TP1_DP4 | TP1_DP4 | 6.42% | 3.92% (19/3) | 9.30% (7/15) |
| TP1_DP4 | TP4_DP1 | 6.48% | 4.50% (1/21) | 9.88% (11/11) |
| TP1_DP4 | TP2_DP2 | 6.75% | 3.59% (19/3) | 10.87% (11/11) |

## Selected plan

The minimum estimated latency was obtained with an even 11/11 layer split and `TP1_DP4` in both stages:

| Stage | Layers | Strategy | Estimated stage cost |
|---|---|---|---:|
| 0 | `[0, 11)` | TP1_DP4 | 593.25 ms |
| 1 | `[11, 22)` | TP1_DP4 | 594.60 ms |

The estimated 1F1B latency was 5,350.04 ms for eight microbatches. This was a **homogeneous** plan. TinyLlama's relatively uniform Transformer blocks did not make a heterogeneous plan optimal in this experiment.

The supported conclusion is therefore not that heterogeneity improved performance. The contribution was expanding the runtime and search space so heterogeneous plans could be executed, modeled, and compared.

## Measurement limitations

- The experiment covered one model and one eight-GPU scale point.
- Accelerator model and interconnect metadata were not retained in a publication-safe artifact.
- The archived prototype used a leader-based activation replication path; the experiment did not establish unique-sample data-parallel scaling.
- Raw rank logs are excluded from the public reconstruction pending ownership and privacy review.
- The estimator assumes repeated blocks have stable marginal cost.
