# Plotting and Statistical Claim Guide

This project uses figures to support paper claims, not to duplicate numeric tables.
Exact benchmark matrices belong in generated tables; plots should make one claim
legible in a first pass through the paper or README.

## Claim-first figure set

1. **Protocol changes recommendations**
   - Preferred plot: diverging rank-change barplot, not a dense slopegraph.
   - Metric: `single_thread_rank - peak_dataloader_rank`.
   - Positive values mean the decoder moves up under `DataLoader`; negative values mean it moves down.
   - Use representative Intel, AMD, and ARM panels unless a claim explicitly needs two generations from one vendor.
   - Caption must say that full throughput matrices are in the tables.

2. **AMD generations change worker-count effectiveness**
   - Preferred plot: paired AMD panels showing `% change from w=4 to w=8`.
   - Use a zero line; encode positive and negative directions with distinct colors.
   - Do not write "AMD scales poorly"; write that Zen 4 and Zen 5 differ under this worker policy.

3. **TensorFlow ARM penalty**
   - Preferred plot: normalized bar/dot chart.
   - Metric: TensorFlow single-thread throughput as `% of the platform single-thread winner`.
   - This avoids mixing CPU speed with decoder behavior and makes the ARM-specific drop visible.

4. **Overall performance and recommendations**
   - Preferred plot: range/lollipop plot of each decoder's peak `DataLoader` throughput as `% of platform winner`.
   - Show min/mean/max across platforms and individual platform points.
   - Use tables for exact values and robustness; the recommendation figure should emphasize stable near-optimality.

## General claim-to-plot mapping

Use this section when the paper claims change. Pick the plot from the evidence
shape, not from habit.

| Claim shape | Preferred plot | Use when | Avoid |
| --- | --- | --- | --- |
| A vs B changes the conclusion | Difference barplot, diverging barplot, paired dot plot | Comparing protocols, workers, settings, before/after, or ablations | Side-by-side grouped bars when readers must mentally subtract |
| Ranking changes | Rank-change barplot or small bump chart with few highlighted items | Showing that evaluation protocol changes rank/tier | Dense slopegraphs with many decoders |
| One item moves up/down relative to a baseline | Delta barplot or waterfall-style difference chart | Showing improvement/regression by decoder, worker count, platform, or version | Raw absolute bars unless absolute scale is the point |
| Time/worker scaling curve | Small-multiple line chart with shared axes | Showing monotonicity, saturation, collapse, or nonlinearity over workers/threads | One crowded line chart with all platforms and all decoders |
| Hardware/platform interaction | Small multiples or interaction plot | Showing behavior changes by CPU generation, vendor, ISA, SMT, or architecture | Collapsing hardware into one average |
| Cross-platform stability | Range/lollipop, box/strip plot, min/mean/max dot range | Showing robust defaults or platform sensitivity | Ranking by mean alone |
| Near-optimal recommendation | Normalized range plot: `% of local winner` | Choosing a practical default across heterogeneous platforms | Absolute throughput across CPUs without normalization |
| Single-platform leaderboard | Horizontal bar chart with error bars | Showing one platform's descriptive ordering | Claiming deployment recommendations from it |
| Full benchmark matrix | Table; optional supplementary heatmap | Preserving exact values across decoders × platforms | Main-text heatmap for directional claims |
| Distribution/noise | Dot/strip plot of raw runs, CI interval plot, violin only if enough samples | Showing run variance, outliers, or uncertainty | Hiding raw samples behind only means |
| Robustness/failures | Table with counts and example errors | Skips, unsupported modes, deadlocks, eligibility | Treating rare failures as a throughput scatter unless failure rate is the claim |
| Cost/runtime tradeoff | Pareto scatter or frontier plot | Throughput vs setup cost, dependency cost, memory, failure rate | Single scalar score unless weighting is justified |
| Composition/share | Stacked bar only when parts sum to a meaningful whole | Breaking wall time into decode/I/O/overhead components | Stacked bars for unrelated metrics |
| Correlation | Scatter with regression only if causal story is not implied | Exploring relation between image size, throughput, or variance | Causal wording from correlation |
| Categorical recommendation | Decision table or rubric | Translating benchmark evidence into user choices | Overloaded figure with too many dimensions |

Default hierarchy:

1. If the claim is comparative, plot the **difference**.
2. If the claim is about recommendation, plot **normalized distance from the local best/top tier**.
3. If the claim is about exact values, use a **table**.
4. If the claim is about uncertainty, show **raw samples or intervals**.
5. If the claim is about many platforms or decoders, use **small multiples** rather than one crowded panel.

## Visual rules

- Use **tables** for exact values and **figures** for claim support.
- Use aligned bars or dot/range plots for comparisons. Avoid grouped bars when the target task is a difference.
- Avoid slopegraphs/bump charts with many decoders unless only one or two highlighted lines matter.
- Avoid heatmaps as main evidence for directional claims; heatmaps are acceptable as supplementary overviews.
- Sort bars by the quantity that supports the claim, usually delta or normalized performance, not alphabetically.
- Use shared axes across small multiples when the viewer is expected to compare panels.
- Normalize within platform when comparing decoders across different CPUs.
- Keep absolute throughput when the claim is capacity planning for one known machine.
- Put units in axis labels and captions; avoid encoding units only in title text.
- Use log scale only when multiplicative differences are the claim and explain it in the caption.
- Include the denominator in normalized plots, e.g. `% of platform winner`.
- Color should encode meaning:
  - improvement or moving up: green/blue
  - regression or moving down: red/orange
  - background or non-highlighted items: gray
- Label only the largest movers or the claim-critical decoders. Too many labels turn plots into tables.
- Use neutral titles that state the measured effect, not a causal story the experiment cannot isolate.
- Captions must state the protocol, platform subset, normalization, and where the full table lives.

These rules follow common visualization evidence from comparison-chart research:
difference charts make A-vs-B changes explicit, crowded rank charts have high
cognitive load, and benchmark papers typically separate complete numeric tables
from compact claim figures.

## Statistical methodology

The raw JSON contains the samples needed for uncertainty-aware comparisons:

- single-thread rows: `benchmark_results.raw_throughput_ips`, usually 5 runs
- `DataLoader` rows: `worker_results[*].raw_throughput_ips`, usually 3 runs per worker count

Use this policy for paper claims:

1. Report descriptive tables as `mean ± sample std`.
2. For "faster/slower" claims, compare raw run samples, not only means.
3. Apply practical-significance thresholds before using strict faster/slower language:
   - single-thread practical threshold: 1% relative throughput
   - `DataLoader` practical threshold: 5% relative throughput, because worker-process variance is larger
4. Treat decoders as the same top tier when the measured gap is below the practical threshold or the raw-run samples make the ordering uncertain.
5. Prefer "top tier", "near-optimal", "not clearly separated", and "rank order changes" over strict winner language.
6. For peak `DataLoader` worker count, choose the highest mean for generated tables, but text should mention ties/top tiers when adjacent worker counts are uncertain or practically indistinguishable.
7. Do not mention confidence intervals, bootstrap intervals, or error bars in paper claims unless the implementation actually computes those intervals for the reported comparison.

The current checked-in artifact has 5 single-thread runs and 3 `DataLoader`
runs per worker count. That is enough to prevent obvious overclaiming, but it is
not enough to justify fine-grained claims about 1-2% differences in `DataLoader`
throughput.

## Bad claims to avoid

- "Decoder X is universally fastest."
- "Single-thread winner is the best training-loader choice."
- "AMD scales poorly."
- "ARM is slow for JPEG decode."
- "TensorFlow is always slow."
- "OpenCV is fastest."
- "PyVips is broken." Use "unsupported in this fork-based PyTorch DataLoader harness."
- "Strict native JPEG decoders are wrong." Use "they require an explicit fallback policy for uncommon JPEG modes."
- "Rank 1 beats rank 2" when their uncertainty or practical gap makes them a top tier.
- Cross-platform absolute throughput comparisons without explaining CPU speed and microarchitecture.
