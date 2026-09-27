# Figures

Every graph this project produced. The numbered set is the main sequence, cited by section in `Plan_X/RESULTS.md`; `runs/` holds the incidental ones written by `Tester.py`, `score_all.py` and `analyze_run.py` during individual runs.

Regenerate the collection with `python collect_graphs.py`.

## Main sequence

| figure | shows | RESULTS.md |
|---|---|---|
| [`1_learning_curves.png`](1_learning_curves.png) | Design A and B training curves: reward, soft-min and constraint terms per iteration | 3, 4 |
| [`1b_softmin_comparable.png`](1b_softmin_comparable.png) | Design A vs B on the same soft-min metric, since the two optimised different rewards | 4 |
| [`2_per_detector.png`](2_per_detector.png) | Per-detector movement under Design A -- the aggregator left AASIST undefended and it degraded | 3 |
| [`3_constraints.png`](3_constraints.png) | Similarity and WER against their limits; neither ever bound | 4 |
| [`4_action_scale.png`](4_action_scale.png) | Edit magnitude over training: the max_delta cap was never approached | 4 |
| [`5_diagnostics.png`](5_diagnostics.png) | SAC internals -- the entropy coefficient rising for nine hours, the signature of a critic that cannot rank | 4 |
| [`6_pgd_ceiling.png`](6_pgd_ceiling.png) | What white-box PGD reaches, establishing the intervention point works | 2 |
| [`7_per_voice_detector.png`](7_per_voice_detector.png) | Design A vs B per voice on detector scores | 5 |
| [`8_per_voice_quality.png`](8_per_voice_quality.png) | What A and B cost in similarity and WER | 5 |
| [`9_distill_curves.png`](9_distill_curves.png) | Distillation: regression loss, direction agreement, and the fraction of PGD's gain recovered | 7 |
| [`10_distill_vs_rl.png`](10_distill_vs_rl.png) | Distilled model vs RL on unseen voices and sentences | 7, 7h |
| [`11_distill_cost.png`](11_distill_cost.png) | Similarity, WER, PESQ and STOI for the distilled edit | 7 |
| [`12_distill_sweep.png`](12_distill_sweep.png) | Capacity sweep -- smaller networks won when imitating fixed targets | 7 |
| [`13_amplify.png`](13_amplify.png) | Applying the edit repeatedly compounds; scaling it up does not. The step-operator result | 7b, 7h |
| [`14_per_voice_analysis.png`](14_per_voice_analysis.png) | Which voices the edit helps, against reference acoustics that mostly fail to predict it | 7l |
| [`15_margin_tradeoff.png`](15_margin_tradeoff.png) | Evasion against PESQ cost for the margin objective, with the quality budget marked | 7m, 7n |
| [`16_margin_per_voice.png`](16_margin_per_voice.png) | Per voice at both operating points: 14 of 15 clear soft-min 0.5 when quality is not constrained | 7n |
| [`17_frontier.png`](17_frontier.png) | Every method on one axis: soft-min bought per unit of audio damage | 7q |
| [`18_longform.png`](18_longform.png) | Long audio attacked against the whole-file score: the failure at length was the objective, not the audio | 7r |

## From individual runs

| figure | source |
|---|---|
| [`runs/discriminator_heatmap_riven_20260903_172055.png`](runs/discriminator_heatmap_riven_20260903_172055.png) | `discriminator/heatmap/riven/20260903_172055.png` |
| [`runs/discriminator_human_vs_ai_riven_20260903_172055.png`](runs/discriminator_human_vs_ai_riven_20260903_172055.png) | `discriminator/human/vs/ai/riven/20260903_172055.png` |
| [`runs/generators_metrics_similarity_6voices_20260902_232005.png`](runs/generators_metrics_similarity_6voices_20260902_232005.png) | `generators/metrics/similarity/6voices/20260902_232005.png` |
| [`runs/generators_metrics_similarity_riven_20260903_172113.png`](runs/generators_metrics_similarity_riven_20260903_172113.png) | `generators/metrics/similarity/riven/20260903_172113.png` |
| [`runs/generators_metrics_wer_6voices_20260902_232051.png`](runs/generators_metrics_wer_6voices_20260902_232051.png) | `generators/metrics/wer/6voices/20260902_232051.png` |
| [`runs/generators_metrics_wer_riven_20260903_172113.png`](runs/generators_metrics_wer_riven_20260903_172113.png) | `generators/metrics/wer/riven/20260903_172113.png` |
| [`runs/rl_runs_designA_budget4h_trend.png`](runs/rl_runs_designA_budget4h_trend.png) | `rl/runs/designA/budget4h_trend.png` |
