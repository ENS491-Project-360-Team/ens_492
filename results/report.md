# Pancreatic MatchMaker report

## Inputs

- Summary metrics: `results/summary_metrics.csv`
- Per-seed metrics: `results/per_split_seed_metrics.csv`

## Best stable scenario

- Regression winner: dataset=`None`, split=`None`, MSE=`1.2420`, Pearson=`0.5257`
- Classification winner: dataset=`None`, split=`None`, F1=`0.6990`
- Ranking winner (AUC/AUPRC): dataset=`None`, split=`None`, AUC=`0.7270`, AUPRC=`0.7985`

## Split difficulty trend

- Mean MSE by split:
  - lto: 1.4319
  - lpo: 2.0940
  - lco: 1.3591
  - lodo: 1.9797
  - ldo: 2.2823

## Disagreement filter impact

- `lco`: MSE delta(filtered-unfiltered)=-0.2342; F1 delta(filtered-unfiltered)=-0.0158
- `ldo`: MSE delta(filtered-unfiltered)=0.4276; F1 delta(filtered-unfiltered)=-0.2265
- `lodo`: MSE delta(filtered-unfiltered)=-0.0645; F1 delta(filtered-unfiltered)=-0.0076
- `lpo`: MSE delta(filtered-unfiltered)=-0.3704; F1 delta(filtered-unfiltered)=-0.0325
- `lto`: MSE delta(filtered-unfiltered)=-0.1477; F1 delta(filtered-unfiltered)=-0.0354
