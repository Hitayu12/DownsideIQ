# DownsideIQ — Modeling

## Council 2 — three models, three questions
1. **XGBoost downside classifier** — P(meaningful downside over the horizon).
   Target: volatility-adjusted `downside_label = 1 if future_return <
   -max(floor, vol_mult·recent_vol)`. NaN-native; class imbalance via
   `scale_pos_weight`.
2. **GARCH/EGARCH volatility** — 1-step forecast vol + parametric VaR &
   expected shortfall (Student-t tails). Fit on-demand per request.
3. **Quantile regression (HistGradientBoosting, pinball loss)** — 5th/10th
   percentile of next-session return → downside-tail severity.

Hyperparameters and versions live in `config/model_config.yaml`.

## Validation
- **Walk-forward only** (`src/backtesting/walk_forward.py`): expanding windows,
  time-ordered, never a random split. Metrics (AUC, Brier, precision/recall) are
  recorded to `model_performance` and the registry.
- Honest results: single-stock daily AUC ≈ 0.54 — modest by design. The edge is
  the risk-controlled decision, not the raw predictor.
- **No model metric ever comes from mock data.** Mocks are used only for
  interface/failure tests.

## Model registry & governance (spec §11)
`ModelRegistry` stores artifacts in `models_store/` and a JSON index; every
prediction carries `model_name, model_version, training_date,
feature_set_version, prediction_timestamp`. Predictions are persisted to
`model_predictions` for full lineage (snapshot → predictions → signal).

## Ensemble + news overlay
```
base_log_odds = logit(p_downside)
news_shift    = α₁·company_news + α₂·macro_news + α₃·net_catalyst + α₄·pv_confirm
                (hard-capped by max_logodds_shift)
adjusted_p    = sigmoid(base_log_odds + news_shift)
base_risk     = 0.45·adjusted_p + 0.30·garch_dvr + 0.25·tail_score
adjusted_risk = base_risk · agreement · data_confidence · regime_adj − uncertainty_penalty
```
News cannot override the model unless credibility, relevance, **and**
price/volume confirmation are all strong. Weights/α/cap are config-driven
(`model_config.yaml`, `settings.yaml`).
