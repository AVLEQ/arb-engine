"""
backtester/ml_filter.py — ML Confidence Scoring
XGBoost classifier predicts P(spread reverts within N days).
Used as a filter on top of z-score signals.
"""

import sys
import logging
import warnings
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import CFG

log = logging.getLogger("ml_filter")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ML feature matrix from the signals DataFrame.
    Only uses features available AT signal time (no lookahead).
    """
    feats = pd.DataFrame(index=df.index)

    feats["zscore"]            = df["zscore"].abs()
    feats["zscore_signed"]     = df["zscore"]
    feats["spread_mom_5d"]     = df.get("spread_mom_5d", pd.Series(np.nan, index=df.index))
    feats["spread_mom_10d"]    = df.get("spread_mom_10d", pd.Series(np.nan, index=df.index))
    feats["vol_ratio"]         = df.get("vol_ratio", pd.Series(1.0, index=df.index)).clip(0.1, 10)
    feats["vol_rank_min"]      = df.get("vol_rank_min", pd.Series(0.5, index=df.index))
    feats["vix"]               = df.get("vix", pd.Series(18.0, index=df.index))
    feats["days_since_signal"] = df.get("days_since_signal", pd.Series(10, index=df.index))
    feats["fx_vol_20d"]        = df.get("fx_vol_20d", pd.Series(0.05, index=df.index))
    feats["leg1_mom_10d"]      = df.get("leg1_mom_10d", pd.Series(0.0, index=df.index))
    feats["leg2_mom_10d"]      = df.get("leg2_mom_10d", pd.Series(0.0, index=df.index))
    feats["coint_pval"]        = df.get("coint_pval", pd.Series(0.05, index=df.index))
    feats["spread_std"]        = df.get("spread_std", pd.Series(0.01, index=df.index))

    # Rolling mean reversion speed (half-life approximation)
    # Half-life = -log(2) / log(1 + beta) where beta is AR(1) coefficient
    ar_betas = []
    for i in range(30, len(df)):
        window = df["spread_ols"].iloc[i-30:i].dropna()
        if len(window) < 10:
            ar_betas.append(np.nan)
            continue
        try:
            y = window.diff().dropna()
            x = window.shift(1).dropna()
            min_len = min(len(y), len(x))
            beta = np.cov(y[-min_len:], x[-min_len:])[0, 1] / max(np.var(x[-min_len:]), 1e-10)
            ar_betas.append(beta)
        except Exception:
            ar_betas.append(np.nan)
    ar_betas = [np.nan] * 30 + ar_betas
    feats["ar_beta"] = ar_betas[:len(df)]

    return feats.ffill().fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_labels(df: pd.DataFrame, horizon: int = None) -> pd.Series:
    """
    Label = 1 if spread reverts to within exit threshold within `horizon` days.
    This is our prediction target: did the signal work?
    """
    horizon = horizon or CFG.ml.reversion_horizon
    exit_thresh = CFG.signal.zscore_exit
    labels = pd.Series(0, index=df.index)

    for i in range(len(df) - horizon):
        current_z = df["zscore"].iloc[i]
        if pd.isna(current_z) or abs(current_z) < CFG.signal.zscore_entry:
            continue
        # Look ahead: did z-score cross zero / exit threshold?
        future_z = df["zscore"].iloc[i + 1 : i + horizon + 1]
        if (future_z.abs() < exit_thresh).any():
            labels.iloc[i] = 1

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class MLConfidenceModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.train_auc = None
        self.test_auc  = None

    def train(self, df: pd.DataFrame) -> "MLConfidenceModel":
        """
        Train on data up to CFG.ml.train_end.
        Evaluate on CFG.ml.test_start onwards.
        """
        log.info("Building ML features and labels...")
        feats = build_features(df)
        labels = build_labels(df)

        self.feature_names = feats.columns.tolist()

        # Only train on signal days (avoid class imbalance from no-signal days)
        signal_mask = df["raw_signal"].abs() > 0
        feats_sig  = feats[signal_mask]
        labels_sig = labels[signal_mask]

        if len(feats_sig) < 50:
            log.warning("Fewer than 50 signal observations — skipping ML training")
            self.is_trained = False
            return self

        # Temporal train/test split (no shuffling — time series!)
        train_mask = feats_sig.index <= CFG.ml.train_end
        test_mask  = feats_sig.index >= CFG.ml.test_start

        X_train = feats_sig[train_mask].values
        y_train = labels_sig[train_mask].values
        X_test  = feats_sig[test_mask].values
        y_test  = labels_sig[test_mask].values

        if len(X_train) < 30 or len(X_test) < 10:
            log.warning("Not enough data for ML split — skipping")
            self.is_trained = False
            return self

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        log.info(f"Training XGBoost: {len(X_train)} train, {len(X_test)} test observations")
        self.model.fit(X_train_s, y_train)

        train_probs = self.model.predict_proba(X_train_s)[:, 1]
        test_probs  = self.model.predict_proba(X_test_s)[:, 1]

        try:
            self.train_auc = roc_auc_score(y_train, train_probs)
            self.test_auc  = roc_auc_score(y_test, test_probs)
            log.info(f"  Train AUC: {self.train_auc:.3f} | Test AUC: {self.test_auc:.3f}")
        except Exception:
            pass

        self.is_trained = True
        return self

    def predict_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Return P(reversion within horizon) for every row.
        Returns 0.5 (neutral) on non-signal days or if not trained.
        """
        if not self.is_trained:
            return pd.Series(0.5, index=df.index)

        feats = build_features(df)
        X = self.scaler.transform(feats.values)
        probs = self.model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=df.index)

    def feature_importance(self) -> pd.Series:
        """Return feature importances as a sorted Series."""
        if not self.is_trained:
            return pd.Series()
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def ablation_study(self, results_with_ml: pd.DataFrame, results_no_ml: pd.DataFrame) -> Dict:
        """Compare metrics: ML-filtered vs pure z-score."""
        from backtester.metrics import compute_metrics
        m_ml = compute_metrics(results_with_ml)
        m_base = compute_metrics(results_no_ml)
        return {
            "sharpe_with_ml":    m_ml.get("sharpe_ratio"),
            "sharpe_without_ml": m_base.get("sharpe_ratio"),
            "sharpe_improvement": round(
                (m_ml.get("sharpe_ratio", 0) - m_base.get("sharpe_ratio", 0)) /
                max(abs(m_base.get("sharpe_ratio", 1)), 0.01) * 100, 1
            ),
            "win_rate_with_ml":    m_ml.get("win_rate_pct"),
            "win_rate_without_ml": m_base.get("win_rate_pct"),
        }

