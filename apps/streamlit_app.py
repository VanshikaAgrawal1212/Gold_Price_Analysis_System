"""Streamlit dashboard for gold price analysis (step 7, option A).

Run from the project root:

    py -m streamlit run apps/streamlit_app.py

Layout follows how a reviewer actually interrogates a forecasting model:
forecast first, then the evidence (back-test), then the diagnostics (EDA).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import settings as S  # noqa: E402
from src import service  # noqa: E402
from src.service import ModelNotTrained  # noqa: E402

st.set_page_config(
    page_title="Gold Price Analysis & Prediction",
    page_icon="🪙",
    layout="wide",
)

FIG = S.FIGURES_DIR


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Pulling market data...")
def load_history(refresh: bool) -> pd.DataFrame:
    from src.data_loader import build_dataset
    from src.eda import clean

    return clean(build_dataset(use_cache=not refresh))


@st.cache_data(show_spinner=False)
def load_corr() -> pd.DataFrame:
    from src.eda import correlation_matrix

    df = load_history(refresh=False)
    return correlation_matrix(df, method="spearman")


@st.cache_data(show_spinner=False)
def run_retrain(horizon: int):
    from src.pipeline import run

    return run(synthetic=False, with_lstm=False, horizon=horizon, do_cv=False)


def main() -> None:
    st.title("Gold Price Analysis & Prediction")
    st.caption(
        "Daily gold futures modelled against USD/INR, crude oil, S&P 500, "
        "US 10Y yield, the dollar index, silver and CPI."
    )

    with st.sidebar:
        st.header("Controls")
        model_name = None
        try:
            names = service.available_models()
            default = names.index(service.load_bundle().get("best_model", ""))
        except ModelNotTrained:
            st.error(
                "No trained model found. Run `py -m src.pipeline` in a terminal "
                "first, then refresh this page."
            )
            st.stop()
        except ValueError:
            default = 0

        model_name = st.selectbox("Model", names, index=max(default, 0))
        refresh = st.toggle("Fetch latest data", value=False)
        horizon = st.number_input(
            "Trained horizon (business days)",
            min_value=1,
            max_value=63,
            value=int(service.load_bundle().get("horizon", S.HORIZON_DAYS)),
        )
        st.divider()
        if st.button("Retrain models", use_container_width=True):
            st.info("Retraining on the full history - a few seconds for XGBoost.")
            with st.spinner("Training..."):
                run_retrain(int(horizon))
            service.clear_caches()
            st.cache_data.clear()
            st.success("Done.")
            st.rerun()

    tab_fc, tab_bt, tab_eda, tab_feat = st.tabs(
        ["Forecast", "Back-test", "EDA", "Features & importance"]
    )

    hist = load_history(refresh)

    # ------------------------------------------------------------- forecast
    with tab_fc:
        try:
            f = service.forecast(
                model_name=model_name, horizon=int(horizon), refresh_data=refresh
            )
        except Exception as exc:  # surface the reason, never a blank page
            st.error(f"Forecast failed: {exc}")
            return

        entry = float(f["entry_price_usd"])
        pred = float(f["predicted_price_usd"])
        ret = float(f["predicted_return"])
        up = ret >= 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spot (as of)", f"${entry:,.2f}", help=f["as_of_date"])
        c2.metric(
            f"Predicted · {f['trained_horizon_days']}d",
            f"${pred:,.2f}",
            delta=f"{ret * 100:+.2f}%",
        )
        c3.metric("Direction", "UP" if up else "DOWN")
        c4.metric("Target date", f["target_date"])

        st.caption(
            "The model predicts the **forward return**, which is then applied to "
            "the last observed price. Error bars below come from held-out RMSE."
        )

        rmse_ret = float(f.get("metrics", {}).get("rmse_ret") or float("nan"))
        band_lo = entry * (1 + ret - rmse_ret)
        band_hi = entry * (1 + ret + rmse_ret)

        recent = hist.tail(180)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent.index, y=recent[S.TARGET_COL],
                name="Actual gold", line=dict(color="#b8860b", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[recent.index[-1], pd.Timestamp(f["target_date"])],
                y=[entry, pred],
                name="Forecast",
                line=dict(color="crimson", width=2, dash="dot"),
                mode="lines+markers",
            )
        )
        if pd.notna(rmse_ret):
            # a proper confidence band: lower bound trace, then the upper bound
            # filled *to* it, which is how plotly renders a shaded interval.
            # It fans out from today's real price, which is the only point the
            # band is certain about.
            t0, t1 = recent.index[-1], pd.Timestamp(f["target_date"])
            fig.add_trace(
                go.Scatter(
                    x=[t0, t1], y=[entry, band_lo],
                    mode="lines", line=dict(width=0), showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[t0, t1], y=[entry, band_hi],
                    mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(220, 20, 60, 0.12)",
                    name="±1 RMSE band",
                )
            )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if "usdinr" in hist.columns:
            fx = float(hist["usdinr"].iloc[-1])
            per_gram = pred / 31.1035 * fx
            st.metric(
                "Implied 24K gold price (INR / gram)",
                f"₹{per_gram:,.0f}",
                help=f"at USD/INR {fx:.2f}, 1 troy oz = 31.1035 g",
            )
        st.write(
            f"Model **{model_name}** · trained "
            f"{service.load_meta().get('trained_at', 'n/a')}"
        )

    # ------------------------------------------------------------ back-test
    with tab_bt:
        st.subheader("Held-out performance")
        table = service.metrics_table()
        if table is None:
            st.warning("Run the pipeline to generate metrics.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.caption(
                "`skill_vs_naive` is the fraction of the naive-persistence USD "
                "error that the model removes. **Anything near 0 means no edge** "
                "- gold is close to efficiently priced over short horizons."
            )
            for img in ("04_predictions.png", "05_errors.png", "06_walk_forward.png"):
                p = FIG / img
                if p.exists():
                    st.image(str(p), use_container_width=True)

        cv = service.walk_forward_table()
        if cv is not None:
            st.subheader("Walk-forward summary")
            st.dataframe(
                cv.groupby("model")[["rmse_usd", "skill_vs_naive"]]
                .mean()
                .round(4),
                use_container_width=True,
            )

    # ------------------------------------------------------------------ EDA
    with tab_eda:
        st.subheader("Price history & rolling statistics")
        chart = px.line(
            hist.reset_index(), x="date", y=S.TARGET_COL,
            title="Gold futures close (full history)", height=380,
        )
        st.plotly_chart(chart, use_container_width=True)

        for img in ("01_price_trend.png", "03_distributions.png",
                    "00_missing_pattern.png"):
            p = FIG / img
            if p.exists():
                st.image(str(p), use_container_width=True)

        st.subheader("Correlation")
        corr = load_corr()
        returns_mode = st.toggle("Use daily returns instead of levels", value=False)
        if returns_mode:
            from src.eda import correlation_matrix

            corr = correlation_matrix(hist, method="pearson", returns=True)
        st.plotly_chart(
            px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu",
                zmin=-1, zmax=1, aspect="auto", height=560,
            ),
            use_container_width=True,
        )

    # -------------------------------------------------------------- features
    with tab_feat:
        st.subheader("Feature set")
        cols = service.load_bundle()["feature_cols"]
        st.write(f"**{len(cols)} engineered features** (lags, rolling stats, "
                 f"returns, calendar, macro spreads).")
        groups = {
            "Lags": [c for c in cols if "_lag_" in c],
            "Rolling mean/std/z": [c for c in cols if "_rm_" in c or "_rstd_" in c or "_z_" in c],
            "Returns": [c for c in cols if "_ret_" in c or "logret" in c],
            "Calendar": [c for c in cols if c in
                         ("dow", "month", "quarter", "month_sin", "month_cos",
                          "is_festive_season", "is_year_end", "is_month_start",
                          "is_month_end", "week_of_year", "day_of_year", "year",
                          "dow_sin", "dow_cos")],
            "Macro ratios": [c for c in cols if "ratio" in c or "yield" in c or "chg" in c],
        }
        for label, items in groups.items():
            with st.expander(f"{label} ({len(items)})"):
                st.code("\n".join(items) or "-", language="text")

        p = FIG / "07_feature_importance.png"
        if p.exists():
            st.image(str(p), use_container_width=True)
        imp_path = S.ARTIFACTS_DIR / "feature_importance.csv"
        if imp_path.exists():
            st.dataframe(
                pd.read_csv(imp_path).head(25), use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
