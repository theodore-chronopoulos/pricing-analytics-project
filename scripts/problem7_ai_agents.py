import pandas as pd
import numpy as np
from scipy.optimize import minimize

FEATURE_COLS = [
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score",
    "prop_accesibility_score",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
]

def scaled_rank(series: pd.Series) -> pd.Series:
    r = series.rank(method="average", pct=True)
    if r.max() == r.min():
        return pd.Series([0.5] * len(series), index=series.index)
    return (r - r.min()) / (r.max() - r.min())

def generate_ai_choices(df: pd.DataFrame, sample_queries: int = 100) -> pd.DataFrame:
    query_ids = df["srch_id"].drop_duplicates().iloc[:sample_queries].tolist()
    sample = df[df["srch_id"].isin(query_ids)].copy()

    out_groups = []
    for _, g in sample.groupby("srch_id", sort=True):
        g = g.copy().reset_index(drop=True)
        g["row_id"] = np.arange(1, len(g) + 1)

        star = scaled_rank(g["prop_starrating"])
        review = scaled_rank(g["prop_review_score"])
        brand = g["prop_brand_bool"].astype(float)
        location = scaled_rank(g["prop_location_score"])
        accessibility = g["prop_accesibility_score"].astype(float)
        promo = g["promotion_flag"].astype(float)
        price_rank = scaled_rank(g["price_usd"])

        booking_window = float(g["srch_booking_window"].iloc[0])
        adults = float(g["srch_adults_count"].iloc[0])
        children = float(g["srch_children_count"].iloc[0])
        rooms = float(g["srch_room_count"].iloc[0])

        bw_norm = min(booking_window, 60.0) / 60.0
        price_sensitivity = (
            0.62
            + 0.12 * bw_norm
            - 0.08 * (children > 0)
            - 0.07 * (rooms > 1)
            - 0.04 * (adults >= 3)
            - 0.05 * (booking_window <= 3)
        )

        quality = (
            0.55 * review
            + 0.35 * star
            + 0.12 * brand
            + 0.18 * location
            + 0.04 * accessibility
            + 0.08 * promo
        )
        score = quality - price_sensitivity * price_rank

        chosen_idx = int(score.idxmax())
        g["ai_booking"] = 0
        g.loc[chosen_idx, "ai_booking"] = 1
        g["ai_reason"] = ""
        g.loc[chosen_idx, "ai_reason"] = "Selected for strongest quality-value tradeoff in the query."
        out_groups.append(g.drop(columns=["row_id"]))

    return pd.concat(out_groups, ignore_index=True)

def fit_mnl(df: pd.DataFrame, booking_col: str) -> pd.Series:
    X = df[FEATURE_COLS].astype(float).to_numpy()
    y = df[booking_col].astype(int).to_numpy()
    groups = []
    for _, idx in df.groupby("srch_id", sort=False).indices.items():
        idx = np.asarray(idx)
        chosen_local = int(np.where(y[idx] == 1)[0][0]) if y[idx].sum() == 1 else -1
        groups.append((idx[0], len(idx), chosen_local))

    def obj_grad(beta):
        util = beta[0] + X.dot(beta[1:])
        ll = 0.0
        grad = np.zeros(X.shape[1] + 1)
        for s, l, c in groups:
            u = util[s:s+l]
            Xi = X[s:s+l]
            m = max(0.0, float(u.max()))
            eu = np.exp(u - m)
            e0 = np.exp(-m)
            denom = e0 + eu.sum()
            probs = eu / denom
            exp_x = probs @ Xi
            if c >= 0:
                ll += u[c] - (m + np.log(denom))
                grad[0] += 1.0 - probs.sum()
                grad[1:] += Xi[c] - exp_x
            else:
                ll += -(m + np.log(denom))
                grad[0] += -probs.sum()
                grad[1:] += -exp_x
        return -ll, -grad

    res = minimize(lambda b: obj_grad(b), np.zeros(X.shape[1] + 1), method="L-BFGS-B", jac=True)
    return pd.Series(res.x, index=["intercept"] + FEATURE_COLS)

def main():
    data = pd.read_csv("data.csv")
    ai_df = generate_ai_choices(data, sample_queries=100)
    ai_coef = fit_mnl(ai_df, "ai_booking")
    human_coef = fit_mnl(data, "booking_bool")

    compare = pd.DataFrame({
        "parameter": human_coef.index,
        "human_mnl": human_coef.values,
        "ai_mnl": ai_coef.values,
    })
    compare["difference_ai_minus_human"] = compare["ai_mnl"] - compare["human_mnl"]

    compare.to_csv("results/problem7_human_vs_ai_mnl_comparison.csv", index=False)
    ai_df.to_csv("results/problem7_ai_generated_sample.csv", index=False)

if __name__ == "__main__":
    main()
