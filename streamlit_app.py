import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LAGRANGE INTERPOLATION
# ============================================================

def lagrange_interpolate(x_known, y_known, x_query):
    x_known = np.array(x_known, dtype=float)
    y_known = np.array(y_known, dtype=float)
    n = len(x_known)

    scalar_input = np.isscalar(x_query)
    x_query = np.atleast_1d(np.array(x_query, dtype=float))
    result = np.zeros_like(x_query)

    for i in range(n):
        Li = np.ones_like(x_query)

        for j in range(n):
            if j == i:
                continue

            Li *= (
                (x_query - x_known[j])
                / (x_known[i] - x_known[j])
            )

        result += y_known[i] * Li

    return float(result[0]) if scalar_input else result


# ============================================================
# NEWTON DIVIDED DIFFERENCE
# ============================================================

def build_divided_diff_table(x, y):
    n = len(x)

    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (
                table[i + 1][j - 1] - table[i][j - 1]
            ) / (x[i + j] - x[i])

    return table


def newton_divided_diff(x_known, y_known, x_query):
    x_known = np.array(x_known, dtype=float)
    y_known = np.array(y_known, dtype=float)

    n = len(x_known)

    table = build_divided_diff_table(x_known, y_known)
    coeffs = table[0, :]

    scalar_input = np.isscalar(x_query)
    x_query = np.atleast_1d(np.array(x_query, dtype=float))

    result = np.zeros_like(x_query)

    for i in range(n):
        term = np.ones_like(x_query) * coeffs[i]

        for j in range(i):
            term *= (x_query - x_known[j])

        result += term

    return float(result[0]) if scalar_input else result


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Web Traffic Trend Analyzer",
    page_icon="🌐",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌐 Web Traffic Trend Analyzer")

st.markdown(
    """
    **Data Analytics & Prediction Project**

    Analyze web traffic data using **Lagrange Interpolation**
    and **Newton's Divided Difference** methods.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📊 Traffic Data")

days_input = st.sidebar.text_area(
    "Days",
    value="1, 2, 4, 6, 8, 10, 13, 16, 20",
    help="Enter day values separated by commas."
)

visitors_input = st.sidebar.text_area(
    "Visitors",
    value="120, 180, 250, 310, 400, 370, 430, 510, 600",
    help="Enter visitor values separated by commas."
)

method = st.sidebar.radio(
    "Interpolation Method",
    [
        "Newton's Divided Difference",
        "Lagrange Interpolation"
    ]
)

future_days = st.sidebar.slider(
    "Days to Predict",
    min_value=1,
    max_value=10,
    value=5
)


# ============================================================
# PARSE DATA
# ============================================================

try:
    days = [
        float(x.strip())
        for x in days_input.split(",")
        if x.strip()
    ]

    visitors = [
        float(x.strip())
        for x in visitors_input.split(",")
        if x.strip()
    ]

except ValueError:
    st.error("❌ Please enter only numeric values.")
    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if len(days) < 3:
    st.error("❌ Please enter at least 3 data points.")
    st.stop()

if len(days) != len(visitors):
    st.error(
        f"❌ Number of days ({len(days)}) must equal "
        f"number of visitor values ({len(visitors)})."
    )
    st.stop()

if any(days[i] >= days[i + 1] for i in range(len(days) - 1)):
    st.error("❌ Days must be strictly increasing.")
    st.stop()

if any(v < 0 for v in visitors):
    st.error("❌ Visitor counts cannot be negative.")
    st.stop()


# ============================================================
# SELECT METHOD
# ============================================================

if method == "Lagrange Interpolation":
    interpolation_function = lagrange_interpolate
else:
    interpolation_function = newton_divided_diff


# ============================================================
# INTERPOLATION
# ============================================================

x_smooth = np.linspace(
    days[0],
    days[-1],
    400
)

y_smooth = interpolation_function(
    days,
    visitors,
    x_smooth
)


# ============================================================
# FUTURE PREDICTION
# ============================================================

x = np.array(days, dtype=float)
y = np.array(visitors, dtype=float)

degree = min(3, len(x) - 1)

poly_fit = np.poly1d(
    np.polyfit(x, y, deg=degree)
)

x_future = np.arange(
    x[-1] + 1,
    x[-1] + future_days + 1
)

y_future = poly_fit(x_future)


# ============================================================
# SUMMARY METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Last Known Day",
        int(days[-1])
    )

with col2:
    st.metric(
        "Last Known Visitors",
        f"{int(visitors[-1]):,}"
    )

with col3:
    st.metric(
        "Prediction Days",
        future_days
    )


# ============================================================
# GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(
    days,
    visitors,
    s=100,
    label="Actual Data"
)

ax.plot(
    x_smooth,
    y_smooth,
    linewidth=2.5,
    label=method
)

ax.scatter(
    x_future,
    y_future,
    s=90,
    marker="D",
    label="Predicted Visitors"
)

ax.plot(
    x_future,
    y_future,
    linestyle="--",
    linewidth=1.5
)

ax.axvline(
    x=days[-1],
    linestyle="--",
    linewidth=1.3,
    label="Prediction Start"
)

ax.set_title(
    f"Web Traffic Trend Analyzer — {method}"
)

ax.set_xlabel("Day Number")
ax.set_ylabel("Number of Visitors")

ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()

st.pyplot(fig)


# ============================================================
# PREDICTION TABLE
# ============================================================

st.subheader("📊 Predicted Traffic")

prediction_data = []

last_known = visitors[-1]

for day, prediction in zip(x_future, y_future):

    change = (
        (prediction - last_known)
        / last_known
        * 100
        if last_known
        else 0
    )

    prediction_data.append({
        "Day": f"Day {int(day)}",
        "Predicted Visitors": f"{int(prediction):,}",
        "Trend vs Last Known": f"{change:+.1f}%"
    })

st.table(prediction_data)


# ============================================================
# DOWNLOAD GRAPH
# ============================================================

st.subheader("💾 Download Graph")

image_path = "web_traffic_prediction.png"

fig.savefig(
    image_path,
    dpi=200,
    bbox_inches="tight"
)

with open(image_path, "rb") as file:

    st.download_button(
        label="📥 Download Graph as PNG",
        data=file,
        file_name="web_traffic_prediction.png",
        mime="image/png"
    )


# ============================================================
# ABOUT
# ============================================================

with st.expander("ℹ️ About this project"):

    st.write(
        """
        This project analyzes web traffic and predicts future
        visitor counts.

        Interpolation Methods:

        • Lagrange Interpolation
        • Newton's Divided Difference

        The interpolation methods are implemented from scratch
        using Python and NumPy.

        Short-range future predictions are generated using
        polynomial fitting.
        """
    )