"""
=============================================================
  WEB TRAFFIC TREND ANALYZER
  Methods: Lagrange Interpolation & Newton's Divided Difference
           (implemented from scratch, as per syllabus)

  Libraries: numpy, matplotlib, tkinter (built-in)
  Author   : Student Project
  Date     : 2026
=============================================================

  NOTE ON EXTRAPOLATION:
  Lagrange and Newton's Divided Difference are *interpolation* methods —
  they are designed to find values WITHIN the known data range.
  Extrapolating far beyond the data can cause large errors (Runge's
  phenomenon). This project predicts only a few steps beyond the last
  known point to keep results meaningful.
"""

# ── Standard / built-in imports ──────────────────────────────────────────────
import tkinter as tk                                        # GUI framework
from tkinter import ttk, messagebox, filedialog             # Extra Tk widgets

# ── Third-party imports ───────────────────────────────────────────────────────
import numpy as np                                          # Array math
import matplotlib                                           # Plotting
matplotlib.use("TkAgg")                                     # Tk-compatible backend
import matplotlib.pyplot as plt                             # Pyplot interface
import matplotlib.ticker                                    # Axis formatting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # Embed in Tk

# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLE DATA
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_DAYS     = [1, 2, 4, 6, 8, 10, 13, 16, 20]
SAMPLE_VISITORS = [120, 180, 250, 310, 400, 370, 430, 510, 600]


# ═════════════════════════════════════════════════════════════════════════════
#  INTERPOLATION METHODS  (implemented from scratch – no polyfit)
# ═════════════════════════════════════════════════════════════════════════════

def lagrange_interpolate(x_known, y_known, x_query):
    """
    Lagrange Interpolating Polynomial.

    Formula:
        P(x) = Σᵢ  yᵢ · Lᵢ(x)

    where the basis polynomial Lᵢ(x) is:
        Lᵢ(x) = Π_{j≠i}  (x − xⱼ) / (xᵢ − xⱼ)

    This passes exactly through every known data point.

    Parameters
    ----------
    x_known, y_known : known data arrays
    x_query          : single value or array of values to evaluate P(x) at

    Returns
    -------
    Interpolated value(s)
    """
    x_known = np.array(x_known, dtype=float)
    y_known = np.array(y_known, dtype=float)
    n       = len(x_known)

    scalar_input = np.isscalar(x_query)
    x_query = np.atleast_1d(np.array(x_query, dtype=float))
    result  = np.zeros_like(x_query)

    for i in range(n):              # Loop over each known point
        Li = np.ones_like(x_query) # Basis polynomial starts at 1
        for j in range(n):
            if j == i:
                continue            # Skip i == j (would give division by zero)
            # Multiply numerator (x − xⱼ) and denominator (xᵢ − xⱼ) term by term
            Li *= (x_query - x_known[j]) / (x_known[i] - x_known[j])
        result += y_known[i] * Li   # Add contribution of the i-th term

    return float(result[0]) if scalar_input else result


def _build_divided_diff_table(x, y):
    """
    Build the Divided Difference table for Newton's method.

    Table structure (example for 4 points):
        f[x0]
        f[x0,x1]   f[x1]
        f[x0,x1,x2]  f[x1,x2]   f[x2]
        ...

    The first element of each column is the Newton coefficient.

    Formula:
        f[xᵢ, ..., xᵢ₊ₖ] = (f[xᵢ₊₁,...,xᵢ₊ₖ] − f[xᵢ,...,xᵢ₊ₖ₋₁]) / (xᵢ₊ₖ − xᵢ)
    """
    n     = len(x)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y          # 0th order divided difference = y values

    for j in range(1, n):   # j = order of divided difference
        for i in range(n - j):
            # Apply the divided difference formula recursively
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


def newton_divided_diff(x_known, y_known, x_query):
    """
    Newton's Divided Difference Interpolating Polynomial.

    Formula:
        P(x) = f[x₀]
              + f[x₀,x₁]·(x−x₀)
              + f[x₀,x₁,x₂]·(x−x₀)(x−x₁)
              + …

    The coefficients f[x₀], f[x₀,x₁], … are the first row of the
    divided difference table.

    Parameters
    ----------
    x_known, y_known : known data arrays
    x_query          : evaluation point(s)

    Returns
    -------
    Interpolated value(s)
    """
    x_known = np.array(x_known, dtype=float)
    y_known = np.array(y_known, dtype=float)
    n       = len(x_known)

    table  = _build_divided_diff_table(x_known, y_known)
    coeffs = table[0, :]   # Newton coefficients = first row

    scalar_input = np.isscalar(x_query)
    x_query = np.atleast_1d(np.array(x_query, dtype=float))
    result  = np.zeros_like(x_query)

    for i in range(n):
        # Compute coefficient × product term (x−x₀)(x−x₁)…(x−xᵢ₋₁)
        term = np.ones_like(x_query) * coeffs[i]
        for j in range(i):
            term *= (x_query - x_known[j])   # Multiply by each (x − xⱼ)
        result += term                         # Accumulate the term

    return float(result[0]) if scalar_input else result


# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED WRAPPER: Interpolation + Extrapolation
# ─────────────────────────────────────────────────────────────────────────────
def interpolate_and_predict(days, visitors, method, future_days=5):
    """
    1. Interpolation: use Lagrange/Newton to draw a smooth curve
       WITHIN the known data range (where both methods are accurate).

    2. Extrapolation: use numpy polynomial fit (degree 2 – quadratic)
       to predict BEYOND the last known data point.
       This gives stable predictions while still using the syllabus
       methods for the interpolation part.

    Returns
    -------
    x_smooth, y_smooth : smooth curve over the known range
    x_future, y_future : predicted values beyond the last known day
    """
    x = np.array(days, dtype=float)
    y = np.array(visitors, dtype=float)

    # ── Select interpolation function ────────────────────────────────────
    if method == "lagrange":
        interp_fn = lambda q: lagrange_interpolate(x, y, q)
    else:
        interp_fn = lambda q: newton_divided_diff(x, y, q)

    # ── Smooth curve WITHIN known range (interpolation zone) ─────────────
    x_smooth = np.linspace(x[0], x[-1], 400)   # 400 evenly-spaced points
    y_smooth = interp_fn(x_smooth)              # Evaluate interpolation

    # ── Future predictions BEYOND the data (extrapolation zone) ──────────
    # Use degree-2 polynomial fit for stable short-range extrapolation
    degree    = min(3, len(x) - 1)             # Degree ≤ (n−1) always
    poly_fit  = np.poly1d(np.polyfit(x, y, deg=degree))   # Fit polynomial
    x_future  = np.arange(x[-1] + 1, x[-1] + future_days + 1)
    y_future  = poly_fit(x_future)              # Predict with fitted poly

    return x_smooth, y_smooth, x_future, y_future


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_traffic(ax, days, visitors, x_smooth, y_smooth, x_future, y_future, method):
    """Draw data, interpolation curve, and predictions on the axes."""
    ax.clear()

    method_label = ("Lagrange Interpolation" if method == "lagrange"
                    else "Newton's Divided Difference")

    # Shade the interpolation vs extrapolation zones
    ax.axvspan(days[0], days[-1], alpha=0.06, color="royalblue",
               label="Interpolation zone")
    ax.axvspan(days[-1], x_future[-1] + 1, alpha=0.06, color="red",
               label="Prediction zone")

    # ── Original data points ──────────────────────────────────────────────
    ax.scatter(days, visitors, color="royalblue", s=110, zorder=5,
               edgecolors="white", linewidths=0.6, label="Actual Data")

    # ── Interpolated smooth curve ─────────────────────────────────────────
    ax.plot(x_smooth, y_smooth, color="darkorange", linewidth=2.5,
            label=f"{method_label} (interpolation)")

    # ── Future predicted points ───────────────────────────────────────────
    ax.scatter(x_future, y_future, color="crimson", s=90, marker="D",
               zorder=5, edgecolors="white", linewidths=0.5,
               label="Predicted (Polynomial Fit)")
    # Connect predicted points with a dashed line
    ax.plot(x_future, y_future, color="crimson", linewidth=1.5,
            linestyle="--", alpha=0.7)

    # Vertical separator between known data and predictions
    ax.axvline(x=days[-1], color="gray", linestyle="--",
               linewidth=1.3, label="Prediction start")

    # Annotate each prediction
    for xf, yf in zip(x_future, y_future):
        ax.annotate(f"Day {int(xf)}\n{int(yf):,}",
                    xy=(xf, yf), xytext=(8, 8),
                    textcoords="offset points", fontsize=7, color="darkred")

    # ── Axes styling ──────────────────────────────────────────────────────
    ax.set_title(f"🌐  Web Traffic Trend Analyzer  [{method_label}]",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Day Number", fontsize=11)
    ax.set_ylabel("Number of Visitors", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("#f8f9fa")


# ═════════════════════════════════════════════════════════════════════════════
#  TKINTER APPLICATION
# ═════════════════════════════════════════════════════════════════════════════
class WebTrafficApp(tk.Tk):
    """Main Tkinter window for the Web Traffic Trend Analyzer."""

    def __init__(self):
        super().__init__()
        self.title("Web Traffic Trend Analyzer")
        self.geometry("1160x800")
        self.resizable(True, True)
        self.configure(bg="#1e1e2e")
        self.fig = None
        self.ax  = None
        self._build_header()
        self._build_main_area()

    # ── Header ────────────────────────────────────────────────────────────
    def _build_header(self):
        h = tk.Frame(self, bg="#6c63ff", pady=12)
        h.pack(fill="x")
        tk.Label(h, text="🌐  Web Traffic Trend Analyzer",
                 font=("Segoe UI", 20, "bold"), bg="#6c63ff", fg="white").pack()
        tk.Label(h,
                 text="Lagrange Interpolation  ·  Newton's Divided Difference  ·  numpy + matplotlib + tkinter",
                 font=("Segoe UI", 10), bg="#6c63ff", fg="#dde").pack()

    # ── Main layout ───────────────────────────────────────────────────────
    def _build_main_area(self):
        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=10, pady=10)
        main.columnconfigure(0, minsize=305)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        self._build_left_panel(main)
        self._build_right_panel(main)

    # ── LEFT PANEL ────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        lf = tk.Frame(parent, bg="#2a2a3e")
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        def section_label(text):
            tk.Label(lf, text=text, font=("Segoe UI", 9, "bold"),
                     bg="#2a2a3e", fg="#9aa5ce").pack(anchor="w", padx=12, pady=(8, 2))

        def text_box():
            t = tk.Text(lf, height=3, font=("Consolas", 10),
                        bg="#1a1b26", fg="#c0caf5", insertbackground="white",
                        relief="flat", bd=4)
            t.pack(fill="x", padx=12, pady=(0, 8))
            return t

        tk.Label(lf, text="📋  Enter Traffic Data",
                 font=("Segoe UI", 12, "bold"),
                 bg="#2a2a3e", fg="#a9b1d6", pady=8).pack(fill="x", padx=10)

        section_label("Days  (comma-separated, e.g. 1, 3, 5, 7)")
        self.days_entry = text_box()

        section_label("Visitors  (same count as days)")
        self.visitors_entry = text_box()

        # ── Method selector ───────────────────────────────────────────────
        tk.Label(lf, text="Interpolation Method",
                 font=("Segoe UI", 10, "bold"),
                 bg="#2a2a3e", fg="#a9b1d6").pack(anchor="w", padx=12, pady=(4, 2))

        self.method_var = tk.StringVar(value="newton")
        for text, val in [("Lagrange Interpolation", "lagrange"),
                           ("Newton's Divided Difference", "newton")]:
            tk.Radiobutton(
                lf, text=text, variable=self.method_var, value=val,
                bg="#2a2a3e", fg="#c0caf5", selectcolor="#6c63ff",
                activebackground="#2a2a3e", font=("Segoe UI", 9)
            ).pack(anchor="w", padx=20)

        # Live formula display
        self.formula_var = tk.StringVar()
        tk.Label(lf, textvariable=self.formula_var,
                 font=("Consolas", 8), bg="#1a1b26", fg="#7aa2f7",
                 justify="left", wraplength=270, pady=5).pack(fill="x", padx=12, pady=(4, 8))
        self.method_var.trace_add("write", self._update_formula)
        self._update_formula()

        # ── Future days slider ────────────────────────────────────────────
        section_label("Days to Predict  (1 – 10)")
        self.future_var = tk.IntVar(value=5)
        tk.Scale(lf, from_=1, to=10, orient="horizontal", variable=self.future_var,
                 bg="#2a2a3e", fg="#c0caf5", troughcolor="#6c63ff",
                 highlightthickness=0, font=("Segoe UI", 9),
                 label="").pack(fill="x", padx=12, pady=(0, 10))

        # ── Buttons ───────────────────────────────────────────────────────
        bs = dict(font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", pady=7)
        tk.Button(lf, text="📂  Load Sample Data", bg="#7aa2f7", fg="#1e1e2e",
                  command=self._load_sample_data, **bs).pack(fill="x", padx=12, pady=3)
        tk.Button(lf, text="📈  Generate Prediction", bg="#9ece6a", fg="#1e1e2e",
                  command=self._generate, **bs).pack(fill="x", padx=12, pady=3)
        tk.Button(lf, text="💾  Save Graph as PNG", bg="#e0af68", fg="#1e1e2e",
                  command=self._save_graph, **bs).pack(fill="x", padx=12, pady=3)
        tk.Button(lf, text="🗑️  Clear", bg="#f7768e", fg="white",
                  command=self._clear_all, **bs).pack(fill="x", padx=12, pady=3)

        tk.Label(lf,
                 text="ℹ️  Tips\n• At least 3 data points needed\n"
                      "• Days must be strictly increasing\n"
                      "• Blue zone = interpolation (accurate)\n"
                      "• Red zone  = extrapolation (estimate)",
                 font=("Segoe UI", 8), bg="#2a2a3e", fg="#565f89",
                 justify="left", pady=6).pack(fill="x", padx=12, pady=(12, 0))

    def _update_formula(self, *_):
        if self.method_var.get() == "lagrange":
            self.formula_var.set(
                "P(x) = Σ yᵢ·Lᵢ(x)\n"
                "Lᵢ(x) = Π(x−xⱼ)/(xᵢ−xⱼ),  j≠i"
            )
        else:
            self.formula_var.set(
                "P(x) = f[x₀] + f[x₀,x₁](x−x₀)\n"
                "     + f[x₀,x₁,x₂](x−x₀)(x−x₁) + …"
            )

    # ── RIGHT PANEL ───────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        rf = tk.Frame(parent, bg="#1e1e2e")
        rf.grid(row=0, column=1, sticky="nsew")
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(0, weight=3)
        rf.rowconfigure(1, weight=1)

        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 5), facecolor="#1e1e2e")
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.09)
        self.canvas = FigureCanvasTkAgg(self.fig, master=rf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._show_placeholder()

        # Results table
        tab_frame = tk.Frame(rf, bg="#1e1e2e")
        tab_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        tab_frame.columnconfigure(0, weight=1)

        tk.Label(tab_frame, text="📊  Predicted Traffic Table",
                 font=("Segoe UI", 10, "bold"),
                 bg="#1e1e2e", fg="#a9b1d6").pack(anchor="w", padx=6)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("T.Treeview",
                         background="#2a2a3e", foreground="#c0caf5",
                         fieldbackground="#2a2a3e", rowheight=22, font=("Segoe UI", 9))
        style.configure("T.Treeview.Heading",
                         background="#6c63ff", foreground="white",
                         font=("Segoe UI", 9, "bold"))
        style.map("T.Treeview", background=[("selected", "#414868")])

        cols = ("Day", "Predicted Visitors", "Trend vs Last Known")
        self.tree = ttk.Treeview(tab_frame, columns=cols, show="headings",
                                 height=4, style="T.Treeview")
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=165)

        sb = ttk.Scrollbar(tab_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=6)
        sb.pack(side="right", fill="y")

    def _show_placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor("#f0f4ff")
        self.ax.text(0.5, 0.5, "Enter data and click\n'Generate Prediction'",
                     ha="center", va="center", fontsize=14, color="#aaa",
                     transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    # ── CALLBACKS ─────────────────────────────────────────────────────────

    def _load_sample_data(self):
        self.days_entry.delete("1.0", tk.END)
        self.days_entry.insert(tk.END, ", ".join(map(str, SAMPLE_DAYS)))
        self.visitors_entry.delete("1.0", tk.END)
        self.visitors_entry.insert(tk.END, ", ".join(map(str, SAMPLE_VISITORS)))

    def _parse_input(self):
        """Validate and parse text-box inputs. Raises ValueError on failure."""
        rd = self.days_entry.get("1.0", tk.END).strip()
        rv = self.visitors_entry.get("1.0", tk.END).strip()

        if not rd or not rv:
            raise ValueError("Both 'Days' and 'Visitors' fields must be filled.")

        try:
            days     = [float(v.strip()) for v in rd.split(",") if v.strip()]
            visitors = [float(v.strip()) for v in rv.split(",") if v.strip()]
        except ValueError:
            raise ValueError("All values must be numeric — no letters or symbols.")

        if len(days) < 3:
            raise ValueError("Please enter at least 3 data points.")
        if len(days) != len(visitors):
            raise ValueError(f"Days count ({len(days)}) ≠ Visitors count ({len(visitors)}).")
        if any(days[i] >= days[i + 1] for i in range(len(days) - 1)):
            raise ValueError("Day values must be strictly increasing (e.g. 1, 3, 5, 7).")
        if any(v < 0 for v in visitors):
            raise ValueError("Visitor counts cannot be negative.")

        return days, visitors

    def _generate(self):
        """Main action: validate → interpolate → plot → fill table."""
        try:
            days, visitors = self._parse_input()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        method      = self.method_var.get()
        future_days = self.future_var.get()

        try:
            x_smooth, y_smooth, x_future, y_future = interpolate_and_predict(
                days, visitors, method, future_days
            )
        except Exception as e:
            messagebox.showerror("Computation Error", str(e))
            return

        plot_traffic(self.ax, days, visitors,
                     x_smooth, y_smooth, x_future, y_future, method)
        self.canvas.draw()
        self._fill_table(x_future, y_future, last_known=visitors[-1])

    def _fill_table(self, x_future, y_future, last_known):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for day, pred in zip(x_future, y_future):
            chg = ((pred - last_known) / last_known) * 100 if last_known else 0
            tag = "up" if chg >= 0 else "down"
            self.tree.insert("", "end",
                             values=(f"Day {int(day)}", f"{int(pred):,}", f"{chg:+.1f}%"),
                             tags=(tag,))
        self.tree.tag_configure("up",   foreground="#9ece6a")   # Green = growth
        self.tree.tag_configure("down", foreground="#f7768e")   # Red = decline

    def _save_graph(self):
        if self.fig is None:
            messagebox.showinfo("Nothing to Save", "Generate a prediction first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            title="Save Graph As", initialfile="web_traffic_prediction.png"
        )
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="#1e1e2e")
            messagebox.showinfo("Saved!", f"Graph saved:\n{path}")

    def _clear_all(self):
        self.days_entry.delete("1.0", tk.END)
        self.visitors_entry.delete("1.0", tk.END)
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._show_placeholder()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = WebTrafficApp()   # Create and open the window
    app.mainloop()          # Run the Tkinter event loop
