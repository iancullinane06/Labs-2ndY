import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
import Styling

wavelength_cmap = sns.color_palette("rainbow", as_cmap=True)
figure_size = (8, 8)


data_balmer = pd.read_csv("Code/Data/Balmer.csv")
save_directory = "Reports/Report 5 - Balmer/images"

slits_d = 0.001/600

def plot_lambda(data):
    fig, ax = plt.subplots(figsize=figure_size)
    line_orders = np.linspace(0, data["eta"].max(), 100)

    # Group measurements by their wavelength estimates rather than row order.
    wavelength_estimates = (
        data["sin theta"].to_numpy() / data["eta"].to_numpy() * slits_d * 1e9
    )
    number_of_lights = 3
    labels = KMeans(
        n_clusters=number_of_lights, random_state=0, n_init=10
    ).fit_predict(wavelength_estimates.reshape(-1, 1))
    cluster_order = sorted(
        np.unique(labels),
        key=lambda label: wavelength_estimates[labels == label].mean()
    )

    light_fits = []
    for light_number, label in enumerate(cluster_order, start=1):
        measurements = data.iloc[np.flatnonzero(labels == label)]
        orders = measurements["eta"].to_numpy()
        sin_theta = measurements["sin theta"].to_numpy()
        slope = np.dot(orders, sin_theta) / np.dot(orders, orders)
        wavelength_nm = slope * slits_d * 1e9
        light_fits.append((light_number, orders, sin_theta, slope, wavelength_nm))

    wavelengths = [fit[4] for fit in light_fits]
    wavelength_min, wavelength_max = min(wavelengths), max(wavelengths)
    if wavelength_min == wavelength_max:
        wavelength_max = wavelength_min + 1
    wavelength_range = wavelength_max - wavelength_min

    for light_number, orders, sin_theta, slope, wavelength_nm in light_fits:
        color = wavelength_cmap((wavelength_nm - wavelength_min) / wavelength_range)
        ax.plot(
            line_orders,
            slope * line_orders,
            label=f"Light {light_number} (λ ≈ {wavelength_nm:.0f} nm)",
            color=color
        )

    scatter_orders = np.concatenate([fit[1] for fit in light_fits])
    scatter_sin_theta = np.concatenate([fit[2] for fit in light_fits])
    scatter_wavelengths = np.concatenate([
        np.full(fit[1].shape, fit[4]) for fit in light_fits
    ])
    points = ax.scatter(
        scatter_orders,
        scatter_sin_theta,
        c=scatter_wavelengths,
        cmap=wavelength_cmap,
        vmin=wavelength_min,
        vmax=wavelength_max,
        s=35
    )

    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Wavelength (nm)")

    Styling._style_axis(
        ax,
        "Diffraction measurements for three lights",
        "Diffraction order, n",
        "sin(θ)"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    Styling._finish_plot(fig, "lambda_plot.png", save_directory)
    
    # Save the new frequencies and errors
    new_data = []
    for light_number, (_, orders, sin_theta, slope, wavelength_nm) in enumerate(light_fits, start=1):
        residuals = sin_theta - slope * orders
        slope_se = np.sqrt(
            np.sum(residuals**2)
            / ((len(orders) - 1) * np.dot(orders, orders))
        )
        u_wavelength_nm = slope_se * slits_d * 1e9
        new_data.append({
            "Light": light_number,
            "Wavelength (nm)": wavelength_nm,
            "Uncertainty (nm)": u_wavelength_nm
        })
    new_data_df = pd.DataFrame(new_data)
    new_data_df.to_csv(f"Code/Data/lambda_new_frequencies.csv", index=False)

def plot_balmer(data):
    fig, (ax, ax_residuals) = plt.subplots(2, 1, figsize=figure_size, gridspec_kw={'height_ratios': [3, 1]})
    sns.scatterplot(x="1/nf^2-1/ni^2", y="1/lambda", data=data, ax=ax, c=wavelength_cmap(np.linspace(0, 1, len(data))))
    # Fit a linear model to the data
    x = data["1/nf^2-1/ni^2"].to_numpy()
    y = data["1/lambda"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(x, slope * x + intercept, color="red", linestyle="--", label=f"Linear fit: R_H ≈ {slope:.2e}")
    ax.scatter([0], [intercept], color="red", zorder=4, label=f"Y-intercept ≈ {intercept:.2e}")
    ax.annotate(
        f"(0, {intercept:.2e})",
        xy=(0, intercept),
        xytext=(8, -12),
        textcoords="offset points",
        color="red"
    )
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    Styling._style_axis(ax, "Balmer Series", "1/n_f^2-1/n_i^2", "1/λ")

    # Plot residuals
    residuals = y - (slope * x + intercept)
    sns.scatterplot(x=x, y=residuals, ax=ax_residuals, c=wavelength_cmap(np.linspace(0, 1, len(data))))
    ax_residuals.axhline(0, color="red", linestyle="--")
    ax_residuals.set_xlim(left=0)
    Styling._style_axis(ax_residuals, "Residuals", "1/nf^2-1/ni^2", "Residuals")

    Styling._finish_plot(fig, "balmer_plot.png", save_directory)

PLANCK = 6.62607015e-34
LIGHT_SPEED = 2.99792458e8
ELECTRON_CHARGE = 1.602176634e-19
RYDBERG_THEORY = 1.0973731568e7  # m^-1


def analyse_balmer(data):
    """Wavelengths, mean/uncertainty per line, Rydberg fit, % error,
    series limit and intercept check."""
    data = data.copy()
    # 1. Wavelength of each observed line (nm), from the 1/lambda column
    data["lambda_nm"] = 1e9 / data["1/lambda"]

    # 2. Mean and standard error for each spectral line (grouped by n_i)
    data["n_i"] = np.round(1 / np.sqrt(data["1/ni^2"])).astype(int)
    summary = data.groupby("n_i")["lambda_nm"].agg(["count", "mean", "std"])
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    print("Wavelength per line (nm):")
    print(summary.round(2).to_string())

    # 3. Linear fit: 1/lambda = R_H * (1/nf^2 - 1/ni^2) + c
    x = data["1/nf^2-1/ni^2"].to_numpy()
    y = data["1/lambda"].to_numpy()
    (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    u_slope, u_intercept = np.sqrt(np.diag(cov))
    print(f"\nR_H = ({slope:.4e} ± {u_slope:.1e}) m^-1")

    # 4. Percentage error against the accepted value
    percent_error = abs(slope - RYDBERG_THEORY) / RYDBERG_THEORY * 100
    print(f"Percentage error: {percent_error:.2f}% "
          f"(theory {RYDBERG_THEORY:.6e} m^-1)")
    print(f"Fit intercept c = ({intercept:.3e} ± {u_intercept:.1e}) m^-1 "
          "(should be 0)")

    # 5. Series limit (n_i -> infinity): 1/lambda = R_H / 4
    inv_limit = slope / 4
    u_inv_limit = u_slope / 4
    limit_nm = 1e9 / inv_limit
    u_limit_nm = limit_nm * u_inv_limit / inv_limit
    limit_ev = PLANCK * LIGHT_SPEED * inv_limit / ELECTRON_CHARGE
    u_limit_ev = limit_ev * u_inv_limit / inv_limit
    theory_nm = 4e9 / RYDBERG_THEORY
    theory_ev = PLANCK * LIGHT_SPEED * RYDBERG_THEORY / 4 / ELECTRON_CHARGE
    print(f"\nSeries limit: lambda = {limit_nm:.1f} ± {u_limit_nm:.1f} nm "
          f"(theory {theory_nm:.1f} nm)")
    print(f"Limit energy: E = {limit_ev:.3f} ± {u_limit_ev:.3f} eV "
          f"(theory {theory_ev:.3f} eV)")

    # 6. Intercept analysis: 1/lambda = R_H/nf^2 - R_H/ni^2, regress on 1/ni^2
    inv_ni2 = data["1/ni^2"].to_numpy()
    (m, c), cov2 = np.polyfit(inv_ni2, y, 1, cov=True)
    u_m, u_c = np.sqrt(np.diag(cov2))
    r_h = -m
    nf = np.sqrt(r_h / c)
    u_nf = 0.5 * nf * np.sqrt((u_m / m) ** 2 + (u_c / c) ** 2)
    print(f"\nIntercept analysis: R_H = {r_h:.4e} m^-1, "
          f"intercept = R_H/nf^2 = {c:.4e} m^-1")
    print(f"Implied final level n_f = {nf:.2f} ± {u_nf:.2f} (expected 2)")

    pd.DataFrame([{
        "R_H (m^-1)": slope, "u(R_H)": u_slope,
        "Percentage error (%)": percent_error,
        "Limit wavelength (nm)": limit_nm, "u(limit nm)": u_limit_nm,
        "Limit energy (eV)": limit_ev, "u(limit eV)": u_limit_ev,
        "Implied n_f": nf, "u(n_f)": u_nf,
    }]).to_csv("Code/Data/balmer_analysis.csv", index=False)
    summary.to_csv("Code/Data/balmer_line_wavelengths.csv")


U_THETA_DEG = 2 / 60  # assumed angle reading uncertainty: 2 arcmin
D_REL_UNCERTAINTY = 0.01  # assumed 1% tolerance on the 600 lines/mm grating
GRATING_LINES = 600 * 1000  # lines per metre
GRATING_WIDTH = 0.02  # assumed illuminated grating width (m)
M_E_OVER_M_P = 1 / 1836.15267343
THEORY_NM = {n: 1e9 / (RYDBERG_THEORY * (0.25 - 1 / n**2)) for n in (3, 4, 5)}
LINE_COLOURS = {3: "red", 4: "cyan", 5: "blue"}


def _wls(x, y, u_y):
    """Weighted linear fit y = m x + c. Returns params, covariance, chi2."""
    A = np.column_stack([x, np.ones_like(x)])
    W = np.diag(1 / u_y**2)
    cov = np.linalg.inv(A.T @ W @ A)
    params = cov @ A.T @ W @ y
    chi2 = float(np.sum(((y - A @ params) / u_y) ** 2))
    return params, cov, chi2


def _prepare(data):
    df = data.copy()
    df["n_i"] = np.round(1 / np.sqrt(df["1/ni^2"])).astype(int)
    theta = df["theta rads"].to_numpy()
    order = df["eta"].to_numpy()
    df["lam"] = slits_d * np.sin(theta) / order
    # Angle reading is statistical; grating spacing is a common systematic
    df["u_lam_stat"] = slits_d * np.cos(theta) * np.radians(U_THETA_DEG) / order
    df["u_lam"] = np.hypot(df["u_lam_stat"], df["lam"] * D_REL_UNCERTAINTY)
    df["y"] = 1 / df["lam"]
    df["u_y"] = df["y"] * df["u_lam"] / df["lam"]
    df["theory_nm"] = df["n_i"].map(THEORY_NM)
    return df


def plot_extended(data):
    df = _prepare(data)
    x = df["1/nf^2-1/ni^2"].to_numpy()
    y = df["y"].to_numpy()
    u_y = df["u_y"].to_numpy()
    n = len(df)
    colours = df["n_i"].map(LINE_COLOURS)

    # OLS and WLS fits
    (m_ols, c_ols), cov_ols = np.polyfit(x, y, 1, cov=True)
    res_ols = y - (m_ols * x + c_ols)
    (m_w, c_w), cov_w, chi2 = _wls(x, y, u_y)
    u_m_w = np.sqrt(cov_w[0, 0])
    dof = n - 2
    chi2_red = chi2 / dof
    print("\n=== Extended analysis ===")
    print(f"OLS R_H = {m_ols:.4e} m^-1   WLS R_H = {m_w:.4e} ± {u_m_w:.1e} m^-1")
    print(f"WLS chi2 = {chi2:.2f}, dof = {dof}, reduced chi2 = {chi2_red:.2f}, "
          f"p = {stats.chi2.sf(chi2, dof):.3f}")

    # Fit forced through zero (n_f = 2 fixed)
    m_0 = np.sum(x * y / u_y**2) / np.sum(x**2 / u_y**2)
    u_m_0 = 1 / np.sqrt(np.sum(x**2 / u_y**2))
    print(f"Through-origin WLS R_H = {m_0:.4e} ± {u_m_0:.1e} m^-1 "
          f"({abs(m_0 - RYDBERG_THEORY) / RYDBERG_THEORY * 100:.2f}% error)")

    # Uncertainty budget on R_H
    _, cov_a, _ = _wls(x, y, (df["y"] * df["u_lam_stat"] / df["lam"]).to_numpy())
    print("Uncertainty budget on R_H (m^-1): "
          f"angle reading = {np.sqrt(cov_a[0, 0]):.2e}, "
          f"grating d = {m_w * D_REL_UNCERTAINTY:.2e}, "
          f"fit scatter (OLS) = {np.sqrt(cov_ols[0, 0]):.2e}")

    # Per-line comparison with theory
    rows = []
    for n_i, g in df.groupby("n_i"):
        lam_nm = g["lam"].mean() * 1e9
        sem = g["lam"].std(ddof=1) / np.sqrt(len(g)) * 1e9
        u_tot = np.hypot(sem, lam_nm * D_REL_UNCERTAINTY)
        th = THEORY_NM[n_i]
        rows.append({
            "n_i": n_i, "mean (nm)": lam_nm, "u (nm)": u_tot,
            "theory (nm)": th, "diff (nm)": lam_nm - th,
            "diff (sigma)": (lam_nm - th) / u_tot,
            "error (%)": (lam_nm - th) / th * 100,
        })
    per_line = pd.DataFrame(rows)
    print("\nPer-line comparison with theory:")
    print(per_line.round(2).to_string(index=False))
    per_line.to_csv("Code/Data/balmer_per_line.csv", index=False)

    # Bootstrap
    rng = np.random.default_rng(0)
    inv_ni2 = df["1/ni^2"].to_numpy()
    boots, nfs = [], []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        if len(np.unique(inv_ni2[idx])) < 2:
            continue
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
        mb, cb = np.polyfit(inv_ni2[idx], y[idx], 1)
        if cb > 0 and mb < 0:
            nfs.append(np.sqrt(-mb / cb))
    boots, nfs = np.array(boots), np.array(nfs)
    print(f"\nBootstrap R_H = {boots.mean():.4e} ± {boots.std():.1e} m^-1 "
          f"(95% CI {np.percentile(boots, 2.5):.3e} to {np.percentile(boots, 97.5):.3e})")
    print(f"Bootstrap n_f = {nfs.mean():.2f} ± {nfs.std():.2f}")

    # Theory-related numbers
    print(f"\nReduced-mass R_H = {RYDBERG_THEORY / (1 + M_E_OVER_M_P):.6e} m^-1 "
          f"({M_E_OVER_M_P / (1 + M_E_OVER_M_P) * 100:.3f}% below R_inf)")
    ry_ev = PLANCK * LIGHT_SPEED * m_ols / ELECTRON_CHARGE
    ry_th = PLANCK * LIGHT_SPEED * RYDBERG_THEORY / ELECTRON_CHARGE
    print(f"Rydberg energy hcR_H = {ry_ev:.2f} ± "
          f"{ry_ev * np.sqrt(cov_ols[0, 0]) / m_ols:.2f} eV (theory {ry_th:.2f} eV, "
          f"error {abs(ry_ev - ry_th) / ry_th * 100:.2f}%)")
    print(f"Approx. H/D isotope shift at H-alpha = "
          f"{THEORY_NM[3] * M_E_OVER_M_P / 2:.2f} nm")
    for order in (1, 2, 3):
        power = order * GRATING_LINES * GRATING_WIDTH
        print(f"Resolving power order {order}: R = {power:.0f} -> "
              f"d_lambda(656 nm) = {THEORY_NM[3] / power:.3f} nm")

    # Free n_i per measurement (n_f = 2, theory R_H)
    df["n_i_free"] = 1 / np.sqrt(0.25 - df["y"] / RYDBERG_THEORY)
    print("\nFree n_i per line (mean, std):")
    print(df.groupby("n_i")["n_i_free"].agg(["mean", "std"]).round(2).to_string())

    # 1. Energy-level diagram
    fig, ax = plt.subplots(figsize=figure_size)
    for lvl in range(2, 7):
        e = -13.6 / lvl**2
        ax.hlines(e, 0, 1, color="black")
        ax.text(1.02, e, f"n={lvl}", va="center")
    for k, (n_i, g) in enumerate(df.groupby("n_i")):
        xpos = 0.2 + 0.25 * k
        ax.annotate(
            "", xy=(xpos, -13.6 / 4), xytext=(xpos, -13.6 / n_i**2),
            arrowprops=dict(arrowstyle="->", color=LINE_COLOURS[n_i], lw=2))
        ax.text(xpos + 0.02, -13.6 / 4 - 0.35 * (k + 1),
                f"{g['lam'].mean() * 1e9:.0f} nm\n(theory {THEORY_NM[n_i]:.0f})",
                color=LINE_COLOURS[n_i], fontsize=9)
    ax.set_xlim(0, 1.15)
    Styling._style_axis(ax, "Balmer transitions", "", "Energy (eV)")
    Styling._finish_plot(fig, "energy_levels.png", save_directory)

    # 2. Measured vs theoretical wavelength
    fig, ax = plt.subplots(figsize=figure_size)
    ax.errorbar(df["theory_nm"], df["lam"] * 1e9, yerr=df["u_lam"] * 1e9,
                fmt="none", ecolor="grey", capsize=3)
    ax.scatter(df["theory_nm"], df["lam"] * 1e9, c=colours, zorder=3)
    ax.plot([400, 700], [400, 700], "k--", label="y = x")
    ax.legend()
    Styling._style_axis(ax, "Measured vs theoretical wavelength",
                        "Theoretical λ (nm)", "Measured λ (nm)")
    Styling._finish_plot(fig, "measured_vs_theory.png", save_directory)

    # 3. Fit with confidence/prediction band, extended to series limit
    fig, ax = plt.subplots(figsize=figure_size)
    xs = np.linspace(0, 0.25, 200)
    fit = m_ols * xs + c_ols
    s = np.sqrt(np.sum(res_ols**2) / dof)
    tval = stats.t.ppf(0.975, dof)
    sxx = np.sum((x - x.mean()) ** 2)
    se_mean = s * np.sqrt(1 / n + (xs - x.mean()) ** 2 / sxx)
    se_pred = s * np.sqrt(1 + 1 / n + (xs - x.mean()) ** 2 / sxx)
    ax.fill_between(xs, fit - tval * se_pred, fit + tval * se_pred,
                    alpha=0.15, label="95% prediction")
    ax.fill_between(xs, fit - tval * se_mean, fit + tval * se_mean,
                    alpha=0.35, label="95% confidence")
    ax.plot(xs, fit, "r--", label=f"OLS fit, R_H = {m_ols:.3e}")
    ax.errorbar(x, y, yerr=u_y, fmt="none", ecolor="grey", capsize=3)
    ax.scatter(x, y, c=colours, zorder=3)
    ax.axvline(0.25, color="purple", ls=":", label="Series limit (n_i→∞)")
    ax.set_xlim(0, 0.26)
    ax.legend()
    Styling._style_axis(ax, "Balmer fit with confidence band",
                        "1/n_f^2-1/n_i^2", "1/λ (m⁻¹)")
    Styling._finish_plot(fig, "balmer_fit_band.png", save_directory)

    # 4. Residual diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].errorbar(x, res_ols, yerr=u_y, fmt="none", ecolor="grey", capsize=3)
    axes[0].scatter(x, res_ols, c=colours, zorder=3)
    axes[0].axhline(0, color="red", ls="--")
    axes[0].axhspan(-s, s, alpha=0.15, label="±1σ")
    axes[0].legend()
    Styling._style_axis(axes[0], "Residuals", "1/n_f^2-1/n_i^2", "y - fit (m⁻¹)")
    axes[1].hist(res_ols, bins=6, edgecolor="black")
    Styling._style_axis(axes[1], "Residual histogram", "Residual (m⁻¹)", "Count")
    stats.probplot(res_ols, dist="norm", plot=axes[2])
    Styling._style_axis(axes[2], "Q-Q plot", "Theoretical quantiles", "Ordered residuals")
    Styling._finish_plot(fig, "residual_diagnostics.png", save_directory)

    # 5. Spectrum-style plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for n_i, g in df.groupby("n_i"):
        mean = g["lam"].mean() * 1e9
        sem = g["lam"].std(ddof=1) / np.sqrt(len(g)) * 1e9
        ax.axvspan(mean - sem, mean + sem, color=LINE_COLOURS[n_i], alpha=0.3)
        ax.axvline(mean, color=LINE_COLOURS[n_i], lw=2, label=f"{n_i}→2 measured")
        ax.axvline(THEORY_NM[n_i], color="black", ls=":")
    ax.plot([], [], "k:", label="theory")
    ax.set_xlim(380, 720)
    ax.set_yticks([])
    ax.legend()
    Styling._style_axis(ax, "Balmer spectrum", "Wavelength (nm)", "")
    Styling._finish_plot(fig, "spectrum.png", save_directory)

    # 6. Wavelength by diffraction order
    fig, ax = plt.subplots(figsize=figure_size)
    for n_i, g in df.groupby("n_i"):
        ax.errorbar(g["eta"], g["lam"] * 1e9, yerr=g["u_lam"] * 1e9, fmt="o",
                    color=LINE_COLOURS[n_i], capsize=3, label=f"{n_i}→2")
        ax.axhline(THEORY_NM[n_i], color=LINE_COLOURS[n_i], ls=":")
    ax.set_xticks([1, 2, 3])
    ax.legend()
    Styling._style_axis(ax, "Wavelength by diffraction order", "Order", "λ (nm)")
    Styling._finish_plot(fig, "lambda_by_order.png", save_directory)

    # 7. Fit comparison
    fig, ax = plt.subplots(figsize=figure_size)
    ax.errorbar(x, y, yerr=u_y, fmt="o", capsize=3, color="black")
    ax.plot(xs, m_ols * xs + c_ols, label="OLS")
    ax.plot(xs, m_w * xs + c_w, label=f"WLS (χ²ᵣ={chi2_red:.2f})")
    ax.plot(xs, m_0 * xs, label="Through origin (n_f=2)")
    ax.plot(xs, RYDBERG_THEORY * xs, "k:", label="Theory")
    ax.legend()
    Styling._style_axis(ax, "Fit comparison", "1/n_f^2-1/n_i^2", "1/λ (m⁻¹)")
    Styling._finish_plot(fig, "fit_comparison.png", save_directory)


if __name__ == "__main__":
    choice = input(
        "Do you want to (p)lot, (a)nalyse or (e)xtended-analyse the Balmer data? "
    ).strip().lower()
    if choice == "p":
        plot_lambda(data_balmer)
        plot_balmer(data_balmer)
    elif choice == "a":
        analyse_balmer(data_balmer)
    elif choice == "e":
        plot_extended(data_balmer)