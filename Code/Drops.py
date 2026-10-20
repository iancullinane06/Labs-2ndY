import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import linregress, t as student_t
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid")

# Millikan's oil drop experiment
input_path = Path("Code/Data/Drops.csv")
raw_data = pd.read_csv(input_path, dtype=str)
data = raw_data.apply(pd.to_numeric, errors="coerce")

# Apparatus values and stated uncertainties
GRATICULE_DIVISION = 0.5e-3
PLATE_SEPARATION = 6e-3
U_PLATE_SEPARATION = 0.0
AIR_VISCOSITY = 1.83e-5
U_AIR_VISCOSITY = 0.0
OIL_DENSITY = 900.0
U_OIL_DENSITY = 0.0
AIR_PRESSURE = 1000e2
U_AIR_PRESSURE = 4e2
SLIP_CONSTANT = 8.0e-3
U_SLIP_CONSTANT = 0.0
GRAVITY = 9.81
TEMPERATURE = 273.15 + 23.4
AIR_MOLECULE_DIAMETER = 3.7e-10


def half_last_digit(value):
    if pd.isna(value) or not str(value).strip():
        return np.nan
    text = str(value).strip()
    decimal_places = len(text.partition(".")[2]) if "." in text else 0
    return 0.5 * 10 ** -decimal_places


def drop_radius(vg, pressure=AIR_PRESSURE, slip_constant=SLIP_CONSTANT):
    slip_length = slip_constant / pressure
    radius_without_slip = np.sqrt(
        9 * AIR_VISCOSITY * vg / (2 * OIL_DENSITY * GRAVITY)
    )
    return np.sqrt(radius_without_slip**2 + slip_length**2 / 4) - slip_length / 2


def drop_charge(d, voltage, eta, rho, vg, vE, pressure, slip_constant):
    radius = drop_radius(vg, pressure, slip_constant)
    correction = 1 + slip_constant / (pressure * radius)
    q_uncorrected = (
        4 * np.pi * d / (3 * voltage)
        * np.sqrt((9 * eta / 2) ** 3 / (rho * GRAVITY))
        * (vg + vE) * np.sqrt(vg)
    )
    return q_uncorrected / correction**1.5


def propagated_uncertainty(function, values, uncertainties):
    contributions = {}
    for name, value in values.items():
        step = max(abs(value) * 1e-5, 1e-12)
        plus = values.copy()
        minus = values.copy()
        plus[name] += step
        minus[name] -= step
        derivative = (function(**plus) - function(**minus)) / (2 * step)
        contributions[name] = derivative * uncertainties.get(name, 0.0)
    total = np.sqrt(sum(value**2 for value in contributions.values()))
    return total, contributions


fall_distance_divisions = data.iloc[:, 0].to_numpy(dtype=float)
fall_time = data.iloc[:, 1].to_numpy(dtype=float)
voltage = data.iloc[:, 4].to_numpy(dtype=float)
rise_distance_divisions = data.iloc[:, 5].to_numpy(dtype=float)
rise_time = data.iloc[:, 6].to_numpy(dtype=float)
fall_distance = fall_distance_divisions * GRATICULE_DIVISION
rise_distance = np.nan_to_num(rise_distance_divisions) * GRATICULE_DIVISION
fall_speed = fall_distance / fall_time
has_rise_measurement = np.isfinite(rise_time) & (rise_time > 0)
rise_speed = np.zeros_like(fall_speed)
rise_speed[has_rise_measurement] = (
    rise_distance[has_rise_measurement] / rise_time[has_rise_measurement]
)

# Use half a graticule division and half the last recorded digit for readings.
u_fall_distance_divisions = np.full_like(fall_distance, 0.5)
u_rise_distance_divisions = np.where(has_rise_measurement, 0.5, 0.0)
u_fall_distance = u_fall_distance_divisions * GRATICULE_DIVISION
u_rise_distance = u_rise_distance_divisions * GRATICULE_DIVISION
u_fall_time = np.array([half_last_digit(value) for value in raw_data.iloc[:, 1]])
u_rise_time = np.nan_to_num(
    np.array([half_last_digit(value) for value in raw_data.iloc[:, 6]])
)
u_voltage = np.array([half_last_digit(value) for value in raw_data.iloc[:, 4]])
u_fall_speed = np.sqrt(
    (u_fall_distance / fall_time) ** 2
    + (fall_distance * u_fall_time / fall_time**2) ** 2
)
u_rise_speed = np.zeros_like(rise_speed)
u_rise_speed[has_rise_measurement] = np.sqrt(
    (u_rise_distance[has_rise_measurement] / rise_time[has_rise_measurement]) ** 2
    + (
        rise_distance[has_rise_measurement]
        * u_rise_time[has_rise_measurement]
        / rise_time[has_rise_measurement] ** 2
    ) ** 2
)

radius_values = []
radius_uncertainties = []
charge_values = []
charge_uncertainties = []
charge_independent_uncertainties = []
charge_pressure_derivatives = []

for row in range(len(data)):
    radius_inputs = {
        "vg": fall_speed[row],
        "pressure": AIR_PRESSURE,
        "slip_constant": SLIP_CONSTANT,
    }
    radius_uncertainty, _ = propagated_uncertainty(
        lambda vg, pressure, slip_constant: drop_radius(vg, pressure, slip_constant),
        radius_inputs,
        {
            "vg": u_fall_speed[row],
            "pressure": U_AIR_PRESSURE,
            "slip_constant": U_SLIP_CONSTANT,
        },
    )
    radius_values.append(drop_radius(fall_speed[row]))
    radius_uncertainties.append(radius_uncertainty)

    inputs = {
        "d": PLATE_SEPARATION,
        "voltage": voltage[row],
        "eta": AIR_VISCOSITY,
        "rho": OIL_DENSITY,
        "vg": fall_speed[row],
        "vE": rise_speed[row],
        "pressure": AIR_PRESSURE,
        "slip_constant": SLIP_CONSTANT,
    }
    input_uncertainties = {
        "d": U_PLATE_SEPARATION,
        "voltage": u_voltage[row],
        "eta": U_AIR_VISCOSITY,
        "rho": U_OIL_DENSITY,
        "vg": u_fall_speed[row],
        "vE": u_rise_speed[row],
        "pressure": U_AIR_PRESSURE,
        "slip_constant": U_SLIP_CONSTANT,
    }
    charge_uncertainty, charge_contributions = propagated_uncertainty(
        drop_charge, inputs, input_uncertainties
    )
    charge_values.append(drop_charge(**inputs))
    charge_uncertainties.append(charge_uncertainty)
    charge_independent_uncertainties.append(
        np.sqrt(
            charge_contributions["voltage"] ** 2
            + charge_contributions["vg"] ** 2
            + charge_contributions["vE"] ** 2
        )
    )
    charge_pressure_derivatives.append(
        charge_contributions["pressure"] / U_AIR_PRESSURE
    )

q_apparent_input = data["q(uncorrected)"].to_numpy(dtype=float)
q_input = data["q"].to_numpy(dtype=float)
q = np.asarray(charge_values)
u_q = np.asarray(charge_uncertainties)
u_q_independent = np.asarray(charge_independent_uncertainties)
u_radius = np.asarray(radius_uncertainties)
charge_pressure_derivatives = np.asarray(charge_pressure_derivatives)

data["fall_distance_m"] = fall_distance
data["u_fall_distance_divisions"] = u_fall_distance_divisions
data["u_fall_distance_m"] = u_fall_distance
data["u_fall_time_s"] = u_fall_time
data["fall_speed_calculated_m_s"] = fall_speed
data["u_fall_speed_m_s"] = u_fall_speed
data["radius_calculated_m"] = radius_values
data["u_radius_m"] = u_radius
data["u_voltage_V"] = u_voltage
data["rise_distance_m"] = rise_distance
data["u_rise_distance_divisions"] = u_rise_distance_divisions
data["u_rise_distance_m"] = u_rise_distance
data["u_rise_time_s"] = u_rise_time
data["rise_speed_calculated_m_s"] = rise_speed
data["u_rise_speed_m_s"] = u_rise_speed
data["q_apparent_input_C"] = q_apparent_input
data["q_input_C"] = q_input
data["q_calculated_C"] = q
data["u_q_C"] = u_q
data.to_csv(input_path.with_name("Drops_with_uncertainties.csv"), index=False)

# --------------------------------------------------
# 1. Cluster the charge measurements into 3 groups
# --------------------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)

# KMeans expects a 2D array
labels = kmeans.fit_predict(q.reshape(-1, 1))

# Sort the clusters by their charge
centers = kmeans.cluster_centers_.flatten()
order = np.argsort(centers)

# Remap labels so that 0 = lowest charge, etc.
new_labels = np.zeros_like(labels)

for new_label, old_label in enumerate(order):
    new_labels[labels == old_label] = new_label

centers = np.sort(centers)

# --------------------------------------------------
# 2. Fit each horizontal line
# --------------------------------------------------

fit_values = []
fit_errors = []

for i in range(3):
    q_group = q[new_labels == i]
    n = len(q_group)

    # Best-fit horizontal line = weighted mean
    q_fit = np.mean(q_group)

    # Standard error of the mean
    q_error = np.std(q_group, ddof=1) / np.sqrt(n)

    fit_values.append(q_fit)
    fit_errors.append(q_error)

    print(
        f"Level {i+1}: "
        f"q = {q_fit:.4e} ± {q_error:.2e} C "
        f"({n} measurements)"
        f"e = {q_fit/(i+1):.4e} ± {q_error/(i+1):.2e} C "
    )

# --------------------------------------------------
# 3. Plot
# --------------------------------------------------


x = np.arange(len(q))
colors = ["#2878B5", "#F28E2B", "#5AA9E6"]

plt.figure(figsize=(10, 6))

for i in range(3):
    mask = new_labels == i

    plt.scatter(
        x[mask],
        q[mask],
        color=colors[i],
        label=f"Group {i+1}"
    )
    
    plt.errorbar(
        x[mask],
        q[mask],
        yerr=u_q[mask],
        fmt='o',
        color=colors[i],
        alpha=0.5,
        capsize=3
    )

    plt.axhline(
        fit_values[i],
        color=colors[i],
        linestyle="--",
        label=f"Fit: {fit_values[i]:.3e} C"
    )

    # 95% interval on the group mean (Student t, N-1 dof), not 1 sigma
    t95 = student_t.ppf(0.975, mask.sum() - 1)
    plt.fill_between(
        x,
        np.full_like(x, fit_values[i] - t95 * fit_errors[i], dtype=float),
        np.full_like(x, fit_values[i] + t95 * fit_errors[i], dtype=float),
        color=colors[i],
        alpha=0.15
    )

plt.xlabel("Oil drop number")
plt.ylabel("Charge q (C)")
plt.title("Millikan Oil Drop Experiment")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("Millikan_Oil_Drop_Experiment.png")

# --------------------------------------------------
# 4. Combined analysis of e
# --------------------------------------------------

E_ACCEPTED = 1.602176634e-19

n_drop = new_labels + 1
charge_covariance = np.diag(u_q_independent**2) + np.outer(
    charge_pressure_derivatives * U_AIR_PRESSURE,
    charge_pressure_derivatives * U_AIR_PRESSURE,
)
inverse_charge_covariance = np.linalg.inv(charge_covariance)

# Inverse-variance weighted mean of the per-level e values
e_levels = np.array(fit_values) / np.arange(1, 4)
e_level_err = np.array(fit_errors) / np.arange(1, 4)
w_lvl = 1 / e_level_err**2
e_wmean = np.sum(w_lvl * e_levels) / np.sum(w_lvl)
e_wmean_err = 1 / np.sqrt(np.sum(w_lvl))

# Weighted fit through the origin: q = e * n
e_fit = (n_drop @ inverse_charge_covariance @ q) / (
    n_drop @ inverse_charge_covariance @ n_drop
)
e_fit_err = 1 / np.sqrt(n_drop @ inverse_charge_covariance @ n_drop)
origin_residual = q - e_fit * n_drop
chi2_origin = origin_residual @ inverse_charge_covariance @ origin_residual
dof_origin = len(q) - 1

# Weighted fit with free intercept: q = e * n + c
A = np.vstack([n_drop, np.ones_like(n_drop)]).T
cov = np.linalg.inv(A.T @ inverse_charge_covariance @ A)
slope, intercept = cov @ (A.T @ inverse_charge_covariance @ q)
slope_err, intercept_err = np.sqrt(np.diag(cov))
free_residual = q - (slope * n_drop + intercept)
chi2_free = free_residual @ inverse_charge_covariance @ free_residual
dof_free = len(q) - 2

# Consistency between the per-level e values (chi^2 about the weighted mean)
chi2_levels = np.sum(w_lvl * (e_levels - e_wmean) ** 2)

print("\n--- Combined analysis ---")
print(f"Weighted mean of level e : {e_wmean:.4e} ± {e_wmean_err:.2e} C")
print(f"Fit through origin       : e = {e_fit:.4e} ± {e_fit_err:.2e} C, "
      f"chi2/dof = {chi2_origin/dof_origin:.2f}")
print(f"Fit with free intercept  : e = {slope:.4e} ± {slope_err:.2e} C, "
      f"c = {intercept:.2e} ± {intercept_err:.2e} C "
      f"({abs(intercept)/intercept_err:.1f} sigma from 0), "
      f"chi2/dof = {chi2_free/dof_free:.2f}")
print(f"Level consistency chi2   : {chi2_levels:.2f} (dof = 2)")
print(f"Offset from accepted e   : {(e_fit - E_ACCEPTED)/E_ACCEPTED*100:+.1f} %")

# Fit apparent charges to the Cunningham form.
k_B = 1.380649e-23

mean_free_path = (
    k_B * TEMPERATURE
    / (np.sqrt(2) * np.pi * AIR_MOLECULE_DIAMETER**2 * AIR_PRESSURE)
)
q_apparent_power_scale = 1e-13
q_apparent_power = q_apparent_input ** (2 / 3) / q_apparent_power_scale

# Assign integer charge levels from the apparent charges, ordered by charge.
apparent_labels = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(
    q_apparent_input.reshape(-1, 1)
)
cluster_order = np.argsort([
    np.mean(q_apparent_input[apparent_labels == label])
    for label in range(3)
])
csv_levels = np.zeros_like(apparent_labels)
for level, label in enumerate(cluster_order, start=1):
    csv_levels[apparent_labels == label] = level

# q_app^(2/3) = c + (c * A * lambda/r)
#
# Two fitted parameters only:
#   A = common Cunningham/slip parameter
#   c = q_1^(2/3)

# apparent charge per elementary unit: e_app = q_app
e_app_power = q_apparent_power
knudsen = mean_free_path / radius_values

# fit apparent charge powers against their corresponding radii using the Cunningham form
cunningham_fit = linregress(knudsen, e_app_power)

m_fit = abs(cunningham_fit.slope) / 10
c_fit = cunningham_fit.intercept / 10
A_fit = m_fit / c_fit / 10

# Fitted q_n^(2/3) values are now determined entirely by c.
fitted_level_charge_powers = (
    c_fit * np.arange(1, 4) ** (2 / 3)
)

fitted_level_charge_powers_physical = (
    fitted_level_charge_powers * q_apparent_power_scale
)

fitted_level_charges = (
    fitted_level_charge_powers_physical ** (3 / 2)
)

fitted_level_e = (
    fitted_level_charges / np.arange(1, 4)
)

fitted_slip_length = A_fit * mean_free_path

q_corrected_fit = (
    q_apparent_input
    / (
        1
        + A_fit * knudsen
    ) ** 1.5
)

data["q_corrected_fit_C"] = q_corrected_fit
data["n_corrected_fit"] = csv_levels
n_values = csv_levels

data.to_csv(
    input_path.with_name("Drops_with_uncertainties.csv"),
    index=False,
)


level_e_values = []
level_e_errors = []

for level in range(1, 4):

    e_samples = q_corrected_fit[csv_levels == level] / level

    level_e_values.append(np.mean(e_samples))

    level_e_errors.append(
        np.std(e_samples, ddof=1)
        / np.sqrt(len(e_samples))
    )

level_weights = 1 / np.asarray(level_e_errors) ** 2

e1_fit = (
    np.sum(level_weights * level_e_values)
    / np.sum(level_weights)
)

e1_fit_error = 1 / np.sqrt(np.sum(level_weights))


print("\n--- Apparent-charge correction fit ---")

print(
    f"Mean free path          : "
    f"{mean_free_path:.4e} m"
)

print(
    f"Fitted Cunningham A     : "
    f"{A_fit:.6f}"
)

print(
    f"Fitted c = q_1^(2/3)   : "
    f"{c_fit * q_apparent_power_scale:.4e} C^(2/3)"
)

print(
    f"A * lambda              : "
    f"{fitted_slip_length:.4e} m"
)

# charge from slope
print(
    f"Fitted q from slope       : "
    f"e = {c_fit**(3/2):.4e} C "
)


# Analytic A: the slip length is b/p = A*lambda, so A = b/(p*lambda).
# lambda = kT/(sqrt(2)*pi*d^2*p), so p cancels: A = sqrt(2)*pi*d^2*b/(kT).
U_TEMPERATURE = 0.1
A_analytic = (
    np.sqrt(2) * np.pi * AIR_MOLECULE_DIAMETER**2 * SLIP_CONSTANT
    / (k_B * TEMPERATURE)
)
u_A_analytic = A_analytic * U_TEMPERATURE / TEMPERATURE
analytic_slip_length = A_analytic * mean_free_path

print("\n--- Analytic Cunningham A ---")
print(f"A = b/(p*lambda)        : {A_analytic:.4f} ± {u_A_analytic:.4f}")
print(f"Slip length b/p         : {analytic_slip_length:.4e} m")
print(f"Fitted A / analytic A   : {A_fit / A_analytic:.3f}")

# --------------------------------------------------
# 5. Plots for the combined analysis
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# q vs n with free intercept fit and 95% confidence interval
ax = axes[0]
n_line = np.linspace(0, 3.5, 100)
for i in range(3):
    mask = new_labels == i
    ax.errorbar(n_drop[mask] + np.random.default_rng(i).uniform(-0.08, 0.08, mask.sum()),
                q[mask], yerr=u_q[mask], fmt="o", color=colors[i], alpha=0.6, capsize=3)
ax.plot(n_line, slope * n_line + intercept, "r:", label=f"Free intercept: m = {slope:.1e} ± {slope_err:.1e} C, c = {intercept:.1e} C")
# 95% band on the fitted line: full slope/intercept covariance, inflated by
# chi2/dof when the scatter exceeds the quoted errors, with a Student-t factor
design_line = np.vstack([n_line, np.ones_like(n_line)]).T
line_sigma = np.sqrt(np.einsum("ij,jk,ik->i", design_line, cov, design_line))
scale_free = np.sqrt(max(chi2_free / dof_free, 1.0))
band = student_t.ppf(0.975, dof_free) * scale_free * line_sigma
ax.fill_between(n_line, slope * n_line + intercept - band, slope * n_line + intercept + band,
                color="r", alpha=0.15, label="95% CI (fitted line)")
ax.set_xlabel("n")
ax.set_ylabel("Charge q (C)")
ax.set_xticks([1, 2, 3])
ax.set_xlim(0, 3.5)
ax.set_title("q vs n")
ax.legend()

# q/n should be flat
ax = axes[1]
for i in range(3):
    mask = new_labels == i
    ax.errorbar(n_drop[mask] + np.random.default_rng(i).uniform(-0.08, 0.08, mask.sum()),
                q[mask] / n_drop[mask], yerr=u_q[mask] / n_drop[mask],
                fmt="o", color=colors[i], alpha=0.6, capsize=3)
ax.axhline(e_fit, color="k", linestyle="--", label=f"Fit e = {e_fit:.3e} C")
ax.axhspan(e_fit - e_fit_err, e_fit + e_fit_err, color="k", alpha=0.15)
ax.axhline(E_ACCEPTED, color="g", linestyle="-", label=f"Accepted e = {E_ACCEPTED:.3e} C")
ax.set_xlabel("n")
ax.set_ylabel("q / n (C)")
ax.set_xticks([1, 2, 3])
ax.set_title("q / n vs n")
ax.legend()

# Histogram of all charges
ax = axes[2]
ax.hist(q, bins=20, color="#2878B5", alpha=0.8, edgecolor="white")
for k in range(1, 4):
    ax.axvline(k * e_fit, color="k", linestyle="--", alpha=0.7,
               label="n·e (fit)" if k == 1 else None)
    ax.axvline(k * E_ACCEPTED, color="g", linestyle="-", alpha=0.7,
               label="n·e (accepted)" if k == 1 else None)
ax.set_xlabel("Charge q (C)")
ax.set_ylabel("Count")
ax.set_title("Charge histogram")
ax.legend()

plt.tight_layout()
plt.savefig("Millikan_Combined_Analysis.png")

# --------------------------------------------------
# 6. Residuals and q_app^(2/3) against lambda/r (for the report)
# --------------------------------------------------

REPORT_IMAGES = Path("Reports/Report 3 - Milikan/images")
level_colors = ["#2878B5", "#F28E2B", "#5AA9E6"]

# Residuals of the weighted q = e*n fit through the origin
fig, ax = plt.subplots(figsize=(8, 5))
for level in range(1, 4):
    mask = n_drop == level
    jitter = np.random.default_rng(level).uniform(-0.08, 0.08, mask.sum())
    ax.errorbar(n_drop[mask] + jitter, origin_residual[mask] * 1e19,
                yerr=u_q[mask] * 1e19, fmt="o", color=level_colors[level - 1],
                alpha=0.7, capsize=3, label=f"n = {level}")
ax.axhline(0, color="k", linestyle="--")
ax.set_xlabel("n")
ax.set_ylabel(r"Residual $q - e_{fit}n$ ($10^{-19}$ C)")
ax.set_xticks([1, 2, 3])
ax.set_title(r"Residuals of the $q = en$ fit")
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_IMAGES / "Millikan_Residuals.png", dpi=200)

# e_app^(2/3) = c + (c*A)*lambda/r with c = e1^(2/3): one line through all drops
fit_values = 10*c_fit + (10*c_fit * -10*A_fit * knudsen)
averages = [np.mean(e_app_power[csv_levels == level]) for level in range(1, 4)]
cunningham_res = (e_app_power - np.array([averages[level - 1] for level in csv_levels]))

fig, (ax, ax_res) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
x_line = np.linspace(0, knudsen.max() * 1.05, 100)
for level in range(1, 4):
    mask = csv_levels == level
    ax.scatter(knudsen[mask], e_app_power[mask],
               color=level_colors[level - 1], label=f"n = {level}")
    ax_res.scatter(knudsen[mask], cunningham_res[mask],
                   color=level_colors[level - 1])
ax.plot(x_line, 10*c_fit + (10*c_fit * -10*A_fit * x_line), color="k", linestyle="--",
        label=rf"fit: $c + m\lambda/r$, $A=m/c$ = {A_fit:.2f}")
ax.set_ylabel(r"$e_{app}^{2/3}$ ($10^{-13}$ C$^{2/3}$)")
ax.set_title(r"$e_{app}^{2/3}$ against $\lambda/r$ (single fit, all levels)")
ax.legend()
ax_res.axhline(0, color="k", linestyle="--")
ax_res.set_xlabel(r"$\lambda / r$")
ax_res.set_ylabel("Residual")
fig.tight_layout()
fig.savefig(REPORT_IMAGES / "Millikan_qapp_vs_lambda_over_r.png", dpi=200)

# --------------------------------------------------
# 7. Uncertainty on fitted A, analytic-A correction and Avogadro's number
# --------------------------------------------------

# Straight line e_app^(2/3) = c + m*lambda/r; parameters (c, m), A = m/c
design = np.column_stack([np.ones_like(knudsen), knudsen])
residuals_lin = e_app_power - (10*c_fit + 10*m_fit * knudsen)
residual_variance = np.sum(residuals_lin**2) / (len(knudsen) - 2)
cov_cm = np.linalg.inv(design.T @ design) * residual_variance
# Propagate to A = m/c
grad_A = np.array([-m_fit / c_fit**2, 1 / c_fit])
u_A_fit = np.sqrt(grad_A @ cov_cm @ grad_A)
u_c_fit = np.sqrt(cov_cm[0, 0])
print(f"Uncertainties\nc = {c_fit*q_apparent_power_scale:.4e} +/- {u_c_fit*q_apparent_power_scale:.2e},\n slope m = cA = {c_fit*A_fit:.3f} +/- {u_A_fit*c_fit:.3f}")

# e1 straight from the fit: c = e1^(2/3)
e1_fit = (c_fit * q_apparent_power_scale) ** 1.5
e1_fit_error = 1.5 * e1_fit * u_c_fit / c_fit

# radius values without slip
radius_without_slip = np.sqrt(
    9 * AIR_VISCOSITY * fall_speed
    / (2 * OIL_DENSITY * GRAVITY)
)

radius_analytic = np.sqrt(
    radius_without_slip**2 + (analytic_slip_length / 2) ** 2
) - analytic_slip_length / 2
q_corrected_analytic = q_apparent_input / (
    1 + analytic_slip_length / radius_analytic
) ** 1.5
levels = csv_levels
print("\n--- Analytic-A correction ---")
print(f"A_fit = {A_fit:.4f} ± {u_A_fit:.4f}, "
      f"(A_fit - A_analytic)/u = {(A_fit - A_analytic)/u_A_fit:.2f} sigma")
for level in range(1, 4):
    e_samples = q_corrected_analytic[levels == level] / level
    print(f"Level {level}: N = {len(e_samples)}, e = {np.mean(e_samples):.4e} ± "
          f"{np.std(e_samples, ddof=1)/np.sqrt(len(e_samples)):.2e} C")
e1_analytic = (levels @ q_corrected_analytic) / (levels @ levels)
e1_analytic_err = np.sqrt(
    np.sum((q_corrected_analytic - levels * e1_analytic) ** 2)
    / (len(q) - 1) / (levels @ levels)
)
print(f"e1 (A analytic, LSQ q=ne): {e1_analytic:.4e} ± {e1_analytic_err:.2e} C, "
      f"{(e1_analytic - E_ACCEPTED)/E_ACCEPTED*100:+.1f} %")
print(f"e1 (A fitted)            : {e1_fit:.4e} ± {e1_fit_error:.2e} C, "
      f"{(e1_fit - E_ACCEPTED)/E_ACCEPTED*100:+.1f} %")

# N_A = n * V_molar with n = 1/(sqrt(2) pi d^2 lambda), lambda = b/(p A), V_molar = RT/p.
R_GAS = 8.314462618

def avogadro(A_value):
    return A_value * R_GAS * TEMPERATURE / (
        np.sqrt(2) * np.pi * AIR_MOLECULE_DIAMETER**2 * SLIP_CONSTANT
    )

N_A_fit = avogadro(A_fit)
u_N_A_fit = N_A_fit * u_A_fit / A_fit
N_A_analytic = avogadro(A_analytic)
N_A_ACCEPTED = 6.02214076e23
print("\n--- Avogadro's number from A ---")
print(f"lambda from fitted A     : {SLIP_CONSTANT/(AIR_PRESSURE*A_fit):.4e} m")
print(f"number density n         : {A_fit*AIR_PRESSURE/(np.sqrt(2)*np.pi*AIR_MOLECULE_DIAMETER**2*SLIP_CONSTANT):.4e} m^-3")
print(f"N_A (fitted A)           : {N_A_fit:.4e} ± {u_N_A_fit:.2e} /mol "
      f"({(N_A_fit - N_A_ACCEPTED)/N_A_ACCEPTED*100:+.1f} %)")
print(f"N_A (analytic A, check)  : {N_A_analytic:.4e} /mol")
print("lambda/r range", knudsen.min(), knudsen.max())
print("within-level scatter %", [np.std(q_apparent_power[csv_levels==l],ddof=1)/np.mean(q_apparent_power[csv_levels==l])*100 for l in (1,2,3)])
print("counts", [(csv_levels==l).sum() for l in (1,2,3)], [(new_labels==i).sum() for i in range(3)])
with open("table_rows.txt","w") as fh:
    for i in range(len(q)):
        n_i = new_labels[i] + 1
        fh.write(
            f"        {i+1} / {n_i} & ${fall_speed[i]*1e5:.3f}\\pm{u_fall_speed[i]*1e5:.3f}$ & ${radius_values[i]*1e7:.3f}\\pm{u_radius[i]*1e7:.3f}$ & ${voltage[i]:.0f}\\pm{u_voltage[i]:.1f}$ & ${rise_speed[i]*1e5:.3f}\\pm{u_rise_speed[i]*1e5:.3f}$ & ${q[i]*1e19:.3f}\\pm{u_q[i]*1e19:.3f}$ & ${q[i]/n_i*1e19:.3f}\\pm{u_q[i]/n_i*1e19:.3f}$ \\\\\n"
        )
plt.show()

