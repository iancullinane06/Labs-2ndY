from math import cos, isfinite, sqrt
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from labplotlib import style_axis, finish_plot, symbols, apply_theme

apply_theme()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = ROOT / "Code" / "Data"
PLOT_PATH = ROOT / "Reports" / "Report 7 - Inertia" / "images" / "inertia_analysis.png"

# Errors associated with disk-and-axle measurements
u_diameter = 0.000005 # Example uncertainty in diameter
diameter = 0.00808    # Axle diameter
u_t = 0.002           # Uncertainty in time
t = 8.198888889       # Mean measured time
alpha = 0.244978663   # Incline angle
u_l = 0.0005          # Uncertainty in length
l = 1.212             # Length of the incline

DISK_THEORETICAL_INERTIA = {
    "Wood": 0.002094229,
    "Metal": 0.001996254,
}

def _sample_summary(values: pd.Series) -> tuple[int, float, float, float]:
    count = int(values.count())
    if count < 2:
        raise ValueError("At least two measurements are required for sample statistics.")

    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = standard_deviation / sqrt(count)
    return count, mean, standard_deviation, standard_error


def analyse_inertia(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated disk-and-axle measurements and compare with theory."""
    numeric_data = data.apply(pd.to_numeric, errors="coerce")
    measurements = numeric_data.dropna(subset=["I"])
    expected_count = 2 * 9
    if len(measurements) != expected_count:
        raise ValueError(
            f"Expected {expected_count} disk-and-axle measurements, "
            f"found {len(measurements)}."
        )

    results = []
    for material, group in zip(
        DISK_THEORETICAL_INERTIA,
        (measurements.iloc[:9], measurements.iloc[9:]),
    ):
        count, mean, standard_deviation, standard_error = _sample_summary(group["I"])
        error = mean * ((standard_error/mean)**2 + (u_diameter/diameter)**2 + (u_t/t)**2 + alpha*(u_l/l)**2)**0.5
        theoretical = DISK_THEORETICAL_INERTIA[material]
        difference = mean - theoretical
        results.append(
            {
                "material": material,
                "n": count,
                "mean_time_s": float(group["t"].mean()),
                "mean_inertia": mean,
                "standard_deviation": standard_deviation,
                "error": error,
                "theoretical_inertia": theoretical,
                "difference": difference,
                "percent_difference": 100 * difference / theoretical,
                "z_by_experimental_sem": difference / standard_error,
            }
        )

    return pd.DataFrame(results).set_index("material")


def analyse_inertia_balls(data: pd.DataFrame) -> pd.DataFrame:
    """Compare each ball's predicted inertia with its five calculated values."""
    numeric_data = data.apply(pd.to_numeric, errors="coerce")
    if len(numeric_data.index) != 8 or len(numeric_data.columns) != 5:
        raise ValueError(
            "Expected five ball columns, one prediction row, five calculated "
            "value rows, and two uncertainty rows."
        )

    predicted_values = numeric_data.iloc[0]
    calculated_values = numeric_data.iloc[1:6]
    calculated_uncertainties = numeric_data.iloc[6]
    calculated_relative_uncertainties = numeric_data.iloc[7]
    results = []
    for ball in numeric_data.columns:
        values = calculated_values[ball].dropna()
        if len(values) != 5:
            raise ValueError(
                f"Expected five calculated values for ball {ball}, found {len(values)}."
            )
        count, mean, standard_deviation, standard_error = _sample_summary(values)
        predicted = float(predicted_values[ball])
        calculated_uncertainty = float(calculated_uncertainties[ball])
        relative_uncertainty = float(calculated_relative_uncertainties[ball])
        if not all(
            isfinite(value)
            for value in (predicted, calculated_uncertainty, relative_uncertainty)
        ):
            raise ValueError(f"Prediction or uncertainty for ball {ball} is not finite.")
        if calculated_uncertainty <= 0:
            raise ValueError(f"Calculated uncertainty for ball {ball} must be positive.")
        difference = mean - predicted
        normalized_difference = difference / calculated_uncertainty
        results.append(
            {
                "ball": ball,
                "n": count,
                "mean_inertia": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "calculated_uncertainty": calculated_uncertainty,
                "calculated_relative_uncertainty": relative_uncertainty,
                "predicted_inertia": predicted,
                "difference": difference,
                "percent_difference": 100 * difference / predicted,
                "difference_by_calculated_uncertainty": normalized_difference,
                "comparison": (
                    "within 2 calculated uncertainties"
                    if abs(normalized_difference) <= 2
                    else "more than 2 calculated uncertainties"
                ),
            }
        )

    return pd.DataFrame(results).set_index("ball")


def plot_results(
    disk_results: pd.DataFrame,
    ball_results: pd.DataFrame,
    output_path: Path = PLOT_PATH,
) -> None:
    """Plot trial values and measured/theoretical means for both experiments."""
    raw_disk_data = pd.read_csv(DATA_DIRECTORY / "Intertia.csv")
    disk_measurements = raw_disk_data.apply(pd.to_numeric, errors="coerce").dropna(
        subset=["I"]
    )
    ball_data = pd.read_csv(DATA_DIRECTORY / "Inertia_ball.csv")
    ball_numeric_data = ball_data.apply(pd.to_numeric, errors="coerce")
    ball_trials = ball_numeric_data.iloc[1:6]

    figure, (disk_axis, ball_axis) = plt.subplots(1, 2, figsize=(12, 5))

    disk_positions = range(len(disk_results))
    for position, (_, result) in enumerate(disk_results.iterrows()):
        trials = disk_measurements.iloc[position * 9 : (position + 1) * 9]["I"]
        offsets = [
            position + (index - (len(trials) - 1) / 2) * 0.035
            for index in range(len(trials))
        ]
        disk_axis.scatter(offsets, trials, alpha=0.65, label="Individual trials" if position == 0 else None)
        disk_axis.errorbar(
            position,
            result["mean_inertia"],
            yerr=2*result["error"],
            fmt="o",
            capsize=4,
            color="tab:blue",
            label=f"Mean {symbols['pm']} 2 SEM" if position == 0 else None,
        )
        disk_axis.scatter(
            position,
            result["theoretical_inertia"],
            marker="D",
            color="black",
            label="Theoretical value" if position == 0 else None,
        )

    disk_axis.set_xticks(list(disk_positions), list(disk_results.index))
    style_axis(
        disk_axis,
        "Disk and axle",
        "Material",
        r"Moment of inertia (kg m$^2$)",
    )
    disk_axis.legend(fontsize="small")

    ball_positions = range(len(ball_results))
    for position, (ball, result) in enumerate(ball_results.iterrows()):
        trials = ball_trials[ball].dropna()
        offsets = [
            position + (index - (len(trials) - 1) / 2) * 0.035
            for index in range(len(trials))
        ]
        ball_axis.scatter(
            offsets,
            trials,
            alpha=0.65,
            label="Individual trials" if position == 0 else None,
        )
        ball_axis.errorbar(
            position,
            result["mean_inertia"],
            yerr=2*result["standard_error"],
            fmt="o",
            capsize=4,
            color="tab:blue",
            label=f"Calculated mean {symbols['pm']} 2 SEM" if position == 0 else None,
        )
        ball_axis.scatter(
            position,
            result["predicted_inertia"],
            marker="D",
            color="black",
            label="Predicted value" if position == 0 else None,
        )

    ball_axis.set_xticks(list(ball_positions), [f"Ball {ball}" for ball in ball_results.index])
    ball_axis.set_yscale("log")
    style_axis(
        ball_axis,
        "Ball bearings",
        "Ball",
        r"Moment of inertia (kg m$^2$, log scale)",
    )
    ball_axis.legend(fontsize="small")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    finish_plot(figure, output_path.name, output_path.parent)
    plt.close(figure)


def _print_results(title: str, results: pd.DataFrame, columns: dict[str, str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for index, result in results.iterrows():
        print(f"{index}:")
        for column, label in columns.items():
            value = result[column]
            if isinstance(value, float):
                print(f"  {label}: {value:.6g}")
            else:
                print(f"  {label}: {value}")


if __name__ == "__main__":
    disk_data = pd.read_csv(DATA_DIRECTORY / "Intertia.csv")
    ball_data = pd.read_csv(DATA_DIRECTORY / "Inertia_ball.csv")

    disk_results = analyse_inertia(disk_data)
    ball_results = analyse_inertia_balls(ball_data)
    inertia_unit = f"kg m{symbols['squared']}"

    _print_results(
        "Disk and axle analysis",
        disk_results,
        {
            "n": "number of trials",
            "mean_time_s": "mean travel time (s)",
            "mean_inertia": f"mean experimental inertia ({inertia_unit})",
            "standard_deviation": (
                f"sample standard deviation, {symbols['sigma']} ({inertia_unit})"
            ),
            "error": f"combined error of the mean ({inertia_unit})",
            "theoretical_inertia": f"theoretical inertia ({inertia_unit})",
            "difference": f"experimental minus theoretical ({inertia_unit})",
            "percent_difference": "percent difference",
            "z_by_experimental_sem": "difference / experimental standard error",
        },
    )
    print(
        "Note: disk-and-axle normalised differences use experimental standard "
        "errors only; theoretical uncertainties are not included."
    )

    _print_results(
        "Ball-bearing analysis",
        ball_results,
        {
            "n": "number of trials",
            "mean_inertia": f"mean experimental inertia ({inertia_unit})",
            "standard_deviation": (
                f"sample standard deviation, {symbols['sigma']} ({inertia_unit})"
            ),
            "standard_error": f"standard error of the mean ({inertia_unit})",
            "calculated_uncertainty": f"calculated uncertainty ({inertia_unit})",
            "calculated_relative_uncertainty": "calculated relative uncertainty",
            "predicted_inertia": f"predicted inertia ({inertia_unit})",
            "difference": f"calculated mean minus prediction ({inertia_unit})",
            "percent_difference": "percent difference",
            "difference_by_calculated_uncertainty": (
                "difference / calculated uncertainty"
            ),
            "comparison": "comparison against calculated uncertainty",
        },
    )

    plot_results(disk_results, ball_results)
    print(f"\nComparison plot saved to: {PLOT_PATH}")
