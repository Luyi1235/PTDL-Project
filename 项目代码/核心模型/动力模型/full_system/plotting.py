"""Plots for the complete Python hybrid-system results."""

from pathlib import Path


def plot_results(rows, output_directory, mode):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib; run with --no-plots or install with: python -m pip install matplotlib") from exc
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    stride = max(1, len(rows)//20000)
    data = rows[::stride]
    if data[-1] is not rows[-1]:
        data.append(rows[-1])
    time = [row["Time_s"] for row in data]
    solver = rows[0].get("Solver", "mission_fast")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(time, [row["Pengine_actual_W"] for row in data], label="Engine output")
    ax.plot(time, [row["Pbattery_W"] for row in data], label="Battery output (+ discharge)")
    ax.plot(time, [row["Pmotor_ac_actual_W"] for row in data], label="Motor received (AC)")
    ax.plot(time, [row["Pmotor_demand_table_W"] for row in data], label="Motor demand input")
    ax.set(xlabel="Time (s)", ylabel="Power (W)", title=f"Hybrid system mode {mode} ({solver}): power")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    power_path = output_directory/f"mode{mode}_power.png"
    fig.savefig(power_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time, [row["SOC"] for row in data])
    axes[0].set_ylabel("SOC")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time, [row["FuelFlow_kgps"] for row in data])
    axes[1].set(xlabel="Time (s)", ylabel="Fuel flow (kg/s)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Hybrid system mode {mode} ({solver}): battery and fuel")
    fig.tight_layout()
    state_path = output_directory/f"mode{mode}_soc_fuel.png"
    fig.savefig(state_path, dpi=160)
    plt.close(fig)
    return [power_path, state_path]
