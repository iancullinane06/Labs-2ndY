"""
3D natural convection from a cooling vertical calorimeter.
Finite-difference incompressible flow with a pressure-projection step, explicit
temperature advection/diffusion, and a lumped calorimeter energy balance.

All quantities are in grid-based units unless a physical conversion is supplied.
The outer box is closed/no-slip for momentum and thermally insulated. The
calorimeter is a stationary no-slip solid with a uniform lumped temperature.
This is a research/teaching prototype, not a validated engineering CFD solver.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class FluidSimulation:
    """Explicit pressure-projection solver in grid-based units.

    ``dt`` advances flow, thermal transport, and calorimeter cooling together.
    Grid spacing is one unit, so choose ``dt`` to satisfy the explicit
    diffusion and advection stability limits.
    """

    def __init__(
        self, nx=48, ny=48, nz=96, dt=0.1,
        kinematic_viscosity=0.1, thermal_diffusivity=0.03, buoyancy=1e-4,
        heat_capacity=2e4, calorimeter_radius=None,
        calorimeter_height=None, initial_temperature=1.0,
        ambient_temperature=0.0, seed=42, temperature_noise=0.0,
        pressure_tolerance=1e-5, pressure_max_iterations=500,
    ):
        if min(nx, ny, nz) < 8:
            raise ValueError("Each grid dimension must be at least 8.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if kinematic_viscosity <= 0:
            raise ValueError("kinematic_viscosity must be positive.")
        if thermal_diffusivity <= 0:
            raise ValueError("thermal_diffusivity must be positive.")
        if heat_capacity <= 0:
            raise ValueError("heat_capacity must be positive.")
        if pressure_tolerance <= 0 or pressure_max_iterations < 1:
            raise ValueError("Pressure solver tolerance and iteration limit must be positive.")

        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.dt = float(dt)
        self.nu = float(kinematic_viscosity)
        self.alpha = float(thermal_diffusivity)
        self.pressure_tolerance = float(pressure_tolerance)
        self.pressure_max_iterations = int(pressure_max_iterations)
        self.buoyancy = float(buoyancy)
        self.heat_capacity = float(heat_capacity)
        self.T_inf = float(ambient_temperature)
        self.T_cal = float(initial_temperature)
        self.radius = int(
            max(nx // 6, 3) if calorimeter_radius is None else calorimeter_radius
        )
        self.height = int(nz // 3 if calorimeter_height is None else calorimeter_height)
        if self.radius < 1:
            raise ValueError("calorimeter_radius must be at least 1 grid cell.")
        if self.radius + 2 >= min(nx, ny) / 2:
            raise ValueError("Cylinder is too close to a side wall.")
        if self.height < 2 or self.height >= nz - 2:
            raise ValueError("calorimeter_height must fit inside the box.")

        self.calorimeter, self.solid = self._build_geometry()
        self.fluid = ~self.solid
        self.surface_area = self._surface_area()
        self._fluid_degree = self._build_fluid_degree()

        shape = (self.nx, self.ny, self.nz)
        self.u = np.zeros((3, *shape), dtype=float)
        rng = np.random.default_rng(seed)
        self.T = self.T_inf + temperature_noise * rng.standard_normal(shape)
        self.T[self.solid] = self.T_inf
        self.T[self.calorimeter] = self.T_cal

        self.steps = [0]
        self.times = [0.0]
        self.temperature_history = [self.T_cal]
        self.heat_flux_history = [0.0]
        self.energy_history = [self.total_thermal_energy()]
        self.snapshots = []
        self._sliders = []

    @property
    def kinematic_viscosity(self):
        return self.nu

    @property
    def prandtl_number(self):
        return self.kinematic_viscosity / self.alpha

    def _build_geometry(self):
        x, y = np.mgrid[0:self.nx, 0:self.ny]
        cx, cy = (self.nx - 1) / 2.0, (self.ny - 1) / 2.0
        footprint = (x-cx)**2 + (y-cy)**2 <= self.radius**2
        cal = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        cal[:, :, 1:self.height+1] = footprint[:, :, None]
        solid = cal.copy()
        # Outer shell is stationary and impermeable.
        solid[0, :, :] = solid[-1, :, :] = True
        solid[:, 0, :] = solid[:, -1, :] = True
        solid[:, :, 0] = solid[:, :, -1] = True
        return cal, solid

    def _surface_area(self):
        """Number of exposed six-neighbour calorimeter/fluid faces."""
        fluid = self.fluid
        area = 0
        for axis in range(3):
            for s in (-1, 1):
                area += int((self.calorimeter &
                             np.roll(fluid, s, axis=axis)).sum())
        return area

    def _build_fluid_degree(self):
        degree = np.zeros(self.solid.shape, dtype=float)
        for axis in range(3):
            degree += np.roll(self.fluid, 1, axis=axis)
            degree += np.roll(self.fluid, -1, axis=axis)
        return degree

    def _positive_pressure_laplacian(self, pressure):
        result = self._fluid_degree * pressure
        for axis in range(3):
            for shift in (-1, 1):
                neighbour_fluid = np.roll(self.fluid, shift, axis=axis)
                neighbour_pressure = np.roll(pressure, shift, axis=axis)
                result -= np.where(neighbour_fluid, neighbour_pressure, 0.0)
        result[~self.fluid] = 0.0
        return result

    def _solve_pressure(self, rhs):
        """Solve the Neumann pressure Poisson equation with preconditioned CG."""
        rhs = np.where(self.fluid, rhs, 0.0)
        rhs[self.fluid] -= rhs[self.fluid].mean()
        rhs_norm = float(np.linalg.norm(rhs[self.fluid]))
        if rhs_norm == 0.0:
            return np.zeros_like(rhs)

        pressure = np.zeros_like(rhs)
        residual = rhs.copy()
        degree = np.maximum(self._fluid_degree, 1.0)
        preconditioned = residual / degree
        direction = preconditioned.copy()
        residual_product = float(np.sum(residual[self.fluid] *
                                        preconditioned[self.fluid]))

        for _ in range(self.pressure_max_iterations):
            applied = self._positive_pressure_laplacian(direction)
            denominator = float(np.sum(direction[self.fluid] *
                                       applied[self.fluid]))
            if denominator <= 0.0 or not np.isfinite(denominator):
                raise RuntimeError("Pressure solve encountered a non-positive search direction.")
            step_size = residual_product / denominator
            pressure += step_size * direction
            residual -= step_size * applied
            residual_norm = float(np.linalg.norm(residual[self.fluid]))
            if residual_norm <= self.pressure_tolerance * rhs_norm:
                pressure[self.fluid] -= pressure[self.fluid].mean()
                return pressure

            preconditioned = residual / degree
            next_product = float(np.sum(residual[self.fluid] *
                                        preconditioned[self.fluid]))
            direction = preconditioned + (
                next_product / residual_product
            ) * direction
            residual_product = next_product

        raise RuntimeError(
            f"Pressure solve did not converge in {self.pressure_max_iterations} "
            f"iterations (relative residual {residual_norm / rhs_norm:.3g})."
        )

    def _project_velocity(self, velocity, dt):
        """Project predicted cell velocities onto a discretely divergence-free field."""
        faces = np.zeros_like(velocity)
        for axis in range(3):
            neighbour_fluid = np.roll(self.fluid, -1, axis=axis)
            face_fluid = self.fluid & neighbour_fluid
            faces[axis] = 0.5 * (
                velocity[axis] + np.roll(velocity[axis], -1, axis=axis)
            )
            faces[axis] = np.where(face_fluid, faces[axis], 0.0)

        divergence = np.zeros(self.solid.shape, dtype=float)
        for axis in range(3):
            divergence += faces[axis] - np.roll(faces[axis], 1, axis=axis)
        pressure = self._solve_pressure(-divergence / dt)

        projected = np.zeros_like(faces)
        for axis in range(3):
            neighbour_fluid = np.roll(self.fluid, -1, axis=axis)
            neighbour_pressure = np.roll(pressure, -1, axis=axis)
            face_fluid = self.fluid & neighbour_fluid
            projected[axis] = faces[axis] - dt * (
                neighbour_pressure - pressure
            )
            projected[axis] = np.where(face_fluid, projected[axis], 0.0)

        cell_velocity = np.zeros_like(velocity)
        for axis in range(3):
            cell_velocity[axis] = 0.5 * (
                projected[axis] + np.roll(projected[axis], 1, axis=axis)
            )
        cell_velocity[:, self.solid] = 0.0
        return cell_velocity

    def _flow_step(self):
        velocity = self.u
        predicted = velocity.copy()
        for component in range(3):
            center = velocity[component]
            advection = np.zeros_like(center)
            laplacian = np.zeros_like(center)
            for axis in range(3):
                minus = np.roll(center, 1, axis=axis)
                plus = np.roll(center, -1, axis=axis)
                vel = velocity[axis]
                grad = np.where(vel >= 0.0, center-minus, plus-center)
                advection += vel * grad
                laplacian += minus + plus - 2.0 * center
            acceleration = self.buoyancy * (self.T - self.T_inf) if component == 2 else 0.0
            predicted[component] += self.dt * (
                -advection + self.nu * laplacian + acceleration
            )
        predicted[:, self.solid] = 0.0
        self.u = self._project_velocity(predicted, self.dt)

    def _thermal_step(self):
        """
        Explicit upwind advection plus face-based diffusion over one dt.
        Calorimeter/fluid faces exchange heat with the same flux used to cool
        the lumped solid. Other solid faces are adiabatic.
        """
        T0 = self.T
        fluid = self.fluid
        lap = np.zeros_like(T0)

        # Build face-neighbour values. Calorimeter is held at the old T_cal;
        # non-calorimeter solids are adiabatic (zero normal thermal flux).
        for axis in range(3):
            for shift in (-1, 1):
                Tn = np.roll(T0, shift=shift, axis=axis)
                solid_n = np.roll(self.solid, shift=shift, axis=axis)
                cal_n = np.roll(self.calorimeter, shift=shift, axis=axis)
                Tn = np.where(cal_n, self.T_cal, Tn)
                Tn = np.where(solid_n & ~cal_n, T0, Tn)
                lap += Tn - T0

        # Advection: first-order upwind gradients. Solid nodes have u=0.
        adv = np.zeros_like(T0)
        for axis in range(3):
            vel = self.u[axis]
            Tm = np.roll(T0, 1, axis=axis)
            Tp = np.roll(T0, -1, axis=axis)
            # Do not advect across a solid boundary.
            sm = np.roll(self.solid, 1, axis=axis)
            sp = np.roll(self.solid, -1, axis=axis)
            cm = np.roll(self.calorimeter, 1, axis=axis)
            cp = np.roll(self.calorimeter, -1, axis=axis)
            Tm = np.where(cm, self.T_cal, Tm)
            Tp = np.where(cp, self.T_cal, Tp)
            Tm = np.where(sm & ~cm, T0, Tm)
            Tp = np.where(sp & ~cp, T0, Tp)
            grad = np.where(vel >= 0.0, T0-Tm, Tp-T0)
            adv += vel * grad

        Tnew = T0 + self.dt * (-adv + self.alpha*lap)
        Tnew[~fluid] = T0[~fluid]
        Tnew[self.calorimeter] = self.T_cal
        self.T = Tnew

        # Heat transfer rate uses the same old-time-level face differences as
        # the diffusion update; the calorimeter changes by rate * dt / capacity.
        heat_rate = 0.0
        for axis in range(3):
            for shift in (-1, 1):
                fluid_neighbour = np.roll(fluid, shift=shift, axis=axis)
                faces = self.calorimeter & fluid_neighbour
                Tn = np.roll(T0, shift=shift, axis=axis)
                heat_rate += self.alpha * np.sum((self.T_cal - Tn)[faces])
        self.T_cal -= self.dt * heat_rate / self.heat_capacity
        self.T[self.calorimeter] = self.T_cal
        return float(heat_rate)

    def step(self):
        if self.dt * self.nu > 1.0 / 6.0:
            raise RuntimeError(
                f"Explicit momentum diffusion limit exceeded: dt*nu="
                f"{self.dt * self.nu:.4g} > 1/6."
            )
        if self.dt * self.alpha > 1.0 / 6.0:
            raise RuntimeError(
                f"Explicit thermal diffusion limit exceeded: dt*alpha="
                f"{self.dt * self.alpha:.4g} > 1/6."
            )

        self._flow_step()
        umax = float(np.linalg.norm(self.u, axis=0).max())
        if not np.isfinite(umax):
            raise RuntimeError(
                f"Flow became non-finite: u_max={umax:.4g}."
            )
        flow_courant = self.dt * float(np.max(
            np.sum(np.abs(self.u), axis=0)
        ))
        thermal_courant = flow_courant
        if thermal_courant + 6 * self.dt * self.alpha > 1.0:
            raise RuntimeError(
                f"Explicit thermal stability limit exceeded: "
                f"CFL+6*dt*alpha="
                f"{thermal_courant + 6*self.dt*self.alpha:.4g}."
            )

        heat_rate = self._thermal_step()
        n = self.steps[-1] + 1
        self.steps.append(n)
        self.times.append(self.times[-1] + self.dt)
        self.temperature_history.append(float(self.T_cal))
        self.heat_flux_history.append(heat_rate)
        self.energy_history.append(self.total_thermal_energy())

    def prebake(self, n_steps, dt, progress_every=0):
        """Advance n_steps with a coarse dt, then restore the original dt."""
        original_dt = self.dt
        self.dt = float(dt)
        try:
            for k in range(1, n_steps+1):
                self.step()
                if progress_every and k % progress_every == 0:
                    umax = np.linalg.norm(self.u, axis=0).max()
                    print(f"prebake t={self.times[-1]:.5g} "
                          f"({k:5d}/{n_steps} steps)  "
                          f"u_max={umax:.5f}  T_cal={self.T_cal:.5f}")
        finally:
            self.dt = original_dt
        print(f"Prebake done: {n_steps} steps at dt={dt}, "
              f"t={self.times[-1]:.5g}, T_cal={self.T_cal:.5f}")

    def run(self, n_steps=600, progress_every=50, record_every=10):
        self.snapshots = [(self.times[-1], self.T.astype(np.float32).copy(),
                           np.linalg.norm(self.u, axis=0).astype(np.float32))]
        for k in range(1, n_steps+1):
            self.step()
            if record_every and k % record_every == 0:
                self.snapshots.append((
                    self.times[-1], self.T.astype(np.float32).copy(),
                    np.linalg.norm(self.u, axis=0).astype(np.float32)
                ))
            if progress_every and k % progress_every == 0:
                umax = np.linalg.norm(self.u, axis=0).max()
                print(f"t={self.times[-1]:.5g} ({self.steps[-1]:5d}/{n_steps} steps)  "
                      f"u_max={umax:.5f}  T_cal={self.T_cal:.5f}")

    def mean_temperature(self):
        return float(self.T[self.fluid].mean())

    def reynolds_number(self):
        speed = np.linalg.norm(self.u, axis=0)
        u_char = float(speed[self.fluid].max())
        return u_char * self.height / self.kinematic_viscosity

    def grashof_number(self):
        dT = self.T_cal - self.T_inf
        return self.buoyancy * dT * self.height**3 / self.kinematic_viscosity**2

    def rayleigh_number(self):
        return self.grashof_number() * self.prandtl_number

    def heat_transfer_coefficient(self):
        dT = self.T_cal - self.T_inf
        if dT <= 0 or len(self.heat_flux_history) < 2:
            return 0.0
        return self.heat_flux_history[-1] / (self.surface_area*dT)

    def nusselt_number(self):
        """Grid-based Nusselt estimate using height as characteristic length."""
        return self.heat_transfer_coefficient() * self.height / self.alpha

    def thermal_boundary_layer(self, z=None, threshold=0.01):
        """Radial distance from the first side-wall fluid cell to theta=threshold."""
        if z is None:
            z = max(2, self.height // 2)
        z = int(np.clip(z, 1, self.nz-2))
        cy = int(round((self.ny-1)/2))
        cx = (self.nx-1)/2
        # Search from the rightmost calorimeter cell at this y,z.
        xs = np.arange(self.nx)
        inside = self.calorimeter[:, cy, z]
        ids = np.where(inside)[0]
        if ids.size == 0:
            return np.nan
        wall = int(ids.max())
        fluid_ids = np.arange(wall+1, self.nx-1)
        excess_wall = self.T_cal - self.T_inf
        if excess_wall <= 0 or fluid_ids.size == 0:
            return np.nan
        theta = (self.T[fluid_ids, cy, z] - self.T_inf) / excess_wall
        crossings = np.where(theta < threshold)[0]
        if crossings.size == 0:
            return float(fluid_ids.size)
        j = int(crossings[0])
        if j == 0:
            return 0.0
        # Linear interpolation between adjacent fluid samples.
        t0, t1 = theta[j-1], theta[j]
        frac = 0.0 if t0 == t1 else np.clip((t0-threshold)/(t0-t1), 0, 1)
        return float((fluid_ids[j]-fluid_ids[0]-1) + frac)

    def blasius_boundary_layer(self, x=None):
        """Flat-plate Blasius estimate: delta = 5 x / sqrt(Re_x).

        Returns (velocity_delta, thermal_delta) in grid units. The thermal
        layer uses the standard Pr scaling delta_T = delta / Pr**(1/3).
        """
        x = float(self.height if x is None else x)
        speed = np.linalg.norm(self.u, axis=0)
        re_x = float(speed[self.fluid].max()) * x / self.kinematic_viscosity
        if re_x <= 0:
            return np.nan, np.nan
        delta = 5.0 * x / np.sqrt(re_x)
        return float(delta), float(delta / self.prandtl_number**(1/3))

    def cooling_time_constant(self):
        """Log-linear cooling fit against recorded simulation time."""
        t = np.asarray(self.times, dtype=float)
        excess = np.asarray(self.temperature_history) - self.T_inf
        valid = np.isfinite(excess) & (excess > 1e-8)
        if valid.sum() < 10:
            return np.inf
        slope, _ = np.polyfit(t[valid], np.log(excess[valid]), 1)
        return float(-1/slope) if slope < 0 else np.inf

    def total_thermal_energy(self):
        """Relative thermal energy; solid walls are excluded."""
        fluid_energy = float(np.sum(self.T[self.fluid] - self.T_inf))
        cal_energy = self.heat_capacity * (self.T_cal - self.T_inf)
        return fluid_energy + cal_energy

    def diagnostics(self):
        return {
            "steps": self.steps[-1],
            "time": self.times[-1],
            "dt": self.dt,
            "T_cal": self.T_cal,
            "mean_fluid_temperature": self.mean_temperature(),
            "u_max": float(np.linalg.norm(self.u, axis=0).max()),
            "Re_peak": self.reynolds_number(),
            "Pr": self.prandtl_number,
            "Gr": self.grashof_number(),
            "Ra": self.rayleigh_number(),
            "Nu_grid_estimate": self.nusselt_number(),
            "boundary_layer_grid_units": self.thermal_boundary_layer(),
            "blasius_velocity_layer": self.blasius_boundary_layer()[0],
            "blasius_thermal_layer": self.blasius_boundary_layer()[1],
            "cooling_time_constant": self.cooling_time_constant(),
            "thermal_energy_relative": self.total_thermal_energy(),
        }

    def visualise(self):
        if not self.snapshots:
            self.snapshots = [(self.times[-1], self.T.astype(np.float32).copy(),
                               np.linalg.norm(self.u, axis=0).astype(np.float32))]
        mid_y = self.ny // 2
        vT = max(float(s[1].max()) for s in self.snapshots)
        vu = max(float(s[2].max()) for s in self.snapshots) or 1e-12

        def sl(field):
            return np.ma.masked_where(self.solid, field)[:, mid_y, :].T

        time0, T0, u0 = self.snapshots[0]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.subplots_adjust(bottom=0.16)
        imT = axes[0, 0].imshow(sl(T0), origin="lower", aspect="auto",
                                cmap="inferno", vmin=self.T_inf, vmax=vT)
        axes[0, 0].set_title("Temperature, vertical mid-plane")
        axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("z")
        fig.colorbar(imT, ax=axes[0, 0])
        imU = axes[0, 1].imshow(sl(u0), origin="lower", aspect="auto",
                                cmap="viridis", vmin=0, vmax=vu)
        axes[0, 1].set_title("Speed, vertical mid-plane")
        axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("z")
        fig.colorbar(imU, ax=axes[0, 1])

        axes[1, 0].plot(self.times, self.temperature_history)
        axes[1, 0].set_title("Calorimeter cooling")
        axes[1, 0].set_xlabel("time"); axes[1, 0].set_ylabel("T_cal")
        markerT = axes[1, 0].axvline(time0, color="grey", ls="--")
        axes[1, 1].plot(self.times, self.heat_flux_history)
        axes[1, 1].set_title("Heat transfer from calorimeter")
        axes[1, 1].set_xlabel("time"); axes[1, 1].set_ylabel("heat transfer rate")
        markerQ = axes[1, 1].axvline(time0, color="grey", ls="--")

        sax = fig.add_axes((0.15, 0.06, 0.7, 0.03))
        slider = Slider(sax, "frame", 0, len(self.snapshots)-1,
                        valinit=0, valstep=1)

        def update(_):
            time, T, U = self.snapshots[int(slider.val)]
            imT.set_data(sl(T)); imU.set_data(sl(U))
            markerT.set_xdata([time, time]); markerQ.set_xdata([time, time])
            axes[0, 0].set_title(f"Temperature at t={time:.5g}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        self._sliders.append(slider)
        plt.show()


def run_sanity_tests():
    """Fast structural checks; not a full validation suite."""
    sim = FluidSimulation(
        nx=16, ny=16, nz=24, dt=0.05, kinematic_viscosity=0.1,
        thermal_diffusivity=0.02, buoyancy=0.0,
        initial_temperature=0.0, ambient_temperature=0.0,
        calorimeter_radius=2, calorimeter_height=6,
        heat_capacity=1e4, temperature_noise=0.0,
    )
    for _ in range(3):
        sim.step()
    assert np.isfinite(sim.u).all()
    assert np.isfinite(sim.T).all()
    assert np.allclose(sim.u, 0.0, atol=1e-10)
    assert np.allclose(sim.T, 0.0, atol=1e-10)

    cooling = FluidSimulation(
        nx=16, ny=16, nz=24, dt=0.05,
        thermal_diffusivity=0.02, buoyancy=0.0,
        initial_temperature=1.0, ambient_temperature=0.0,
        calorimeter_radius=2, calorimeter_height=6,
        heat_capacity=1e4, temperature_noise=0.0,
    )
    initial_energy = cooling.total_thermal_energy()
    cooling.step()
    assert cooling.times[-1] == cooling.dt
    assert np.isclose(cooling.total_thermal_energy(), initial_energy, atol=1e-9)
    assert cooling.T_cal < 1.0

    convection = FluidSimulation(
        nx=16, ny=16, nz=24, dt=0.05, kinematic_viscosity=0.1,
        thermal_diffusivity=0.02, buoyancy=0.01,
        initial_temperature=1.0, ambient_temperature=0.0,
        calorimeter_radius=2, calorimeter_height=6, heat_capacity=1e4,
    )
    for _ in range(3):
        convection.step()
    assert np.isfinite(convection.u).all()
    assert np.linalg.norm(convection.u) > 0.0
    print("Pressure-projection and dt sanity tests passed.")


PREBAKES = {
    # "prebake" runs coarse-dt steps first (not recorded), then the main run
    # continues at the normal dt.
    "default": dict(
        sim=dict(nx=48, ny=48, nz=96, dt=1.0,
                 calorimeter_radius=8, calorimeter_height=32),
        prebake=None,
        run=dict(n_steps=200, progress_every=10, record_every=10),
    ),
    # Grid refined 1.5x per axis; 150 coarse (dt=1) prebake steps, then 150
    # steps at dt=0.1 recorded every step.
    "fine_150": dict(
        sim=dict(nx=72, ny=72, nz=144, dt=0.1,
                 calorimeter_radius=12, calorimeter_height=48),
        prebake=dict(n_steps=150, dt=1.0, progress_every=20),
        run=dict(n_steps=150, progress_every=10, record_every=1),
    ),
}


if __name__ == "__main__":
    import sys

    run_sanity_tests()

    # Usage: python simulation.py [default|fine_150]
    prebake = PREBAKES[sys.argv[1] if len(sys.argv) > 1 else "default"]

    sim = FluidSimulation(
        kinematic_viscosity=0.1,
        thermal_diffusivity=0.03,
        buoyancy=5e-04,
        heat_capacity=630,
        initial_temperature=75.0,
        ambient_temperature=20.0,
        temperature_noise=0.0,
        **prebake["sim"],
    )
    if prebake["prebake"]:
        sim.prebake(**prebake["prebake"])
    sim.run(**prebake["run"])
    print("\nDiagnostics (grid-based/nondimensional units):")
    for key, value in sim.diagnostics().items():
        print(f"{key:34s}: {value}")
    sim.visualise()
