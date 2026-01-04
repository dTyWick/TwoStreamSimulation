import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from scipy.stats import linregress
from collections import defaultdict

def gen_normalized_constants(
    L_in_Debye,         
    Time_in_Periods,    
    particles_per_cell, 
):
    """
    Generates constants for a Normalized PIC simulation and performs 
    explicit stability checks.
    """
    
    # --- 1. The Unit System ---
    # We define the simulation such that these are TRUE by definition.
    lambda_D = 1.0
    omega_pe = 1.0
    m_e = 1.0
    
    # --- 2. Derived Numerical Stability Parameters ---
    # Stability Rule 1: Finite Grid Instability
    # We need dx < lambda_D.
    dx = 0.1 
    
    # Stability Rule 2: Leapfrog Accuracy
    # We need omega_pe * dt << 1.
    dt = 0.005
    
    # --- 3. Translate User Inputs to Code Integers ---
    num_cells = int(round(L_in_Debye / dx))
    grid_length_code = num_cells * dx 
    
    total_sim_time = Time_in_Periods * (2 * np.pi)
    timesteps = int(round(total_sim_time / dt))
    
    # --- 4. Particle Setup ---
    num_particles = num_cells * particles_per_cell
    # q_macro calculation: q^2 = dx / ppc (for normalized density)
    q_macro = np.sqrt(dx / particles_per_cell)
    
    # --- 5. Ion Setup ---
    m_i = 1836.0 * m_e
    q_i = q_macro  
    q_e = -q_macro

    # --- 6. Video Smoothness Setting ---
    # Target ~1200 frames for a smooth 45s video at 30fps
    save_interval = max(1, int(timesteps / 1200))

    # --- 7. STABILITY REPORT ---
    print("\n" + "="*50)
    print("SIMULATION PARAMETERS & STABILITY REPORT")
    print("="*50)
    
    # Check 1: Spatial Resolution
    print(f"1. Spatial Resolution (Condition: dx < λ_D)")
    print(f"   • Grid Spacing (dx):      {dx:.4f}")
    print(f"   • Debye Length (λ_D):     {lambda_D:.4f}")
    if dx < lambda_D:
        print(f"   -> STATUS: STABLE (Resolution is sufficient)")
    else:
        print(f"   -> STATUS: UNSTABLE (dx too large, energy will drift!)")

    # Check 2: Temporal Resolution
    print(f"\n2. Temporal Resolution (Condition: ω_pe * dt < 0.2)")
    print(f"   • Timestep (dt):          {dt:.4f}")
    print(f"   • Plasma Freq (ω_pe):     {omega_pe:.4f}")
    print(f"   • ω_pe * dt:              {omega_pe * dt:.4f}")
    if omega_pe * dt < 0.2:
        print(f"   -> STATUS: STABLE (Dynamics resolved)")
    else:
        print(f"   -> STATUS: UNSTABLE (dt too large, aliases plasma wave)")

    # Check 3: Statistics
    print(f"\n3. System Scale")
    print(f"   • System Size:            {L_in_Debye} λ_D ({num_cells} cells)")
    print(f"   • Total Particles:        {num_particles * 2} ({particles_per_cell} PPC)")
    print(f"   • Duration:               {Time_in_Periods} Periods ({timesteps} steps)")
    print("="*50 + "\n")
    
    return {
        'dx': dx,
        'dt': dt,
        'num_grid_points': num_cells,
        'grid_length': grid_length_code,
        'num_electrons': num_particles,
        'num_ions': num_particles,
        'm_e': m_e, 'q_e': q_e,
        'm_i': m_i, 'q_i': q_i,
        'omega_pe': omega_pe,
        'timesteps': timesteps,
        'save_interval': save_interval, 
        'k_array': 2 * np.pi * np.fft.fftfreq(num_cells, d=dx)
    }

def calculate_optimal_v_beam(c):
    """Calculates the ideal beam velocity."""
    chi_peak = 1.006 
    L = c['grid_length']
    wpe = c['omega_pe']
    k = 2 * np.pi / L
    
    v_beam_optimal = (chi_peak * wpe) / k
    
    print(f"-> Optimal Drift Velocity: {v_beam_optimal:.4f}")
    return v_beam_optimal

def assign_charge_cic(x, rho, num_grid_points, dx, charge):
    """Cloud-In-Cell (CIC) charge assignment"""
    x_norm = x / dx
    i_left = x_norm.astype(int) % num_grid_points
    i_right = (i_left + 1) % num_grid_points
    weight_right = x_norm - np.floor(x_norm)
    weight_left = 1.0 - weight_right
    np.add.at(rho, i_left, weight_left * charge)
    np.add.at(rho, i_right, weight_right * charge)

def interpolate_field_cic(x, E_grid, dx, num_grid_points):
    """Cloud-In-Cell (CIC) field interpolation"""
    x_norm = x / dx
    i_left = x_norm.astype(int) % num_grid_points
    i_right = (i_left + 1) % num_grid_points
    weight_right = x_norm - np.floor(x_norm)
    weight_left = 1.0 - weight_right
    return (weight_left * E_grid[i_left] + weight_right * E_grid[i_right])

def solve_poisson_fft(rho, dx, k_array):
    """FFT-based Poisson solver"""
    rho_k = np.fft.fft(rho)
    rho_k[0] = 0.0 
    E_k = np.zeros_like(rho_k, dtype=complex)
    nonzero = k_array != 0
    E_k[nonzero] = -1j * rho_k[nonzero] / k_array[nonzero]
    return np.real(np.fft.ifft(E_k))

def pic_simulation_with_ions(x_e, x_i, v_e, v_i, c, save_phase_space=False):
    dx, dt = c['dx'], c['dt']
    N, L = c['num_grid_points'], c['grid_length']
    k_array = c['k_array']
    kick_e = (c['q_e'] / c['m_e']) * dt
    kick_i = (c['q_i'] / c['m_i']) * dt
    
    raw_data = defaultdict(list)
    rho = np.zeros(N) 

    print("Initializing Leapfrog...")
    # 1. Initial Density
    rho[:] = 0.0
    assign_charge_cic(x_e, rho, N, dx, c['q_e'])
    rho_i = np.zeros_like(rho)
    assign_charge_cic(x_i, rho_i, N, dx, c['q_i'])
    rho += rho_i
    rho /= dx 
    rho -= np.mean(rho) 

    # 2. Initial Field
    E_grid = solve_poisson_fft(rho, dx, k_array)
    
    # 3. Backtrack Velocities
    v_e -= 0.5 * kick_e * interpolate_field_cic(x_e, E_grid, dx, N)
    v_i -= 0.5 * kick_i * interpolate_field_cic(x_i, E_grid, dx, N)

    print(f"Running {c['timesteps']} steps...")
    
    for step in tqdm(range(c['timesteps'])):
        
        # --- Standard PIC Cycle ---
        rho[:] = 0.0
        assign_charge_cic(x_e, rho, N, dx, c['q_e'])
        rho_i[:] = 0.0
        assign_charge_cic(x_i, rho_i, N, dx, c['q_i'])
        rho += rho_i
        rho /= dx 
        rho -= np.mean(rho)
        
        E_grid = solve_poisson_fft(rho, dx, k_array)
        
        E_at_e = interpolate_field_cic(x_e, E_grid, dx, N)
        E_at_i = interpolate_field_cic(x_i, E_grid, dx, N)
        
        v_e += kick_e * E_at_e
        v_i += kick_i * E_at_i
        
        x_e += v_e * dt
        x_i += v_i * dt
        
        np.mod(x_e, L, out=x_e)
        np.mod(x_i, L, out=x_i)
        
        # --- Diagnostics ---
        if step % c['save_interval'] == 0:
            raw_data['time'].append(step * dt)
            
            field_energy = 0.5 * np.sum(E_grid**2) * dx
            ke = 0.5 * c['m_e'] * np.sum(v_e**2) + 0.5 * c['m_i'] * np.sum(v_i**2)
            
            raw_data['field_energy'].append(field_energy)
            raw_data['kinetic_energy'].append(ke)
            raw_data['total_energy'].append(ke + field_energy)
            raw_data['potential_energy'].append(field_energy)
            
            if save_phase_space:
                raw_data['electron_x'].append(x_e.copy())
                raw_data['electron_v'].append(v_e.copy())

    # --- Pack into (Data, Label) Tuples ---
    # CHANGE: Convert Time to Plasma Periods immediately (t / 2pi)
    time_in_periods = np.array(raw_data['time']) / (2 * np.pi)
    
    diagnostics = {}
    diagnostics['time'] = (time_in_periods, r"Time [$T_{pe}$]")
    
    diagnostics['field_energy'] = (np.array(raw_data['field_energy']), r"Field Energy [$m_e v_{th}^2$]")
    diagnostics['kinetic_energy'] = (np.array(raw_data['kinetic_energy']), r"Kinetic Energy [$m_e v_{th}^2$]")
    diagnostics['total_energy'] = (np.array(raw_data['total_energy']), r"Total Energy [$m_e v_{th}^2$]")
    diagnostics['potential_energy'] = (np.array(raw_data['potential_energy']), r"Potential Energy [$m_e v_{th}^2$]")
    
    if save_phase_space:
        diagnostics['electron_x'] = (raw_data['electron_x'], r"Position [$\lambda_D$]")
        diagnostics['electron_v'] = (raw_data['electron_v'], r"Velocity [$v_{th}$]")

    return diagnostics

def plot_growth_rate(diagnostics, omega_plasma, linear_growth_rate=None, fit_range=None):
    """
    Plots field energy growth and optionally performs a linear fit over a 
    specified time range to measure the growth rate and R^2.
    
    Parameters:
    -----------
    fit_range : tuple (t_start, t_end)
        The start and end times (in Plasma Periods) to perform the fit.
    """
    # Unpack tuples
    time, time_label = diagnostics['time']
    field_energy, energy_label = diagnostics['field_energy']
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 1. Plot Raw Data
    ax.semilogy(time, field_energy, label='Simulation Data', color='dodgerblue', linewidth=2, alpha=0.8)
    
    # 2. Perform Fit (if range provided)
    if fit_range is not None:
        t_start, t_end = fit_range
        
        # Filter data within the time window
        mask = (time >= t_start) & (time <= t_end)
        t_fit = time[mask]
        E_fit = field_energy[mask]
        
        if len(t_fit) > 2:
            # Linear Regression on Log(Energy)
            # Model: ln(E) = slope * t + intercept
            # Physics: E ~ exp(2 * gamma * t) -> ln(E) = 2*gamma*t + C
            # Therefore: slope = 2 * gamma
            
            # Note: t_fit is in periods (T_pe). 
            # The growth rate gamma is usually defined in units of omega_pe.
            # Argument of exp is 2 * gamma * t_seconds.
            # t_seconds = t_fit * (2*pi/omega_pe).
            # ln(E) = 2 * gamma * (t_fit * 2*pi/omega_pe) + C
            # ln(E) = [4 * pi * (gamma/omega_pe)] * t_fit + C
            # Slope of regression = 4 * pi * gamma_normalized
            
            res = linregress(t_fit, np.log(E_fit))
            
            measured_gamma_normalized = res.slope / (4 * np.pi)
            r_squared = res.rvalue**2
            
            # Generate Fit Line for Plotting
            fit_line = np.exp(res.intercept + res.slope * t_fit)
            
            # Plot Fit
            label_fit = (r"Fit: $\gamma_{sim} \approx " + f"{measured_gamma_normalized:.4f}$" + 
                         r", $R^2 = " + f"{r_squared:.4f}$")
            ax.semilogy(t_fit, fit_line, 'k--', linewidth=2, label=label_fit)
            
            # Highlight Fit Region
            ax.axvspan(t_start, t_end, color='yellow', alpha=0.1, label='Linear Phase')
            
            # 3. Plot Theory Comparison (Matched to Fit Start)
            if linear_growth_rate:
                # Anchor the theory line to the start of the fit for visual comparison
                y_start = fit_line[0]
                # Theory: exp(2 * gamma_theory * t_raw)
                # t_raw = (t_fit - t_start) * 2*pi
                dt_periods = t_fit - t_start
                y_theory = y_start * np.exp(2 * linear_growth_rate * (dt_periods * 2 * np.pi))
                
                label_theory = r"Theory: $\gamma_{th} = " + f"{linear_growth_rate:.4f}$"
                ax.semilogy(t_fit, y_theory, 'r:', linewidth=2, label=label_theory)

        else:
            print("Warning: Not enough data points in fit_range to perform regression.")

    # Formatting
    ax.set_xlabel(time_label, fontsize=12)
    ax.set_ylabel(energy_label + " (Log Scale)", fontsize=12)
    ax.set_title('Instability Growth Rate Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    plt.show()

def energy_conservation_plot(diagnostics, omega_plasma):
    # Unpack tuples
    times, time_label = diagnostics['time']
    kinetic, _ = diagnostics['kinetic_energy']
    potential, _ = diagnostics['potential_energy']
    total, total_label = diagnostics['total_energy']
    
    E0 = total[0]
    error_pct = 100 * (total - E0) / E0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(times, kinetic, label='Kinetic')
    ax1.plot(times, potential, label='Field')
    ax1.plot(times, total, 'k--', label='Total')
    
    ax1.set_ylabel(total_label) 
    ax1.set_title('Energy Conservation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, error_pct, 'r')
    ax2.set_ylabel('Error [%]')
    ax2.set_xlabel(time_label)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_phase_space_animation_scatter(diagnostics, omega_plasma, save_animation=False, duration_sec=45):
    if 'electron_x' not in diagnostics: return
    
    # Unpack Tuples
    x_list, x_label = diagnostics['electron_x']
    v_list, v_label = diagnostics['electron_v']
    times, time_label = diagnostics['time']
    
    total_frames_available = len(x_list)
    
    fps = 30 
    total_frames_needed = duration_sec * fps
    
    stride = max(1, int(total_frames_available / total_frames_needed))
    indices = np.arange(0, total_frames_available, stride)
    
    print(f"\n--- Animation Generation ---")
    print(f"Available Snapshots: {total_frames_available}")
    print(f"Rendering {len(indices)} frames for {duration_sec}s video.")
    
    sample_indices = np.linspace(0, len(x_list)-1, 10, dtype=int)
    all_x = np.concatenate([x_list[i] for i in sample_indices])
    all_v = np.concatenate([v_list[i] for i in sample_indices])
    x_min, x_max = 0, np.max(all_x)
    v_min, v_max = np.min(all_v), np.max(all_v)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(i):
        idx = indices[i]
        ax.clear()
        
        x = x_list[idx]
        v = v_list[idx]
        t = times[idx] 
        
        ax.scatter(x, v, s=1, alpha=0.2, color='blue', label='Electrons')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(v_min, v_max)
        
        # Title now uses the Time unit stored in the diagnostic (Periods)
        ax.set_title(f'Phase Space t={t:.2f} $T_{{pe}}$')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(v_label)
        return ax,

    anim = FuncAnimation(fig, update, frames=len(indices))
    
    if save_animation:
        print(f"Saving animation to 'instability.mp4'...")
        try:
            with tqdm(total=len(indices), unit="frame", desc="Rendering Video") as pbar:
                anim.save(
                    'instability.mp4', 
                    writer='ffmpeg', 
                    fps=fps, 
                    dpi=100,
                    progress_callback=lambda i, n: pbar.update(1)
                )
            print("Save Complete.")
        except Exception as e:
            print(f"Error saving video: {e}")
            print("Ensure ffmpeg is installed.")
            
    plt.show()

def plot_phase_space_evolution(diagnostics, omega_plasma, 
                                frame_indices=None, 
                                num_frames=6, 
                                save_plots=False):
    if 'electron_x' not in diagnostics: return
    
    x_list, x_label = diagnostics['electron_x']
    v_list, v_label = diagnostics['electron_v']
    times, t_label = diagnostics['time']
    
    total_frames = len(x_list)
    if frame_indices is None:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    n_frames = len(frame_indices)
    ncols = min(3, n_frames)
    nrows = int(np.ceil(n_frames / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_frames == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[idx]
        x = x_list[frame_idx]
        v = v_list[frame_idx]
        t = times[frame_idx]
        
        ax.scatter(x, v, s=1, alpha=0.3, c='blue', edgecolors='none')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(v_label, fontsize=10)
        ax.set_title(f't = {t:.2f} $T_{{pe}}$', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_frames, len(axes)): axes[idx].axis('off')
    
    plt.suptitle('Phase Space Evolution', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('phase_space_evolution.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved: phase_space_evolution.png")
    
    plt.show()

