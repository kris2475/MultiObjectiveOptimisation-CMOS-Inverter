import subprocess
import numpy as np
import os
import re
import shutil
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION & PATHS ----------------
# !!! IMPORTANT: VERIFIED PATH FROM PREVIOUS STEPS !!!
LTSPICE_EXE = r"C:\Users\Kris\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

# File names
NETLIST_FILE = "cmos_optimization.cir"
LOG_FILE = "cmos_optimization.log" 
LOG_DIR = "failed_sim_logs" 

# --- MOSFET physical limits ---
MIN_W = 1e-6    # 1 µm
MAX_W = 50e-6   # 50 µm 
MIN_L = 0.5e-6  # 0.5 µm
MAX_L = 2e-6    # 2 µm 
BOUNDS = [(MIN_W, MAX_W), (MIN_L, MAX_L)] # (W, L)

# --- TRANSIENT PARAMETERS ---
VDD = 5.0 
CL = 100e-15   # Load Capacitance (100 fF)
PULSE_PERIOD = 50e-9 # 50 ns 
PULSE_DELAY = 1e-9   # 1 ns 
PULSE_RISE_FALL = 100e-12 # 100 ps 

# --- FIXED PMOS LOAD (Ratio optimization only happens on NMOS) ---
PMOS_W = 20e-6 # Fixed Width 
PMOS_L = 1e-6  # Fixed Length

# --- Multi-Objective Weights ---
# We minimize both delay and power/current. Delay is generally weighted higher.
WEIGHT_DELAY = 1.0 
WEIGHT_CURRENT = 0.05 
# Define a large penalty value for failed simulations
PENALTY_VALUE = 1e9

# --- Optimization History ---
OPTIMIZATION_HISTORY = []

# ----------------- LTspice Netlist Template -----------------
NETLIST_TEMPLATE = """
* CMOS Inverter Transient Optimization (Speed vs. Power)
.title Automated CMOS Inverter Optimisation
Vdd Vdd 0 DC {VDD_val}

* --- Transistor Model Definitions (Stable Level 3 Models) ---
.model n_mos NMOS (LEVEL=3 VTO=0.5 KP=200u GAMMA=0.5)
.model p_mos PMOS (LEVEL=3 VTO=-0.5 KP=80u GAMMA=0.5)

* --- Input Pulse Source ---
Vin IN 0 PULSE(0 {VDD_val} {PULSE_DELAY_val} {PULSE_RISE_FALL_val} {PULSE_RISE_FALL_val} 
+ {PULSE_PERIOD_half_val} {PULSE_PERIOD_val})

* --- Transistor Definitions ---
* M1 (NMOS Pull-Down) - Optimized W and L
M1 OUT IN 0 0 n_mos L={L_val}u W={W_val}u
* M2 (PMOS Pull-Up) - Fixed
M2 OUT IN Vdd Vdd p_mos L={PMOS_L_val}u W={PMOS_W_val}u

* --- Load ---
CL OUT 0 {CL_val} 

* --- Transient Analysis and MEASUREMENTS ---
.tran 0 {PULSE_PERIOD_2x_val} 0 

* Objective 1: Propagation Delay (Low to High)
.measure tran tplh TRIG V(IN) VAL={VDD_half_val} FALL=1 TARG V(OUT) VAL={VDD_half_val} RISE=1 

* Objective 2: Propagation Delay (High to Low)
.measure tran tphl TRIG V(IN) VAL={VDD_half_val} RISE=1 TARG V(OUT) VAL={VDD_half_val} FALL=1 

* Objective 3: Average Propagation Delay (TP) - Used for speed objective
.measure tran tp PARAM (tplh + tphl) / 2

* Objective 4: Minimize Average Power / Current
.measure tran Iavg AVG I(Vdd) FROM {PULSE_PERIOD_val} TO {PULSE_PERIOD_2x_val}

.end
"""
# ----------------- LTspice Helper Functions -----------------
def generate_netlist(W, L):
    """Generates the LTspice netlist file with specific W and L values."""
    W = np.clip(W, MIN_W, MAX_W)
    L = np.clip(L, MIN_L, MAX_L)
    
    # Pre-calculate and prepare values for formatting
    format_data = {
        'W_val': W * 1e6, 
        'L_val': L * 1e6,
        'VDD_val': VDD,
        'VDD_half_val': VDD / 2,
        'PULSE_DELAY_val': PULSE_DELAY,
        'PULSE_RISE_FALL_val': PULSE_RISE_FALL,
        'PULSE_PERIOD_val': PULSE_PERIOD,
        'PULSE_PERIOD_half_val': PULSE_PERIOD / 2,
        'PULSE_PERIOD_2x_val': PULSE_PERIOD * 2,
        'PMOS_W_val': PMOS_W * 1e6,
        'PMOS_L_val': PMOS_L * 1e6,
        'CL_val': CL 
    }

    with open(NETLIST_FILE, "w") as f:
        f.write(NETLIST_TEMPLATE.format(**format_data))

def run_ltspice():
    """Executes LTspice simulation in batch mode."""
    SIMULATION_TIMEOUT = 15
    try:
        absolute_netlist_path = os.path.abspath(NETLIST_FILE)
        
        if not os.path.exists(LTSPICE_EXE):
             print(f"Error: LTspice executable not found at '{LTSPICE_EXE}'.")
             return False

        if os.path.exists(LOG_FILE): os.remove(LOG_FILE) 
        
        command_list = [LTSPICE_EXE, "-b", absolute_netlist_path]
        
        # FIX: Changed check=True to check=False to prevent Python from crashing on LTspice's non-zero exit codes.
        subprocess.run(command_list, check=False, shell=False, timeout=SIMULATION_TIMEOUT, 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if not os.path.exists(LOG_FILE):
             return False
        return True
    except:
        return False

def parse_ltspice_log():
    """Reads the .log file to extract measured average delay (TP) and average current (IAVG)."""
    delay = PENALTY_VALUE
    i_avg = PENALTY_VALUE
    
    if not os.path.exists(LOG_FILE):
        return delay, i_avg

    try:
        with open(LOG_FILE, 'r') as f:
            content = f.read()

        # 1. Extract Average Propagation Delay (TP) - Robust Regex verified by user run
        tp_match = re.search(r'tp:.*?=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
        if tp_match and tp_match.group(1):
             delay = abs(float(tp_match.group(1)))

        # 2. Extract Average Current (IAVG) - Robust Regex verified by user run
        iavg_match = re.search(r'iavg:.*?=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
        if iavg_match and iavg_match.group(1):
            i_avg = abs(float(iavg_match.group(1)))
            
    except:
        return PENALTY_VALUE, PENALTY_VALUE
    
    return delay, i_avg

# ----------------- OBJECTIVE FUNCTION -----------------
def objective_function(ind):
    """
    The main evaluation function for the optimizer.
    It takes an individual (W, L) and returns a single cost (fitness) value to minimize.
    """
    W, L = ind
    
    # 1. Generate Netlist
    generate_netlist(W, L)
    
    # 2. Run LTspice Simulation
    if not run_ltspice():
        # Assign huge penalty if simulation fails to run/timeout
        print(f"Simulation failed for W={W:.2e}, L={L:.2e}. Applying penalty.")
        OPTIMIZATION_HISTORY.append((W, L, PENALTY_VALUE))
        return PENALTY_VALUE

    # 3. Parse Results
    delay, i_avg = parse_ltspice_log()
    
    # 4. Check for Bad Results (e.g., non-switching waveform)
    if delay >= 1e-6: # Check if delay is unreasonably high (>1µs, indicating failure)
        cost = PENALTY_VALUE
    else:
        # 5. Calculate Weighted Cost
        # Cost = (Weight_Delay * Delay) + (Weight_Current * Current)
        cost = (WEIGHT_DELAY * delay) + (WEIGHT_CURRENT * i_avg)
    
    # Log the result for plotting later
    OPTIMIZATION_HISTORY.append((W, L, cost))
    
    print(f"W={W*1e6:.1f}µm, L={L*1e6:.1f}µm -> Delay={delay*1e9:.2f}ns, Iavg={i_avg*1e6:.1f}µA. Cost={cost:.4e}")
    
    return cost

# ----------------- PLOTTING FUNCTION -----------------
def plot_optimization_results():
    """Generates a 3D plot of the optimization landscape."""
    if not OPTIMIZATION_HISTORY:
        print("\nNo history recorded for plotting.")
        return

    history_array = np.array(OPTIMIZATION_HISTORY)
    W_hist = history_array[:, 0] * 1e6 # Convert to µm
    L_hist = history_array[:, 1] * 1e6 # Convert to µm
    Cost_hist = history_array[:, 2]

    # Create the figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(W_hist, L_hist, Cost_hist, 
                         c=Cost_hist, cmap='viridis', marker='o', depthshade=True)

    # Labeling - FIX: Use raw strings for LaTeX compatibility and to remove SyntaxWarnings
    ax.set_xlabel(r'NMOS Width W ($\mu m$)')
    ax.set_ylabel(r'NMOS Length L ($\mu m$)')
    ax.set_zlabel('Optimization Cost (Weighted Delay + Current)')
    ax.set_title('Multi-Objective Optimization Landscape')

    # Add a color bar
    cbar = fig.colorbar(scatter, pad=0.1)
    cbar.set_label('Cost (to be minimized)')

    # Highlight the minimum cost point (the optimal solution)
    min_cost_index = np.argmin(Cost_hist)
    ax.scatter(W_hist[min_cost_index], L_hist[min_cost_index], Cost_hist[min_cost_index], 
               color='red', marker='*', s=300, label='Optimal Point')

    ax.legend()
    plt.show()

# ----------------- MAIN EXECUTION -----------------
def main():
    print("=======================================================")
    print("=== Starting CMOS Inverter Multi-Objective Optimization ===")
    print("--- Minimizing (Delay * 1.0) + (Current * 0.05) ---")
    print("=======================================================")

    # Run Differential Evolution Optimization
    # maxiter: number of generations
    # popsize: number of individuals in each generation
    # FIX: Set workers=1 to disable parallel processing and resolve conflicts 
    # when calling LTspice (subprocess) multiple times simultaneously.
    result = differential_evolution(
        objective_function, 
        bounds=BOUNDS, 
        maxiter=15, 
        popsize=10, 
        workers=1, # Run sequentially to ensure stable file I/O with LTspice
        disp=True,
        polish=True
    )

    if result.success:
        W_opt, L_opt = result.x
        final_cost = result.fun

        # Re-run simulation at the optimal point to get the clean metrics for display
        generate_netlist(W_opt, L_opt)
        run_ltspice()
        t_p, i_avg = parse_ltspice_log()
        
        print(f"\n✅ Optimization Successful in {result.nit} generations.")
        print("--- Optimal Transistor Parameters (NMOS) ---")
        print(f"Optimal W: {W_opt*1e6:.4f} μm")
        print(f"Optimal L: {L_opt*1e6:.4f} μm")
        print(f"W/L Ratio: {W_opt/L_opt:.2f}")
        print("---------------------------------------------")
        print(f"Final Cost (Minimized): {final_cost:.6e}")
        print(f"Resulting Avg Delay (t_p): {t_p*1e9:.3f} ns")
        print(f"Resulting Avg Current (I_avg): {i_avg*1e6:.3f} μA")
    else:
        print("\n❌ Optimization Failed:")
        print(result.message)
        final_cost = PENALTY_VALUE

    print("\n=======================================================")

    # Plot results if the cost landscape is reasonable
    if final_cost < 1e6:
        plot_optimization_results()
    else:
        print("Skipping plot generation due to high failure rate or cost.")


if __name__ == "__main__":
    try:
        # Check for required library imports (matplotlib, scipy)
        import numpy as np
        from scipy.optimize import differential_evolution 
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D 
        
        main()
    except ImportError as e:
        print(f"Required library missing: {e}")
        print("Please ensure numpy, scipy, and matplotlib are installed (e.g., pip install numpy scipy matplotlib).")
    except Exception as e:
        print(f"\nAn unhandled error occurred during main execution: {e}")




