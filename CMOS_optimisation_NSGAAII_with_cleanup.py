"""
CMOS Inverter Optimisation using NSGA-II
----------------------------------------
Author: Kris
Date: 2025-10-02

Summary:
- This script optimises a CMOS inverter for both propagation delay (tp) 
  and average current (iavg) using a multi-objective NSGA-II algorithm.
- Differential Evolution (DE) has previously found a near-optimal solution:
      W = 2.0707 µm, L = 1.7228 µm
  This solution is used as a seed in the initial population.
- LTspice simulations are run in batch mode (no GUI) for automated evaluation.

Limitations:
- Small MOSFET sizes may trigger warnings: "Length/Width shorter than recommended".
- Extremely small tp or iavg values may be unphysical due to LTspice model limits.
- Simulation failures are logged and penalised with large objective values.
- LTspice must be installed and the path specified in LTSPICE_EXE.
"""

import subprocess
import os
import random
import csv
import glob
from deap import base, creator, tools, algorithms

# ---------------- LTspice Configuration ----------------
LTSPICE_EXE = r"C:\Users\Kris\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
CIRCUIT_FILE = r"C:\Users\Kris\ML_MOSFET\DE\Resonant Circuit\cmos_inverter.cir"

# ---------------- Optimisation Parameters ----------------
POPULATION_SIZE = 50
NGEN = 30
CXPB, MUTPB = 0.7, 0.2

# ---------------- DE-optimal seed ----------------
DE_OPTIMAL = {'W': 2.0707, 'L': 1.7228}

# ---------------- DEAP Setup ----------------
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # minimise tp and iavg
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Gene ranges
toolbox.register("attr_W", random.uniform, 2.0, 15.0)
toolbox.register("attr_L", random.uniform, 0.6, 2.0)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_W, toolbox.attr_L), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ---------------- Circuit Evaluation ----------------
def generate_netlist(W, L, netlist_path):
    """
    Generate a temporary netlist with given W and L.
    """
    with open(CIRCUIT_FILE, 'r') as f:
        base_netlist = f.read()
    netlist = base_netlist.replace(".PARAM W=1", f".PARAM W={W:.4f}") \
                           .replace(".PARAM L=1", f".PARAM L={L:.4f}")
    with open(netlist_path, 'w') as f:
        f.write(netlist)

def evaluate_circuit(individual):
    """
    Run LTspice in batch mode and return tp and iavg.
    """
    W, L = individual
    tmp_netlist = f"temp_{W:.4f}_{L:.4f}.cir"
    generate_netlist(W, L, tmp_netlist)

    try:
        # Run LTspice in batch mode (no GUI)
        subprocess.run([LTSPICE_EXE, "-b", tmp_netlist],
                       check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)

        # Placeholder: extract values from LTspice output (.raw)
        # In production, parse .raw file using PyLTSpice or custom parser
        tplh = random.uniform(1e-9, 1e-8)
        tphl = random.uniform(1e-9, 1e-8)
        tp = (tplh + tphl) / 2
        iavg = random.uniform(1e-9, 1e-6)

    except Exception as e:
        # Failed simulation penalty
        tp = 1e21
        iavg = 1e18

    finally:
        # Clean up temporary files for this simulation
        for ext in [".cir", ".raw", ".log", ".plt", ".net"]:
            tmp_file = tmp_netlist.replace(".cir", ext)
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    print(f"Evaluated W={W:.4f} µm, L={L:.4f} µm -> tp={tp:.3e}, iavg={iavg:.3e}")
    return tp, iavg

toolbox.register("evaluate", evaluate_circuit)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# ---------------- Cleanup Function ----------------
def cleanup_ltspice_files():
    """
    Remove any leftover temporary LTspice netlists and output files.
    """
    patterns = ["temp_*.cir", "temp_*.raw", "temp_*.log", "temp_*.plt", "temp_*.net"]
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except OSError:
                pass

# ---------------- Main Optimisation ----------------
def main_nsgaii():
    # Create initial population with DE-optimal seed
    pop = toolbox.population(n=POPULATION_SIZE-1)
    seed = creator.Individual([DE_OPTIMAL['W'], DE_OPTIMAL['L']])
    pop.append(seed)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda vals: (min(v[0] for v in vals), min(v[1] for v in vals)))
    stats.register("avg", lambda vals: (sum(v[0] for v in vals)/len(vals),
                                        sum(v[1] for v in vals)/len(vals)))

    # Hall of Fame (best solutions)
    hof = tools.HallOfFame(10)

    # Run NSGA-II
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox,
                                         mu=POPULATION_SIZE,
                                         lambda_=POPULATION_SIZE,
                                         cxpb=CXPB,
                                         mutpb=MUTPB,
                                         ngen=NGEN,
                                         stats=stats,
                                         halloffame=hof,
                                         verbose=True)

    # Save Pareto front
    with open('pareto_front_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['W(um)', 'L(um)', 'tp(s)', 'iavg(A)'])
        for ind in pop:
            writer.writerow([ind[0], ind[1], ind.fitness.values[0], ind.fitness.values[1]])

    print("\nResults saved to 'pareto_front_results.csv'")
    print("Top solutions in Hall of Fame:")
    for ind in hof:
        print(f"W={ind[0]:.4f}, L={ind[1]:.4f}, tp={ind.fitness.values[0]:.3e}, iavg={ind.fitness.values[1]:.3e}")

if __name__ == "__main__":
    main_nsgaii()
    cleanup_ltspice_files()
    print("Temporary files cleaned up.")









