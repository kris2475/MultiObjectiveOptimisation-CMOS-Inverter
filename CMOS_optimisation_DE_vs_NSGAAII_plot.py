import matplotlib.pyplot as plt

# NSGA-II Hall of Fame results (tp in ns, iavg in µA)
nsga_tp = [1.099e-09, 1.119e-09, 1.173e-09, 1.203e-09, 1.204e-09, 1.222e-09]  # tp in seconds
nsga_iavg = [5.205e-08, 4.624e-07, 7.398e-07, 1.677e-07, 3.585e-07, 7.127e-07]  # iavg in A

# DE solution
de_tp = 7.284e-09   # example from previous DE evaluation, tp in seconds
de_iavg = 8.009e-07 # example from previous DE evaluation, iavg in A

plt.figure(figsize=(8,6))

# Plot NSGA-II Pareto points
plt.scatter([tp*1e9 for tp in nsga_tp], [i*1e6 for i in nsga_iavg], color='blue', label='NSGA-II Pareto', s=100)

# Plot DE solution
plt.scatter(de_tp*1e9, de_iavg*1e6, color='red', marker='*', s=200, label='DE Solution')

plt.xlabel('Propagation Delay tp (ns)')
plt.ylabel('Average Current iavg (µA)')
plt.title('Comparison of NSGA-II Pareto Front vs DE Solution')
plt.legend()
plt.grid(True)
plt.show()








