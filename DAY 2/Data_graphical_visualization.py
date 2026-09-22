import numpy as np
import matplotlib.pyplot as plt
time = np.linspace(0,10,500)
amplitude=np.sin(2*np.pi*time)
fig,ax=plt.subplots(figsize=(8,4.5),dpi=100)
ax.plot(time, amplitude, color='green', linewidth=2, label="Sensor A Displacement")
ax.plot(time, amplitude, color='green', linewidth=2, label="Sensor A Displacement")
ax.set_title('Damped Harmonic Response of Structure', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Displacement (mm)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', frameon=True, shadow=True)
plt.show()