# filtros_signales.py
"""
Diseño y análisis de filtros digitales:
- Señal compuesta + ruido
- Filtros: FIR (ventana), Butterworth (IIR), Chebyshev I (IIR)
- Gráficas: señal en tiempo, espectro (FFT), respuesta en frecuencia, señal filtrada
- Guarda figuras en carpeta ./figs
Requisitos: numpy, scipy, matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# -------------------------
# Parámetros generales
# -------------------------
fs = 2000.0        # frecuencia de muestreo (Hz)
T = 2.0            # duración en segundos
t = np.arange(0, T, 1/fs)
n = t.size

# Crear carpeta para figuras
os.makedirs("figs", exist_ok=True)

# -------------------------
# Señal de prueba: suma de senos + ruido blanco
# -------------------------
f1 = 50.0    # componente baja frecuencia (Hz)
f2 = 300.0   # componente media (Hz)
f3 = 700.0   # componente alta (Hz)
amp1, amp2, amp3 = 1.0, 0.8, 0.6

signal_clean = amp1*np.sin(2*np.pi*f1*t) + amp2*np.sin(2*np.pi*f2*t) + amp3*np.sin(2*np.pi*f3*t)
np.random.seed(0)
noise = 0.8*np.random.normal(scale=1.0, size=n)
x = signal_clean + noise

# -------------------------
# Funciones utilitarias
# -------------------------
def plot_time_signal(tt, sig, title, filename, xlim=None):
    plt.figure(figsize=(8,3))
    plt.plot(tt, sig)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(title)
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_fft(sig, fs, title, filename):
    N = len(sig)
    S = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    S_mag = np.abs(S) / N
    plt.figure(figsize=(8,3))
    plt.semilogy(freqs, S_mag + 1e-16)  # semilogy para ver amplio rango
    plt.xlim(0, fs/2)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud (espectral)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_freq_response(b, a, fs, title, filename):
    w, h = signal.freqz(b, a, worN=8000)
    freqs = w * fs / (2*np.pi)
    plt.figure(figsize=(8,3))
    plt.plot(freqs, 20*np.log10(np.abs(h)+1e-16))
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title(title)
    plt.xlim(0, fs/2)
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# -------------------------
# Graficas iniciales
# -------------------------
plot_time_signal(t, x, "Señal ruidosa (completa)", "figs/01_senal_tiempo_total.png")
plot_time_signal(t, x, "Señal ruidosa (zoom 0-0.05s)", "figs/02_senal_tiempo_zoom.png", xlim=(0,0.05))
plot_fft(x, fs, "Espectro de la señal sin filtrar", "figs/03_espectro_senal_sin_filtrar.png")

# -------------------------
# 1) FILTRO PASA-BAJOS (Butterworth IIR)
# -------------------------
fc_lp = 200.0  # fc cutoff lowpass (Hz)
order_lp = 4
wn = fc_lp / (fs/2)  # frecuencia normalizada
b_lp, a_lp = signal.butter(order_lp, wn, btype='low', analog=False)
y_lp = signal.filtfilt(b_lp, a_lp, x)

plot_freq_response(b_lp, a_lp, fs, f"Respuesta en frecuencia - Butterworth LP (fc={fc_lp}Hz, order={order_lp})", "figs/04_freqresp_butter_lp.png")
plot_time_signal(t, y_lp, "Señal filtrada - Pasa bajos (Butterworth)", "figs/05_senal_lp_tiempo.png", xlim=(0,0.05))
plot_fft(y_lp, fs, "Espectro señal - Pasa bajos (Butterworth)", "figs/06_espectro_lp.png")

# -------------------------
# 2) FILTRO PASA-ALTOS (Chebyshev I IIR)
# -------------------------
fc_hp = 250.0  # cutoff highpass (Hz)
order_hp = 4
ripple = 1  # dB ripple
wn_hp = fc_hp / (fs/2)
b_hp, a_hp = signal.cheby1(order_hp, ripple, wn_hp, btype='high', analog=False)
y_hp = signal.filtfilt(b_hp, a_hp, x)

plot_freq_response(b_hp, a_hp, fs, f"Respuesta en frecuencia - Chebyshev I HP (fc={fc_hp}Hz, order={order_hp})", "figs/07_freqresp_cheby_hp.png")
plot_time_signal(t, y_hp, "Señal filtrada - Pasa altos (Chebyshev I)", "figs/08_senal_hp_tiempo.png", xlim=(0,0.05))
plot_fft(y_hp, fs, "Espectro señal - Pasa altos (Chebyshev I)", "figs/09_espectro_hp.png")

# -------------------------
# 3) FILTRO PASA-BANDA (FIR con ventana)
# -------------------------
f1_pb = 280.0
f2_pb = 320.0
numtaps = 251  # número de coeficientes FIR (ajustable)
bands = [0, f1_pb-10, f1_pb, f2_pb, f2_pb+10, fs/2]
desired = [0, 0, 1, 1, 0, 0]  # ideal banda entre f1_pb y f2_pb

# Normalizar bandas para firwin2
bands_norm = [b/(fs/2) for b in bands]
# Usar remez (Parks-McClellan) o firwin2; aquí firwin2 para simplicidad:
from scipy.signal import firwin
# diseño pasa banda con firwin (bandpass)
b_pb = firwin(numtaps, [f1_pb, f2_pb], pass_zero=False, fs=fs)
a_pb = [1.0]
y_pb = signal.filtfilt(b_pb, a_pb, x)

# Respuesta en frecuencia FIR
w, h = signal.freqz(b_pb, a_pb, worN=8000)
freqs = w * fs / (2*np.pi)
plt.figure(figsize=(8,3))
plt.plot(freqs, 20*np.log10(np.abs(h)+1e-16))
plt.title(f"Respuesta - FIR Pasa-banda {f1_pb}-{f2_pb} Hz (numtaps={numtaps})")
plt.xlabel("Frecuencia [Hz]"); plt.ylabel("Magnitud [dB]"); plt.xlim(0, fs/2)
plt.grid(True, ls='--', alpha=0.3); plt.tight_layout()
plt.savefig("figs/10_freqresp_fir_pb.png", dpi=150); plt.close()

plot_time_signal(t, y_pb, f"Señal filtrada - Pasa banda {f1_pb}-{f2_pb} Hz (FIR)", "figs/11_senal_pb_tiempo.png", xlim=(0,0.05))
plot_fft(y_pb, fs, f"Espectro señal - Pasa banda {f1_pb}-{f2_pb} Hz (FIR)", "figs/12_espectro_pb.png")

# -------------------------
# 4) Comparación gráfica (subplots)
# -------------------------
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t, x); plt.title("Señal sin filtrar (ruidosa)"); plt.xlim(0,0.02)
plt.subplot(3,1,2)
plt.plot(t, y_lp); plt.title("Filtro Pasa-bajos (Butterworth)"); plt.xlim(0,0.02)
plt.subplot(3,1,3)
plt.plot(t, y_hp); plt.title("Filtro Pasa-altos (Chebyshev I)"); plt.xlim(0,0.02)
plt.tight_layout(); plt.savefig("figs/13_comparacion_tiempo.png", dpi=150); plt.close()

# -------------------------
# 5) Guardar coeficientes y resumen breve
# -------------------------
np.savetxt("figs/b_lp_coeffs.txt", b_lp)
np.savetxt("figs/a_lp_coeffs.txt", a_lp)
np.savetxt("figs/b_hp_coeffs.txt", b_hp)
np.savetxt("figs/a_hp_coeffs.txt", a_hp)
np.savetxt("figs/b_pb_coeffs.txt", b_pb)

with open("figs/resumen.txt", "w") as f:
    f.write("Resumen de filtros aplicados\n")
    f.write(f"Fs={fs} Hz, duración={T}s\n")
    f.write(f"LP Butterworth: fc={fc_lp}Hz, order={order_lp}\n")
    f.write(f"HP Chebyshev I: fc={fc_hp}Hz, order={order_hp}, ripple={ripple} dB\n")
    f.write(f"PB FIR: {f1_pb}-{f2_pb} Hz, numtaps={numtaps}\n")

print("Ejecucion completada. Figuras guardadas en ./figs y coeficientes en ./figs/*.txt")
