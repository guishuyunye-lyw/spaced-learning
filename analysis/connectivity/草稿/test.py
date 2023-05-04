import numpy as np
import matplotlib.pyplot as plt

# 创建信号
fs = 1000
t = np.linspace(0, 1, fs)
a1, a2 = 1, 1
w = 2 * np.pi * 5
signal = a1 * np.cos(w * t + 1) + a2 * np.cos(2 * w * t + 2)

# 计算相位谱
fft_signal = np.fft.rfft(signal)
phase_spectrum = np.angle(fft_signal)
frequencies = np.fft.rfftfreq(len(t), 1/fs)

# 绘制相位谱
plt.plot(frequencies, phase_spectrum)
plt.title('Phase Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase Angle (Radians)')
plt.grid()
plt.show()
