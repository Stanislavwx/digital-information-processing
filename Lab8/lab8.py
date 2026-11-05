"""
Лабораторна робота №8 з дисципліни «Цифрова обробка інформації»
Тема: Швидке дискретне перетворення Фур’є (ШПФ)
Студент: Чепара Станіслав, група ФеП-32
Варіант: 11
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import time
import numpy as np
import matplotlib.pyplot as plt

# Базова директорія для збереження графіків
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def W(k, N, n=1):
    """
    Обчислення поворотного множника W_N^(k*n) = exp(-2*pi*j*k*n / N)
    """
    return np.exp(-2 * np.pi * 1j * k * n / N)


def dft(x):
    """
    Пряме дискретне перетворення Фур'є (ДПФ) за базовим визначенням O(N^2)
    """
    N = len(x)
    return np.array([sum(x[n] * W(k, N, n) for n in range(N)) for k in range(N)])


def fft(x):
    """
    Швидке перетворення Фур'є (ШПФ) за алгоритмом Кулі-Тюкі (прорідження у часі)
    """
    N = len(x)
    if N % 2 > 0:
        raise ValueError("Кількість відліків повинна бути степенем 2")

    # Базовий випадок рекурсії (операція 'метелик' для N = 2)
    if N == 2:
        return [x[0] + x[1], x[0] - x[1]]

    # Рекурсивне розбиття на парні та непарні відліки
    x_even = fft(x[::2])
    x_odd = fft(x[1::2])

    Y1 = []
    Y2 = []
    for k in range(N // 2):
        twiddle = W(k, N) * x_odd[k]
        Y1.append(x_even[k] + twiddle)
        Y2.append(x_even[k] - twiddle)

    return Y1 + Y2


def ifft(X):
    """
    Обернене швидке перетворення Фур'є (ОШПФ)
    Використовує властивість комплексної спряженості: ifft(X) = conj(fft(conj(X))) / N
    """
    N = len(X)
    X_conj = [np.conj(val) for val in X]
    res = fft(X_conj)
    return [np.real(np.conj(val)) / N for val in res]


def main():
    print("=" * 70)
    print("  Лабораторна робота №8: Швидке дискретне перетворення Фур’є (ШПФ)")
    print("  Студент: Чепара Станіслав, група ФеП-32")
    print("=" * 70)

    # 1. Формування модельного ряду (формула 12 з методички)
    # X(t) = cos(2*pi*t/10 + 1) + cos(2*pi*t/40 + pi/2), t in [0, 480], dt = 0.7
    dt = 0.7
    t = np.arange(0, 480, dt)
    x = np.cos(2 * np.pi * t / 10 + 1) + np.cos(2 * np.pi * t / 40 + np.pi / 2)

    # Вибірка N = 512 відліків (найближча степінь двійки)
    N = 512
    xp2 = x[:N]
    t_p2 = t[:N]

    print(f"Початкова кількість точок t < 480 (dt = {dt}): {len(t)}")
    print(f"Кількість відліків для ШПФ (степінь 2):        {N}")

    # 2. Вимірювання часу виконання ДПФ та ШПФ
    print("\n[1] Обчислення спектрів та вимірювання часу...")

    t_start = time.perf_counter()
    X_dft = dft(xp2)
    t_dft = time.perf_counter() - t_start

    t_start = time.perf_counter()
    X_fft = np.array(fft(xp2))
    t_fft = time.perf_counter() - t_start

    t_start = time.perf_counter()
    X_numpy = np.fft.fft(xp2)
    t_numpy = time.perf_counter() - t_start

    speedup = t_dft / t_fft if t_fft > 0 else 0
    max_diff = np.max(np.abs(X_dft - X_fft))

    print(f"  Час ДПФ (O(N^2)):                  {t_dft:.5f} с")
    print(f"  Час ШПФ Кулі-Тюкі (O(N*log2(N))):  {t_fft:.5f} с")
    print(f"  Час бібліотечного numpy.fft:        {t_numpy:.6f} с")
    print(f"  Експериментальне прискорення:      {speedup:.2f}x")
    print(f"  Теоретичне прискорення (2N/log2N): {2 * N / np.log2(N):.2f}x")
    print(f"  Максимальна різниця |ДПФ - ШПФ|:   {max_diff:.2e}")

    # 3. Зворотне перетворення (ОШПФ) та оцінка точності
    print("\n[2] Відновлення сигналу за допомогою ОШПФ...")
    x_reconstructed = np.array(ifft(X_fft))
    reconstruction_error = np.max(np.abs(xp2 - x_reconstructed))
    print(f"  Максимальна похибка реконструкції: {reconstruction_error:.2e}")

    # 4. Розрахунок періодів та амплітудно-фазового спектра
    nu = np.arange(N) / dt / N
    with np.errstate(divide="ignore"):
        T = 1.0 / nu
    amplitude = np.abs(X_fft)
    phase = np.angle(X_fft)

    # Визначення домінуючих гармонік у першій половині спектра
    half = N // 2
    top_indices = np.argsort(amplitude[1:half])[-2:][::-1] + 1
    print("\n[3] Домінуючі піки у спектрі:")
    for idx in sorted(top_indices):
        print(f"  Пік k = {idx:2d}: частота f = {nu[idx]:.4f} Гц, період T = {T[idx]:.2f} с, "
              f"амплітуда = {amplitude[idx]:.2f}, фаза = {phase[idx]:.4f} рад")

    # 5. Побудова графіків
    print("\n[4] Генерація графіків для звіту...")

    # Рис. 1. Графік початкового модельного сигналу
    plt.figure(figsize=(10, 3.8))
    plt.plot(t, x, color="#1f77b4", linewidth=1.2)
    plt.title("Рис. 1. Графічний вигляд функції X(t) = cos(2πt/10 + 1) + cos(2πt/40 + π/2)")
    plt.xlabel("Час t, с")
    plt.ylabel("X(t)")
    plt.xlim(0, 480)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_signal.png"), dpi=300)
    plt.close()

    # Рис. 2. Графік дійсної та уявної частини поворотного множника W_N^k (N = 20)
    k_range = np.arange(100)
    w_values = [W(ki, 20) for ki in k_range]
    plt.figure(figsize=(10, 4.8))
    plt.subplot(2, 1, 1)
    plt.plot(k_range, np.real(w_values), color="#1f77b4", linewidth=1.2)
    plt.title("Рис. 2. Дійсна (Re) та уявна (Im) частини функції W_N^k (N = 20)")
    plt.ylabel("Re(W)")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(k_range, np.imag(w_values), color="#ff7f0e", linewidth=1.2)
    plt.xlabel("k")
    plt.ylabel("Im(W)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_twiddle.png"), dpi=300)
    plt.close()

    # Рис. 3. Періодограма та фазовий спектр за допомогою ДПФ
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.semilogy(T[0:N // 2], np.abs(X_dft[0:N // 2]), color="#2ca02c", linewidth=1.3)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100, step=10))
    plt.title("Рис. 3. Періодограма, отримана за допомогою класичного ДПФ")
    plt.ylabel("Амплітуда |X(T)|")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(T[0:N // 2], np.angle(X_dft[0:N // 2]), color="#d62728", linewidth=1.3)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100, step=10))
    plt.xlabel("Період T, с")
    plt.ylabel("Фаза, рад")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_dft_periodogram.png"), dpi=300)
    plt.close()

    # Рис. 4. Періодограма та фазовий спектр за допомогою ШПФ
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.semilogy(T[0:N // 2], amplitude[0:N // 2], color="#2ca02c", linewidth=1.3)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100, step=10))
    plt.title("Рис. 4. Періодограма, отримана за допомогою ШПФ Кулі-Тюкі")
    plt.ylabel("Амплітуда |X(T)|")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    plt.plot(T[0:N // 2], phase[0:N // 2], color="#d62728", linewidth=1.3)
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 100, step=10))
    plt.xlabel("Період T, с")
    plt.ylabel("Фаза, рад")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_fft_periodogram.png"), dpi=300)
    plt.close()

    # Рис. 5. Результати відновлення сигналу через ОШПФ
    plt.figure(figsize=(10, 4.5))
    plt.subplot(2, 1, 1)
    plt.plot(t_p2, xp2, label="Початковий сигнал", color="#1f77b4", linewidth=1.4)
    plt.plot(t_p2, x_reconstructed, "--", label="Відновлений сигнал (ОШПФ)", color="#d62728", linewidth=1.2)
    plt.title("Рис. 5. Точність відновлення сигналу за допомогою алгоритму ОШПФ")
    plt.ylabel("Амплітуда")
    plt.xlim(0, t_p2[-1])
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(2, 1, 2)
    diff_signal = xp2 - x_reconstructed
    plt.plot(t_p2, diff_signal, color="#7f7f7f", linewidth=1.0)
    plt.xlabel("Час t, с")
    plt.ylabel("Похибка (Δx)")
    plt.xlim(0, t_p2[-1])
    plt.ylim(-1e-14, 1e-14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_reconstructed.png"), dpi=300)
    plt.close()

    # Рис. 6. Зведене порівняння ДПФ та ШПФ (структура і швидкість)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5))
    axes[0].plot(t_p2, xp2, color="#1f77b4", linewidth=1.2)
    axes[0].set_title("1. Модельний сигнал x(t) [перші 512 відліків]")
    axes[0].set_ylabel("Амплітуда")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].semilogy(T[0:N // 2], amplitude[0:N // 2], color="#2ca02c", linewidth=1.3)
    axes[1].set_title("2. Періодограма ШПФ (виділені гармоніки з T = 10 с та T = 40 с)")
    axes[1].set_xlim(0, 100)
    axes[1].set_xticks(np.arange(0, 100, step=10))
    axes[1].set_ylabel("|X(T)|")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)

    methods = ["ДПФ O(N²)", "ШПФ Кулі-Тюкі", "NumPy FFT"]
    times = [t_dft, t_fft, t_numpy]
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    bars = axes[2].barh(methods, times, color=colors, height=0.55)
    axes[2].set_xscale("log")
    axes[2].set_title(f"3. Порівняння часу обчислення (Прискорення ШПФ відносно ДПФ: {speedup:.1f}x)")
    axes[2].set_xlabel("Час виконання, секунди (логарифмічна шкала)")
    for bar in bars:
        width = bar.get_width()
        axes[2].text(width * 1.2, bar.get_y() + bar.get_height() / 2,
                     f"{width:.5f} с", va="center", ha="left", fontsize=9, fontweight="bold")
    axes[2].grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_summary.png"), dpi=300)
    plt.close()

    print(f"Всі графіки успішно збережено в папку: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

