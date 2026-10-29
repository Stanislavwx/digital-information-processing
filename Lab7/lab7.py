"""
Лабораторна робота №7 з дисципліни «Цифрова обробка інформації»
Тема: Дискретне перетворення Фур’є (ДПФ)
Варіант 11: X(t) = sin(2*pi*t/20 + pi/4) + sin(2*pi*t/20 + pi/2),  t in [0, 480]
Студент: Чепара Станіслав, група ФеП-32
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Базова директорія для збереження результатів
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 65)
    print("  Лабораторна робота №7: Дискретне перетворення Фур’є (ДПФ)")
    print("  Варіант 11: X(t) = sin(2πt/20 + π/4) + sin(2πt/20 + π/2)")
    print("  Часовий інтервал: t ∈ [0, 480]")
    print("=" * 65)

    # 1. Параметри сигналу (згідно з варіантом 11)
    # Задаємо N = 480 відліків (t = 0, 1, ..., 479)
    N = 480
    t = np.arange(N)

    # 2. Модельний сигнал (Варіант 11)
    # X(t) = sin(2*pi*t/20 + pi/4) + sin(2*pi*t/20 + pi/2)
    x = np.sin(2 * np.pi * t / 20 + np.pi / 4) + np.sin(2 * np.pi * t / 20 + np.pi / 2)

    # 3. Пряме дискретне перетворення Фур'є (ДПФ)
    X = np.fft.fft(x)
    freq = np.arange(N)

    # 4. Амплітудний і фазовий спектри
    amplitude = np.abs(X)
    phase = np.angle(X)

    # 5. Зворотне дискретне перетворення Фур'є (ОДПФ)
    x_reconstructed = np.fft.ifft(X).real

    # 6. Розрахунок характеристик та похибки відновлення
    max_error = np.max(np.abs(x - x_reconstructed))
    peak_idx = int(np.argmax(amplitude[: N // 2]))
    peak_amp = float(amplitude[peak_idx])
    norm_amp = float(2 * peak_amp / N)
    theoretical_amp = float(2 * np.cos(np.pi / 8))
    peak_phase = float(phase[peak_idx])
    theoretical_phase = float(-np.pi / 8)

    print(f"Кількість точок (N):                 {N}")
    print(f"Індекс спектрального піку (k):       {peak_idx} (дзеркальний: {N - peak_idx})")
    print(f"Циклічна частота піку f_k:          {peak_idx / N:.4f} Гц (період T = {N / peak_idx:.1f} відліків)")
    print(f"Амплітуда спектрального відліку:     {peak_amp:.2f}")
    print(f"Відновлена амплітуда (2|X|/N):       {norm_amp:.4f} (Теоретична: {theoretical_amp:.4f})")
    print(f"Фаза спектрального піку:             {peak_phase:.4f} рад (Теоретична: {theoretical_phase:.4f} рад)")
    print(f"Максимальна похибка ОДПФ:            {max_error:.2e}")
    print("=" * 65)

    # 7. Побудова зведеного графіка (структура з 4 підграфіків відповідно до методички)
    plt.figure(figsize=(12, 10))

    # Рис. 1. Початковий сигнал
    plt.subplot(4, 1, 1)
    plt.plot(t, x, color="#1f77b4", linewidth=1.5)
    plt.title("Рис.1. Графічний вираз функції (Варіант 11, t ∈ [0, 480])")
    plt.xlabel("n")
    plt.ylabel("Амплітуда")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Рис. 2. Фаза ДПФ
    plt.subplot(4, 1, 2)
    plt.plot(freq, phase, color="#ff7f0e", linewidth=1.2)
    plt.title("Рис.2. Графік фази ДПФ")
    plt.xlabel("k")
    plt.ylabel("Фаза, рад")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Рис. 3. Амплітуда ДПФ (логарифмічна шкала)
    plt.subplot(4, 1, 3)
    plt.semilogy(freq, amplitude, color="#2ca02c", linewidth=1.5)
    plt.title("Рис.3. Графік амплітуди ДПФ (логарифмічна шкала)")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    # Рис. 4. Зворотне перетворення
    plt.subplot(4, 1, 4)
    plt.plot(t, x_reconstructed, color="#d62728", linewidth=1.5)
    plt.title("Рис.4. Зворотне перетворення (відновлений сигнал)")
    plt.xlabel("n")
    plt.ylabel("Амплітуда")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "dft_variant11_summary.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"Зведений графік збережено: {summary_path}")

    # 8. Окремі графіки високої чіткості для включення у звіт
    # Графік 1: Початковий сигнал
    plt.figure(figsize=(9, 3.8))
    plt.plot(t, x, color="#1f77b4", linewidth=1.5)
    plt.title("Рис.1. Графічний вираз функції (Варіант 11)")
    plt.xlabel("Час t, відліки n")
    plt.ylabel("X(t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_signal.png"), dpi=300)
    plt.close()

    # Графік 2: Фазовий спектр
    plt.figure(figsize=(9, 3.8))
    plt.plot(freq, phase, color="#ff7f0e", linewidth=1.2)
    plt.title("Рис.2. Графік фази ДПФ")
    plt.xlabel("Індекс частоти k")
    plt.ylabel("arg(X[k]), рад")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_phase.png"), dpi=300)
    plt.close()

    # Графік 3: Амплітудний спектр
    plt.figure(figsize=(9, 3.8))
    plt.semilogy(freq, amplitude, color="#2ca02c", linewidth=1.5)
    plt.title("Рис.3. Графік амплітуди ДПФ (періодограма)")
    plt.xlabel("Індекс частоти k")
    plt.ylabel("|X[k]|")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_amplitude.png"), dpi=300)
    plt.close()

    # Графік 4: Відновлений сигнал
    plt.figure(figsize=(9, 3.8))
    plt.plot(t, x_reconstructed, color="#d62728", linewidth=1.5)
    plt.title("Рис.4. Зворотне дискретне перетворення Фур'є (ОДПФ)")
    plt.xlabel("Час t, відліки n")
    plt.ylabel("x_rec[n]")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_reconstructed.png"), dpi=300)
    plt.close()

    print("Всі індивідуальні графіки збережено в папку output/")


if __name__ == "__main__":
    main()
