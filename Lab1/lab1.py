"""
Лабораторна робота №1 з дисципліни «Цифрова обробка інформації»
Тема: ДЕТЕРМІНОВАНІ СИГНАЛИ ДЛЯ ОБРОБКИ ІНФОРМАЦІЇ

Виконав: студент Чепара Станіслав
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Створюємо теку для збереження графіків (для звіту)
os.makedirs('figures', exist_ok=True)


# =====================================================================
# ЧАСТИНА 1. ЗАВДАННЯ 2 (Код з додатку до методички)
# Реалізація прямокутного сигналу та його аналітичного Фур'є-образу (sinc)
# =====================================================================

def Rectangular(x: float, width=1.0):
    """Прямокутна функція (прямокутний імпульс) шириною width."""
    if abs(x) <= width / 2.0:
        return 1.0
    else:
        return 0.0

def sinc_function(x: float):
    """Функція sinc: sin(πx)/(πx)."""
    if x == 0:
        return 1.0  # границя при x -> 0
    else:
        return math.sin(math.pi * x) / (math.pi * x)

def run_task2():
    print("-> Виконання Завдання 2 (код з додатку)...")
    width = 1.0         # ширина імпульсу
    sample_rate = 0.01  # крок дискретизації
    time_range = 3.0    # діапазон часу

    time = np.arange(-time_range, time_range, sample_rate)
    rect_signal = [Rectangular(t, width) for t in time]

    frequency = np.arange(-4.0, 4.0, sample_rate)
    analytic_ft = [width * sinc_function(width * f) for f in frequency]

    # Графік 1: Часова та частотна області поруч
    plt.figure(figsize=(11, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(time, rect_signal, 'b-', linewidth=2)
    plt.title('Прямокутний сигнал\n(часова область)')
    plt.xlabel('Час (t)')
    plt.ylabel('Амплітуда')
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-0.2, 1.2)

    plt.subplot(1, 2, 2)
    plt.plot(frequency, analytic_ft, 'r-', linewidth=2)
    plt.title('Перетворення Фур\'є\n(sinc функція)')
    plt.xlabel('Частота (f)')
    plt.ylabel('Амплітуда')
    plt.grid(True)
    plt.xlim(-4, 4)
    plt.tight_layout()
    plt.savefig('figures/appendix_rect_sinc.png', dpi=300)

    # Графік 2: Детальний графік sinc-функції
    plt.figure(figsize=(7, 4.5))
    plt.plot(frequency, analytic_ft, 'r-', linewidth=2)
    plt.title('Sinc функція - перетворення Фур\'є прямокутного сигналу')
    plt.xlabel('Частота (f)')
    plt.ylabel('Амплітуда')
    plt.text(0, 0.8, f'sinc(f) = sin(π·{width}·f)/(π·{width}·f)',
             fontsize=11, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True)
    plt.xlim(-4, 4)
    plt.tight_layout()
    plt.savefig('figures/appendix_sinc_detailed.png', dpi=300)


# =====================================================================
# ЧАСТИНА 2. ЗАВДАННЯ 3 (Моделювання одномірних детермінованих сигналів)
# =====================================================================

def plot_harmonic():
    """1. Періодичні сигнали cos(pi*t) та sin(pi*t) та їх спектри."""
    t = np.linspace(-3, 3, 1000)
    cos_sig = np.cos(np.pi * t)
    sin_sig = np.sin(np.pi * t)
    f0 = 0.5  # оскільки cos(pi*t) = cos(2*pi*f0*t) => f0 = 0.5 Гц

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # cos(pi*t)
    axes[0, 0].plot(t, cos_sig, 'b-', linewidth=1.8)
    axes[0, 0].set_title('Сигнал: cos(πt) — Часова область')
    axes[0, 0].set_xlabel('Час t (с)')
    axes[0, 0].set_ylabel('Амплітуда')
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim(-1.3, 1.3)

    # Дійсна частина спектра косинуса: 0.5*delta(f+0.5) + 0.5*delta(f-0.5)
    axes[0, 1].stem([-f0, f0], [0.5, 0.5], linefmt='b-', markerfmt='b^', basefmt='k-')
    axes[0, 1].set_title('Спектр F[cos(πt)] = 0.5·δ(f+0.5) + 0.5·δ(f-0.5)')
    axes[0, 1].set_xlabel('Частота f (Гц)')
    axes[0, 1].set_ylabel('Re{X(f)}')
    axes[0, 1].grid(True)
    axes[0, 1].set_xlim(-1.5, 1.5)
    axes[0, 1].set_ylim(0, 0.7)

    # sin(pi*t)
    axes[1, 0].plot(t, sin_sig, 'g-', linewidth=1.8)
    axes[1, 0].set_title('Сигнал: sin(πt) — Часова область')
    axes[1, 0].set_xlabel('Час t (с)')
    axes[1, 0].set_ylabel('Амплітуда')
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(-1.3, 1.3)

    # Уявна частина спектра синуса: 0.5i*delta(f+0.5) - 0.5i*delta(f-0.5)
    axes[1, 1].stem([-f0, f0], [0.5, -0.5], linefmt='g--', markerfmt='g^', basefmt='k-')
    axes[1, 1].set_title('Спектр F[sin(πt)] — Уявна частина Im{X(f)}')
    axes[1, 1].set_xlabel('Частота f (Гц)')
    axes[1, 1].set_ylabel('Im{X(f)}')
    axes[1, 1].grid(True)
    axes[1, 1].set_xlim(-1.5, 1.5)
    axes[1, 1].set_ylim(-0.7, 0.7)

    plt.tight_layout()
    plt.savefig('figures/1_harmonic.png', dpi=300)

def plot_dirac_delta():
    """2. Дельта-функція Дірака та її рівномірний спектр."""
    dt = 0.005
    t = np.arange(-2, 2 + dt, dt)
    delta_sig = np.zeros_like(t)
    idx_zero = np.argmin(np.abs(t))
    delta_sig[idx_zero] = 1.0 / dt  # площа під імпульсом = 1

    f = np.linspace(-50, 50, 1000)
    spectrum = np.ones_like(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(t, delta_sig, 'b-', linewidth=2)
    ax1.set_title('Дельта-функція Дірака δ(t) — Часова область')
    ax1.set_xlabel('Час t (с)')
    ax1.set_ylabel('Амплітуда (апроксимація 1/dt)')
    ax1.grid(True)
    ax1.set_xlim(-1, 1)

    ax2.plot(f, spectrum, 'r-', linewidth=2)
    ax2.set_title('Спектр дельта-функції: F[δ(t)] = 1')
    ax2.set_xlabel('Частота f (Гц)')
    ax2.set_ylabel('|X(f)| (Рівномірний спектр)')
    ax2.grid(True)
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig('figures/2_dirac_delta.png', dpi=300)

def plot_rectangular():
    """3. Прямокутний імпульс rect(t/T) та його спектр sinc."""
    T = 2.0
    t = np.linspace(-3, 3, 1000)
    rect_sig = np.where(np.abs(t) <= T / 2.0, 1.0, 0.0)

    f = np.linspace(-4, 4, 1000)
    spectrum = T * np.sinc(T * f)  # np.sinc(x) = sin(pi*x)/(pi*x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(t, rect_sig, 'b-', linewidth=2)
    ax1.set_title(f'Прямокутний імпульс rect(t/{T:.0f}) — Часова область')
    ax1.set_xlabel('Час t (с)')
    ax1.set_ylabel('Амплітуда')
    ax1.grid(True)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.2, 1.2)

    ax2.plot(f, spectrum, 'r-', linewidth=2)
    ax2.set_title(f'Спектр прямокутного імпульсу: {T:.0f}·sinc({T:.0f}f)')
    ax2.set_xlabel('Частота f (Гц)')
    ax2.set_ylabel('X(f)')
    ax2.grid(True)
    ax2.set_xlim(-4, 4)

    plt.tight_layout()
    plt.savefig('figures/3_rectangular.png', dpi=300)

def plot_sinc():
    """4. Функція відліків sinc(t) та прямокутний спектр rect(f)."""
    t = np.linspace(-6, 6, 1000)
    sinc_sig = np.sinc(t)

    f = np.linspace(-2, 2, 1000)
    spectrum = np.where(np.abs(f) <= 0.5, 1.0, 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(t, sinc_sig, 'b-', linewidth=2)
    ax1.set_title('Функція відліків sinc(t) — Часова область')
    ax1.set_xlabel('Час t (с)')
    ax1.set_ylabel('Амплітуда')
    ax1.grid(True)
    ax1.set_xlim(-6, 6)

    ax2.plot(f, spectrum, 'r-', linewidth=2)
    ax2.set_title('Спектр функції sinc(t): F[sinc(t)] = rect(f)')
    ax2.set_xlabel('Частота f (Гц)')
    ax2.set_ylabel('X(f) (Ідеальний НЧ-фільтр)')
    ax2.grid(True)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig('figures/4_sinc.png', dpi=300)

def plot_signum():
    """5. Функція signum sgn(t) та її спектр 1/(j*pi*f)."""
    t = np.linspace(-3, 3, 1000)
    sgn_sig = np.sign(t)

    f_pos = np.linspace(0.05, 3, 500)
    f_neg = np.linspace(-3, -0.05, 500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(t, sgn_sig, 'b-', linewidth=2)
    ax1.set_title('Функція signum sgn(t) — Часова область')
    ax1.set_xlabel('Час t (с)')
    ax1.set_ylabel('Амплітуда')
    ax1.grid(True)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1.3, 1.3)

    ax2.plot(f_neg, -1.0 / (np.pi * f_neg), 'r-', linewidth=2)
    ax2.plot(f_pos, -1.0 / (np.pi * f_pos), 'r-', linewidth=2)
    ax2.set_title('Спектр F[sgn(t)] — Уявна частина 1/(j·π·f)')
    ax2.set_xlabel('Частота f (Гц)')
    ax2.set_ylabel('Im{X(f)}')
    ax2.grid(True)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-5, 5)

    plt.tight_layout()
    plt.savefig('figures/5_signum.png', dpi=300)

def plot_triangular():
    """6. Трикутна функція Lambda(t/T) та її спектр sinc^2."""
    T = 2.0
    t = np.linspace(-3, 3, 1000)
    tri_sig = np.maximum(0.0, 1.0 - np.abs(t) / T)

    f = np.linspace(-3, 3, 1000)
    spectrum = T * (np.sinc(T * f) ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(t, tri_sig, 'b-', linewidth=2)
    ax1.set_title(f'Трикутна функція Λ(t/{T:.0f}) — Часова область')
    ax1.set_xlabel('Час t (с)')
    ax1.set_ylabel('Амплітуда')
    ax1.grid(True)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.1, 1.2)

    ax2.plot(f, spectrum, 'r-', linewidth=2)
    ax2.set_title(f'Спектр трикутної функції: {T:.0f}·sinc²({T:.0f}f)')
    ax2.set_xlabel('Частота f (Гц)')
    ax2.set_ylabel('X(f)')
    ax2.grid(True)
    ax2.set_xlim(-3, 3)

    plt.tight_layout()
    plt.savefig('figures/6_triangular.png', dpi=300)


def main():
    print("==================================================================")
    print("Лабораторна робота №1: Детерміновані сигнали для обробки інформації")
    print("==================================================================")

    # 1. Завдання 2
    run_task2()

    # 2. Завдання 3
    print("-> Виконання Завдання 3 (моделювання детермінованих сигналів)...")
    plot_harmonic()
    plot_dirac_delta()
    plot_rectangular()
    plot_sinc()
    plot_signum()
    plot_triangular()

    print("-> Усі графіки успішно побудовано та збережено в папку 'figures/'!")
    plt.show()

if __name__ == '__main__':
    main()
