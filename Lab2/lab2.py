"""
Лабораторна робота №2 з дисципліни «Цифрова обробка інформації»
Тема: ОСНОВНІ ОПЕРАЦІЇ НАД СИГНАЛАМИ
Студент: Чепара С.

Програма містить:
1. Реалізацію основних операцій для неперервних (аналогових) сигналів (на основі коду з додатку).
2. Реалізацію основних операцій для дискретних сигналів (дискретний час, децимація, stem-графіки).
"""

import os
import math
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

# Створюємо директорію для збереження графіків
os.makedirs('figures', exist_ok=True)


# =====================================================================
# ЧАСТИНА 1. ОСНОВНІ ОПЕРАЦІЇ НАД НЕПЕРЕРВНИМИ СИГНАЛАМИ (з додатку)
# =====================================================================

def frange(start, stop=None, step=None):
    """Генератор чисел з плаваючою крапкою для часової сітки."""
    start = float(start)
    if stop is None:
        stop = start + 0.0
        start = 0.0
    if step is None:
        step = 1.0
    count = 0
    while True:
        temp = float(start + count * step)
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield temp
        count += 1


class Signal:
    """Клас для генерації та виконання базових операцій над неперервними сигналами."""

    def __init__(self):
        # Ініціалізуємо список окремо для кожного екземпляра
        self.SignalReal = [[], []]

    def SignalGenerator(self, Xboundary0, Xboundary1, Yboundary0, Yboundary1, variant):
        """
        Генерація тестового сигналу:
        variant = 0: випадковий шум
        variant = 1: синусоїдальний сигнал
        variant = 2: косинусоїдальний сигнал
        """
        self.SignalReal = [[], []]
        step = abs(Xboundary0 - Xboundary1) / 1000.0

        if variant == 0:
            for i in frange(Xboundary0, Xboundary1, step):
                self.SignalReal[0].append(i)
                self.SignalReal[1].append(rnd.uniform(Yboundary0, Yboundary1))
        elif variant == 1:
            for i in frange(Xboundary0, Xboundary1, step):
                self.SignalReal[0].append(i)
                self.SignalReal[1].append(math.sin(i + Yboundary0) * Yboundary1)
        elif variant == 2:
            for i in frange(Xboundary0, Xboundary1, step):
                self.SignalReal[0].append(i)
                self.SignalReal[1].append(math.cos(i + Yboundary0) * Yboundary1)

    def Scaling(self, InSignal, Scale):
        """1. Масштабування амплітуди: y(t) = a * x(t)."""
        Scaled = [InSignal[0].copy(), [i * Scale for i in InSignal[1]]]
        return Scaled

    def TimeReversal(self, InSignal):
        """2. Реверс за часом: y(t) = x(-t). Віддзеркалення відносно осі часу."""
        Reversed = [[-i for i in reversed(InSignal[0])], list(reversed(InSignal[1]))]
        return Reversed

    def TimeShift(self, InSignal, Shift):
        """3. Зсув у часі: y(t) = x(t - Shift). При Shift > 0 затримка (вправо)."""
        Shifted = [[i + Shift for i in InSignal[0]], InSignal[1].copy()]
        return Shifted

    def Widen(self, InSignal, WidthFactor):
        """4. Розширення / масштабування осі часу: t_new = t * WidthFactor."""
        Widened = [[i * WidthFactor for i in InSignal[0]], InSignal[1].copy()]
        return Widened

    def Combination(self, InSignal1, InSignal2):
        """5.1 Додавання сигналів: y(t) = x1(t) + x2(t)."""
        t1, s1 = InSignal1[0], InSignal1[1]
        t2, s2 = InSignal2[0], InSignal2[1]
        t_start = max(min(t1), min(t2))
        t_end = min(max(t1), max(t2))
        t_common = np.linspace(t_start, t_end, 1000)
        y1 = np.interp(t_common, t1, s1)
        y2 = np.interp(t_common, t2, s2)
        return [list(t_common), list(y1 + y2)]

    def Multiplication(self, InSignal1, InSignal2):
        """5.2 Множення сигналів: y(t) = x1(t) * x2(t)."""
        t1, s1 = InSignal1[0], InSignal1[1]
        t2, s2 = InSignal2[0], InSignal2[1]
        t_start = max(min(t1), min(t2))
        t_end = min(max(t1), max(t2))
        t_common = np.linspace(t_start, t_end, 1000)
        y1 = np.interp(t_common, t1, s1)
        y2 = np.interp(t_common, t2, s2)
        return [list(t_common), list(y1 * y2)]


def run_continuous_signals():
    """Виконання операцій для неперервних сигналів та побудова графіків."""
    print("-> Запуск операцій для неперервних сигналів...")

    # Базовий сигнал: косинусоїда (варіант 2, інтервал [0, 10], амплітуда 1)
    gen = Signal()
    gen.SignalGenerator(0, 10, 0, 1, 2)
    orig = gen.SignalReal

    # Коефіцієнти для операцій
    shift_val = 3.0       # зсув на 3 одиниці праворуч
    scale_val = 2.5       # підсилення амплітуди у 2.5 рази
    width_val = 2.0       # розширення осі часу у 2 рази

    sig_shift = gen.TimeShift(orig, shift_val)
    sig_rev = gen.TimeReversal(orig)
    sig_scale = gen.Scaling(orig, scale_val)
    sig_widen = gen.Widen(orig, width_val)

    # 1. Окремі графіки для кожної операції
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(orig[0], orig[1], 'b-', label='Оригінал x(t)')
    axs[0, 0].plot(sig_shift[0], sig_shift[1], 'g--', label=f'Зсув x(t - {shift_val})')
    axs[0, 0].set_title('Часовий зсув (затримка)')
    axs[0, 0].set_xlabel('Час (t)')
    axs[0, 0].set_ylabel('Амплітуда')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(orig[0], orig[1], 'b-', label='Оригінал x(t)')
    axs[0, 1].plot(sig_rev[0], sig_rev[1], 'm--', label='Реверс x(-t)')
    axs[0, 1].set_title('Часовий реверс (відображення)')
    axs[0, 1].set_xlabel('Час (t)')
    axs[0, 1].set_ylabel('Амплітуда')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(orig[0], orig[1], 'b-', label='Оригінал x(t)')
    axs[1, 0].plot(sig_scale[0], sig_scale[1], 'r--', label=f'Масштабування {scale_val}·x(t)')
    axs[1, 0].set_title('Масштабування амплітуди (підсилення)')
    axs[1, 0].set_xlabel('Час (t)')
    axs[1, 0].set_ylabel('Амплітуда')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(orig[0], orig[1], 'b-', label='Оригінал x(t)')
    axs[1, 1].plot(sig_widen[0], sig_widen[1], 'c--', label=f'Розширення часу (k={width_val})')
    axs[1, 1].set_title('Часове масштабування (розширення)')
    axs[1, 1].set_xlabel('Час (t)')
    axs[1, 1].set_ylabel('Амплітуда')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('figures/continuous_individual.png', dpi=300)
    plt.close()

    # 2. Зведений графік усіх перетворень (як у додатку)
    plt.figure(figsize=(10, 5.5))
    plt.plot(orig[0], orig[1], 'k-', linewidth=2, label='Оригінальний сигнал x(t)')
    plt.plot(sig_shift[0], sig_shift[1], 'g--', label=f'Зсув у часі (+{shift_val})')
    plt.plot(sig_rev[0], sig_rev[1], 'y-.', label='Реверс у часі x(-t)')
    plt.plot(sig_scale[0], sig_scale[1], 'b:', label=f'Масштабування амплітуди (×{scale_val})')
    plt.plot(sig_widen[0], sig_widen[1], 'r-', alpha=0.7, label=f'Розширення часу (×{width_val})')
    plt.title('Порівняння основних операцій над неперервним сигналом')
    plt.xlabel('Час (t)')
    plt.ylabel('Амплітуда')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figures/continuous_summary.png', dpi=300)
    plt.close()

    # 3. Накладання неперервних сигналів: додавання та множення
    gen2 = Signal()
    gen2.SignalGenerator(0, 10, 0, 0.7, 1)  # синусоїдальний сигнал амплітуди 0.7
    sig2 = gen2.SignalReal

    sig_add = gen.Combination(orig, sig2)
    sig_mul = gen.Multiplication(orig, sig2)

    plt.figure(figsize=(11, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sig_add[0], sig_add[1], 'indigo', label='y(t) = x₁(t) + x₂(t)')
    plt.title('Операція додавання неперервних сигналів (інтерференція)')
    plt.xlabel('Час (t)')
    plt.ylabel('Амплітуда')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(sig_mul[0], sig_mul[1], 'darkgreen', label='y(t) = x₁(t) · x₂(t)')
    plt.title('Операція множення неперервних сигналів (модуляція)')
    plt.xlabel('Час (t)')
    plt.ylabel('Амплітуда')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/continuous_superposition.png', dpi=300)
    plt.close()


# =====================================================================
# ЧАСТИНА 2. ОСНОВНІ ОПЕРАЦІЇ НАД ДИСКРЕТНИМИ СИГНАЛАМИ
# =====================================================================

class DiscreteSignal:
    """Клас для представлення та обробки дискретних сигналів x[n]."""

    def __init__(self, n=None, x=None):
        self.n = np.array(n, dtype=int) if n is not None else np.array([], dtype=int)
        self.x = np.array(x, dtype=float) if x is not None else np.array([], dtype=float)

    def scale(self, a):
        """1. Амплітудне масштабування: y[n] = a * x[n]."""
        return DiscreteSignal(self.n.copy(), self.x * float(a))

    def time_reversal(self):
        """2. Часовий реверс: y[n] = x[-n]."""
        return DiscreteSignal(-self.n[::-1], self.x[::-1])

    def time_shift(self, N):
        """3. Часовий зсув: y[n] = x[n - N]. При N > 0 зсув вправо (затримка)."""
        return DiscreteSignal(self.n + int(N), self.x.copy())

    def decimate(self, M):
        """
        4. Децимація (стиснення у часі з проріджуванням): y[m] = x[M * m].
        Зберігаються тільки відліки, для яких індекс n ділиться нацело на M.
        """
        M = int(M)
        mask = (self.n % M == 0)
        new_n = self.n[mask] // M
        new_x = self.x[mask]
        return DiscreteSignal(new_n, new_x)

    def add(self, other):
        """5.1 Додавання дискретних послідовностей: y[n] = x1[n] + x2[n]."""
        n_min = min(self.n.min(), other.n.min())
        n_max = max(self.n.max(), other.n.max())
        n_all = np.arange(n_min, n_max + 1)

        dict1 = dict(zip(self.n, self.x))
        dict2 = dict(zip(other.n, other.x))
        y = np.array([dict1.get(k, 0.0) + dict2.get(k, 0.0) for k in n_all])
        return DiscreteSignal(n_all, y)

    def multiply(self, other):
        """5.2 Множення дискретних послідовностей: y[n] = x1[n] * x2[n]."""
        n_min = min(self.n.min(), other.n.min())
        n_max = max(self.n.max(), other.n.max())
        n_all = np.arange(n_min, n_max + 1)

        dict1 = dict(zip(self.n, self.x))
        dict2 = dict(zip(other.n, other.x))
        y = np.array([dict1.get(k, 0.0) * dict2.get(k, 0.0) for k in n_all])
        return DiscreteSignal(n_all, y)


def run_discrete_signals():
    """Виконання операцій для дискретних сигналів та побудова stem-графіків."""
    print("-> Запуск операцій для дискретних сигналів...")

    # Базовий тестовий сигнал: асиметричне згасаюче гармонічне коливання
    # x[n] = 0.85^n * cos(0.2 * π * n), n in [0, 12]
    n_base = np.arange(0, 13)
    x_base = (0.85 ** n_base) * np.cos(0.2 * np.pi * n_base)
    d_orig = DiscreteSignal(n_base, x_base)

    # Параметри перетворень
    d_scale = d_orig.scale(2.0)          # масштабування a = 2.0
    d_shift = d_orig.time_shift(3)       # зсув на 3 відліки праворуч
    d_rev = d_orig.time_reversal()       # реверс у часі
    d_dec = d_orig.decimate(2)           # децимація з коефіцієнтом M = 2

    # 1. Графік базових дискретних операцій
    fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=False)

    # Оригінал
    markerline, stemlines, baseline = axs[0].stem(d_orig.n, d_orig.x, basefmt="k-")
    plt.setp(stemlines, 'color', 'b', 'linewidth', 1.8)
    plt.setp(markerline, 'color', 'b', 'markersize', 6)
    axs[0].set_title('Оригінальний дискретний сигнал x[n]')
    axs[0].set_ylabel('x[n]')
    axs[0].grid(True, linestyle=':', alpha=0.7)

    # Масштабування
    markerline, stemlines, baseline = axs[1].stem(d_scale.n, d_scale.x, basefmt="k-")
    plt.setp(stemlines, 'color', 'r', 'linewidth', 1.8)
    plt.setp(markerline, 'color', 'r', 'markersize', 6)
    axs[1].set_title('Масштабування амплітуди: y[n] = 2.0 · x[n]')
    axs[1].set_ylabel('y[n]')
    axs[1].grid(True, linestyle=':', alpha=0.7)

    # Часовий зсув
    markerline, stemlines, baseline = axs[2].stem(d_shift.n, d_shift.x, basefmt="k-")
    plt.setp(stemlines, 'color', 'g', 'linewidth', 1.8)
    plt.setp(markerline, 'color', 'g', 'markersize', 6)
    axs[2].set_title('Часовий зсув (затримка на N=3): y[n] = x[n - 3]')
    axs[2].set_ylabel('y[n]')
    axs[2].grid(True, linestyle=':', alpha=0.7)

    # Часовий реверс
    markerline, stemlines, baseline = axs[3].stem(d_rev.n, d_rev.x, basefmt="k-")
    plt.setp(stemlines, 'color', 'm', 'linewidth', 1.8)
    plt.setp(markerline, 'color', 'm', 'markersize', 6)
    axs[3].set_title('Часовий реверс: y[n] = x[-n]')
    axs[3].set_ylabel('y[n]')
    axs[3].grid(True, linestyle=':', alpha=0.7)

    # Децимація
    markerline, stemlines, baseline = axs[4].stem(d_dec.n, d_dec.x, basefmt="k-")
    plt.setp(stemlines, 'color', 'darkorange', 'linewidth', 1.8)
    plt.setp(markerline, 'color', 'darkorange', 'markersize', 6)
    axs[4].set_title('Часове масштабування / Децимація (M=2): y[m] = x[2m]')
    axs[4].set_xlabel('Індекс відліку (n або m)')
    axs[4].set_ylabel('y[m]')
    axs[4].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/discrete_operations.png', dpi=300)
    plt.close()

    # 2. Накладання дискретних сигналів (додавання та множення)
    # Другий сигнал: синусоїда на інтервалі [0, 12]
    x_second = 0.6 * np.sin(0.3 * np.pi * n_base)
    d_second = DiscreteSignal(n_base, x_second)

    d_added = d_orig.add(d_second)
    d_mult = d_orig.multiply(d_second)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Вхідні сигнали разом
    axs[0].stem(d_orig.n - 0.1, d_orig.x, linefmt='b-', markerfmt='bo', basefmt="k-", label='x₁[n]')
    axs[0].stem(d_second.n + 0.1, d_second.x, linefmt='c--', markerfmt='cs', basefmt="k-", label='x₂[n]')
    axs[0].set_title('Вхідні дискретні сигнали x₁[n] та x₂[n]')
    axs[0].set_ylabel('Амплітуда')
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].legend()

    # Додавання
    axs[1].stem(d_added.n, d_added.x, linefmt='m-', markerfmt='mo', basefmt="k-")
    axs[1].set_title('Додавання дискретних сигналів: y[n] = x₁[n] + x₂[n]')
    axs[1].set_ylabel('y[n]')
    axs[1].grid(True, linestyle=':', alpha=0.7)

    # Множення
    axs[2].stem(d_mult.n, d_mult.x, linefmt='g-', markerfmt='go', basefmt="k-")
    axs[2].set_title('Множення дискретних сигналів: y[n] = x₁[n] · x₂[n]')
    axs[2].set_xlabel('Індекс відліку n')
    axs[2].set_ylabel('y[n]')
    axs[2].grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('figures/discrete_superposition.png', dpi=300)
    plt.close()


def main():
    print("=====================================================")
    print("  Лабораторна робота №2: ОСНОВНІ ОПЕРАЦІЇ НАД СИГНАЛАМИ")
    print("=====================================================")
    run_continuous_signals()
    run_discrete_signals()
    print("-> Усі графіки успішно згенеровано та збережено у папку 'figures/'.")
    print("Готово!")


if __name__ == '__main__':
    main()
