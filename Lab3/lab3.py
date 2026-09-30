import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===== Завдання 1 =====
# Обчислення взаємної кореляції за формулою (2): r12 = (1/N) * sum(x1(n)*x2(n))

def task1_cross_correlation():
    """Обчислення взаємної кореляції для даних з Табл.1."""
    print("=" * 55)
    print("  Завдання 1: Взаємна кореляція (Табл.1, формула 2)")
    print("=" * 55)

    # Дані з Табл.1 (n = 1..9)
    x1 = np.array([4, 2, -1, 3, -2, -6, -5, 4, 5])
    x2 = np.array([7, 4, -2, -8, -2, -1, 0, 0, 0])
    N = len(x1)

    # r12 = (1/N) * sum(x1(n) * x2(n))
    products = x1 * x2
    r12 = np.sum(products) / N

    print(f"  x1 = {x1}")
    print(f"  x2 = {x2}")
    print(f"  N  = {N}")
    print()
    print("  Поелементні добутки x1(n) * x2(n):")
    for i in range(N):
        print(f"    n={i+1}: {x1[i]:3} * {x2[i]:3} = {products[i]:6.1f}")
    print(f"  Сума добутків = {np.sum(products):.1f}")
    print(f"  r12 = {np.sum(products):.1f} / {N} = {r12:.4f}")
    print()
    return r12


# ===== Завдання 2 =====
# Нормувальний коефіцієнт (формула 4) та нормований коефіцієнт кореляції (формула 5)

def task2_normalized_correlation():
    """Обчислення нормованого коефіцієнта кореляції для даних з Табл.2."""
    print("=" * 55)
    print("  Завдання 2: Нормований коефіцієнт кореляції (Табл.2)")
    print("=" * 55)

    # Дані з Табл.2 (n = 0..8)
    x1 = np.array([0, 3, 5, 5, 5, 2, 0.5, 0.25, 0])
    x2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    x3 = np.array([0, 9, 15, 15, 15, 6, 1.5, 0.75, 0])
    x4 = np.array([2, 2, 2, 2, 2, 0, 0, 0, 0])
    N = len(x1)

    def compute_normalized_corr(xa, xb, label_a, label_b):
        """Обчислення r та rho для пари сигналів."""
        # Взаємна кореляція r (формула 2, при j=0)
        r = np.sum(xa * xb) / N

        # Нормувальний коефіцієнт (формула 4)
        norm_coeff = (1.0 / N) * np.sqrt(np.sum(xa**2) * np.sum(xb**2))

        # Нормований коеф. кореляції (формула 5)
        rho = r / norm_coeff if norm_coeff != 0 else 0

        print(f"\n  Пара {label_a}, {label_b}:")
        print(f"    {label_a} = {xa}")
        print(f"    {label_b} = {xb}")
        print(f"    r_{label_a}{label_b}  = (1/{N}) * sum({label_a}*{label_b}) = {r:.4f}")
        print(f"    sum({label_a}^2)  = {np.sum(xa**2):.4f}")
        print(f"    sum({label_b}^2)  = {np.sum(xb**2):.4f}")
        print(f"    Нормув. коеф. = (1/{N}) * sqrt({np.sum(xa**2):.2f} * {np.sum(xb**2):.2f}) = {norm_coeff:.4f}")
        print(f"    rho = {r:.4f} / {norm_coeff:.4f} = {rho:.4f}")
        return r, norm_coeff, rho

    r12, norm12, rho12 = compute_normalized_corr(x1, x2, "x1", "x2")
    r34, norm34, rho34 = compute_normalized_corr(x3, x4, "x3", "x4")

    print(f"\n  Порівняння:")
    print(f"    r12 = {r12:.4f},  r34 = {r34:.4f}   (відрізняються через амплітуди)")
    print(f"    rho12 = {rho12:.4f}, rho34 = {rho34:.4f} (однакові — сигнали подібні)")
    print()
    return rho12, rho34


# ===== Завдання 3 =====
# Кореляція та автокореляція випадкових сигналів (N=200, j=0..20)

def cross_corr(x, y, j):
    """Взаємна кореляція r_xy(j) = (1/N) * sum(x(n) * y(n+j)), n=0..N-1-j."""
    N = len(x)
    if j >= N:
        return 0.0
    return np.sum(x[:N - j] * y[j:N]) / N


def auto_corr(x, j):
    """Автокореляція r_xx(j) = (1/N) * sum(x(n) * x(n+j)), n=0..N-1-j."""
    return cross_corr(x, x, j)


def task3_random_signals():
    """Кореляція та автокореляція двох масивів випадкових чисел N=200."""
    print("=" * 55)
    print("  Завдання 3: Кореляція випадкових сигналів (N=200)")
    print("=" * 55)

    np.random.seed(42)
    N = 200
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)

    shifts = np.arange(0, 21)  # j = 0..20

    # Взаємна кореляція r12(j)
    r12 = np.array([cross_corr(x1, x2, j) for j in shifts])

    # Автокореляція r11(j) та r22(j)
    r11 = np.array([auto_corr(x1, j) for j in shifts])
    r22 = np.array([auto_corr(x2, j) for j in shifts])

    print(f"  N = {N}")
    print(f"  Діапазон зсувів j = 0..20")
    print(f"\n  Взаємна кореляція r12(j):")
    for j in shifts:
        print(f"    j={j:2d}: r12 = {r12[j]:8.4f}")
    print(f"\n  Автокореляція r11(j):")
    for j in shifts:
        print(f"    j={j:2d}: r11 = {r11[j]:8.4f}")

    # --- Графік 1: Сигнали x1 та x2 ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].plot(x1, color='b', linewidth=0.8)
    axes[0].set_title('Випадковий сигнал x1(n), N=200')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('x1(n)')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(x2, color='r', linewidth=0.8)
    axes[1].set_title('Випадковий сигнал x2(n), N=200')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('x2(n)')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('figures/random_signals.png', dpi=200)
    print("\n  -> Збережено: figures/random_signals.png")
    plt.close()

    # --- Графік 2: Взаємна кореляція ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stem(shifts, r12, linefmt='g-', markerfmt='go', basefmt='k-')
    ax.set_title('Взаємна кореляція r12(j) випадкових сигналів')
    ax.set_xlabel('Зсув j')
    ax.set_ylabel('r12(j)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(y=0, color='k', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('figures/cross_correlation.png', dpi=200)
    print("  -> Збережено: figures/cross_correlation.png")
    plt.close()

    # --- Графік 3: Автокореляція ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].stem(shifts, r11, linefmt='b-', markerfmt='bo', basefmt='k-')
    axes[0].set_title('Автокореляція r11(j) сигналу x1')
    axes[0].set_xlabel('Зсув j')
    axes[0].set_ylabel('r11(j)')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].stem(shifts, r22, linefmt='r-', markerfmt='ro', basefmt='k-')
    axes[1].set_title('Автокореляція r22(j) сигналу x2')
    axes[1].set_xlabel('Зсув j')
    axes[1].set_ylabel('r22(j)')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('figures/autocorrelation.png', dpi=200)
    print("  -> Збережено: figures/autocorrelation.png")
    plt.close()

    print()
    return x1, x2, r12, r11, r22


# ===== Завдання 4 =====
# Обчислення енергії сигналів за формулою (7): r11(0) = (1/N) * sum(x1^2) = S

def task4_energy(x1, x2):
    """Обчислення нормованої енергії через автокореляцію при j=0."""
    print("=" * 55)
    print("  Завдання 4: Енергія сигналів (формула 7)")
    print("=" * 55)

    N = len(x1)
    S1 = np.sum(x1**2) / N   # r11(0) = енергія x1
    S2 = np.sum(x2**2) / N   # r22(0) = енергія x2

    print(f"  r11(0) = (1/{N}) * sum(x1^2) = S1 = {S1:.4f}")
    print(f"  r22(0) = (1/{N}) * sum(x2^2) = S2 = {S2:.4f}")
    print(f"  |S1 - S2| = {abs(S1 - S2):.4f}")
    print()

    # Перевірка властивості r11(0) >= r11(j) для кількох j
    print("  Перевірка властивості r11(0) >= r11(j):")
    for j in [1, 5, 10, 15, 20]:
        r11_j = np.sum(x1[:N - j] * x1[j:N]) / N
        ok = "✓" if S1 >= r11_j else "✗"
        print(f"    j={j:2d}: r11(j) = {r11_j:8.4f}  {ok}  (r11(0) = {S1:.4f})")
    print()
    return S1, S2


# ===== Завдання 5 =====
# Пояснення результатів

def task5_explanation():
    """Текстове пояснення отриманих результатів."""
    print("=" * 55)
    print("  Завдання 5: Пояснення результатів")
    print("=" * 55)
    print("""
  1. Взаємна кореляція (Завдання 1):
     Додатне значення r12 вказує на додатну кореляцію —
     зміни x1 та x2 в цілому узгоджені за напрямком.

  2. Нормований коефіцієнт (Завдання 2):
     Незважаючи на різні амплітуди, нормовані коефіцієнти
     rho12 і rho34 збігаються, що підтверджує подібність
     форми сигналів незалежно від масштабу.

  3. Кореляція випадкових сигналів (Завдання 3):
     Взаємна кореляція двох незалежних випадкових сигналів
     флуктуює біля нуля для всіх зсувів j, що свідчить
     про відсутність лінійного зв'язку між ними.
     Автокореляція має максимум при j=0 і швидко спадає,
     що є типовою ознакою білого шуму.

  4. Енергія сигналів (Завдання 4):
     Значення r11(0) та r22(0) дають нормовану енергію
     відповідних сигналів. Для випадкових сигналів з
     однаковим розподілом енергії мають бути близькими.
     Виконується властивість r11(0) >= r11(j) для всіх j.
""")


def main():
    print("=====================================================")
    print("  Лабораторна робота №3: КОРЕЛЯЦІЯ ТА АВТОКОРЕЛЯЦІЯ")
    print("  СИГНАЛІВ")
    print("=====================================================\n")

    task1_cross_correlation()
    task2_normalized_correlation()
    x1, x2, r12, r11, r22 = task3_random_signals()
    task4_energy(x1, x2)
    task5_explanation()

    print("-> Усі завдання виконано. Графіки збережено у figures/")


if __name__ == '__main__':
    main()
