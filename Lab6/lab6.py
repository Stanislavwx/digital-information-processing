import os
import math
import random
import matplotlib.pyplot as plt

# Базова директорія для збереження графіків
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_sample(n, seed=None):
    """Генерація вибірки випадкових цілих чисел у діапазоні 0..9."""
    if seed is not None:
        random.seed(seed)
    return [random.randint(0, 9) for _ in range(n)]


def compute_statistics(sample):
    """
    Обчислення кількості появ, емпіричних ймовірностей
    та власної кількості інформації для кожної цифри 0..9.
    """
    n = len(sample)
    counts = [0] * 10
    for num in sample:
        counts[num] += 1

    probabilities = [count / n for count in counts]
    info_measures = [-math.log2(p) if p > 0 else 0.0 for p in probabilities]

    return counts, probabilities, info_measures


def compute_entropy(probabilities):
    """Обчислення ентропії Шеннона H = -sum(P_i * log2(P_i))."""
    return sum(-p * math.log2(p) for p in probabilities if p > 0)


def run_experiment_n100(seed=42):
    """Виконання пунктів 1, 2, 3 методички для вибірки N=100."""
    print("=" * 65)
    print("ЕКСПЕРИМЕНТ 1: Вибірка випадкових чисел для N = 100")
    print("=" * 65)

    n100 = 100
    sample100 = generate_sample(n100, seed=seed)
    counts, probs, infos = compute_statistics(sample100)
    entropy100 = compute_entropy(probs)
    h_max = math.log2(10)

    print(f"Згенерований масив (перші 25 значень): {sample100[:25]}...")
    print("\nСтатистичні результати для N = 100:")
    print("-" * 65)
    print(f"{'Число':^7} | {'Кількість n_i':^15} | {'Ймовірність P(i)':^18} | {'I(i) (біт)':^14}")
    print("-" * 65)
    for i in range(10):
        print(f"{i:^7d} | {counts[i]:^15d} | {probs[i]:^18.3f} | {infos[i]:^14.3f}")
    print("-" * 65)
    print(f"Сумарна інформаційна міра sum(I): {sum(infos):.3f} біт")
    print(f"Ентропія Шеннона H(100):          {entropy100:.4f} біт/символ")
    print(f"Теоретичний максимум H_max:        {h_max:.4f} біт/символ (log2(10))")
    print(f"Відхилення від максимуму:          {h_max - entropy100:+.4f} біт")

    # Графік 1: Ймовірності появи чисел P(i)
    plt.figure(figsize=(7.2, 3.6))
    bars = plt.bar(range(10), probs, color="#5dade2", edgecolor="#1b4f72", width=0.6, zorder=3)
    plt.axhline(0.1, color="#c0392b", linestyle="--", linewidth=1.5,
                label="Теоретична ймовірність $P = 1/10 = 0.10$")
    plt.title("Ймовірність випадання кожного числа (N = 100)", fontsize=13, fontweight="bold")
    plt.xlabel("Випадкове число $i$", fontsize=11)
    plt.ylabel("Ймовірність $P(i)$", fontsize=11)
    plt.xticks(range(10))
    plt.ylim(0, max(probs) * 1.25)
    plt.grid(axis="y", linestyle=":", alpha=0.7, zorder=0)
    plt.legend(loc="upper right", framealpha=0.9)

    # Підписи значень над стовпчиками
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    prob_path = os.path.join(OUTPUT_DIR, "probabilities_n100.png")
    plt.savefig(prob_path, dpi=300)
    plt.close()
    print(f"-> Графік ймовірностей збережено у: {prob_path}")

    # Графік 2: Інформаційна міра I(i)
    plt.figure(figsize=(7.2, 3.6))
    bars_info = plt.bar(range(10), infos, color="#58d68d", edgecolor="#145a32", width=0.6, zorder=3)
    plt.axhline(h_max, color="#8e44ad", linestyle="--", linewidth=1.5,
                label=f"Теоретичне $I_{{теор}} = \\log_2(10) \\approx {h_max:.3f}$ біт")
    plt.title("Кількість інформації $I(i) = -\\log_2(P(i))$ для чисел (N = 100)", fontsize=13, fontweight="bold")
    plt.xlabel("Випадкове число $i$", fontsize=11)
    plt.ylabel("Кількість інформації $I(i)$ (біт)", fontsize=11)
    plt.xticks(range(10))
    plt.ylim(0, max(infos) * 1.25)
    plt.grid(axis="y", linestyle=":", alpha=0.7, zorder=0)
    plt.legend(loc="upper right", framealpha=0.9)

    for bar in bars_info:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    info_path = os.path.join(OUTPUT_DIR, "information_n100.png")
    plt.savefig(info_path, dpi=300)
    plt.close()
    print(f"-> Графік кількості інформації збережено у: {info_path}")

    return sample100, counts, probs, infos, entropy100


def run_experiment_sample_sizes(sample_sizes=None, seed=42):
    """
    Виконання пункту 5 методички:
    Обчислення ентропії для N = [50, 100, 200, 500, 1000, 2000, 3000].
    """
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 500, 1000, 2000, 3000]

    print("\n" + "=" * 65)
    print("ЕКСПЕРИМЕНТ 2: Залежність ентропії від обсягу вибірки N")
    print("=" * 65)

    h_max = math.log2(10)
    entropies = []
    diffs = []

    if seed is not None:
        random.seed(seed)

    print(f"{'N (обсяг)':^10} | {'Ентропія H(N) (біт)':^22} | {'Відхилення від H_max':^24}")
    print("-" * 65)
    for n in sample_sizes:
        sample = [random.randint(0, 9) for _ in range(n)]
        counts = [0] * 10
        for num in sample:
            counts[num] += 1
        probs = [c / n for c in counts]
        h = compute_entropy(probs)
        entropies.append(h)
        diff = h_max - h
        diffs.append(diff)
        print(f"{n:^10d} | {h:^22.4f} | {diff:^+24.4f}")
    print("-" * 65)
    print(f"Теоретичний максимум H_max = log2(10) = {h_max:.4f} біт/символ")

    # Графік 3: H(N) = f(N)
    plt.figure(figsize=(7.2, 3.6))
    plt.plot(sample_sizes, entropies, marker="o", color="#e74c3c", linewidth=2,
             markersize=6, label="Експериментальна ентропія $H(N)$", zorder=4)
    plt.axhline(h_max, color="#2980b9", linestyle="--", linewidth=1.8,
                label=f"Теоретична максимальна ентропія $H_{{max}} = \\log_2(10) \\approx {h_max:.4f}$ біт", zorder=3)

    plt.title("Залежність ентропії від розміру вибірки $H(N) = f(N)$", fontsize=13, fontweight="bold")
    plt.xlabel("Розмір вибірки $N$", fontsize=11)
    plt.ylabel("Середня кількість інформації $H$ (біт/символ)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="lower right", framealpha=0.9)

    # Анотації значень на точках
    for n, h in zip(sample_sizes, entropies):
        plt.annotate(f"{h:.3f}", (n, h), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)

    plt.tight_layout()
    entropy_path = os.path.join(OUTPUT_DIR, "entropy_vs_n.png")
    plt.savefig(entropy_path, dpi=300)
    plt.close()
    print(f"-> Графік залежності ентропії збережено у: {entropy_path}")

    # Зведений графік 3-в-1 для компактного перегляду
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Ймовірності
    sample100 = generate_sample(100, seed=seed)
    counts100, probs100, infos100 = compute_statistics(sample100)
    axes[0].bar(range(10), probs100, color="#5dade2", edgecolor="#1b4f72", width=0.6)
    axes[0].axhline(0.1, color="#c0392b", linestyle="--", label="$P_{теор} = 0.10$")
    axes[0].set_title("а) Ймовірності $P(i)$ (N=100)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Число $i$")
    axes[0].set_ylabel("Ймовірність $P(i)$")
    axes[0].set_xticks(range(10))
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8)

    # 2. Кількість інформації
    axes[1].bar(range(10), infos100, color="#58d68d", edgecolor="#145a32", width=0.6)
    axes[1].axhline(h_max, color="#8e44ad", linestyle="--", label="$I_{теор} \\approx 3.322$ біт")
    axes[1].set_title("б) Кількість інформації $I(i)$ (N=100)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Число $i$")
    axes[1].set_ylabel("$I(i)$ (біт)")
    axes[1].set_xticks(range(10))
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8)

    # 3. Ентропія H(N)
    axes[2].plot(sample_sizes, entropies, marker="o", color="#e74c3c", linewidth=1.8, markersize=5)
    axes[2].axhline(h_max, color="#2980b9", linestyle="--", label=f"$H_{{max}} \\approx {h_max:.3f}$ біт")
    axes[2].set_title("в) Залежність ентропії $H(N)$", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Розмір вибірки $N$")
    axes[2].set_ylabel("Ентропія $H(N)$ (біт)")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "lab6_summary.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"-> Зведений графік (3-в-1) збережено у: {summary_path}")

    return sample_sizes, entropies, diffs


def main():
    print("=" * 65)
    print("  Цифрова обробка інформації (ЦОІ) — Лабораторна робота №6")
    print("  Тема: Кількісна оцінка інформації")
    print("  Студент: Чепара С., група ФеП-32")
    print("=" * 65 + "\n")

    # 1. Аналіз вибірки N = 100
    run_experiment_n100(seed=42)

    # 2. Аналіз залежності ентропії від розміру вибірки N
    sizes = [50, 100, 200, 500, 1000, 2000, 3000]
    run_experiment_sample_sizes(sample_sizes=sizes, seed=42)

    print("\n" + "=" * 65)
    print("ВИСНОВОК ЩОДО ОТРИМАНИХ РЕЗУЛЬТАТІВ:")
    print("1. Згідно з законом великих чисел, при збільшенні обсягу вибірки N")
    print("   відносні частоти появи цифр 0..9 збігаються до їх істинної")
    print("   ймовірності P = 0.1, а емпірична ентропія асимптотично наближається")
    print(f"   до максимально можливої H_max = log2(10) ≈ {math.log2(10):.4f} біт/символ.")
    print("2. Власна інформація I(i) обернено пропорційна до ймовірності: рідкісні")
    print("   події несуть більше інформації, а часті — менше.")
    print("=" * 65)


if __name__ == "__main__":
    main()
