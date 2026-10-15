"""
Лабораторна робота №5 з дисципліни «Цифрова обробка інформації» (ЦОІ)
Тема: «Маркування інформації»
Виконав: студент групи ФЕП-32 Чепара С.

Програма суворо реалізує завдання та алгоритми з методичних вказівок:
1. Завдання 1: Накладання видимого оптичного маркеру (Alpha Blending) за допомогою OpenCV:
   img = cv2.imread("32.png"), logo = cv2.imread("wt1.jpg") -> cv2.imwrite("watermarked.jpg", img).
2. Завдання 2: Стеганографічне нанесення та вилучення оптичного маркеру за методом LSB
   у просторовій області:
   - Вбудовування: заміна 2 молодших бітів (LSB) кожного колірного каналу (R, G, B) контейнера.
   - Вилучення: зчитування 2 LSB бітів з каналів R, G, B, усереднення та відновлення маркеру.
3. Оцінка метрик спотворення та якості маркування (MSE, PSNR).
4. Графічний інтерфейс програми (GUI на базі Tkinter) з режимами Apply mode та Read mode.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

# Шляхи до робочих каталогів
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# Завдання 1: Накладання видимого оптичного маркеру (cv2.addWeighted)
# =====================================================================

def apply_visible_watermark(img_path, logo_path, output_path, alpha=1.0, beta=0.7, gamma=0, margin=10):
    """
    Накладає видимий логотип на базове зображення (як у коді методички):
    destination = img[h - wH - 10:h - 10, w - wW - 10:w - 10]
    result = cv2.addWeighted(destination, 1, logo, 0.7, 0)
    img[h - wH - 10:h - 10, w - wW - 10:w - 10] = result
    """
    img = cv2.imread(img_path)
    logo = cv2.imread(logo_path)

    if img is None:
        raise FileNotFoundError(f"Не знайдено базове зображення: {img_path}")
    if logo is None:
        raise FileNotFoundError(f"Не знайдено логотип: {logo_path}")

    h, w = img.shape[:2]
    wH, wW = logo.shape[:2]

    # Якщо логотип більший за базове зображення — масштабуємо
    if wH > h - margin or wW > w - margin:
        scale = min((h - 2 * margin) / wH, (w - 2 * margin) / wW)
        new_w = max(1, int(wW * scale))
        new_h = max(1, int(wH * scale))
        logo = cv2.resize(logo, (new_w, new_h))
        wH, wW = logo.shape[:2]

    # Область інтересу (ROI) у правому нижньому куті
    y1 = h - wH - margin
    y2 = h - margin
    x1 = w - wW - margin
    x2 = w - margin

    destination = img[y1:y2, x1:x2]
    result = cv2.addWeighted(destination, alpha, logo, beta, gamma)

    watermarked = img.copy()
    watermarked[y1:y2, x1:x2] = result

    cv2.imwrite(output_path, watermarked)
    return img, logo, watermarked, destination, result


# =====================================================================
# Завдання 2: Приховане маркування за методом LSB
# =====================================================================

def embed_lsb(base_img, marker_img):
    """
    Вбудовує оптичний маркер у 2 молодші біти кожного колірного каналу контейнера:
    base[i, j, c] = (original & 252) | (marker_bits[i, j] & 0b11)
    """
    base = np.array(base_img.convert("RGB"))
    marker = np.array(marker_img.resize(base_img.size))

    # Для контрастного штампа з методички (червоний штамп на темному тлі):
    # виділяємо штамп (значення 0) та фон (значення 1)
    if marker.ndim == 3 and marker.shape[2] >= 3:
        r = marker[:, :, 0].astype(float)
        g = marker[:, :, 1].astype(float)
        b = marker[:, :, 2].astype(float)
        is_stamp = (r > 100) & (r > g * 1.3) & (r > b * 1.3)
        marker_bits = np.where(is_stamp, 0, 1).astype(np.uint8)
    else:
        m_gray = np.array(marker_img.resize(base_img.size).convert("L"))
        marker_bits = (m_gray // 85).astype(np.uint8) & 0b11

    # Заміна 2 молодших бітів (маска 252 = 0xFC = 11111100_2)
    watermarked = (base & 0xFC) | marker_bits[:, :, None]
    return Image.fromarray(watermarked)


def extract_lsb(watermarked_img):
    """
    Витягує маркер з маркованого зображення:
    avg_bit = round((r_bits + g_bits + b_bits) / 3)
    extracted[i, j] = avg_bit * 85
    """
    img = np.array(watermarked_img.convert("RGB"))

    bits = (img[:, :, 0] & 0b11) + (img[:, :, 1] & 0b11) + (img[:, :, 2] & 0b11)
    avg_bit = np.round(bits / 3.0).astype(np.uint8)
    extracted = avg_bit * 85
    return Image.fromarray(extracted)


# =====================================================================
# Оцінка якості та спотворення (MSE, PSNR)
# =====================================================================

def compute_metrics(arr1, arr2):
    """Обчислює середньоквадратичну похибку (MSE) та PSNR між двома зображеннями."""
    a1 = arr1.astype(np.float64)
    a2 = arr2.astype(np.float64)
    mse = np.mean((a1 - a2) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    return mse, psnr


# =====================================================================
# Побудова зведених графіків
# =====================================================================

def generate_comparison_plots(base_vis, logo_vis, wm_vis, base_pil, marker_pil, wm_lsb_pil, ext_marker_pil):
    """Створює порівняльні рисунки результатів для звіту та README."""

    # 1. Видимий водяний знак (Alpha Blending)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(cv2.cvtColor(base_vis, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Оригінальне зображення (32.png)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(logo_vis, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Оптичний маркер (wt1.jpg)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(wm_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Результат нанесення ОМ (watermarked.jpg)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    vis_plot_path = os.path.join(OUTPUT_DIR, "visible_comparison.png")
    plt.savefig(vis_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Прихований LSB водяний знак
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(base_pil)
    axes[0].set_title("Оригінальне зображення", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(marker_pil)
    axes[1].set_title("Оптичний маркер (ОМ)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(wm_lsb_pil)
    axes[2].set_title("Зображення з ОМ (LSB)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(ext_marker_pil, cmap="gray")
    axes[3].set_title("Виокремлений ОМ", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    lsb_plot_path = os.path.join(OUTPUT_DIR, "lsb_comparison.png")
    plt.savefig(lsb_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    return vis_plot_path, lsb_plot_path


# =====================================================================
# Автоматичний запуск (CLI)
# =====================================================================

def run_tasks():
    """Виконує повний цикл маркування строго згідно з методичкою."""
    print("=" * 60)
    print("  Лабораторна робота №5: МАРКУВАННЯ ІНФОРМАЦІЇ")
    print("  Виконав: студент гр. ФЕП-32 Чепара С.")
    print("=" * 60)

    # --- Завдання 1: Видимий оптичний маркер ---
    print("\n--- Завдання 1: Нанесення видимого ОМ (OpenCV) ---")
    img_32_path = os.path.join(IMAGE_DIR, "32.png")
    wt1_path = os.path.join(IMAGE_DIR, "wt1.jpg")
    wm_visible_out = os.path.join(OUTPUT_DIR, "watermarked.jpg")

    base_vis, logo_vis, wm_vis, dest_roi, res_roi = apply_visible_watermark(
        img_32_path, wt1_path, wm_visible_out, alpha=1.0, beta=0.7, gamma=0, margin=10
    )
    mse_vis, psnr_vis = compute_metrics(base_vis, wm_vis)
    mse_roi, psnr_roi = compute_metrics(dest_roi, res_roi)

    print(f"  [+] Видимий маркер нанесено на 32.png: {wm_visible_out}")
    print(f"      Параметри: alpha=1.0, beta=0.7, margin=10 px")
    print(f"      По всьому зображенню: MSE = {mse_vis:.4f}, PSNR = {psnr_vis:.2f} дБ")
    print(f"      У зоні накладання ROI: MSE = {mse_roi:.4f}, PSNR = {psnr_roi:.2f} дБ")

    # --- Завдання 2: Стеганографічне LSB маркування ---
    print("\n--- Завдання 2: Стеганографічне маркування методом LSB ---")
    orig_path = os.path.join(IMAGE_DIR, "original.png")
    marker_path = os.path.join(IMAGE_DIR, "marker.png")

    base_pil = Image.open(orig_path).convert("RGB")
    marker_pil = Image.open(marker_path).convert("RGB")

    # Вбудовування маркеру
    wm_lsb_pil = embed_lsb(base_pil, marker_pil)
    wm_lsb_out = os.path.join(OUTPUT_DIR, "watermarked.png")
    wm_lsb_pil.save(wm_lsb_out)
    print(f"  [+] Зображення з ОМ збережено: {wm_lsb_out}")

    # Вилучення маркеру
    ext_marker_pil = extract_lsb(wm_lsb_pil)
    ext_marker_out = os.path.join(OUTPUT_DIR, "extracted_marker.png")
    ext_marker_pil.save(ext_marker_out)
    print(f"  [+] Виокремлений ОМ збережено: {ext_marker_out}")

    # Оцінка якості контейнера
    mse_lsb, psnr_lsb = compute_metrics(np.array(base_pil), np.array(wm_lsb_pil))
    print(f"  [+] Метрики якості маркованого контейнера:")
    print(f"      MSE  = {mse_lsb:.4f}")
    print(f"      PSNR = {psnr_lsb:.2f} дБ (значення > 40 дБ свідчить про повну непомітність)")

    # Генерація порівняльних графіків
    vis_plot, lsb_plot = generate_comparison_plots(
        base_vis, logo_vis, wm_vis, base_pil, marker_pil, wm_lsb_pil, ext_marker_pil
    )
    print(f"\n  [+] Порівняльні рисунки збережено:")
    print(f"      - {vis_plot}")
    print(f"      - {lsb_plot}")

    print("\n" + "=" * 60)
    print("  Усі завдання успішно виконано!")
    print("=" * 60)

    return {
        "mse_vis": mse_vis,
        "psnr_vis": psnr_vis,
        "mse_roi": mse_roi,
        "psnr_roi": psnr_roi,
        "mse_lsb": mse_lsb,
        "psnr_lsb": psnr_lsb,
    }


# =====================================================================
# Графічний інтерфейс користувача (WatermarkApp)
# =====================================================================

class WatermarkApp:
    """Графічний інтерфейс програми згідно з вимогами методичних вказівок."""

    def __init__(self, root):
        self.root = root
        self.root.title("LSB Watermarking — Лабораторна робота №5")
        self.root.geometry("860x640")
        self.root.minsize(800, 560)

        self.mode = tk.StringVar(value="apply")
        self.original_img = None
        self.marker_img = None
        self.result_img = None

        self.build_gui()
        self.load_examples()

    def build_gui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill="x", side="top")

        self.btn_apply_mode = ttk.Button(
            control_frame, text="Apply mode", command=lambda: self.set_mode("apply")
        )
        self.btn_apply_mode.pack(side="left", padx=5)

        self.btn_read_mode = ttk.Button(
            control_frame, text="Read mode", command=lambda: self.set_mode("read")
        )
        self.btn_read_mode.pack(side="left", padx=5)

        self.btn_process = ttk.Button(
            control_frame, text="Apply", command=self.process
        )
        self.btn_process.pack(side="left", padx=15)

        self.lbl_status = ttk.Label(control_frame, text="Режим: Вбудовування маркеру (Apply mode)")
        self.lbl_status.pack(side="left", padx=10)

        btn_exit = ttk.Button(control_frame, text="Exit", command=self.root.quit)
        btn_exit.pack(side="right", padx=5)

        self.canvas_frame = ttk.Frame(self.root, padding=10)
        self.canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#f5f5f5", relief="sunken", bd=1)
        self.canvas.pack(fill="both", expand=True)

    def set_mode(self, new_mode):
        self.mode.set(new_mode)
        if new_mode == "apply":
            self.lbl_status.config(text="Режим: Вбудовування маркеру (Apply mode)")
            self.btn_process.config(text="Apply")
        else:
            self.lbl_status.config(text="Режим: Зчитування маркеру (Read mode)")
            self.btn_process.config(text="Read")
        self.load_examples()

    def load_examples(self):
        self.original_img = None
        self.marker_img = None
        self.result_img = None

        orig_path = os.path.join(IMAGE_DIR, "original.png")
        marker_path = os.path.join(IMAGE_DIR, "marker.png")
        wm_path = os.path.join(OUTPUT_DIR, "watermarked.png")

        if self.mode.get() == "apply":
            if os.path.exists(orig_path):
                self.original_img = Image.open(orig_path)
            if os.path.exists(marker_path):
                self.marker_img = Image.open(marker_path)
        else:
            if os.path.exists(wm_path):
                self.original_img = Image.open(wm_path)
            elif os.path.exists(orig_path):
                self.original_img = Image.open(orig_path)

        self.update_canvas()

    def update_canvas(self):
        self.canvas.delete("all")
        self.canvas.image_refs = []

        if self.mode.get() == "apply":
            images = [
                ("Оригінальне зображення", self.original_img),
                ("Оптичний маркер (ОМ)", self.marker_img),
                ("Зображення з ОМ", self.result_img)
            ]
        else:
            images = [
                ("Зображення з ОМ", self.original_img),
                ("Виокремлений ОМ", self.result_img)
            ]

        card_w, card_h = 250, 250
        gap = 25
        start_x = 20
        start_y = 40

        for idx, (title, img) in enumerate(images):
            x = start_x + idx * (card_w + gap)
            y = start_y

            self.canvas.create_text(x + card_w // 2, y - 15, text=title, font=("Arial", 11, "bold"))
            self.canvas.create_rectangle(x, y, x + card_w, y + card_h, fill="#ffffff", outline="#cccccc")

            if img is not None:
                resized = img.copy().resize((card_w - 10, card_h - 10))
                tk_img = ImageTk.PhotoImage(resized)
                self.canvas.image_refs.append(tk_img)
                self.canvas.create_image(x + 5, y + 5, anchor="nw", image=tk_img)
            else:
                self.canvas.create_text(
                    x + card_w // 2, y + card_h // 2, text="(не завантажено)", fill="#888888"
                )

    def process(self):
        if self.mode.get() == "apply":
            if self.original_img and self.marker_img:
                self.result_img = embed_lsb(self.original_img, self.marker_img)
                out_path = os.path.join(OUTPUT_DIR, "watermarked.png")
                self.result_img.save(out_path)
                self.update_canvas()
                messagebox.showinfo("Успіх", f"Маркер успішно нанесено!\nЗбережено: {out_path}")
            else:
                messagebox.showerror("Помилка", "Необхідно завантажити базове зображення та маркер.")
        else:
            if self.original_img:
                self.result_img = extract_lsb(self.original_img)
                out_path = os.path.join(OUTPUT_DIR, "extracted_marker.png")
                self.result_img.save(out_path)
                self.update_canvas()
                messagebox.showinfo("Успіх", f"Оптичний маркер успішно вилучено!\nЗбережено: {out_path}")
            else:
                messagebox.showerror("Помилка", "Не завантажено марковане зображення.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Лабораторна робота №5: Маркування інформації")
    parser.add_argument("--gui", action="store_true", help="Запустити графічний інтерфейс")
    parser.add_argument("--cli", action="store_true", help="Виконати в консольному режимі")
    args = parser.parse_args()

    # За замовчуванням виконуємо завдання
    results = run_tasks()

    if args.gui:
        root = tk.Tk()
        app = WatermarkApp(root)
        root.mainloop()
