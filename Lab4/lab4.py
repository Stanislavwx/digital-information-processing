"""
Лабораторна робота №4: Синтез вербальної інформації
Голосовий синтезатор на базі gTTS (Google Text-to-Speech)
"""

from gtts import gTTS
import os
import time
import random


# Папка для збереження аудіофайлів
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def save_audio(text, lang="uk", filename=None):
    """Синтезує мовлення з тексту та зберігає у mp3-файл."""
    if filename is None:
        timestamp = str(int(time.time()))
        rnd = str(random.randint(0, 99999))
        filename = f"voice_{timestamp}_{rnd}.mp3"

    filepath = os.path.join(AUDIO_DIR, filename)
    voice = gTTS(text, lang=lang)
    voice.save(filepath)
    print(f"  [Збережено] {filepath}")
    return filepath


def play_audio(filepath):
    """Відтворює аудіофайл через playsound або системні програвачі."""
    # Спочатку намагаємося відтворити через playsound (як у методичці)
    try:
        from playsound import playsound
        playsound(filepath)
        return
    except Exception:
        pass

    # Якщо playsound не спрацював на Linux (наприклад, через відсутність gi/GStreamer),
    # використовуємо нативні системні утиліти (PipeWire, PulseAudio або FFmpeg)
    import shutil
    import subprocess
    for player, args in [
        ("pw-play", ["pw-play", filepath]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath]),
        ("paplay", ["paplay", filepath]),
        ("cvlc", ["cvlc", "--play-and-exit", "--quiet", filepath]),
    ]:
        if shutil.which(player):
            try:
                res = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    return
            except Exception:
                pass

    print(f"  [i] Файл збережено: {filepath} (аудіоплеєр не знайдено)")


def say(text, lang="uk", filename=None):
    """Синтезує, зберігає та відтворює повідомлення."""
    print(f"  Асистент: {text}")
    filepath = save_audio(text, lang=lang, filename=filename)
    play_audio(filepath)
    return filepath


def listen_command():
    """Зчитує команду від користувача з клавіатури."""
    return input("\nВведіть команду: ").strip()


def process_command(message):
    """Обробляє введену команду та генерує відповідь."""
    msg = message.lower()

    if "привіт" in msg or "вітаю" in msg:
        say("Привіт! Радий тебе чути. Чим можу допомогти?")

    elif "час" in msg:
        current_time = time.strftime("%H:%M:%S")
        say(f"Зараз {current_time}")

    elif "дата" in msg:
        current_date = time.strftime("%d.%m.%Y")
        say(f"Сьогодні {current_date}")

    elif "як справи" in msg:
        say("Дякую, все добре! Я готовий працювати.")

    elif "допомога" in msg:
        say("Доступні команди: привіт, час, дата, як справи, допомога, бувай.")

    elif "бувай" in msg or "до побачення" in msg:
        say("До побачення! Гарного дня!")
        return False

    else:
        say("Вибачте, не розумію команду. Скажіть допомога для списку команд.")

    return True


def generate_demo_files():
    """Генерує демонстраційні аудіофайли для звіту."""
    print("=" * 50)
    print("  Генерація демонстраційних аудіофайлів")
    print("=" * 50)

    demos = [
        ("Привіт! Це голосовий синтезатор.", "uk", "demo_uk_hello.mp3"),
        ("Лабораторна робота номер чотири. Синтез вербальної інформації.", "uk", "demo_uk_lab.mp3"),
        ("Hello! This is a voice synthesizer.", "en", "demo_en_hello.mp3"),
        ("Частота дискретизації вербального сигналу складає вісім кілогерц.", "uk", "demo_uk_frequency.mp3"),
    ]

    for text, lang, fname in demos:
        print(f"\n  Текст: \"{text}\"")
        print(f"  Мова: {lang}")
        save_audio(text, lang=lang, filename=fname)

    print(f"\n  Усього згенеровано: {len(demos)} файлів")
    print(f"  Папка: {AUDIO_DIR}/")
    print()


def interactive_mode():
    """Інтерактивний режим голосового помічника."""
    print("=" * 50)
    print("  Голосовий помічник (інтерактивний режим)")
    print("  Команди: привіт, час, дата, як справи,")
    print("           допомога, бувай")
    print("=" * 50)

    say("Вітаю! Я голосовий помічник. Чим допомогти?",
        filename="greeting.mp3")

    while True:
        command = listen_command()
        if not command:
            continue
        if not process_command(command):
            break


def main():
    print("=" * 50)
    print("  Лабораторна робота №4")
    print("  СИНТЕЗ ВЕРБАЛЬНОЇ ІНФОРМАЦІЇ")
    print("=" * 50)
    print()
    print("  Оберіть режим:")
    print("  1 — Генерація демонстраційних аудіофайлів")
    print("  2 — Інтерактивний голосовий помічник")
    print()

    choice = input("  Ваш вибір (1/2): ").strip()

    if choice == "1":
        generate_demo_files()
    elif choice == "2":
        interactive_mode()
    else:
        print("  Невірний вибір. Генерую демо-файли...")
        generate_demo_files()

    print("-> Роботу завершено.")


if __name__ == '__main__':
    main()
