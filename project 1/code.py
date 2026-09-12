import os
import sys

# Fix Windows console encoding so non-ASCII (Hindi, Arabic, etc.) display correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from deep_translator import GoogleTranslator
    from gtts import gTTS
except ImportError as exc:
    raise SystemExit(
        "Missing required package(s). Install them with:\n"
        "py -m pip install deep-translator gTTS\n"
    ) from exc

LANGUAGES = {
    "1": ("Hindi",   "hi"),
    "2": ("French",  "fr"),
    "3": ("Spanish", "es"),
    "4": ("German",  "de"),
    "5": ("Arabic",  "ar"),
    "6": ("Japanese","ja"),
    "7": ("Chinese", "zh-CN"),
    "8": ("Urdu",    "ur"),
}


def choose_language():
    print("\nChoose target language:")
    for key, (name, code) in LANGUAGES.items():
        print(f"  {key}. {name}")
    choice = input("\nEnter number (default=1 Hindi): ").strip() or "1"
    return LANGUAGES.get(choice, LANGUAGES["1"])


def translate_text(text, target_code):
    return GoogleTranslator(source="auto", target=target_code).translate(text)


def speak_translation(translated_text, lang_code):
    output_file = "translated_voice.mp3"
    speech = gTTS(text=translated_text, lang=lang_code, slow=False)
    speech.save(output_file)
    os.system(f'start "" "{output_file}"')
    print(f"  Audio saved & playing: {output_file}")


def main():
    print("=" * 55)
    print("        Language Translation Tool")
    print("        (Type text to translate)")
    print("=" * 55)

    while True:
        print("\nOptions:")
        print("  [T] Translate text")
        print("  [Q] Quit")
        action = input("\nYour choice: ").strip().upper()

        if action == "Q":
            print("Goodbye!")
            break

        elif action == "T":
            text = input("\nEnter text to translate: ").strip()
            if not text:
                print("  No text entered. Please try again.")
                continue

            lang_name, lang_code = choose_language()
            print(f"\nTranslating to {lang_name}...")

            try:
                result = translate_text(text, lang_code)
                print("\n" + "-" * 55)
                print(f"  Original   : {text}")
                print(f"  Translated : {result}")
                print("-" * 55)

                speak = input("\nPlay audio of translation? (y/n): ").strip().lower()
                if speak == "y":
                    speak_translation(result, lang_code)

            except Exception as e:
                print(f"  Translation error: {e}")

        else:
            print("  Invalid option. Press T to translate or Q to quit.")


if __name__ == "__main__":
    main()