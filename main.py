#!/usr/bin/env python
import argparse
import json
import logging
import re
from http.cookies import SimpleCookie
from pathlib import Path

import pyaudio
import requests
import speech_recognition as sr
import Stemmer
import webrtcvad

# Константы
RATE = 16000
FRAME_DURATION_MS = 30  # Длительность одного фрейма в миллисекундах
FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)  # 480 байт = 30мс
FORMAT = pyaudio.paInt16
CHANNELS = 1
MIN_SPEECH_FRAMES = 10
MAX_SILENCE_FRAMES = 5
BUFFER_THRESHOLD = 30_000 / FRAME_DURATION_MS * FRAME_SIZE
MIN_RECOGNIZE_FRAMES = 10

vad = webrtcvad.Vad()
vad.set_mode(2)

stemmer = Stemmer.Stemmer("russian")


def cookies_as_dict(cookie_string):
    """Парсит строку с куками и возвращает их в виде словаря."""
    cookie = SimpleCookie()
    cookie.load(cookie_string)
    return {key: morsel.value for key, morsel in cookie.items()}


class BlackboxError(Exception):
    pass


class BlackboxChat:
    chat_endpoint: str = "https://www.blackbox.ai/api/chat"

    def __init__(
        self,
        chat_id: str,
        cookies: dict,
        validated: str,
        session: requests.Session = None,
    ):
        self.chat_id = chat_id
        self.cookies = cookies
        self.validated = validated
        self.session = session or self.get_session()

    def get_session(self) -> requests.Session:
        session = requests.session()
        session.headers.update(
            {
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9,ru;q=0.8,bg;q=0.7",
                "content-type": "application/json",
                "origin": "https://www.blackbox.ai",
                "priority": "u=1, i",
                "referer": "https://www.blackbox.ai/",
                "sec-ch-ua": '"Brave";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Linux"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "sec-gpc": "1",
                "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            }
        )
        return session

    def send_message(self, message: str) -> str:
        """Отправляет распознанный текст на API BlackBox.ai."""
        data = {
            "messages": [
                {"id": self.chat_id, "content": message, "role": "user"}
            ],
            "id": self.chat_id,
            "previewToken": None,
            "userId": None,
            "codeModelMode": True,
            "agentMode": {},
            "trendingAgentMode": {},
            "isMicMode": False,
            "maxTokens": 1024,
            "playgroundTopP": None,
            "playgroundTemperature": None,
            "isChromeExt": False,
            "githubToken": "",
            "clickedAnswer2": False,
            "clickedAnswer3": False,
            "clickedForceWebSearch": False,
            "visitFromDelta": False,
            "mobileClient": False,
            "userSelectedModel": None,
            "validated": self.validated,
            "imageGenerationMode": False,
            "webSearchModePrompt": False,
        }

        try:
            response = self.session.post(
                self.chat_endpoint, json=data, cookies=self.cookies
            )
            return response.text
        except requests.exceptions.RequestException as ex:
            raise BlackboxError() from ex


def split_words(text: str) -> list[str]:
    return re.findall(r"\w+(?:-\w+)*", text)


def is_question(sentence: str) -> bool:
    question_stems = {
        "как",
        "поч",
        "кто",
        "когд",
        "что",
        "где",
        "скольк",
        "испра",
        "расскаж",
        "выполн",
        "напеч",
        "доба",
        "пожалуйс",
        "допиш",
        "улучш",
        "помог",
        "подгот",
        "продемонстрир",
        "настр",
        "подскаж",
        "определ",
        "ответ",
        "дополн",
        "сдел",
        "реализоват",
        "созд",
        "изм",
        "провер",
        "перепис",
        "убед",
        "наход",
        "упрост",
        "объясн",
        "оптимиз",
        "запуск",
        "напис",
        "нарис",
    }

    words = split_words(sentence.lower())
    stems = set(stemmer.stemWords(words))
    return bool(stems & question_stems)


def process_speech(chat: BlackboxChat) -> None:
    """Основная функция для захвата аудио, распознавания речи и отправки данных на сервер."""

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    )
    logging.info("Начало захвата звука. Говорите!")

    recognizer = sr.Recognizer()
    audio_buffer = bytearray()
    speech_frames = 0
    silence_frames = 0
    speech_recognition_started = False

    try:
        while True:
            # Захват аудио данных
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            audio_buffer.extend(data)
            # Проверка, является ли фрейм речью
            if vad.is_speech(data, RATE):
                silence_frames = 0
                speech_frames += 1
                if (
                    speech_frames >= MIN_SPEECH_FRAMES
                    and not speech_recognition_started
                ):
                    logging.info("Начинаем распознавание речи")
                    speech_recognition_started = True
                    audio_buffer = audio_buffer[
                        : -FRAME_SIZE * MIN_SPEECH_FRAMES
                    ]
            else:
                speech_frames = 0
                silence_frames += 1
                # Если запись не начата, то очистим буфер
                if not speech_recognition_started:
                    audio_buffer.clear()
                    continue

            # Проверка на конец речи
            if speech_recognition_started and (
                silence_frames >= MAX_SILENCE_FRAMES
                or len(audio_buffer) >= BUFFER_THRESHOLD
            ):
                logging.info("Конец распознавания речи")

                if len(audio_buffer) // FRAME_SIZE >= MIN_RECOGNIZE_FRAMES:
                    try:
                        # Распознавание речи
                        audio_data = sr.AudioData(audio_buffer, RATE, 2)
                        text = recognizer.recognize_google(
                            audio_data, language="ru-RU"
                        )
                        logging.info(f"Распознанный текст: {text}")

                        # Отправка данных на сервер
                        if is_question(text):
                            try:
                                answer = chat.send_message(text)
                                answer = answer.split("$~~~$")[-1]
                                print(answer.strip())
                            except BlackboxError:
                                logging.error("Ошибка Blackbox.ai")
                        else:
                            logging.warning("Текст не является вопросом")
                    except sr.UnknownValueError:
                        logging.warning("Не удалось распознать речь.")
                    except sr.RequestError as e:
                        logging.error(f"Ошибка сервиса распознавания речи: {e}")
                else:
                    logging.warning("Нечего распознавать.")
                audio_buffer.clear()
                speech_frames = 0
                silence_frames = 0
                speech_recognition_started = False

    except KeyboardInterrupt:
        logging.info("Программа остановлена пользователем.")
    except Exception as e:
        logging.error(f"Непредвиденная ошибка: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        logging.info("Аудиопоток закрыт. Программа завершена.")


def setup_logging(verbosity):
    """
    Устанавливает уровень логирования в зависимости от количества флагов -v.
    Чем больше флагов -v, тем ниже уровень логирования.
    """
    level = max(logging.DEBUG, logging.ERROR - logging.DEBUG * verbosity)
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Распознает текст из речи для получения ответов от ChatBox.AI."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Увеличить уровень детализации логов (добавьте -v для WARNING, -vv для INFO, -vvv для DEBUG).",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    config_path: Path = Path.cwd() / "config.json"

    if not config_path.exists():
        print(
            """\
Откройте сайт https://blackbox.ai.

Введите какой-нибудь запрос чтобы создался чат.

 * Выполните `copy(document.cookie)`, чтобы копировать cookie.
 * Скопируйте идентефикатор чата из адреса (наюор символов без слешем).
 * Во вкладке `Network` найдите `chat` и в `Payload` скопируйте параметр `validated`.
              """
        )

        cookie = input("Cookie: ")
        chat_id = input("Chat ID: ")
        validated = input("Validated: ")

        with config_path.open("w") as fp:
            json.dump(
                {
                    "cookie": cookie,
                    "chat_id": chat_id,
                    "validated": validated,
                },
                fp,
                ensure_ascii=True,
                indent=2,
            )
    else:
        with config_path.open() as fp:
            config = json.load(fp)

        cookie = config["cookie"]
        chat_id = config["chat_id"]
        validated = config["validated"]

    cookies = cookies_as_dict(cookie)
    chat = BlackboxChat(chat_id, cookies, validated)
    process_speech(chat)


if __name__ == "__main__":
    main()
