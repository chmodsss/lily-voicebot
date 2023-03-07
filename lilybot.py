import os
import openai
import pygame
import whisper
import logging
import regex as re
from io import BytesIO
import numpy as np
from gtts import gTTS
import soundfile as sf
import speech_recognition as sr

openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


wake_word = "lily"
whisper_model = whisper.load_model("base")
recognizer = sr.Recognizer()

''' Returns responses using openai chatgpt API'''
def get_chat(user_input):
   completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": user_input}]
      )
   return completion['choices'][0]['message']['content']

''' Returns transcribed text from audio using whisper module '''
def stt(audiodata):
   wav_stream = BytesIO(audiodata.get_wav_data(convert_rate=16000))
   audio_array, _ = sf.read(wav_stream)
   audio_array = audio_array.astype(np.float32)
   text_recognized = whisper_model.transcribe(audio_array, fp16=False, language='english')
   return text_recognized['text']

''' plays audio file from text using google Text to speech API '''
def tts(ip_txt):
  tts = gTTS(text=ip_txt, lang='en')
  fp = BytesIO()
  tts.write_to_fp(fp)
  fp.seek(0)
  pygame.mixer.init()
  pygame.mixer.music.load(fp)
  pygame.mixer.music.play()
  while pygame.mixer.music.get_busy():
      pygame.time.Clock().tick(10)

''' Listens continuously for the wake word. Runs indefinitely and can only be stopped forcefully.'''
while True:
    logger.info("Say the wake word {} to start.. ".format(wake_word))
    with sr.Microphone() as source:
      audio = recognizer.listen(source)
      try:
         recognized = stt(audio)
         re.sub('[^A-Za-z]+', '', recognized).lower()
         logger.info("Recognized wake word: {}".format(recognized))
         if recognized.strip().lower() == wake_word:
            logger.info("Yes I am listening... ")
            with sr.Microphone() as source:
              audio = recognizer.listen(source)
              recognized = stt(audio)
              logger.info("Recognized text: {}".format(recognized))
              chat_output = get_chat(recognized)
              logger.info("GPT response: {}".format(chat_output))
              tts(chat_output)

      except Exception as e:
         logger.error("Error : {}".format(e))
