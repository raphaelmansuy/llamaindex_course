import gradio as gr
from transformers import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition",
                       model="openai/whisper-base.en")


def transcribe(stream, new_chunk):
    """ Transcribe the audio stream and return the text. """
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]

def analyse_text(text):
    """ Append 'Toto' to the transcribed text. """
    return text + " Toto"    


demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)



demo.launch()
