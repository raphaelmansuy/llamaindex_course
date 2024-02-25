import os
import whisper

# Assuming DecodingOptions is the correct class to set the language
from whisper import DecodingOptions

# Create a DecodingOptions instance with the language set to English
decoding_options = DecodingOptions(language="en")

VOICE_FILE = "demo.m4a"

# Get current path
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))


FILE_PATH = os.path.join(CURRENT_PATH, VOICE_FILE)

# Load the model
model = whisper.load_model("base.en")

# Decode the audio
result = whisper.transcribe(model, FILE_PATH)

print(result)
