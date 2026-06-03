from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


TTS_MODEL_PATH = r"C:/Users/AHG5HC/Documents/Code/transcript_video_eng/model/Qwen3-TTS-12Hz-1.7B-CustomVoice"

model = Qwen3TTSModel.from_pretrained(
    TTS_MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

wavs, sr = model.generate_custom_voice(
    text="Visual Studio Code will automatically open and import the test workspace folder of the item you need to create.",
    language="English",
    speaker="Ryan",
    instruct="Speak clearly and naturally.",
)

out_path = Path("test_local_tts.wav")
sf.write(str(out_path), wavs[0], sr)

print(f"Saved: {out_path.resolve()}")