from __future__ import annotations

import argparse
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


DEFAULT_TTS_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"

DEFAULT_TEST_TEXT = (
    "Today, we will learn how to create and manage a test workspace in Visual Studio Code. "
    "First, make sure that your project folder is opened correctly. "
    "A test workspace helps us organize source code, test cases, and output files in one clear structure. "
    "When the workspace is ready, we can generate test cases for each important function. "
    "Each test case should include an input, an expected output, and a clear pass or fail condition. "
    "If a test fails, do not rush to change the code immediately. "
    "Instead, read the error message carefully and compare the actual result with the expected result. "
    "This habit will help you debug more systematically and write more reliable software."
)

DEFAULT_INSTRUCT = (
    "Use a calm, clear, professional teaching voice. "
    "Keep a steady medium pace and neutral tone. "
    "Do not laugh, dramatize, or add emotions. "
    "Read exactly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick local Qwen3-TTS test with a teaching-style script."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TTS_MODEL_PATH,
        help="Local Qwen3-TTS CustomVoice model folder.",
    )
    parser.add_argument(
        "--output",
        default="test_local_tts_teaching.wav",
        help="Output WAV file path.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEST_TEXT,
        help="Text to synthesize. Default is an 8-sentence teaching script.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="TTS language. Default: English.",
    )
    parser.add_argument(
        "--speaker",
        default="Aiden",
        help="Qwen CustomVoice speaker. Examples: Aiden, Ryan, Vivian, Serena.",
    )
    parser.add_argument(
        "--instruct",
        default=DEFAULT_INSTRUCT,
        help="Voice style instruction.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--attn",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help="Attention implementation. Use eager for safest Windows testing.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype. float16 is recommended for RTX 3060.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available. Falling back to CPU.")
        device = "cpu"

    dtype = resolve_dtype(args.dtype)
    if device == "cpu":
        dtype = torch.float32

    print("========== Qwen3-TTS Local Test ==========")
    print(f"Model     : {model_path}")
    print(f"Output    : {output_path}")
    print(f"Device    : {device}")
    print(f"DType     : {dtype}")
    print(f"Attention : {args.attn}")
    print(f"Speaker   : {args.speaker}")
    print(f"Language  : {args.language}")
    print("==========================================")

    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map="cuda:0" if device == "cuda" else "cpu",
        dtype=dtype,
        attn_implementation=args.attn,
    )

    wavs, sr = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sr)

    print(f"Saved: {output_path}")
    print(f"Sample rate: {sr}")


if __name__ == "__main__":
    main()
