import argparse
import os
import subprocess
import imageio_ffmpeg

class SubtitleSegment:
    """Class dùng chung để đồng bộ định dạng segment giữa các engine"""
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text

def format_timestamp(seconds):
    """Chuyển đổi giây sang định dạng SRT: HH:MM:SS,mmm"""
    if seconds is None:
        seconds = 0.0
    td_h = int(seconds // 3600)
    td_m = int((seconds % 3600) // 60)
    td_s = int(seconds % 60)
    td_ms = int((seconds % 1) * 1000)
    return f"{td_h:02}:{td_m:02}:{td_s:02},{td_ms:03}"

def create_srt(segments, srt_path):
    """Tạo file phụ đề .srt"""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            f.write(f"{i}\n{start} --> {end}\n{segment.text.strip()}\n\n")

def burn_subtitles(video_in, srt_in, video_out):
    """Dùng FFmpeg khép kín để chèn cứng phụ đề vào video"""
    srt_in_escaped = srt_in.replace("\\", "/").replace(":", "\\:")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    command = [
        ffmpeg_path, '-y',
        '-i', video_in,
        '-vf', f"subtitles='{srt_in_escaped}'",
        '-c:a', 'copy', 
        video_out
    ]
    subprocess.run(command, check=True)

# --- MAIN PROCESS ---
parser = argparse.ArgumentParser(description="Transcribe and burn subtitles into a video.")
parser.add_argument("input_video", help="Path to the input video file")
# CLI ARGUMENT: Mặc định vẫn là bản faster, ông có thể đổi qua CLI
parser.add_argument("--model", default="C:\\Users\\AHG5HC\\.faster-whisper-large-v3", help="Đường dẫn tới thư mục model")
args = parser.parse_args()

input_video = args.input_video
video_dir = os.path.dirname(os.path.abspath(input_video))
video_stem = os.path.splitext(os.path.basename(input_video))[0]

temp_srt = os.path.join(video_dir, video_stem + "_sub.srt")
output_video = os.path.join(video_dir, video_stem + "_ENG_SUB.mp4")

# Check if SRT already exists
if os.path.exists(temp_srt):
    print(f"SRT file already exists: {temp_srt}")
    print("Skipping transcription, going straight to burn step.")
else:
    print("Step 1: Transcribing and Translating (Vietnamese -> English)...")
    
    # Kiểm tra xem thư mục truyền vào là loại model nào
    is_faster_whisper = os.path.exists(os.path.join(args.model, "model.bin"))
    is_hf_whisper = os.path.exists(os.path.join(args.model, "model.safetensors")) or os.path.exists(os.path.join(args.model, "pytorch_model.bin"))
    
    final_segments = []
    
    # THƯỜNG HỢP 1: Chạy bằng engine faster-whisper ngon bổ rẻ
    if is_faster_whisper:
        print(f"--> Engine detect: [faster-whisper] tại {os.path.basename(args.model)}")
        from faster_whisper import WhisperModel
        model = WhisperModel(args.model, device="cuda", compute_type="int8")
        segments, info = model.transcribe(input_video, task="translate", beam_size=5)
        for s in segments:
            final_segments.append(SubtitleSegment(s.start, s.end, s.text))
            
    # TRƯỜNG HỢP 2: Chạy bằng bản gốc Hugging Face (Phù hợp với thư mục mới của ông)
    elif is_hf_whisper:
        print(f"--> Engine detect: [Standard Hugging Face] tại {os.path.basename(args.model)}")
        print("Lưu ý: Bản gốc PyTorch này ngốn nhiều VRAM và chạy chậm hơn bản 'faster' nhé!")
        
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        # Mẹo né AppLocker: Trích xuất âm thanh ra file tạm trước để đưa vào Pipeline độc lập
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        temp_audio = os.path.join(video_dir, "temp_audio_extract.wav")
        print("Extracting audio stream safely...")
        extract_cmd = [
            ffmpeg_path, '-y', '-i', input_video, 
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio
        ]
        subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load model bằng cấu hình chuẩn PyTorch
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to("cuda")
        processor = AutoProcessor.from_pretrained(args.model)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            device="cuda",
            dtype=torch_dtype
        )
        
        # Thực hiện dịch thuật thuật toán
        result = pipe(temp_audio, generate_kwargs={"task": "translate"}, return_timestamps=True)
        
        # Xóa file âm thanh tạm
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
            
        for chunk in result.get("chunks", []):
            ts = chunk.get("timestamp")
            if ts is not None:
                final_segments.append(SubtitleSegment(ts[0], ts[1], chunk.get("text")))
    else:
        raise ValueError(f"Không nhận diện được định dạng model tại đường dẫn: {args.model}")

    # 2. Save to SRT
    print("Step 2: Generating SRT file...")
    create_srt(final_segments, temp_srt)

# 3. Burn to Video
print("Step 3: Burning subtitles into video with FFmpeg...")
try:
    burn_subtitles(input_video, temp_srt, output_video)
    print(f"DONE! Video output: {output_video}")
except Exception as e:
    print(f"Error during burning: {e}")