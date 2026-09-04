# Tài liệu Transcript Video

[English](README.md) · [Tiếng Việt](README.vi.md)

Transcript Video là pipeline chạy local, ưu tiên GPU để nhận dạng giọng nói, dịch tiếng Việt sang tiếng Anh, render phụ đề, tạo thuyết minh bằng Qwen TTS và ghép nhiều video thành một khóa học.

> Bản tiếng Anh là tài liệu chính được ưu tiên hiển thị trên GitHub. Bạn có thể đổi ngôn ngữ bằng liên kết phía trên.

## Tổng quan command

```text
transcript-video
├── process [VIDEO]
├── inspect VIDEO
├── doctor
├── config show|validate
└── course create|build|tui
```

Các option toàn cục phải đặt trước command: `-q`, `-v`, `-vv`, `--no-color`, `--log-file PATH` và `--json`. Typer cũng hỗ trợ `--install-completion` và `--show-completion`.

Log mặc định trên terminal được rút gọn. `-v` hiện thêm chẩn đoán, còn `-vv` hiện vị trí source và traceback. Log DEBUG chi tiết được rotate tại `logs/transcript-video.log`, hoặc đường dẫn truyền qua `--log-file`. Cả biến `NO_COLOR` và option `--no-color` đều được hỗ trợ.

## Quy trình thường dùng

```powershell
uv run transcript-video process
uv run transcript-video process lesson.mp4 --profile gpu-tts
uv run transcript-video process lesson.mp4 --dry-run
uv run transcript-video process lesson.mp4 --force transcription --force tts
uv run transcript-video doctor
uv run transcript-video inspect data/input/lesson.mp4
uv run transcript-video config show --sources
uv run transcript-video config validate --profile gpu-tts
```

Dry-run kiểm tra config, input path và kế hoạch thực thi nhưng không tạo folder, lưu config, load model AI hay chạy encode FFmpeg. `--force` có thể lặp lại với `transcription`, `translation`, `tts`, `render`; các cờ overwrite cũ vẫn dùng được.

Profile là file TOML không cần khai báo đủ mọi field, đặt tại `configs/profiles/<tên>.toml` hoặc truyền đường dẫn trực tiếp. Thứ tự ghi đè là: mặc định, config gốc, profile, rồi option CLI.

Command chỉ đọc hỗ trợ JSON khi đặt `--json` trước command:

```powershell
uv run transcript-video --json inspect data/input/lesson.mp4
uv run transcript-video --json doctor
```

Exit code: `0` thành công, `1` lỗi runtime/môi trường chưa sẵn sàng, `2` dùng CLI sai và `130` khi người dùng hủy.

## Course Wizard và TUI đầy đủ

`transcript-video course create` mở wizard Questionary gọn nhẹ. Ở bước review có thể sửa title/number, đổi thứ tự, xóa session và quay lại mà không phải chạy lại từ đầu.

`transcript-video course tui` mở ứng dụng Textual gồm ba màn hình Course Metadata, Session Editor và Review/Build. Đọc metadata video và build course đều chạy background worker. Phím tắt chính: `Ctrl+S` lưu, `Esc` quay lại, `A/E/Delete` thêm/sửa/xóa, `U/D` đổi thứ tự và `Q` thoát. Khi còn thay đổi chưa lưu, ứng dụng sẽ hỏi xác nhận.

## Kiến trúc

Presentation nằm trong `cli.py`, `ui/`, `course/wizard.py` và `tui/`. Các application service trong `application/` xử lý config, diagnostics và inspection. `events.py` định nghĩa stage/observer không phụ thuộc UI. `process_runner.py` chịu trách nhiệm subprocess, ffprobe JSON, parse FFmpeg `-progress pipe:1`, lỗi và hủy tiến trình. Logic media/model nằm trong các module processing và course.

## Quy trình phát triển

```powershell
uv sync
just format
just lint
just test
just check
uv run pre-commit install
```

Các pytest marker gồm `integration`, `gpu`, `slow`; `just test-fast` loại cả ba nhóm. Ruff vừa format vừa lint. Pre-commit chạy Ruff cùng kiểm tra TOML/YAML và whitespace.

## Mục lục

- [Chức năng](#chức-năng)
- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt](#cài-đặt)
- [Tăng tốc GPU](#tăng-tốc-gpu)
- [Cấu hình](#cấu-hình)
- [Quy trình transcription](#quy-trình-transcription)
- [Quy trình TTS](#quy-trình-tts)
- [Course builder](#course-builder)
- [File đầu ra](#file-đầu-ra)
- [Kiểm tra chất lượng](#kiểm-tra-chất-lượng)
- [Xử lý sự cố](#xử-lý-sự-cố)

## Chức năng

- Nhận dạng giọng nói local bằng faster-whisper hoặc Hugging Face Whisper.
- Dịch trực tiếp bằng Whisper hoặc dùng riêng VinAI để dịch tiếng Việt sang tiếng Anh.
- Tạo, kiểm tra, làm sạch và tái sử dụng file SRT.
- Render hard subtitle bằng FFmpeg.
- Qwen3-TTS với các chế độ simple, timed, full và fixed chunk để dễ kiểm tra.
- Thay thế hoặc trộn giọng TTS với audio gốc.
- Ghép course video với mục lục, session card, video chuẩn hóa và MP4 chapter.
- Lưu cấu hình chạy bằng TOML và cấu hình course bằng JSON.
- Chạy inference bằng CUDA và tự động dùng NVIDIA NVENC khi có thể.

## Cấu trúc project

```text
transcript-video/
├── assets/                         # ảnh dùng cho card/mục lục
├── configs/
│   ├── transcription.toml          # profile xử lý mặc định
│   └── courses/                    # profile JSON của course builder
├── data/
│   ├── input/                      # video nguồn
│   ├── subtitles/                  # file SRT
│   ├── audio/                      # WAV TTS và các chunk kiểm tra
│   ├── output/                     # video đã render
│   ├── temp/                       # audio tạm dùng cho ASR
│   └── compilation/                # file tạm và course hoàn chỉnh
├── docs/                           # tài liệu Anh/Việt và prompt chỉnh subtitle
├── scripts/                        # tiện ích smoke test thủ công
├── src/transcript_video/
│   ├── cli.py                      # CLI chính
│   ├── application/                # settings, inspection, diagnostics
│   ├── ui/                         # Rich console, logging, progress
│   ├── tui/                        # ứng dụng course Textual đầy đủ
│   ├── events.py                   # pipeline event độc lập UI
│   ├── process_runner.py           # subprocess/FFmpeg/ffprobe boundary
│   ├── processing/                 # ASR, dịch, subtitle, media, TTS
│   └── course/                     # course builder, card, timeline, wizard
├── tests/                          # automated test ít phụ thuộc
├── pyproject.toml                  # metadata và dependency trực tiếp
├── ruff.toml                       # chính sách lint/format
├── justfile                        # task runner cho developer
├── .pre-commit-config.yaml         # kiểm tra trước commit
└── uv.lock                         # dependency lock tái lập được
```

## Cài đặt

### Yêu cầu

- Windows và Python 3.12.
- Rất nên dùng NVIDIA GPU.
- NVIDIA driver mới, tương thích với PyTorch CUDA 12.4 đã lock.
- Model Whisper local và model VinAI/Qwen local nếu dùng các tính năng tương ứng.

Cài uv:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Tạo `.venv` và cài đúng môi trường đã lock:

```powershell
uv sync
```

Kiểm tra CLI:

```powershell
uv run transcript-video --help
uv run transcript-video course --help
uv run transcript-video course create --help
```

## Tăng tốc GPU

Profile mặc định ưu tiên GPU:

```toml
[hardware]
device = "cuda"
compute_type = "int8_float16"
video_encoder = "auto"
```

### Bảng phân bổ workload

| Workload | Nơi chạy ưu tiên | Ghi chú |
| --- | --- | --- |
| faster-whisper ASR | CUDA INT8/FP16 | `compute_type = "int8_float16"` giữ phần không quantize ở FP16 và giảm bộ nhớ model. |
| Hugging Face Whisper | CUDA FP16 | Transformers pipeline được đặt trên GPU 0. |
| VinAI translation | CUDA FP16 | Trọng số model và batch token đều chuyển lên CUDA. |
| Qwen3-TTS | CUDA FP16 | Model dùng `device_map="cuda:0"`. |
| Encode H.264 cho subtitle/course | NVIDIA NVENC | `auto` kiểm tra `h264_nvenc` bằng một lần encode thật. |
| Filter subtitle/libass | CPU | Subtitle renderer chuẩn của FFmpeg chạy CPU; bước encode cuối vẫn dùng NVENC. |
| Filter scale/pad/fps | CPU | Các filter này cấp frame cho NVENC; encode GPU đã loại bỏ phần CPU nặng có thể tránh được. |
| AAC, WAV, NumPy timing/mixing | CPU | Các bước này nhẹ hoặc tốn nhiều chi phí truyền dữ liệu nếu đưa lên GPU. |
| Vẽ course card bằng Pillow | CPU | Chi phí nhỏ hơn đáng kể so với encode video. |
| Stream copy/mux/chapter metadata | Không nặng compute | Packet đã encode được copy, không re-encode video. |

Nên giữ `video_encoder = "auto"`. Ở lần dùng đầu tiên, project yêu cầu FFmpeg encode thử một frame nhỏ. Chỉ khi encoder, NVIDIA driver và GPU thật sự hoạt động thì `h264_nvenc` mới được chọn. Nếu không, chương trình ghi log và dùng `libx264`.

Thứ tự chọn FFmpeg là `TRANSCRIPT_VIDEO_FFMPEG`, sau đó đến `ffmpeg` trong `PATH` (bao gồm `bin/ffmpeg.exe` sau khi project khởi động), cuối cùng mới là binary của `imageio-ffmpeg`. Binary bundled có thể không chứa NVENC. Để bật GPU video encoding, hãy đặt bản FFmpeg Windows có NVENC tại `bin/ffmpeg.exe`, thêm nó vào `PATH`, hoặc đặt:

```powershell
$env:TRANSCRIPT_VIDEO_FFMPEG = "C:\path\to\ffmpeg.exe"
```

Các policy hỗ trợ:

- `auto`: ưu tiên NVENC, fallback an toàn sang libx264.
- `h264_nvenc`: yêu cầu NVENC nhưng vẫn fallback nếu runtime probe thất bại.
- `libx264`: chủ động encode bằng CPU.

Override cho một lần chạy:

```powershell
uv run transcript-video --device cuda --compute-type int8_float16 --video-encoder h264_nvenc
```

Theo dõi GPU khi đang xử lý video thật:

```powershell
nvidia-smi -l 1
```

Fallback giúp chương trình vẫn chạy trên máy khác, nhưng cảnh báo fallback CUDA đồng nghĩa workload AI sẽ chậm hơn rất nhiều. Nên xử lý nguyên nhân trong phần troubleshooting thay vì chấp nhận fallback khi chạy production.

## Cấu hình

Lệnh chính tự động đọc [profile mặc định](../configs/transcription.toml). Đường dẫn tương đối được resolve từ `project.root`.

```toml
[project]
root = "."
model = "models/faster-whisper-large-v3"
# video = "lesson-01.mp4"
# translation_model = "models/vinai-translate-vi2en-v2"

[hardware]
device = "cuda"
compute_type = "int8_float16"
video_encoder = "auto"

[transcription]
task = "transcribe"
language = "vi"
translation_batch_size = 8
overwrite_srt = false
skip_burn = false

[tts]
enabled = false
overwrite = false
mode = "timed"
generation_mode = "chunked"
model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
language = "English"
speaker = "Aiden"
instruct = "Speak clearly and professionally..."
attn_implementation = "auto"
audio_mode = "replace"
split_audio = true
chunk_minutes = 5
max_speedup = 1.15
chunk_tail_seconds = 10.0
context_max_sentences = 4
context_max_chars = 450
context_break_seconds = 3.0
```

Giá trị từ CLI override TOML nhưng không sửa file:

```powershell
uv run transcript-video --video lesson-02.mp4 --task translate --enable-tts
```

Lưu profile hiệu lực, bao gồm các override:

```powershell
uv run transcript-video `
  --video lesson-02.mp4 `
  --task translate `
  --enable-tts `
  --save-config configs/lesson-02.toml
```

Dùng lại ở lần sau:

```powershell
uv run transcript-video --config configs/lesson-02.toml
```

## Quy trình transcription

Đặt video vào `data/input`, cấu hình đúng đường dẫn model rồi chạy:

```powershell
uv run transcript-video
```

Chỉ xử lý một video:

```powershell
uv run transcript-video --video lesson.mp4
```

Chỉ tạo/tái sử dụng SRT, không render video:

```powershell
uv run transcript-video --video lesson.mp4 --skip-burn
```

Tạo lại SRT đã tồn tại:

```powershell
uv run transcript-video --video lesson.mp4 --overwrite-srt
```

Dùng chuỗi language rỗng để Whisper tự nhận diện ngôn ngữ:

```powershell
uv run transcript-video --language ""
```

Khi dùng VinAI riêng, Whisper sẽ transcribe ngôn ngữ nguồn trước, sau đó VinAI dịch từng batch subtitle:

```powershell
uv run transcript-video `
  --video lesson.mp4 `
  --task translate `
  --translation-model models/vinai-translate-vi2en-v2
```

SRT đã có sẽ được tái sử dụng trừ khi bật `overwrite_srt`.

## Quy trình TTS

Bật thuyết minh tiếng Anh:

```powershell
uv run transcript-video --video lesson.mp4 --task translate --enable-tts
```

Chế độ `chunked` được khuyến nghị. Nó tạo các cửa sổ thời gian cố định để kiểm tra, giữ một khoảng tail an toàn tại biên chunk, rồi dựng lại WAV bằng cách overlay theo timeline. Sau khi nghe kiểm tra, tạo lại một chunk với index bắt đầu từ 0:

```powershell
uv run transcript-video --rerun-tts-chunk 3
```

Các option quan trọng:

- `tts.mode = "timed"`: đặt mỗi câu vào timestamp subtitle tương ứng.
- `tts.mode = "simple"`: tạo một track voice-over liên tục.
- `tts.generation_mode = "chunked"`: tạo chunk có thể tái sử dụng và kiểm tra.
- `tts.generation_mode = "full"`: tạo toàn bộ track trong một lượt.
- `tts.audio_mode = "replace"`: thay audio nguồn.
- `tts.audio_mode = "mix"`: trộn audio nguồn với TTS.
- `tts.max_speedup`: mức tăng tốc giữ nguyên cao độ bằng FFmpeg `atempo` để câu vừa slot.
- `tts.chunk_tail_seconds`: khoảng dự phòng cho câu nằm gần cuối chunk.
- `tts.context_max_sentences` / `context_max_chars`: giới hạn mỗi lần Qwen sinh theo ngữ cảnh.
- `tts.context_break_seconds`: chỉ khoảng nghỉ lớn hơn mức này mới ngắt ngữ cảnh âm học.

TTS timed align mỗi context bằng model faster-whisper đã cấu hình, rồi đặt từng câu đã tách về
đúng timestamp bắt đầu trong SRT. Các lỗi cần kiểm tra thủ công được ghi vào
`data/audio/<video>_tts_review.jsonl`; lời nói không bị âm thầm cắt ngắn.

Chạy smoke test TTS độc lập:

```powershell
uv run python scripts/tts_smoke_test.py --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## Course builder

Tạo JSON profile bằng giao diện terminal:

```powershell
uv run transcript-video course create
```

Build course:

```powershell
uv run transcript-video course build --config configs/courses/training_course.json
```

Mỗi course profile hỗ trợ:

- danh sách session video có thứ tự;
- tiêu đề course và session;
- theme image và custom font tùy chọn;
- phân trang và thời lượng mục lục;
- resolution, frame rate, bitrate và audio sample rate;
- `render.video_encoder` nhận `auto`, `h264_nvenc` hoặc `libx264`;
- MP4 chapter metadata tùy chọn.

Mọi đường dẫn tương đối trong JSON được resolve từ repository root. Video encoder `auto` cũng được áp dụng cho TOC card và bước chuẩn hóa session video.

## File đầu ra

| Artifact | Vị trí mặc định |
| --- | --- |
| SRT | `data/subtitles/<video>_<model>.srt` |
| Video có hard subtitle | `data/output/<video>_vi-dub_en-sub.mp4` |
| WAV TTS hoàn chỉnh | `data/audio/<video>_tts.wav` |
| Chunk TTS để kiểm tra | `data/audio/<video>_tts_chunks/` |
| Log kiểm tra timing/alignment TTS | `data/audio/<video>_tts_review.jsonl` |
| Video TTS cuối | `data/output/<video>_en-dub_en-sub.mp4` |
| File tạm/course cuối | `data/compilation/` |

## Kiểm tra chất lượng

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Áp dụng format:

```powershell
uv run ruff format .
uv run ruff check --fix .
```

VS Code đã được cấu hình dùng interpreter trong `.venv` và Ruff format-on-save.

## Xử lý sự cố

### CUDA không khả dụng

Chạy:

```powershell
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Nếu `nvidia-smi` chạy được nhưng PyTorch trả về `False`, hãy tạo lại môi trường đã lock bằng `uv sync` và kiểm tra xem có Python environment khác đang active hay không.

### NVENC fallback sang libx264

Runtime probe đã thất bại. Nguyên nhân thường gặp gồm NVIDIA driver cũ, FFmpeg không có NVENC (bao gồm một số binary của `imageio-ffmpeg`), GPU đang hết resource hoặc chương trình chạy trên máy không có NVIDIA GPU. Hãy cài FFmpeg có NVENC vào `bin/ffmpeg.exe` hoặc chọn bằng `TRANSCRIPT_VIDEO_FFMPEG`. `auto` vẫn tiếp tục job bằng CPU encode. Các stage AI vẫn có thể dùng CUDA độc lập.

### CUDA hết bộ nhớ

- Đóng ứng dụng khác đang dùng GPU.
- Giảm `translation_batch_size`.
- Giữ `compute_type = "int8_float16"` để giảm VRAM của faster-whisper.
- Dùng TTS chunked mode.
- Dùng model nhỏ hơn nếu có.

GTX 1650 Ti có VRAM hạn chế, vì vậy không nên giữ Whisper, VinAI và Qwen trong VRAM cùng lúc. Pipeline hiện nạp model theo từng stage thay vì cố tình giữ tất cả model resident.

### Vẫn thấy CPU hoạt động

Đây là hành vi bình thường. Decode media, raster subtitle bằng libass, FFmpeg filter, AAC audio, vẽ card bằng Pillow, file I/O và ghép waveform bằng NumPy vẫn dùng CPU. Phần inference model nặng và encode H.264 được hỗ trợ mới là những phần được đưa sang GPU.

### Artifact cũ bị tái sử dụng ngoài mong muốn

Dùng `--overwrite-srt` hoặc `--overwrite-tts`. Với TTS chunked, dùng `--rerun-tts-chunk INDEX` để chỉ tạo lại chunk cần thiết.
