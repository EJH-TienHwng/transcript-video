# Tài liệu Transcript Video

[English](README.md) · [Tiếng Việt](README.vi.md)

Transcript Video là pipeline chạy local, ưu tiên GPU để nhận dạng tiếng Việt, dùng phụ đề tiếng Anh được dịch/chỉnh sửa bên ngoài, tạo thuyết minh bằng Qwen TTS và ghép nhiều video thành một khóa học.

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

Các option toàn cục phải đặt trước command: `-q`, `-v`, `-vv`, `--no-color`, `--plain`, `--log-file PATH` và `--json`. Typer cũng hỗ trợ `--install-completion` và `--show-completion`.

Log mặc định trên terminal được rút gọn. `-v` hiện thêm chẩn đoán, còn `-vv` hiện vị trí source và traceback. Log DEBUG chi tiết được rotate tại `logs/transcript-video.log`, hoặc đường dẫn truyền qua `--log-file`. Cả biến `NO_COLOR` và option `--no-color` đều được hỗ trợ.


Xem [logging/event schema](observability.md): default chỉ hiện status/progress, warning và kết quả; `-q` tắt progress. Mỗi lệnh dài có log DEBUG text + JSONL riêng trong `logs/runs/`. `--events-json FILE` dùng với `process` hoặc `course build` để ghi event JSONL vào file mới; dry-run không ghi file. Wizard hỗ trợ chọn nhiều video, tự tạo title/number, Recommended/Custom và final review trước khi lưu.
## Quy trình thường dùng

```powershell
uv run transcript-video process
uv run transcript-video process lesson.mp4 --profile gpu-tts
uv run transcript-video process lesson.mp4 --dry-run
uv run transcript-video process lesson1.mp4 lesson2.mp4 lesson3.mp4
uv run transcript-video process lesson.mp4 --translated-srt edits/lesson_en.srt
uv run transcript-video process lesson.mp4 --force transcription
uv run transcript-video doctor
uv run transcript-video inspect data/input/lesson.mp4
uv run transcript-video config show --sources
uv run transcript-video config validate --profile gpu-tts
```

Dry-run kiểm tra config, input, model ASR, output và riêng từng vai trò subtitle nhưng không tạo folder, lưu config, load model hay chạy FFmpeg. Nó hiển thị source/translated SRT đang tồn tại hay thiếu và video sẽ render hay dừng ở bước handoff. `--force transcription` chỉ tạo lại Vietnamese source SRT.

## Handoff phụ đề bắt buộc

1. Chạy `process VIDEO`. Whisper ghi Vietnamese source SRT do ứng dụng sở hữu tại `data/subtitles/source/<stem>_vi_<backend>.srt` nếu file chưa tồn tại.
2. Gửi SRT đó cho LLM bên ngoài cùng [`prompts/optimal_prompt.md`](prompts/optimal_prompt.md).
3. Lưu kết quả tiếng Anh đã chỉnh sửa tại `data/subtitles/translated/<stem>_en.srt`, hoặc dùng `--translated-srt PATH` khi xử lý một video.
4. Chạy lại `process`. Timed/chunked Qwen TTS tạo `data/subtitles/retimed/<stem>_en_retimed.srt` từ placement thực tế; SRT dẫn xuất này được burn trước khi mux. TTS disabled/simple burn trực tiếp SRT đã dịch.

Ứng dụng không tự dịch file này và transcription không bao giờ ghi vào `translated/`. Nếu thiếu English SRT, lệnh kết thúc bình thường sau khi tạo/tái sử dụng Vietnamese source SRT, báo rõ đường dẫn cần tạo và không load Qwen.

SRT trong `translated/` là bản chuẩn và không bị ghi lại. Retiming giữ nguyên cue khi narration ngắn hơn, chỉ dời start khi TTS bắt đầu trễ có ý nghĩa, và kéo dài end khi TTS dài hơn. Cue chỉ được kết thúc sớm hơn timing đã dịch để bàn giao sạch cho cue retimed tiếp theo, và không bao giờ trước khi narration của chính cue đó kết thúc. So sánh theo mili giây loại bỏ nhiễu sample; review TTS lưu timing gốc, placement thực tế, timing retimed cuối cùng, độ dịch và lý do.

## Video speed-up đầu ra

`<stem>_vi-dub_en-sub.mp4` dùng audio tiếng Việt với subtitle tiếng Anh. Khi bật TTS, `<stem>_en-dub_en-sub.mp4` dùng English TTS với subtitle tiếng Anh. `process --speedup` áp dụng cùng một timeline `data/speedup/<stem>.speedup.toml` cho cả hai video thường và tạo hai artifact `_speedup.mp4`; khi tắt TTS chỉ xử lý bản audio tiếng Việt. Lệnh `speedup` độc lập tự tìm một hoặc cả hai output chuẩn đang tồn tại.

Các trạng thái spec được phân biệt rõ: thiếu file thì tạo một template và hoãn speed-up; file có 0 segment, kể cả file chỉ chứa `# Reviewed: no speed-up required.`, hoàn tất mà không chạy FFmpeg hay tạo bản sao; có segment hợp lệ thì tạo output; TOML hoặc interval không hợp lệ thì báo lỗi. Dry-run chỉ báo trạng thái và không tạo file.

Profile là file TOML không cần khai báo đủ mọi field, đặt tại `configs/profiles/<tên>.toml` hoặc truyền đường dẫn trực tiếp. Thứ tự ghi đè là: mặc định, config gốc, profile, rồi option CLI.

Command chỉ đọc hỗ trợ JSON khi đặt `--json` trước command:

```powershell
uv run transcript-video --json inspect data/input/lesson.mp4
uv run transcript-video --json doctor
```

Exit code: `0` thành công, `1` lỗi runtime/môi trường chưa sẵn sàng, `2` dùng CLI sai và `130` khi người dùng hủy.

## Course Wizard và TUI đầy đủ

`transcript-video course create` mở wizard Questionary gọn nhẹ. Ở bước review có thể Add session (dùng lại validation/cache, chặn video trùng), sửa title/number, đổi thứ tự, xóa session và quay lại mà không phải chạy lại từ đầu.

`transcript-video course tui` mở ứng dụng Textual gồm ba màn hình Course Metadata, Session Editor và Review/Build. Đọc metadata video và build course đều chạy background worker. Phím tắt chính: `Ctrl+S` lưu, `Esc` quay lại, `A/E/Delete` thêm/sửa/xóa, `U/D` đổi thứ tự và `Q` thoát. Khi còn thay đổi chưa lưu, ứng dụng sẽ hỏi xác nhận.

Browser chỉ liệt kê media trong `data/input` và `data/output`, không quét đệ quy. Enter điền form,
Add xác nhận; vẫn có manual path. Edit giữ nguyên session đến khi Save changes hợp lệ; `Ctrl+E`
hủy edit, giữ nguyên vị trí và number. Phải save/cancel trước khi remove/reorder/review.
Metadata chạy worker, cache theo path/mtime/size. Settings có card duration, chapters, TOC và
advanced rendering/font. Review có duration từng session, tổng nguồn và ước tính gồm card/TOC.
ProgressBar hiển thị số stage hoàn tất, progress thực của stage/FFmpeg và session; total chưa biết
thì busy. `Ctrl+B` build tại review; trong khi build, khóa save/navigation/quit đến khi worker xong.
JSON hỏng báo lỗi kèm path và không ghi đè; Wizard/TUI/builder dùng chung domain validation,
atomic save giữ unknown fields. TUI no-color là grayscale, vẫn cần điều khiển cursor fullscreen.

`inspect` dùng bảng dễ đọc và hoạt động khi thiếu model; subtitle artifact có thể unresolved.
`--json config validate` trả kết quả machine-readable; config sai exit 1. Doctor tách configured
encoder/NVENC listing/runtime/fallback, hiển thị PASS/WARN/FAIL cho môi trường, model và storage.
CI Windows/Linux dùng `.github/requirements-test.txt` + editable `--no-deps`, không cần CUDA wheels.
Xem [README chính](../README.md) để biết đầy đủ behavior và recipe mới.

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

- [Tài liệu Transcript Video](#tài-liệu-transcript-video)
  - [Tổng quan command](#tổng-quan-command)
  - [Quy trình thường dùng](#quy-trình-thường-dùng)
  - [Course Wizard và TUI đầy đủ](#course-wizard-và-tui-đầy-đủ)
  - [Kiến trúc](#kiến-trúc)
  - [Quy trình phát triển](#quy-trình-phát-triển)
  - [Mục lục](#mục-lục)
  - [Chức năng](#chức-năng)
  - [Cấu trúc project](#cấu-trúc-project)
  - [Cài đặt](#cài-đặt)
    - [Yêu cầu](#yêu-cầu)
  - [Tăng tốc GPU](#tăng-tốc-gpu)
    - [Bảng phân bổ workload](#bảng-phân-bổ-workload)
  - [Cấu hình](#cấu-hình)
  - [Quy trình transcription](#quy-trình-transcription)
  - [Quy trình TTS](#quy-trình-tts)
  - [Course builder](#course-builder)
  - [File đầu ra](#file-đầu-ra)
  - [Kiểm tra chất lượng](#kiểm-tra-chất-lượng)
  - [Xử lý sự cố](#xử-lý-sự-cố)
    - [CUDA không khả dụng](#cuda-không-khả-dụng)
    - [NVENC fallback sang libx264](#nvenc-fallback-sang-libx264)
    - [CUDA hết bộ nhớ](#cuda-hết-bộ-nhớ)
    - [Vẫn thấy CPU hoạt động](#vẫn-thấy-cpu-hoạt-động)
    - [Source subtitle cũ được tái sử dụng ngoài mong muốn](#source-subtitle-cũ-được-tái-sử-dụng-ngoài-mong-muốn)

## Chức năng

- Nhận dạng giọng nói local bằng faster-whisper hoặc Hugging Face Whisper.
- Tạo, kiểm tra, làm sạch và tái sử dụng Vietnamese source SRT theo sự tồn tại của file.
- Handoff thủ công cho bước dịch/chỉnh sửa English SRT bên ngoài với quyền sở hữu tách biệt.
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

- Windows và Python theo `.python-version`.
- Rất nên dùng NVIDIA GPU.
- NVIDIA driver mới, tương thích với PyTorch CUDA 13.2 đã lock.
- Model Whisper local và model Qwen local nếu dùng TTS.

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

[hardware]
device = "cuda"
compute_type = "int8_float16"
video_encoder = "auto"

[transcription]
language = "vi"
overwrite_srt = false
skip_burn = false

[tts]
enabled = false
mode = "timed"
generation_mode = "chunked"
model = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
language = "English"
speaker = "Aiden"
instruct = "Speak clearly and professionally..."
attn_implementation = "auto"
audio_mode = "replace"
split_audio = true
chunk_minutes = 5
max_speedup = 1.25
chunk_tail_seconds = 10.0
context_max_sentences = 4
context_max_chars = 450
context_break_seconds = 3.0
```

Giá trị từ CLI override TOML nhưng không sửa file:

```powershell
uv run transcript-video process lesson-02.mp4 --translated-srt edits/lesson-02_en.srt --enable-tts
```

Lưu profile hiệu lực, bao gồm các override:

```powershell
uv run transcript-video process lesson-02.mp4 `
  --enable-tts `
  --save-config configs/lesson-02.toml
```

Dùng lại ở lần sau:

```powershell
uv run transcript-video process --config configs/lesson-02.toml
```

## Quy trình transcription

Đặt video vào `data/input`, cấu hình đúng đường dẫn model rồi chạy:

```powershell
uv run transcript-video process
```

Chỉ xử lý một video:

```powershell
uv run transcript-video process lesson.mp4
```

Chỉ tạo/tái sử dụng SRT, không render video:

```powershell
uv run transcript-video process lesson.mp4 --skip-burn
```

Tạo lại SRT đã tồn tại:

```powershell
uv run transcript-video process lesson.mp4 --force transcription
```

Dùng chuỗi language rỗng để Whisper tự nhận diện ngôn ngữ:

```powershell
uv run transcript-video process lesson.mp4 --language ""
```

Hãy dịch và chỉnh sửa Vietnamese SRT bên ngoài bằng [`prompts/optimal_prompt.md`](prompts/optimal_prompt.md), lưu vào English path được báo, rồi chạy lại:

```powershell
uv run transcript-video process lesson.mp4
```

Source SRT đang tồn tại được tái sử dụng chỉ dựa vào sự tồn tại của file. Thay đổi model/config
không tự invalidate file. `--force transcription` chỉ tạo lại source và giữ nguyên từng byte của
English SRT. File provenance cũ bị bỏ qua và có thể xóa thủ công.

## Quy trình TTS

Bật thuyết minh tiếng Anh:

```powershell
uv run transcript-video process lesson.mp4 --enable-tts
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
gần timestamp bắt đầu trong English SRT nhất có thể, dịch thời điểm câu khi cần để tránh chồng giọng. Các lỗi cần kiểm tra thủ công được ghi vào
`data/report/tts/<video>_tts_review.jsonl`; lời nói không bị âm thầm cắt ngắn.

JSONL giữ một object trên mỗi dòng. Bản dễ đọc có đuôi `.pretty.json` nằm cùng thư mục.
Metadata chunk nằm tại `data/report/tts/<video>_tts_chunks/`; WAV vẫn ở `data/audio/`.
Sidecar cũ cạnh WAV bị bỏ qua. Lần chạy bình thường tạo lại mọi TTS chunk; rerun chunk rõ ràng
vẫn xử lý cả owner của context vượt biên và dựng lại từ các chunk còn lại.
Padding cuối 180 ms có guard bảo vệ onset câu sau; release gap tối thiểu là 120 ms.
Boundary không an toàn vẫn sinh riêng sentence, không hard-trim speech.

Video list giữ thứ tự nhập, bỏ path trùng lặp sau lần đầu, hỗ trợ filename trong `data/input`
và absolute path. Hai file khác nhau trùng stem sẽ bị từ chối vì dùng chung tên output.
Không truyền video vẫn dùng `project.video` nếu đã cấu hình, nếu không sẽ scan `data/input`.
Doctor đọc `.python-version`, cảnh báo nếu pin thiếu/sai và kiểm tra quyền ghi `data/report`.

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
| Vietnamese source SRT (ứng dụng sở hữu) | `data/subtitles/source/<video>_vi_<backend>.srt` |
| English translated SRT (người dùng sở hữu) | `data/subtitles/translated/<video>_en.srt` |
| English retimed SRT (generated) | `data/subtitles/retimed/<video>_en_retimed.srt` |
| Video audio tiếng Việt + subtitle tiếng Anh | `data/output/<video>_vi-dub_en-sub.mp4` |
| WAV TTS hoàn chỉnh | `data/audio/<video>_tts.wav` |
| Chunk TTS để kiểm tra | `data/audio/<video>_tts_chunks/` |
| Log kiểm tra timing/alignment TTS | `data/report/tts/<video>_tts_review.jsonl` |
| Video English TTS + subtitle tiếng Anh | `data/output/<video>_en-dub_en-sub.mp4` |
| Video speed-up tùy chọn | `data/output/<video>_{vi,en}-dub_en-sub_speedup.mp4` |
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
- Giữ `compute_type = "int8_float16"` để giảm VRAM của faster-whisper.
- Dùng TTS chunked mode.
- Dùng model nhỏ hơn nếu có.

Runtime batch do application sở hữu tái sử dụng model, chuyển workload GPU trước đó về CPU
trước khi kích hoạt model tiếp theo; aligner CPU được giữ để dùng lại. Cách này vẫn cần RAM
và thời gian truyền CPU/GPU. Chưa benchmark offload với model thật.

### Vẫn thấy CPU hoạt động

Đây là hành vi bình thường. Decode media, raster subtitle bằng libass, FFmpeg filter, AAC audio, vẽ card bằng Pillow, file I/O và ghép waveform bằng NumPy vẫn dùng CPU. Phần inference model nặng và encode H.264 được hỗ trợ mới là những phần được đưa sang GPU.

### Source subtitle cũ được tái sử dụng ngoài mong muốn

Dùng `--force transcription`. Lần chạy TTS bình thường tái sử dụng artifact hợp lệ; dùng `--force tts` để tạo lại toàn bộ hoặc `--rerun-tts-chunk INDEX` để tạo lại một chunk.

## Cập nhật QA và an toàn artifact

```powershell
uv run transcript-video process lesson.mp4 --profile srt
uv run transcript-video process lesson.mp4 --profile tts-review
uv run transcript-video process one.mp4 two.mp4 --profile srt
```

Profile `tts-review` bật ASR verifier đọc sentence từ WAV cuối đã publish; CLI override vẫn thắng.
Review v4 có từ thiếu/thừa, `missing_final_word`, coverage và edit ratio dùng SequenceMatcher
(không phải minimum-edit WER). Tail RMS là heuristic REVIEW, không phải kết luận mất âm cuối.
Mọi speed-up thực tế có REVIEW và được đếm trong `speed_adjusted`. Report duration ghi nguồn,
WAV, MP4 và delta; không cắt speech để bằng thời lượng nguồn.

SRT/WAV/MP4 được ghi tạm rồi atomic replace. Không còn provenance/fingerprint hay automatic
cache invalidation; chunk rerun vẫn giữ owner context vượt biên. `[subtitle_style]` có validation, mặc định vẫn
`MarginV=25`. FA2 là tùy chọn và fallback SDPA khi không hỗ trợ; không thêm SoX.
Chi tiết review và cách lọc câu speed-up nằm trong [README](../README.md#tts-review-reports).
Các mục phát âm/prosody, giọng nhất quán và natural pauses vẫn cần nghe thủ công;
xem [giới hạn kiểm chứng](quality-validation.md) và [TODO](todo.md).
