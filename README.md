<div align="center">

# Transcript Video

**Chuyển giọng nói trong video thành phụ đề, xuất file SRT và burn hard subtitle vào video bằng Whisper.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-faster--whisper%20%7C%20Transformers-412991)
![FFmpeg](https://img.shields.io/badge/FFmpeg-imageio--ffmpeg-007808?logo=ffmpeg&logoColor=white)

</div>

## Mục lục

- [Tổng quan](#tổng-quan)
- [Luồng xử lý](#luồng-xử-lý)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị model](#chuẩn-bị-model)
- [Sử dụng nhanh](#sử-dụng-nhanh)
- [Tham số CLI](#tham-số-cli)
- [Cách các engine hoạt động](#cách-các-engine-hoạt-động)
- [Output](#output)
- [Lưu ý vận hành](#lưu-ý-vận-hành)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

## Tổng quan

Project cung cấp một pipeline Python chạy local để xử lý video theo batch:

1. Đọc một video cụ thể hoặc toàn bộ video trong `data/input/`.
2. Tự nhận diện định dạng model Whisper.
3. Chuyển giọng nói thành phụ đề nguyên ngữ hoặc dịch sang tiếng Anh.
4. Ghi phụ đề ra file `.srt` trong `data/subtitles/`.
5. Burn phụ đề vào video và ghi kết quả vào `data/output/`.

Pipeline hỗ trợ hai loại model nhận dạng giọng nói local và một model dịch văn bản tùy chọn:

| Engine | File dùng để nhận diện | Vai trò |
| --- | --- | --- |
| `faster-whisper` | `model.bin` | Nhẹ hơn, phù hợp để xử lý nhanh với CTranslate2 |
| Hugging Face Transformers | `model.safetensors` hoặc `pytorch_model.bin` | Dùng model Whisper chuẩn từ Hugging Face |
| VinAI Translate | `pytorch_model.bin` với `model_type: mbart` | Dịch text tiếng Việt từ Whisper sang tiếng Anh theo từng subtitle segment |

## Luồng xử lý

```mermaid
flowchart TD
    A[Video trong data/input/] --> B{Chọn video}
    B -->|Có --video| C[Xử lý một file]
    B -->|Không có --video| D[Xử lý toàn bộ video hợp lệ]
    C --> F{Nhận diện model}
    D --> F
    F -->|Có model.bin| G[faster-whisper<br/>suffix _faster]
    F -->|Có model.safetensors hoặc pytorch_model.bin| H[Hugging Face Transformers<br/>suffix _huggingface]
    G --> E{SRT tương ứng đã tồn tại?}
    H --> E
    E -->|Có và không bật --overwrite-srt| K[Tái sử dụng SRT]
    E -->|Chưa có hoặc cần ghi đè| Q{Engine đã nhận diện}
    Q -->|faster-whisper| J[Tạo subtitle segments]
    Q -->|Hugging Face Transformers| I[Trích xuất WAV mono 16 kHz tạm thời]
    I --> J
    J --> R{Có --translation-model?}
    R -->|Có| S[VinAI Translate dịch text vi2en]
    R -->|Không| L[Ghi data/subtitles/video-name_model.srt]
    S --> L
    K --> M{Có --skip-burn?}
    L --> M
    M -->|Có| N[Kết thúc sau khi tạo SRT]
    M -->|Không| O[Burn subtitle bằng FFmpeg]
    O --> P[Ghi data/output/video-name_ENG_SUB.mp4]
```

## Cấu trúc thư mục

```text
transcript-video/
├── src/                    # Toàn bộ source code
│   ├── __init__.py
│   ├── main.py             # CLI, transcription engines, TTS, orchestration
│   ├── project_config.py   # Path layout, constants, shared data models
│   └── subtitles.py        # SRT parsing, writing, hallucination filtering
├── tests/                  # Mã kiểm thử và test TTS local
├── data/                   # Runtime data, ignored by Git
│   ├── input/              # Video đầu vào
│   ├── audio/              # TTS WAV files and review chunks
│   ├── subtitles/          # File SRT
│   ├── output/             # Video đã burn hard subtitle
│   └── temp/               # Audio tạm thời
├── models/                 # Model local, không commit lên Git
│   ├── faster-whisper-large-v3/
│   ├── whisper-large-v3/
│   └── vinai-translate-vi2en-v2/
├── bin/                    # CUDA DLLs, ignored by Git
├── docs/                   # Notes and prompts
└── requirements.txt
```

Chạy từ root bằng `python -m src.main`; các DLL trong `bin/` được tự động thêm vào `PATH`.

Các định dạng video được hỗ trợ: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`.

Ví dụ tên artifact sau khi xử lý:

```text
data/input/crawl guide.mp4
data/subtitles/crawl guide_faster.srt
data/output/crawl guide_ENG_SUB.mp4
```

## Yêu cầu môi trường

- Windows với PowerShell.
- Python `3.12` (môi trường hiện tại sử dụng Python `3.12.10`).
- Model Whisper đã tải về máy.
- GPU NVIDIA và CUDA được khuyến nghị khi xử lý video dài.

Không cần cài FFmpeg thủ công: project sử dụng executable do `imageio-ffmpeg` cung cấp.

## Cài đặt

Tạo virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài dependency theo môi trường CUDA 12.4 đang được khai báo trong `requirements.txt`:

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

Nếu chỉ sử dụng CPU, hãy cài bản PyTorch phù hợp với CPU trước, sau đó chạy script với `--device cpu`. File `requirements.txt` hiện được pin theo môi trường GPU CUDA 12.4 của project. Model VinAI local dùng file `pytorch_model.bin`, vì vậy cần PyTorch `2.6.0` trở lên để Transformers load model an toàn.

## Chuẩn bị model

Đặt model local vào `models/`. Script nhận diện engine dựa trên file trong thư mục model:

```text
models/
├── faster-whisper-large-v3/
│   ├── config.json
│   ├── model.bin
│   ├── tokenizer.json
│   └── vocabulary.json
├── whisper-large-v3/
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── tokenizer.json
└── vinai-translate-vi2en-v2/
    ├── config.json
    ├── pytorch_model.bin
    └── sentencepiece.bpe.model
```

Model là artifact lớn và đã được ignore khỏi Git. Khi chạy, nên truyền `--model` rõ ràng vì giá trị mặc định trong script đang trỏ đến một đường dẫn local.

## Sử dụng nhanh

Đặt video vào `data/input/`, sau đó chạy model `faster-whisper`:

```powershell
\.\.venv\Scripts\python.exe -m src.main `
  --model .\models\faster-whisper-large-v3
```

Mặc định script sẽ:

- Quét tất cả video hợp lệ trong `data/input/`.
- Dùng `--task translate` để tạo phụ đề tiếng Anh.
- Dùng gợi ý ngôn ngữ nguồn `vi`.
- Chạy inference trên `cuda`.
- Tạo `data/subtitles/<video-name>_<model>.srt`.
- Tạo `data/output/<video-name>_ENG_SUB.mp4`.

### Một số lệnh thường dùng

Chỉ xử lý một video:

```powershell
.\.venv\Scripts\python.exe -m src.main `
  --video "crawl guide.mp4" `
  --model .\model\faster-whisper-large-v3
```

Giữ nguyên ngôn ngữ gốc thay vì dịch sang tiếng Anh:

```powershell
.\.venv\Scripts\python.exe -m src.main `
  --task transcribe `
  --model .\model\faster-whisper-large-v3
```

Chỉ sinh file SRT, không burn subtitle:

```powershell
.\.venv\Scripts\python.exe -m src.main `
  --skip-burn `
  --model .\model\faster-whisper-large-v3
```

Tạo lại SRT kể cả khi file đã tồn tại:

```powershell
.\.venv\Scripts\python.exe -m src.main `
  --overwrite-srt `
  --model .\model\faster-whisper-large-v3
```

Chạy Hugging Face Transformers engine trên CPU:

```powershell
.\.venv\Scripts\python.exe -m src.main `
  --device cpu `
  --model .\model\whisper-large-v3
```

Chép lời tiếng Việt bằng Whisper rồi dịch từng subtitle segment sang tiếng Anh bằng VinAI Translate:

```powershell
.\.venv\Scripts\python.exe -m src.main `
  --task translate `
  --model .\model\faster-whisper-large-v3 `
  --translation-model .\model\vinai-translate-vi2en-v2
```

## Tham số CLI

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--root` | `.` | Thư mục gốc chứa `data/` và `models/` |
| `--video` | Trống | Chỉ xử lý một file cụ thể bên trong `data/input/` |
| `--model` | Đường dẫn local trong script | Thư mục chứa model `faster-whisper` hoặc Hugging Face Whisper |
| `--translation-model` | Trống | Thư mục VinAI Translate vi2en tùy chọn; chỉ dùng cùng `--task translate` |
| `--task` | `translate` | `translate`: dịch sang tiếng Anh; `transcribe`: giữ nguyên ngôn ngữ |
| `--language` | `vi` | Gợi ý ngôn ngữ nguồn cho `faster-whisper`; truyền chuỗi rỗng để auto-detect |
| `--device` | `cuda` | Thiết bị inference: `cuda` hoặc `cpu` |
| `--compute-type` | `int8` | Kiểu tính toán cho `faster-whisper`, thường dùng `int8`, `float16`, `float32` |
| `--translation-batch-size` | `8` | Số subtitle segment được VinAI dịch cùng lúc |
| `--overwrite-srt` | Tắt | Tạo lại SRT nếu file cùng tên đã tồn tại |
| `--skip-burn` | Tắt | Chỉ tạo SRT, không xuất video hard subtitle |

Xem trợ giúp trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m src.main --help
```

## Cách các engine hoạt động

```mermaid
flowchart LR
    A[Thư mục model] --> B{Có model.bin?}
    B -->|Có| C[faster-whisper]
    B -->|Không| D{Có model.safetensors<br/>hoặc pytorch_model.bin?}
    D -->|Có| E[Hugging Face Transformers]
    D -->|Không| F[Lỗi: không nhận diện được model]
    C --> G[Transcribe trực tiếp từ video]
    E --> H[FFmpeg trích xuất WAV]
    H --> I[Đọc PCM thành NumPy float32]
    I --> J[ASR pipeline xử lý theo chunk 30 giây]
    G --> K[Subtitle segments]
    J --> K
    K --> L{Có --translation-model?}
    L -->|Có| M[VinAI Translate dịch text vi2en theo batch]
    L -->|Không| N[Ghi SRT]
    M --> N
```

Với Hugging Face engine, audio WAV mono `16 kHz` chỉ tồn tại tạm thời trong `data/temp/` và được xóa sau khi xử lý xong. Pipeline đọc WAV thành NumPy array trước khi đưa vào Transformers để tránh phụ thuộc vào cơ chế tìm FFmpeg nội bộ của thư viện.

VinAI Translate không nhận audio hoặc video. Khi dùng `--translation-model`, script yêu cầu Whisper chép lời tiếng Việt bằng task `transcribe`, giữ nguyên timestamp, sau đó dịch text theo batch và ghi SRT với suffix `_vinai`.

## Output

Mỗi video có thể tạo ra hai artifact:

| Artifact | Đường dẫn | Nội dung |
| --- | --- | --- |
| Subtitle mềm | `data/subtitles/<video-name>_<model>.srt` | Phụ đề theo chuẩn SRT, có thể chỉnh sửa trước khi burn |
| Video hard subtitle | `data/output/<video-name>_ENG_SUB.mp4` | Video MP4 đã gắn phụ đề trực tiếp vào hình ảnh |

Ví dụ SRT:

```srt
1
00:00:00,000 --> 00:00:07,000
Hello everyone, today I will show you how to crawl the score data...
```

## Lưu ý vận hành

- SRT dùng suffix `_faster.srt` với `faster-whisper` và `_huggingface.srt` với Hugging Face Transformers.
- Khi bật VinAI Translate, SRT có thêm suffix `_vinai`, ví dụ `_faster_vinai.srt`, để không tái sử dụng nhầm SRT được tạo bởi engine khác.
- Nếu SRT tương ứng với model đã tồn tại, script sẽ tái sử dụng file đó. Dùng `--overwrite-srt` khi muốn chạy transcription lại với model hoặc tham số khác.
- File SRT theo quy tắc cũ như `crawl guide.srt` sẽ không tự được tái sử dụng. Đổi tên file sang suffix tương ứng nếu muốn burn lại mà không chạy transcription.
- Có thể chỉnh sửa thủ công file trong `data/subtitles/`, sau đó chạy lại không kèm `--overwrite-srt` để burn bản subtitle đã sửa.
- Tên video đầu ra luôn có suffix `_ENG_SUB.mp4`, kể cả khi chạy `--task transcribe`.
- `--language` hiện chỉ được truyền vào `faster-whisper`. Hugging Face engine sử dụng `task` nhưng không nhận language hint từ CLI.
- Hugging Face engine tự chuyển sang CPU nếu yêu cầu `cuda` nhưng PyTorch không tìm thấy CUDA. Với `faster-whisper`, hãy truyền đúng `--device` phù hợp với máy.
- Audio của video đầu ra được copy nguyên bản bằng FFmpeg (`-c:a copy`).
- Nếu một video lỗi trong chế độ batch, script ghi log lỗi và tiếp tục xử lý video kế tiếp.

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân thường gặp | Cách xử lý |
| --- | --- | --- |
| `Không tìm thấy model folder` | Sai đường dẫn truyền vào `--model` | Kiểm tra thư mục model và dùng đường dẫn tương đối như `.\model\faster-whisper-large-v3` |
| `Không nhận diện được định dạng model` | Thiếu `model.bin`, `model.safetensors` hoặc `pytorch_model.bin` | Kiểm tra model đã tải đầy đủ |
| Model dịch văn bản phải truyền qua `--translation-model` | Đã truyền VinAI Translate vào `--model` | Dùng Whisper cho `--model` và VinAI cho `--translation-model` |
| `Không có video nào trong folder` | `data/input/` trống hoặc extension không được hỗ trợ | Thêm video hợp lệ vào `data/input/` |
| Lỗi CUDA | Máy không có GPU NVIDIA hoặc CUDA/PyTorch không tương thích | Chạy với `--device cpu` hoặc cài lại PyTorch đúng phiên bản CUDA |
| Burn subtitle lỗi | File SRT không tồn tại hoặc FFmpeg không đọc được subtitle | Chạy tạo SRT trước, kiểm tra encoding UTF-8 và đường dẫn file |
