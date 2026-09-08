# Báo cáo hoàn thiện QoL trên nhánh dev

Ngày kiểm tra: 2026-09-06. Giữ kiến trúc CLI/Wizard/Textual → application → processing/course.

## Fixed bugs

- Textual Edit không còn pop session: giữ bản gốc đến khi Save changes hợp lệ, giữ vị trí và number. Cancel/Back/validation thất bại không xóa dữ liệu. Trong lúc edit, chặn remove/reorder/review và nhắc save/cancel.
- Course JSON hỏng không còn được coi là draft mới. Launch báo lỗi có đường dẫn; file không bị ghi đè. File chưa tồn tại vẫn mở draft mới mà chưa ghi disk.
- Wizard và TUI dùng chung `save_course_config`; validate trước khi tạo thư mục/ghi file, ghi tạm cùng thư mục rồi atomic replace. Builder dùng cùng parser. Unknown fields ở root, render, TOC và session được giữ; session extras theo session khi sửa video hoặc đổi thứ tự.
- Bỏ force target không có tác dụng độc lập. Transcription mới làm mất hiệu lực TTS downstream, kể cả cache simple/full và toàn bộ chunks; không tái sử dụng giọng đọc của SRT cũ.
- Dry-run không dùng `<model-suffix>` giả. Kiểm tra input, extension, stem collision, thư mục/định dạng ASR và translation, settings, đường dẫn binary được cấu hình và xung đột output path; không load weights, chạy FFmpeg hoặc tạo artifact/log/event/config/output directory.
- Provenance chạy migration legacy device/compute_type trước khi đánh dấu nguồn, nên nguồn là `hardware.* = config/profile`.
- Inspect không bị chặn bởi model ASR/translation thiếu; artifact subtitle không xác định được hiển thị `unresolved`.
- Wrapper deprecation warning hiện trên stderr; global options được đặt đúng chỗ khi delegate. Lỗi option từ Typer hiện tại được bắt đúng loại và thoát mã 2.
- TUI không khởi chạy build sau khi save/validation thất bại. Worker build kết thúc hoặc lỗi sẽ mở lại các nút điều khiển.
- `--no-color` của TUI được áp dụng lúc Textual tạo bộ lọc monochrome; help CLI cũng xử lý no-color và redirected output.

## Completed QoL

- `EventKind.REUSED` cho subtitle, TTS audio và chunk cache; Rich reducer, plain output, JSON observer và Textual hiểu semantic này.
- `ProcessExecutionError` giữ command, returncode, stdout, stderr, tool, timeout và stage. `str()` ngắn; diagnostics đầy đủ vẫn trong object và DEBUG logs. Có context khi launch thất bại, timeout, FFmpeg/ffprobe failure và ffprobe trả JSON hỏng.
- Lỗi application cho biết stage/tool khi có dữ liệu, hướng đến detailed log và `-vv`. Chỉ gợi ý libx264 khi stderr xác nhận thiếu encoder h264_nvenc.
- Inspect dạng bảng: duration, resolution, FPS, video/audio codec, bitrate, sample rate, channels, size, path và trạng thái artifact. JSON vẫn giữ metadata gốc.
- Doctor tách configured encoder, NVENC listing/runtime và libx264 fallback; thêm config, PyTorch/CUDA build, GPU, ASR compute support, translation/TTS paths và cache. PASS/WARN/FAIL vẫn quyết định bằng required/optional.
- Config validate có bảng các nhóm kiểm tra, lỗi có tên field và JSON machine-readable; config sai exit 1.
- Wizard thêm session từ review, tái dùng selection/validation/duplicate protection/title generation/metadata cache và cấp number chưa dùng.
- Textual có video browser, metadata worker/cache, advanced settings, review durations và progress widgets thật.
- CI Windows/Linux dùng lightweight test dependencies, không cài CUDA local wheels. Thêm recipe `wizard`, `test-tui`, `coverage`.

## Deliberately removed/deferred behavior

- Bỏ `--force translation`: chưa có source transcription cache độc lập. Force transcription chạy lại cả ASR và translation được cấu hình.
- Bỏ `--force render`: rendering luôn chạy, không có render cache để invalidate.
- Không split dependency extras: giữ nguyên setup uv/PyTorch local wheels đang hoạt động. CI dùng editable `--no-deps` và danh sách dependency test riêng.
- Không tách hàng loạt cli.py/tui/app.py hoặc viết lại event/progress/cache architecture. Chỉ thêm presentation helper cho inspect và application error formatter.

## Architecture changes

- `course/config.py`: read/parse/save/document APIs, atomic persistence, validation cho draft settings chưa có session, bảo toàn unknown fields.
- `application/inspection.py`: media inspection dùng chung cho CLI/TUI; metadata cache theo resolved path + mtime_ns + size, bao gồm lỗi probe. FFprobe luôn chạy worker trong TUI.
- `TextualObserver`: callback về UI thread cập nhật widget độc lập với log throttling; không redirect Rich output vào Textual.
- Event JSON giữ schema_version 1; `reused` là kind bổ sung, không đổi các trường hiện hữu. Chunk reuse không hoàn tất parent TTS stage.
- Subprocess exception có structured context; DEBUG diagnostics và cleanup tiến trình khi bị ngắt vẫn được giữ.

## CLI behavior

| Option | Behavior |
| --- | --- |
| `--force transcription` | Rebuild ASR, configured translation và enabled downstream TTS |
| `--force transcription` | Regenerate riêng Vietnamese source SRT |
| TTS bình thường | Regenerate TTS và toàn bộ chunks; không dùng artifact cache |
| `--rerun-tts-chunk` | Rerun chunk chọn lọc theo yêu cầu rõ ràng |
| `--dry-run` | Validate/print plan; không ghi, kể cả `--save-config` và `--events-json` |
| `-q` | Warning/error và kết quả; không progress |
| `-v` | Thêm INFO diagnostics, không traceback |
| `-vv` | DEBUG diagnostics và failure traceback, không locals |
| `--plain` | Status theo dòng, không live redraw |
| `--no-color`, `NO_COLOR` | Tắt màu non-fullscreen; TUI dùng monochrome |
| `--events-json PATH` | Semantic JSONL vào file mới; không ghi đè destination cũ |

Global options đặt trước command. `--force` có thể lặp lại; full TTS force không kết hợp selective chunk rerun. Multi-video order, dedupe, stem protection và continue-on-video-failure vẫn được giữ.

## Wizard behavior

Checkbox selection → review sessions → Recommended/Custom appearance/output → final review → save. Review có Continue, Add session, Edit title/number, Move up/down, Remove, Cancel. Final review cho quay lại sửa settings/sessions; remove và overwrite vẫn xác nhận. Không ghi config trước quyết định tạo.

## TUI behavior

- **Metadata/settings:** title, output, theme, card duration, chapters, TOC enabled/heading/items/page duration; advanced width/height/FPS/encoder/bitrates/audio sample rate/font.
- **Sessions:** browser cạnh danh sách course, chỉ media trong `data/input` và `data/output`, không quét đệ quy. Enter chọn video để điền form; Add xác nhận. Có manual path và chặn video trùng.
- **Editing:** giữ item gốc đến Save changes; Cancel edit giữ nguyên dữ liệu và thứ tự. Số session không tự renumber khi save.
- **Metadata:** duration/resolution/FPS/codecs/bitrates/audio/size/artifacts; worker + cache, lỗi hiển thị unavailable.
- **Review/build:** session paths/durations, tổng source, ước tính output gồm cards/TOC; Back/Save/Build. Overall đếm stage hoàn tất, stage progress dùng current/total thực, FFmpeg dùng seconds, unknown total dùng busy; có session count và log.
- **Bindings:** Ctrl+S save, Esc back, Ctrl+E cancel edit, A/E/Delete add/edit/remove, U/D reorder, Ctrl+B build tại review, Q quit. Phím chữ áp dụng ngoài text input. Unsaved changes có confirm quit. Trong build, khóa save/navigation/quit đến khi worker xong.

## Tests executed

| Command | Result |
| --- | --- |
| `uv run pytest -m "not slow and not gpu"` trước thay đổi | 133 passed |
| Phase 1: cùng command | 157 passed |
| Phase 2: cùng command | 169 passed |
| Phase 3: cùng command | 176 passed |
| `uv run pytest` sau các regression bổ sung | 190 passed |
| `uv run ruff format .` | Hoàn tất |
| `uv run ruff check .` | All checks passed |
| `.venv/qol-ci/Scripts/python.exe -m pytest -m "not gpu and not slow and not integration"` | 185 passed, 5 deselected |
| `git diff --check` | Không có whitespace errors |

Môi trường test nhẹ được cài thật bằng `uv venv --python 3.14 .venv/qol-ci`, `uv pip install --python .venv/qol-ci/Scripts/python.exe -r .github/requirements-test.txt`, rồi editable `--no-deps`. Không cần Torch/Qwen/ASR runtime trong môi trường này.

CLI smoke `uv run transcript-video --help`, `process --help`, `config --help`, `course --help`, `config validate` đều exit 0. `doctor`, `--no-color doctor` và `--no-color --json doctor` exit 1 đúng với ba prerequisite thiếu: ffprobe, ASR model, TTS model.

Suite đầy đủ có real FFmpeg tests cho pitch-preserving speedup và mux không cắt audio dài hơn video, cùng Questionary keyboard flow và Textual Pilot. Doctor xác nhận CUDA, NVENC runtime và libx264 hoạt động trên máy này.

## Tests not executed

- Không chạy Whisper/VinAI/Qwen inference thật: workspace không có thư mục model.
- Không chạy inspect với ffprobe thật: ffprobe không có trên PATH/cạnh FFmpeg; workspace cũng không có input video sẵn. Metadata/inspect được test bằng mocks.
- Không chạy GitHub-hosted workflow hoặc Linux runner từ phiên làm việc này. Chiến lược cài dependency/suite CI đã được xác minh trên Windows local.

## Remaining limitations

- Không có độc lập force translation; phải rebuild ASR + translation cùng nhau.
- Browser chỉ scan một cấp trong hai project media directories; path khác dùng manual entry.
- Overall build progress đếm stages, không phải ước lượng phần trăm encode theo thời gian. Duration estimate là độ dài video, không phải thời gian chờ build.
- TUI không thêm cơ chế hủy build đang chạy; navigation/quit bị chặn đến khi worker hoàn tất. CLI Ctrl+C vẫn cleanup subprocess.
- Textual monochrome vẫn dùng grayscale và fullscreen cursor control, không phải output thuần text.
- Directory usability check là lightweight, không tạo file thử để kiểm tra mọi Windows ACL/network-share condition.

## Files changed

- Core/domain: `course/config.py`, `course/builder.py`, `processing/pipeline.py`, `processing/tts/chunks.py`, `events.py`, `process_runner.py`.
- Application: `application/settings.py`, `processing.py`, `inspection.py`, `diagnostics.py`, `errors.py`.
- Presentation: `cli.py`, course wrappers/wizard, `tui/app.py`, `ui/console.py`, `progress.py`, `theme.py`, `inspection.py`.
- Tests: mới `tests/test_qol.py`, `tests/test_tui.py`; mở rộng các suite dx/observability/wizard/tts hiện hữu.
- Engineering/docs: `.github/workflows/quality.yml`, `.github/requirements-test.txt`, `justfile`, README Anh/Vi và `docs/observability.md`.

Các phần đã đáp ứng từ trước được giữ: semantic events, Rich progress, FFmpeg pipe progress/ETA, batch processing, per-run/rotating logs, TTS review/report/chunk cache/selective rerun, sentence boundaries/timing shifts và NVENC fallback. Không đổi timestamp subtitle, không truncate speech, không thêm streaming TTS.
