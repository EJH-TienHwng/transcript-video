# TODO — transcript-video (`dev`)

> Đối chiếu với nhánh `dev` tại commit `fabdc4cc2d163d2221ff456d513b958ef432052f`.
> Cập nhật working tree ngày 2026-09-07: review v4, provenance v1 và các phase implementation; xem `quality-validation.md`. Các mục acceptance audio/model thật vẫn giữ chưa hoàn thành.
>
> Quy ước:
> - `[x]`: đã có implementation rõ ràng trong code và/hoặc regression test.
> - `[ ]`: chưa hoàn thành hoặc mới chỉ làm một phần.
> - Các mục QA thực tế vẫn để `[ ]` nếu code đã có cơ chế bảo vệ nhưng chưa được kiểm chứng đầy đủ bằng Qwen/Whisper và video thật.

---

## 1. Đã hoàn thành

### TTS — timing, boundary và review

- [x] Sửa cơ chế TTS để không cắt waveform chỉ vì câu dài hơn slot subtitle. `fit_wav_to_available_duration()` chỉ speed-up trong giới hạn và giữ nguyên speech nếu không thể fit.
- [x] Thêm bounded pitch-preserving speed-up bằng FFmpeg `atempo`; mặc định giới hạn bởi `tts.max_speedup = 1.25`.
- [x] Bảo vệ phần đầu/cuối câu khi cắt audio từ context bằng alignment padding.
- [x] Khi alignment không đủ an toàn, regenerate riêng sentence thay vì cắt theo tỷ lệ.
- [x] Có retry/fallback cho câu bị thiếu từ; giữ candidate có coverage tốt hơn.
- [x] Sửa trường hợp âm cuối/fricative của câu trước bị lẹm sang câu sau; đã có regression test gần đúng ví dụ `testcases` → `Here`.
- [x] Không cho hai câu TTS đè lên nhau khi assemble audio; câu sau được shift và giữ `MIN_GAP_SECONDS`.
- [x] Log `timing_shift` và flag khi shift vượt threshold.
- [x] Log câu bị overflow / cần speed-up vượt giới hạn vào TTS review.
- [x] TTS review có machine-readable JSONL và file `.pretty.json` để đọc thủ công.
- [x] TTS review được chuyển khỏi `data/audio` sang `data/report/tts`.
- [x] Chunk review cũng nằm dưới `data/report/tts/...`, không còn rải JSON sidecar trong audio folder.
- [x] Không publish partial timed/chunked TTS như thành công nếu có sentence generation failure.
- [x] Cache chunk cũ thiếu metadata an toàn sẽ bị invalidate/regenerate.
- [x] Mux không dùng cách cắt TTS tại cuối video; speech dài hơn source video được giữ lại.
- [x] Có regression tests cho tail preservation, unsafe alignment fallback, overlap prevention, chunk rebuild, speed-up và mux không cắt audio.

### Build video / CLI

- [x] `process` nhận nhiều video positional trong một lệnh.
- [x] Giữ đúng thứ tự video được truyền vào.
- [x] Deduplicate input bị lặp.
- [x] Phát hiện hai input khác nhau có cùng stem trước khi chúng ghi đè output/cache của nhau.
- [x] Một video fail không làm dừng toàn bộ batch; các video sau vẫn tiếp tục.
- [x] Có `--profile` và cơ chế merge profile/config/CLI override.
- [x] Có `--dry-run`, `--plain`, semantic events, JSON event output và structured diagnostics.

### Course builder / Wizard / TUI

- [x] TOC dùng `assets/table_of_content.png`.
- [x] TOC dùng chữ đen trên background trắng; session card dùng chữ trắng.
- [x] TOC cho course chỉ có 1–3 session đã được compact/center lại, không còn giãn hàng quá xa như trước.
- [x] Report không còn bị ném vào audio folder.
- [x] Wizard hỗ trợ chọn nhiều video bằng checkbox.
- [x] Wizard có review, add/edit/remove/reorder session, metadata preview và estimate duration.
- [x] Wizard/TUI dùng chung course config validation và atomic save.
- [x] TUI có video browser, metadata worker/cache, advanced settings, review duration và progress widgets.
- [x] `--no-color` đã được xử lý cho CLI/TUI.

### Doctor / môi trường

- [x] Doctor không còn hard-code Python version; đọc expected version từ `.python-version`.
- [x] Doctor kiểm tra FFmpeg/ffprobe, encoder, CUDA/PyTorch, ASR compute support, model paths và writable directories.
- [x] Doctor có kiểm tra `data/report`.

### Các task cũ không còn phù hợp

- [x] Không thêm SoX chỉ để speed-up TTS. Pipeline hiện đã dùng FFmpeg `atempo`, không cần thêm một dependency native chỉ để thực hiện chức năng này.

---

## 2. P0 — Làm ngay: TTS quality / acceptance

> Code đã có nhiều guard tốt, nhưng đây vẫn là phần cần kiểm chứng bằng model và media thật. Không nên đánh đồng regression test giả lập với chất lượng audio thực tế.

### 2.1 End-to-end QA bằng audio thật

- [ ] Chạy một bộ video/SRT đại diện bằng Qwen TTS thật và Whisper verifier thật để xác nhận fix boundary hoạt động ngoài thực tế.
- [ ] Xác nhận TTS đọc đủ câu, đặc biệt từ cuối và âm tiết cuối.
- [ ] Xác nhận TTS không tự thêm hoặc bỏ từ.
- [ ] Xác nhận không còn trường hợp âm cuối của câu trước dính sang đầu câu sau.
- [ ] Xác nhận phát âm các technical terms đủ chính xác.
- [ ] Xác nhận giọng giữa các câu đủ nhất quán.
- [ ] Xác nhận không có câu đột nhiên cao giọng/thấp giọng bất thường.
- [ ] Xác nhận tốc độ nói giữa các câu không thay đổi bất thường.
- [ ] Xác nhận khoảng lặng giữa các câu nghe tự nhiên.
- [ ] Xác nhận audio không bị cắt đột ngột ở đầu/cuối sentence.

### 2.2 Tự động hóa QA TTS

- [x] Thêm optional final-audio verifier để so `expected subtitle text` với transcript nhận dạng lại từ audio đã generate/fit.
- [x] Trong report, thêm các metric tối thiểu: word coverage, missing words, added words, first-word match, last-word match, WER/CER hoặc metric tương đương.
- [x] Flag riêng `missing_final_word` / `possible_truncated_tail` để ưu tiên lỗi mất từ cuối.
- [x] Thêm acoustic tail check đơn giản để phát hiện waveform kết thúc khi energy vẫn còn cao.
- [x] Ghi duration/timing QA: `original_start`, `actual_start`, `timing_shift`, `raw_duration`, `final_audio_duration`, `available_duration`, `overflow_duration`.
- [x] Ghi `source_video_duration`, `tts_audio_duration`, `final_output_duration` và `duration_delta` vào summary/report để biết chính xác audio cuối có lệch duration video hay không.
- [x] Không tự tuyên bố “pronunciation/prosody pass” chỉ dựa trên WER; những chỉ số không đo đáng tin cậy phải được flag cho manual review.

### 2.3 Speed-up observability

- [x] Bất kỳ sentence nào thực sự dùng speed-up (`applied_speedup > 1`) đều phải tạo semantic REVIEW/event rõ ràng, không chỉ các câu overflow.
- [x] TTS quality summary phải có thêm số lượng `speed_adjusted`.
- [x] Report phải cho phép lọc nhanh các câu speed-up cao nhất.

---

## 3. P0 — Cache correctness / artifact safety

> Đây là task bổ sung sau code review vì cache hiện chủ yếu dựa vào artifact tồn tại + subtitle text/timestamp, chưa đủ provenance.

### 3.1 Cache provenance / fingerprint

- [x] Thêm versioned provenance/fingerprint cho subtitle cache.
- [x] Fingerprint subtitle phải phản ánh ít nhất input video identity, ASR model/type, translation model, task, language và các setting có ảnh hưởng tới kết quả.
- [x] Thêm versioned provenance/fingerprint cho TTS full audio và từng chunk.
- [x] Fingerprint TTS phải phản ánh ít nhất subtitle content/timestamps, TTS model, speaker, language, instruct, generation mode, timing/context settings và các setting có ảnh hưởng tới waveform.
- [x] Khi fingerprint không match, artifact phải được regenerate thay vì silently reuse.
- [x] Event/log phải giải thích ngắn gọn vì sao cache `REUSED` hoặc `INVALIDATED`.

### 3.2 Atomic media outputs

- [x] Các artifact quan trọng như SRT/WAV/MP4 nên được ghi qua temporary/partial path rồi atomic replace sau khi thành công.
- [x] Run bị Ctrl+C/FFmpeg crash/model crash không được để lại final-path artifact có thể bị hiểu nhầm là cache hợp lệ.
- [x] Cleanup partial files an toàn ở lần chạy tiếp theo. Cleanup bằng `finally`; orphan do hard-kill có tên riêng và bị bỏ qua khi discovery/cache, không tự xóa file có thể thuộc writer khác.

---

## 4. P1 — Hiệu năng batch

- [x] Tránh load lại Whisper model cho từng video trong cùng một batch.
- [x] Tránh load lại translation model cho từng video nếu configuration không đổi.
- [x] Tránh load lại Qwen model/aligner cho từng video nếu có thể reuse an toàn.
- [x] Thiết kế lifecycle model để không giữ nhiều model GPU lớn đồng thời gây OOM.
- [x] Có unit test/mock chứng minh số lần load model giảm khi chạy nhiều video.
- [x] Event/progress vẫn phải biểu diễn đúng từng video/stage sau khi model lifecycle được tối ưu.

---

## 5. P1 — Giảm số tham số phải truyền / profile

### Đã có nền tảng

- [x] CLI đã hỗ trợ `--profile`.
- [x] Profile được merge với base config và CLI override theo precedence rõ ràng.

### Còn thiếu

- [x] Tạo `configs/profiles/` với ít nhất một profile SRT-only để tạo subtitle nhanh mà không phải lặp option.
- [x] Tạo một profile TTS review/quality phù hợp cho workflow kiểm tra giọng.
- [x] Cân nhắc thêm preset `fast` / `balanced` / `quality` nếu chúng có khác biệt cấu hình thực sự và được document rõ. Đã cân nhắc: giữ hai profile có workflow khác nhau; chưa có benchmark để tạo thêm preset tốc độ/chất lượng.
- [x] Common workflow phải chạy được bằng lệnh ngắn, ví dụ `transcript-video process VIDEO --profile srt`.
- [x] Không xóa các advanced CLI flags hiện tại; profile chỉ là shortcut/default layer.
- [x] README phải có ví dụ ngắn cho SRT-only, full processing, TTS review và batch.

---

## 6. P1 — Python version / Doctor consistency

- [x] Doctor đọc `.python-version`.

- [x] Giải quyết phần còn lại của việc Python version đang xuất hiện ở nhiều nơi:
  - `.python-version`
  - `pyproject.toml -> requires-python`
  - `.github/workflows/quality.yml -> python-version`
- [x] Tạo một consistency check/test/script để CI fail với thông báo rõ nếu ba nơi lệch nhau.
- [x] Không hard-code thêm một Python version thứ tư trong source code.

---

## 7. P2 — Subtitle rendering

- [x] Dọn `_subtitle_style` hiện đang gần như dead code trong `processing/media.py`.
- [x] Không tiếp tục “debug từng option” bằng hard-code thủ công; chuyển subtitle style thành cấu hình có validation hoặc một cấu trúc rõ ràng.
- [x] Xác định chính xác các field cần hỗ trợ: font, size, primary/outline color, border, outline, shadow, alignment, margins.
- [x] Tạo hàm build `force_style` và unit tests cho escaping/serialization.
- [x] Giữ default hiện tại ổn định nếu người dùng không cấu hình style.
- [x] Với NVENC + subtitle burn, chỉ thay encoder settings khi có lỗi/chất lượng được reproduce và benchmark; không thay đổi ngẫu nhiên chỉ vì “cân nhắc”.

---

## 8. P2 — FlashAttention 2 / tối ưu runtime

- [x] Nghiên cứu compatibility thực tế của FlashAttention 2 với Python 3.14, Torch/CUDA local wheels và Qwen version đang pin.
- [ ] Nếu hỗ trợ ổn định, cho phép dùng `attn_implementation = "flash_attention_2"` như optimization tùy chọn. Option và probe/fallback đã có test; xác nhận support thực tế còn chờ GPU tương thích và model.
- [x] Không biến FlashAttention 2 thành dependency bắt buộc.
- [x] Nếu không cài được hoặc runtime không hỗ trợ, fallback rõ ràng về `sdpa` và Doctor phải báo WARN thay vì làm tool chết.
- [ ] Benchmark trước/sau trên ít nhất một workload TTS thực tế trước khi coi task này là hoàn thành.

---

## 9. P3 — UI polish

### Wizard

- [x] Polish Wizard thêm nếu không làm tăng số bước hoặc làm workflow chậm hơn.
- [x] Giữ flow hiện tại: Sessions → Appearance → Output → Review.
- [x] Ưu tiên spacing, hierarchy, copy, summary và màu sắc; không rewrite logic đang ổn.

### TUI

- [x] Cải thiện visual hierarchy của TUI.
- [x] Làm form/session list/review/progress dễ đọc hơn trên terminal 80 và 120 cột.
- [x] Giữ keyboard bindings và behavior hiện tại.
- [x] Không hy sinh `--no-color`, accessibility hoặc Windows terminal compatibility chỉ để làm đẹp.

---

## 10. Acceptance checklist trước khi đóng các task TTS

- [ ] Chạy test suite fast trên Windows/Linux. Windows: 278 test pass; chưa chạy Linux, giữ CI matrix hiện tại.
- [x] Chạy integration tests FFmpeg.
- [ ] Chạy real Qwen TTS trên ít nhất một video ngắn và một video dài.
- [ ] Nghe thủ công toàn bộ các sentence bị QA report flag.
- [ ] Xác minh các sentence có `applied_speedup > 1`.
- [ ] Xác minh các sentence có `timing_shift > 0`.
- [ ] Xác minh các sentence có `overflow_duration > 0`.
- [ ] Xác minh final TTS WAV và final muxed MP4 bằng ffprobe.
- [x] Không có partial artifact bị reuse sau một run bị interrupt.
- [x] Chạy `ruff check`, `ruff format --check`, `pytest` và `uv build`. Lượt đầy đủ: 278 passed, lint/format sạch, wheel và sdist build thành công.

---

## 11. Thứ tự thực hiện khuyến nghị

1. **TTS automatic QA + real acceptance**
2. **Cache provenance/fingerprint**
3. **Atomic output**
4. **Batch model reuse**
5. **Speed-up semantic logging**
6. **Profiles / giảm số tham số**
7. **Python version consistency**
8. **Subtitle style cleanup/config**
9. **FlashAttention 2 nếu benchmark chứng minh có lợi**
10. **Wizard/TUI polish**

---

## 12. Ghi chú về các task cũ đã được gộp

Các task cũ sau không còn giữ thành nhiều checkbox trùng nhau:

- “âm thanh bị lỗi”
- “TTS có đọc đủ câu không”
- “đầu/cuối câu có bị cắt không”
- “TTS mất 1–2 từ cuối”
- “âm cuối câu trước dính sang câu sau”

Chúng được gom vào **P0 — TTS quality / acceptance**, trong đó phần boundary handling đã có implementation + regression tests, còn chất lượng thực tế vẫn phải được verify bằng model/media thật.

Tương tự:

- “Doctor cập nhật theo Python hiện tại” đã hoàn thành ở mức Doctor đọc `.python-version`; phần còn thiếu là consistency giữa `.python-version`, `pyproject.toml` và CI.
- “Tạo profile SRT” chưa hoàn thành: engine `--profile` đã có nhưng repo hiện chưa có `configs/profiles/` với preset SRT cụ thể.
- “Cài SoX” được đóng vì không còn cần cho pipeline speed-up hiện tại.
