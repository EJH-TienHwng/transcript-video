# Todo lists

## Làm bây giờ

### Về phần TTS

- [ ] giọng của con AI đã tạm ổn, nhưng hiện tại có một số chỗ nó cắt vội hoặc là cắt sớm, nên có một vài chỗ bị mất đi âm tiết cuối cùng, hoặc là âm tiêt cuối cùng bị dính qua câu sau, ví dụ: câu 1: then we run this testcases. câu 2: Here is the result thì âm 'es' của câu 1 bị lẹm qua đầu câu 2.

### Build course

- [ ] phần TOC, các chữ đại diện cho video tụi nó cách nhau ra hơi nhiều cho các course có cỡ 2-3 vid, bây giờ hãy sửa code để chúng gần nhau hơn 1 tí
- [ ] Wizard ổn rồi, nhưng nếu có thể làm cho nó đẹp hơn nữa thì tốt
- [ ] file report nên để trong folder khác, thay vì ném nó vào folder audio, ví dụ để trong folder data/report, nói chung là tùy bạn nghiên cứu và thấy cấu trúc nào hợp lý thì cứ làm

### Build Video

- [ ] làm sao để chạy với 1 list video được chỉ định, thay vì phải gõ từng cái một
- [ ] file jsonl phải nằm trong 1 folder nào đó, chứ không phải quăng vô folder audio, tôi muốn file này phải được pretty print để tôi dễ nhìn và dễ đọc

### Misc

- [ ] sửa lại doctor cho nó cập nhật theo version python "hiện tại" ?

## Cần phải sửa

- [ ] tôi muốn có cách nào đó để truyền tham số vào càng ít càng tốt, kiểu chỉ chạy bằng 1 ít lệnh thôi, thay vì vẫn phải truyền khá nhiều tham số như hiện tại

- [x] log lại những phần tts bị quá dài, tràn ra khỏi constrain cho phép lại thành 1 file, để tôi sửa lại file srt thủ công

- [x] giúp tôi sửa lại phần ảnh cho table of content, dùng file table_of_content.png

- [x] tôi muốn cái TOC thì chữ đen, tại nền của tôi là nền trắng, còn về cái section thì chữ trắng tại vì phần ảnh nền của nó là nền nhiều màu

- [ ] bất kì cái nào dùng tới speed up cũng nên log lại
- [ ] âm thanh bị lỗi rồi, xét về giọng nó đã bớt nhấn nhá hơn, nhưng có gì đó đã làm cho nó bị lỗi
- [ ] TTS có đọc đủ câu không
- [ ] có tự thêm/bớt từ không
- [ ] phát âm có đúng không
- [ ] giọng giữa các câu có nhất quán không
- [ ] có lúc đột nhiên cao giọng/thấp giọng không
- [ ] tốc độ giữa các câu có thay đổi bất thường không
- [ ] câu dài có bị ép tốc độ quá mức không
- [ ] câu có bắt đầu đúng timestamp subtitle không
- [ ] có đè câu lên nhau không
- [ ] khoảng lặng có tự nhiên không
- [ ] đầu/cuối câu có bị cắt không
- [ ] audio cuối có cùng duration với video không.

- [ ] tạo profile srt để tạo nhanh hơn, tránh lặp lại
- [ ] cài SoX, flash attn 2

- [ ] TUI quá xấu

- [ ] TTS thì xét về chất lượng giọng đọc nó đã đỡ hơn, nhưng nhìn chung thì còn lỗi âm thanh, có rất nhiều câu âm thanh nó chỉ nói tới chữ gần cuối rồi dừng, nó mất đi 1-2 từ ở cuối rồi, rồi log tts có gì đó là lạ

## Từ từ rồi sửa

cập nhật lại doctor, làm sao để ít nhất thì nó có thể đồng bộ được với phiên bản python được ghi trong file .python-version hay sao đấy, vì hiện tại phiên bản python đang nằm ở 3 nơi, nên nếu sau này có update thì nhiều khi sẽ gặp lại lỗi doctor báo fail như hôm nay

debug từng cái option của subtitle style xem cái nào gây ra lỗi

```text
subtitle_style = (
    "FontName=Arial,"
    "FontSize=36,"
    "PrimaryColour=&H00FFFFFF,"
    "OutlineColour=&H00000000,"
    "BorderStyle=1,"
    "Outline=2,"
    "Shadow=0,"
    "Alignment=2,"
    "MarginV=40"
)
```

cân nhắc thay đổi lại cấu hình của NVENC cái config của chữ
