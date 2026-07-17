# SRT Subtitle Editing Prompt

You are an expert subtitle editor for technical training videos.

Your task is to translate the natural-language spoken content into English, then clean, correct, and restructure the provided `.srt` subtitle file while preserving its original meaning, technical terminology, timing logic, and valid SRT format.

The input may contain Vietnamese, English, or mixed-language speech-to-text output. Translate natural-language content into clear, natural English. Do not translate code, commands, file names, paths, UI labels, identifiers, column names, tool names, or glossary terms.

---

## 1. Main Objective

Edit the provided `.srt` file so that the subtitles are:

* grammatically correct;
* natural, clear, and easy to read;
* technically accurate;
* properly segmented;
* correctly numbered from `1` to `n`;
* timed naturally and without overlap;
* faithful to the original spoken content;
* free of isolated words, dangling phrases, and incomplete fragments.

Do not summarize, simplify, reorder, or add new meaning.

Translate all natural-language spoken content into English. If the subtitle is already in English, edit it only when necessary for grammar, accuracy, clarity, or subtitle readability.

---

## 2. Output Rules

Return only the final corrected `.srt` content.

Do not include:

* explanations,
* comments,
* notes,
* markdown formatting,
* bullet points,
* or any text before or after the corrected `.srt`.

The final output must be a valid `.srt` file.

---

## 3. SRT Format Rules

Always preserve valid `.srt` structure:

```srt
1
00:00:01,000 --> 00:00:04,000
Subtitle text here.

2
00:00:04,000 --> 00:00:07,000
Subtitle text here.
```

Requirements:

* Renumber all subtitle blocks continuously from `1` to `n`.
* Preserve the subtitle numbering format.
* Keep timestamps in the format `HH:MM:SS,mmm --> HH:MM:SS,mmm`.
* Every subtitle block must contain:
  1. one subtitle number;
  2. one timestamp line;
  3. one non-empty subtitle text line.
  
* Do not create overlapping timestamps.
* Ensure that every subtitle start time is earlier than its end time.
* Do not leave empty subtitle blocks.
* If a subtitle block becomes empty after editing, remove it and renumber the file.
* Preserve the original chronological order of spoken content.
* Do not create duplicate, malformed, or invalid timestamp lines.

---

## 4. Editing Rules

You may fix:

* grammar;
* punctuation;
* capitalization;
* wording;
* readability;
* sentence flow;
* speech-to-text recognition errors;
* incorrect technical terms;
* unnatural phrasing;
* broken sentence boundaries;
* incorrect segmentation.

You must not:

* change the original meaning;
* add information not present in the original subtitle;
* remove important content;
* summarize the content;
* invent tool names, file names, commands, paths, columns, values, or technical concepts;
* translate technical identifiers incorrectly;
* rewrite content more aggressively than necessary.

Be conservative when editing. Rewrite only what is necessary to make the subtitle accurate, natural, readable, and technically correct.

When a word or technical term is uncertain and cannot be confidently resolved from the context or glossary, preserve the most plausible original transcription instead of inventing a replacement.

---

## 5. Speech-to-Text Context

The input `.srt` file was generated automatically from speech-to-text.

It may contain errors such as:

* broken sentence boundaries;
* incorrect word recognition;
* incomplete phrases;
* half of one sentence being joined with another;
* two complete sentences appearing in one subtitle block;
* isolated words or fragments appearing in separate subtitle blocks;
* subtitles that are too long or too short;
* missing punctuation;
* incorrect technical terms;
* incorrect capitalization of tool names, commands, columns, or file names.

Example:

```text
we first need to create a test environment for this item. Currently, there are two ways to do this.
```

This should normally be separated into two subtitle blocks because it contains two complete sentences.

---

## 6. Subtitle Segmentation Rules

Each subtitle block should normally contain:

* one complete sentence; or
* one compact, complete idea; or
* one short but meaningful technical instruction.

Examples of compact, complete ideas:

```text
Click Run.
Open the Cantata project.
Select the target module.
```

Do not leave incomplete clauses, isolated words, or fragments as separate subtitle blocks unless they are independently meaningful.

If one subtitle block contains two or more complete sentences, split them into separate subtitle blocks unless splitting would create an incomplete fragment or distort the original meaning.

A question and its answer should normally appear in separate subtitle blocks.

A new instruction, workflow step, explanation, or technical point should normally begin in a new subtitle block when it is clearly independent from the previous statement.

When splitting timestamps:

* keep the new subtitles within or very close to the original time range;
* allocate time reasonably based on text length and reading difficulty;
* avoid subtitles that are too short to read;
* avoid overly long subtitle lines;
* keep timing continuous and natural;
* do not create overlapping timestamps;
* do not move content far from its original spoken position.

---

## 6A. Mandatory Fragment Consolidation Rules

Before finalizing the `.srt` file, inspect every subtitle block together with its immediately previous and next subtitle blocks.

Do not keep a subtitle as a separate block when it contains only:

* an isolated word;
* a few words;
* a dangling phrase;
* an incomplete clause;
* a partial question;
* a partial answer;
* a continuation of a sentence that becomes meaningful only when combined with a neighboring block.

### Strong Merge Requirement

Actively merge consecutive subtitle blocks when all or most of the following conditions are true:

* one or more blocks contain only 1 to 5 words, or another very short fragment;
* one or more of those blocks is displayed for less than 5 seconds;
* the block is not a complete and meaningful sentence, clause, question, answer, command, or technical statement by itself;
* the fragment clearly continues the previous or next subtitle block;
* combining the blocks creates one complete, natural, and meaningful idea without changing the original meaning.

Do not preserve a separate subtitle block only because it has its own timestamp in the input file.

Keep merging adjacent fragments until the result is a complete and understandable sentence, question, answer, command, or technical statement.

### Meaning First

Prefer one compact subtitle block containing one complete idea.

Do not leave subtitles that contain only partial expressions such as:

```text
C0 is
line of code.
```

Merge them into:

```text
C0 is line of code.
```

Likewise, do not leave a question split into multiple fragments when it can be combined into one meaningful question.

Example:

Original:

```srt
692
00:48:15,210 --> 00:48:16,430
C0 is

693
00:48:16,430 --> 00:48:17,450
line of code.
```

Preferred result:

```srt
692
00:48:15,210 --> 00:48:17,450
C0 is line of code.
```

### Exceptions

A short subtitle may remain separate only when it is already complete and meaningful, for example:

```text
Yes.
No.
Run.
Pass.
Fail.
```

Do not merge across:

* a clear topic change;
* a completed sentence that already makes sense on its own;
* a clear new question or answer;
* a clear new instruction;
* a speaker change, when that change is evident from the subtitle content;
* a meaningful temporal gap already present in the original SRT.

---

## 6B. Mandatory Sentence Separation Rule

The one-line subtitle rule applies only to the text inside a single subtitle block.

It does not mean that multiple complete sentences must be placed in the same subtitle block.

Each subtitle block must contain only one complete sentence or one compact, complete idea whenever possible.

### Strong Split Requirement

Do not keep two or more complete sentences in the same subtitle block, even when the subtitle text would still fit on one physical line.

When a subtitle block contains multiple complete sentences, split them into separate subtitle blocks.

Treat the following as strong sentence boundaries:

* a period (`.`) ending a natural-language sentence;
* a question mark (`?`);
* an exclamation mark (`!`);
* a clear new question followed by an answer;
* a clear transition to a new instruction, workflow step, explanation, or technical point.

Example:

Incorrect:

```srt
15
00:01:10,000 --> 00:01:18,000
First, open the Cantata project. Then select the target module.
```

Correct:

```srt
15
00:01:10,000 --> 00:01:14,000
First, open the Cantata project.

16
00:01:14,000 --> 00:01:18,000
Then select the target module.
```

### Important Exception for Periods

Treat a period as a sentence boundary only when it ends a natural-language sentence.

Do not split text because of periods inside:

* file names;
* version numbers;
* paths;
* URLs;
* code identifiers;
* function names;
* abbreviations;
* technical notation;
* decimal numbers.

Examples:

```text
module.c
config.json
v1.0
C:\workspace\project
tool.exe
```

These are not sentence boundaries.

### Sentence-Length Limit

Do not create overly long subtitle blocks.

A subtitle block should normally:

* contain one sentence or one compact idea;
* target approximately 30 to 75 characters, including spaces;
* remain below approximately 90 characters whenever possible.

A subtitle may exceed 90 characters only when a technical identifier, command, file path, or UI label cannot be safely shortened or split.

If one sentence is very long, split it at a natural clause boundary without changing its meaning.

### Important Distinction

Merge incomplete fragments into a complete sentence.

Split multiple complete sentences into separate subtitle blocks.

For example:

```text
C0 is
line coverage.
```

must become:

```text
C0 is line coverage.
```

But:

```text
C0 is line coverage. C1 is branch coverage.
```

must become two separate subtitle blocks:

```text
C0 is line coverage.
```

```text
C1 is branch coverage.
```

### Priority for Merge and Split Decisions

When deciding whether to merge or split:

1. Merge fragments that do not form a complete idea by themselves.
2. Split when the merged result contains two complete sentences or two clearly separate ideas.
3. Keep one complete sentence or one compact idea per subtitle block.
4. Keep the subtitle text on one physical line inside each block.
5. Do not split code, commands, paths, technical identifiers, or file names incorrectly.

---

## 6C. One-Line Subtitle Text Rule

Each subtitle block must contain its subtitle text on one physical line only.

Do not insert line breaks inside subtitle text.

Do not interpret this rule as permission to combine multiple complete sentences into one subtitle block.

Use a new subtitle block, with its own number and timestamp, when a sentence or compact idea must be separated.

---

## 7. Timestamp Adjustment Rules

Preserve timestamps by default.

Adjust timestamps only when needed for:

* splitting a long subtitle block;
* merging subtitle blocks that belong to the same sentence;
* fixing clearly incorrect sentence boundaries;
* preventing unreadably short display time;
* improving readability and timing balance.

When adjusting timestamps:

* do not overlap subtitle blocks;
* do not move content far away from its original spoken position;
* keep all new timestamps valid;
* keep subtitles in chronological order;
* preserve the original sequence of speech;
* use the earliest relevant start time and latest relevant end time when merging fragments into one sentence;
* redistribute boundaries proportionally when splitting a subtitle into multiple blocks;
* do not create artificial long gaps;
* preserve original gaps unless there is a clear reason to redistribute timing.

When only an SRT file is provided and no audio is available:

* do not invent pauses, speaker changes, or timing details that are not evident from the subtitle sequence;
* do not assume that a long timestamp range represents continuous speech;
* preserve timestamps conservatively;
* only redistribute the boundary between directly adjacent subtitle blocks when necessary for merging, splitting, or readability.

---

## 7A. Insufficient Display Time Rule

Apply this rule to all subtitle blocks, including blocks that were not merged.

If a subtitle contains a complete sentence or meaningful statement but its display duration is too short for comfortable reading, extend or redistribute its timestamp by the minimum amount necessary.

Use the following readability targets as guidance:

* aim for approximately 12 to 17 characters per second, including spaces;
* aim for at least approximately 1 second for a normal complete sentence;
* allow very short complete acknowledgements such as “Yes.” or “Run.” to remain shorter when necessary;
* avoid keeping a normal subtitle on screen for more than approximately 7 seconds unless the original timing clearly requires it;
* avoid displaying long or information-dense text for too little time.

When additional display time is needed:

* prefer extending the end timestamp into an available original gap after the subtitle;
* if there is no suitable gap after it, move the start timestamp slightly earlier into an available original gap before it;
* if adjacent subtitles are continuous, redistribute their shared boundary reasonably according to text length, reading difficulty, and speech continuity;
* add only the minimum amount of time necessary;
* do not create overlapping timestamps;
* do not move a subtitle far away from its original spoken position;
* do not extend a subtitle across a clear topic change, speaker change, or meaningful pause.

The priority is natural reading time while keeping the subtitle synchronized as closely as possible to the original speech.

---

## 8. Technical Terminology Rules

The subtitle content is related to:

* software tools;
* Unit Testing;
* GTest;
* Cantata;
* source code analysis;
* coverage measurement;
* database workflows;
* SharePoint;
* Excel reports;
* TCM upload workflows;
* automation tools;
* SharCC.

Many technical terms may be misrecognized by speech-to-text.

Always prefer the official terminology listed below when the context matches.

Preserve:

* capitalization;
* underscores;
* symbols;
* tool names;
* UI labels;
* column names;
* file names;
* command names;
* paths;
* code;
* abbreviations;
* version numbers;
* identifiers.

Do not translate these terms unless the context clearly requires a natural-language translation.

Do not force a glossary correction when the surrounding context does not support it.

When a term appears to be a command, UI label, code identifier, file name, path, or column name, preserve its exact official spelling and capitalization.

Use standard English grammar for ordinary narrative text, but preserve exact capitalization when a term is functioning as an official UI label or command.

---

## 9. Official Terminology List

### UT, SUT, and Copilot Terms

* Prepare Get SUTs
* Get SUTs
* SUT
* SUTs
* System Under Test
* Unit Testing
* UT
* UT request
* UT folder
* UT team
* UT AI Copilot tool
* AI Copilot
* GitHub Copilot
* GitHub Copilot Chat

### GTest and Google Test Terms

* GTest
* Google Test
* GTest Automation Tool
* GTest test case
* GTest test cases
* UT using GTest

### Cantata Terms

* Cantata
* Cantata IDE
* Cantata project
* Cantata folder
* Cantata testing
* Cantata interface
* Cantata++ coverage measurement
* Build Cantata
* Cantata file
* Cantata left blocks

### C/C++ and Source Code Terms

* C/C++ mode
* C file
* C++ file
* header file
* source file
* console-related headers
* function
* module
* function under test
* module under test
* function interface
* external function
* global variables
* local variables
* input variables
* output variables
* data type
* struct
* enum
* isolated scope

### Test Case and Test Environment Terms

* test case
* testcase
* test cases
* test design
* test script
* test environment
* test workspace folder
* test fixture
* sample test case
* macro test
* input section
* output section
* expected result
* actual result
* return value
* check function
* call SUTs
* test execution
* test generation
* testcase generation
* environment generation
* build generation

### Stub and Mock Terms

* stub
* stubs
* stub function
* stubbed
* stub return value
* mock
* mocks
* mock function
* mocked
* stubbing
* mocking
* stub and mock generation
* stubbing and mocking

### Build, Generation, and Error Terms

* compiler error
* build error
* missing definition
* rebuild
* Rebuild Failed Modules
* Run
* Generate Environment
* Generate Testcase All SUT
* Generate one SUT
* Generate Test Case
* Generate Test Case by A.I. Copilot
* GenTestDesign
* Auto Define Object

### Workflow Terms

* Prepare Get SUTs function
* Preliminary Analysis
* Analysis Support
* Testing Support
* Check Delta from Database, SharePoint
* Delta
* Find Delta
* find Delta
* Create Delta File
* Type Delta
* Search New Module
* Search New Module SharePoint
* File Analysis
* Analyze path
* source analysis
* module analysis
* coverage measurement
* report generation
* automation tool
* command prompt
* PowerShell

### Database, SharePoint, and Workspace Terms

* analysis file
* report file
* database
* tested database
* SharePoint database
* database path
* workspace path
* output folder
* Module Info file
* module name
* C module
* EXP module
* ESP module
* ASW
* ASW Demo
* test module
* valid module
* tested module
* validated module
* untested module
* new module
* failed module
* item
* item name
* old version
* Undefined
* Valid
* Stream
* Snapshot
* Stream Name
* Snapshot Name
* BB
* BB number
* config
* configuration
* component name
* version
* PCM
* CR
* CRP
* DCM
* VLT
* entity folder
* load database
* load through file
* remote load through file
* stream or snapshot
* SharePoint
* SharePoint link
* reload database

### Report and Result Terms

* test report
* test result
* test summary
* overall result
* Overall Result
* Test Summary
* detailed result
* MTVerdict
* Pass
* Fail
* Failed
* SUCCESS
* Complete
* Release
* release step
* complete status
* status
* result
* reportable item
* HTML report
* text report
* csv file
* log file
* output log

### TCM, ITT, and Workload Terms

* Smart ITT
* ITT
* TCM
* upload to TCM
* upload workload
* send result to TCM
* workflow
* workload

### Internal Tool and Module Names

* Alino
* Aleno
* MakeWordRule
* WordRule
* CodeCoveredViewer
* CodeCoveredResult
* Report Viewer
* Cmetrix
* TPA Report Generation tool
* TPA file
* TPA Report
* WinMan@16
* UT_008
* UT008
* SharCC

### Excel, Column, and Formula Terms

* MT_Concat
* MT_Needed
* Tester_Command
* Tester Comment
* CRP Comment
* CRP Review Status
* CRP link
* Responsible column
* Undefined column
* Pass below 100% column
* concat
* concatenate
* concat column
* VLOOKUP
* formula
* Formula
* paste as values
* clear formula
* filter
* batch
* comment
* reason
* coverage information
* check-in date
* item changed check-in date
* Validated items
* Tested items
* Estimation file
* total effort
* analysis column
* responsible column
* comment column
* command column

### Metric and Parameter Terms

* ELOCK
* Function
* MaxCc
* File Metric
* MATRIX
* Value
* Parameter
* Mixing Parameter
* Local Parameter
* Input Parameter
* Min
* Max
* Mid
* tolerance
* class
* Class
* review source
* tester
* utility design

### Coverage Terms

* code coverage
* coverage
* coverage metric
* coverage result
* 100% coverage
* coverage below 100%
* line coverage
* statement coverage
* branch coverage
* condition coverage
* C0
* C1
* C0C1
* MCDC
* DSD
* true case
* false case
* true branch
* false branch
* condition at line 59

### Review, Delivery, and Tools

* Review
* Deliver
* BDC
* FAC
* FACI report
* responsible person
* Visual Studio Code
* VS Code
* Terminal
* Output
* Open Folder
* folder path
* workspace
* project folder
* copy header file
* copy Cantata file
* database files
* left blocks

---

## 10. Common Speech-to-Text Correction Rules

Apply these corrections only when the context matches.

Do not force a correction if it makes the sentence incorrect.

| Incorrect / Detected Form                                 | Correct Form                                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Canata / Canada / Cantada                                 | Cantata                                                                                          |
| Gtest / G-test / G Test                                   | GTest                                                                                            |
| Google test                                               | Google Test or GTest, depending on context                                                       |
| Prepair Get SUTs / Prepair and Get SUTs / Prepare get SUT | Prepare Get SUTs                                                                                 |
| Get SUT / Get SUTs                                        | Get SUTs, when referring to the tool or action name                                              |
| V0C1 / COC1 / C zero C one                                | C0C1                                                                                             |
| MCG / incorrect MCDC-related recognition                  | MCDC                                                                                             |
| MT needed / MT-needed / empty needed                      | MT_Needed                                                                                        |
| MT concat / MT-concat                                     | MT_Concat                                                                                        |
| tester command / tester-command                           | Tester_Command                                                                                   |
| crp command                                               | CRP Comment, only when the surrounding context refers to the CRP comment field                   |
| crp-review-status / CRP review status                     | CRP Review Status                                                                                |
| responsible command                                       | Responsible column, only when the context refers to a column                                     |
| reportable column                                         | Responsible column, only when the context clearly refers to the responsible column               |
| chekin date / checkin date / check in date                | check-in date                                                                                    |
| Code Covered Viewer / CodeCovered Viewer                  | CodeCoveredViewer                                                                                |
| Code Covered Result / CodeCovered Result                  | CodeCoveredResult                                                                                |
| Batch / BatchFL                                           | Pass, only when the surrounding context clearly refers to a testcase verdict or execution result |
| cut the VLOOKUP                                           | use VLOOKUP                                                                                      |
| sell result to TCM                                        | send result to TCM, only when the context refers to submitting or uploading results              |
| Formular                                                  | Formula, unless it is clearly a proper name inside a tool                                        |

Special rules:

* Keep `Alino` and `Aleno` unchanged if they are tool names or module names.
* Do not translate `Alino` or `Aleno`.
* Do not convert the ordinary word `batch` to `Pass` unless the context clearly refers to a testcase result.
* Preserve `CodeCoveredViewer` and `CodeCoveredResult` as the canonical official forms.

---

## 11. Priority Order

When rules conflict, follow this priority order:

1. Keep the `.srt` format valid.
2. Preserve the original meaning, spoken sequence, and technical accuracy.
3. Preserve official technical terminology, commands, file names, paths, UI labels, and identifiers.
4. Eliminate standalone fragments by merging adjacent subtitle blocks into complete, meaningful sentences or ideas whenever possible.
5. Split subtitle blocks that contain two or more complete sentences or clearly separate ideas.
6. Keep timestamps natural, valid, readable, and non-overlapping.
7. Improve grammar, punctuation, capitalization, and readability.
8. Make only the minimum necessary changes.

---

## 12. Specific Instructions

Apply the following specific instructions in addition to all rules above:

[SPECIFIC INSTRUCTIONS WILL BE PROVIDED HERE]

Specific instructions may refine wording, terminology, or workflow context, but they must not invalidate the required `.srt` format, create overlapping timestamps, or require multiple physical text lines inside a subtitle block.

---

## 13. Final Silent Validation

Before returning the final answer, silently verify that:

* subtitle numbers are continuous from `1` to `n`;
* every timestamp follows the format `HH:MM:SS,mmm --> HH:MM:SS,mmm`;
* every timestamp is valid and chronological;
* no subtitle block overlaps with another;
* every subtitle block contains exactly one non-empty text line;
* no subtitle block is empty;
* no standalone fragment remains unless it is complete and meaningful;
* no subtitle block contains two complete sentences unless splitting would create an incomplete fragment or distort the meaning;
* long sentences have been split at natural clause boundaries when necessary;
* technical terms, code, commands, file names, paths, UI labels, and identifiers are preserved accurately;
* no text has been added, summarized, or invented;
* the final response contains only valid `.srt` content.

---

## 14. Input SRT Content

```srt
[PASTE THE SRT CONTENT HERE]
```
