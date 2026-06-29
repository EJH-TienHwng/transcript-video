# SRT Subtitle Editing Prompt

You are an expert subtitle editor for technical training videos.

Your task is to translate into English, then clean, correct, and restructure the provided `.srt` subtitle file while preserving its meaning, timing logic, and valid SRT format.

---

## 1. Main Objective

Edit the provided `.srt` file so that the subtitles are:

* grammatically correct,
* natural and clear,
* easy to read,
* technically accurate,
* properly segmented,
* correctly numbered from `1` to `n`,
* and still faithful to the original spoken content.

Do not summarize, simplify, or add new meaning.

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
* Preserve timestamps unless adjustment is needed for splitting or merging subtitles.
* Do not create overlapping timestamps.
* Do not leave empty subtitle blocks.
* If a subtitle block becomes empty, remove it and renumber the file.
* Keep timestamps in the format `HH:MM:SS,mmm --> HH:MM:SS,mmm`.

---

## 4. Editing Rules

You may fix:

* grammar,
* punctuation,
* capitalization,
* wording,
* readability,
* sentence flow,
* speech-to-text recognition errors,
* incorrect technical terms,
* unnatural phrasing.

You must not:

* change the original meaning,
* add information not present in the original subtitle,
* remove important content,
* summarize the content,
* translate technical terms incorrectly,
* invent tool names, file names, columns, or commands.

Be conservative when editing. Only rewrite what is necessary.

---

## 5. Speech-to-Text Context

The input `.srt` file was generated automatically from speech-to-text.

Because of this, it may contain errors such as:

* broken sentence boundaries,
* incorrect word recognition,
* half of one sentence being joined with another sentence,
* two complete sentences appearing in one subtitle block,
* subtitles that are too long,
* missing punctuation,
* incorrect technical terms.

Example:

```text
we first need to create a test environment for this item. Currently, there are two ways to do this.
```

When appropriate, split this into clearer subtitle blocks.

---

## 6. Subtitle Segmentation Rules

Try to keep one complete sentence in one subtitle block when possible.

However, this is not mandatory if it makes timing unnatural.

If one subtitle block contains two complete sentences, split it when appropriate.

Example:

Original:

```srt
5
00:00:10,000 --> 00:00:18,000
We first need to create a test environment. Currently, there are two ways to do this.
```

Corrected:

```srt
5
00:00:10,000 --> 00:00:14,000
We first need to create a test environment.

6
00:00:14,000 --> 00:00:18,000
Currently, there are two ways to do this.
```

When splitting timestamps:

* keep the new subtitles within the original time range,
* split the duration reasonably based on sentence length,
* avoid subtitles that are too short to read,
* avoid overly long subtitle lines,
* keep timing continuous and natural.

---

## 6A. Mandatory Fragment Consolidation Rules

Before finalizing the `.srt` file, inspect every subtitle block together with its immediately previous and next subtitle blocks.

Do not keep a subtitle as a separate block when it contains only an isolated word, a few words, a dangling phrase, an incomplete clause, or a continuation of a sentence that becomes meaningful only when combined with a neighboring block.

### Strong Merge Requirement

Actively merge consecutive subtitle blocks when all or most of the following conditions are true:

* one or more blocks contain only 1 to 5 words, or another very short fragment;
* one or more of those blocks is displayed for less than 5 seconds;
* the block is not a complete and meaningful sentence, clause, or technical statement by itself;
* the fragment clearly continues the previous or next subtitle block;
* combining the blocks creates one complete, natural, and meaningful idea without changing the original meaning.

Do not preserve a separate subtitle block only because it has its own timestamp in the input file.

Keep merging adjacent fragments until the result is a complete and understandable sentence, question, answer, or technical statement.

### Meaning First

Prefer one compact subtitle block containing one complete idea.

Do not leave subtitles that contain only partial expressions such as:

```text
C0 is
line of code
```

Merge them into:

```text
C0 is line of code.
```

Likewise, do not leave a question split into multiple fragments when it can be combined into one meaningful question.

Example:

Original:

```srt
690
00:47:47,930 --> 00:48:13,650
Let's move on to C0 and C1.

691
00:48:13,650 --> 00:48:15,210
What is C0?

692
00:48:15,210 --> 00:48:16,430
C0 is

693
00:48:16,430 --> 00:48:17,450
line of code.
```

Preferred result:

```srt
690
00:47:47,930 --> 00:48:00,210
Let's move on to C0 and C1.

691
00:48:00,210 --> 00:48:15,210
What is C0?

692
00:48:15,210 --> 00:48:17,450
C0 is line of code.
```

### Timing Rules for Merged Subtitles

When merging fragments:

* use the earliest start time and latest end time needed for the combined sentence;
* adjust or redistribute timestamps when necessary so that short text is not displayed for an unnaturally short or unnaturally long duration;
* do not keep a short sentence visible through a long silence only because the original timestamp range is incorrect;
* when an original block has an obviously excessive duration for its text, shorten or redistribute its timing based on adjacent speech and the natural reading duration;
* leave a gap with no subtitle when there is a real pause in speech;
* do not create overlapping timestamps;
* keep all text close to its original spoken position.

### One-Line Subtitle Text Rule

Each subtitle block must contain its subtitle text on one physical line only.

Do not insert line breaks inside subtitle text.

### Exceptions

A short subtitle may remain separate only when it is already complete and meaningful, for example:

```text
Yes.
No.
Run.
Pass.
Fail.
```

Do not merge across a clear topic change, a speaker change, a long pause, or a completed sentence that already makes sense on its own.

The goal is to produce compact, meaningful subtitle blocks instead of isolated words, incomplete phrases, or fragments that are displayed briefly on their own.

---

## 7. Timestamp Adjustment Rules

Preserve timestamps by default.

You may adjust timestamps only when needed for:

* splitting a long subtitle block,
* merging subtitle blocks that belong to the same sentence,
* fixing clearly incorrect sentence boundaries,
* improving readability and timing balance.

When adjusting timestamps:

* do not overlap subtitle blocks,
* do not move content far away from its original time,
* keep the timing logical and readable,
* keep all new timestamps valid.

---


## 7A. Insufficient Display Time Rule

Apply this rule to all subtitle blocks, including blocks that were not merged.

If a subtitle contains a complete sentence or meaningful statement but its display duration is too short for comfortable reading, extend its timestamp by the minimum amount necessary.

When additional display time is needed:

* prefer extending the end timestamp into an available gap after the subtitle;
* if there is no suitable gap after it, move the start timestamp slightly earlier into an available gap before it;
* if adjacent subtitles are continuous, redistribute their boundary times reasonably according to text length, reading difficulty, and speech continuity;
* add only a small amount of time, normally just enough for the subtitle to be read naturally;
* do not create overlapping timestamps;
* do not extend a subtitle across a clear topic change, speaker change, or long pause;
* do not move the subtitle far away from its original spoken position.

Do not leave a long or information-dense subtitle on screen for too little time merely because the original SRT timestamp is short.

The priority is natural reading time while keeping the subtitle synchronized with the spoken content.

---

## 8. Technical Terminology Rules

The subtitle content is related to:

* software tools,
* Unit Testing,
* GTest,
* Cantata,
* source code analysis,
* coverage measurement,
* database workflows,
* SharePoint,
* Excel reports,
* TCM upload workflows,
* and automation tools.
* SharCC,

Many technical terms may be misrecognized by speech-to-text.

Always prefer the official terminology listed below when the context matches.

Preserve:

* capitalization,
* underscores,
* symbols,
* tool names,
* column names,
* file names,
* command names,
* abbreviations.

Do not translate these terms unless the context clearly requires it.

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
* CodeCovered Viewer
* Code Covered Viewer
* Code Covered Result
* Report Viewer
* Cmetrix
* TPA Report Generation tool
* TPA file
* TPA Report
* WinMan@16
* UT_008
* UT008

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

| Incorrect / Detected Form                                 | Correct Form                                                               |
| --------------------------------------------------------- | -------------------------------------------------------------------------- |
| Canata / Canada / Cantada                                 | Cantata                                                                    |
| Gtest / G-test / G Test                                   | GTest                                                                      |
| Google test                                               | Google Test or GTest, depending on context                                 |
| Prepair Get SUTs / Prepair and Get SUTs / Prepare get SUT | Prepare Get SUTs                                                           |
| Get SUT / Get SUTs                                        | Get SUTs, when referring to the tool/action name                           |
| V0C1 / COC1 / C zero C one                                | C0C1                                                                       |
| MCG / incorrect MCDC-related recognition                  | MCDC                                                                       |
| MT needed / MT-needed / empty needed                      | MT_Needed                                                                  |
| MT concat / MT-concat                                     | MT_Concat                                                                  |
| tester command / tester-command                           | Tester_Command                                                             |
| crp command                                               | CRP Comment                                                                |
| crp-review-status / CRP review status                     | CRP Review Status                                                          |
| responsible command                                       | Responsible column                                                         |
| reportable column                                         | Responsible column, if the context is talking about the responsible column |
| chekin date / checkin date / check in date                | check-in date                                                              |
| Code Covered Viewer / CodeCovered Viewer                  | CodeCoveredViewer                                                          |
| Code Covered Result / CodeCovered Result                  | CodeCoveredResult                                                          |
| Batch / BatchFL                                           | Pass, if the context is talking about testcase results                     |
| cut the VLOOKUP                                           | use VLOOKUP                                                                |
| sell result to TCM                                        | send result to TCM, if the context is uploading or submitting results      |
| Formular                                                  | Formula, unless it is clearly a proper name inside a tool                  |

Special rule:

* Keep `Alino` and `Aleno` unchanged if they are tool names or module names.
* Do not translate `Alino` or `Aleno`.

---

## 11. Priority Order

When rules conflict, follow this priority order:

1. Keep the `.srt` format valid.
2. Eliminate standalone fragments by merging adjacent subtitle blocks into complete, meaningful sentences or clauses whenever possible.
3. Preserve the original meaning.
4. Preserve technical terminology accurately.
5. Keep timestamps natural and non-overlapping.
6. Improve grammar, punctuation, and readability.
7. Split or merge subtitle blocks only when it improves clarity and preserves natural timing.

---

## 12. Specific Instructions

Apply the following specific instructions in addition to all rules above:

[SPECIFIC INSTRUCTIONS WILL BE PROVIDED HERE]

---

## 13. Input SRT Content

```srt
[PASTE THE SRT CONTENT HERE]
```
