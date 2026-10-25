# Week 6 OCR resources

`chi_sim.traineddata` is the official Tesseract fast integer Chinese model at
[tessdata_fast commit 87416418657359cb625c412a48b6e1d6d41c29bd](https://github.com/tesseract-ocr/tessdata_fast/tree/87416418657359cb625c412a48b6e1d6d41c29bd),
distributed under Apache 2.0. The unmodified upstream `LICENSE` is included.
`eng.traineddata` is an unchanged copy of the English model already installed with
Tesseract on the measurement machine. `provenance.json` records source, byte count
and SHA256 for both models and the license. No model was trained or fine-tuned.

The Chinese file is 2,469,156 bytes, SHA256
`a5fcb6f0db1e1d6d8522f39db4e848f05984669172e584e8d76b6b3141e1f730`.
The English file is 4,113,088 bytes, SHA256
`7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`.

The Windows Tesseract build used here cannot read a tessdata directory containing
non-ASCII characters. Its pytesseract wrapper also retains literal quote marks
when parsing a custom directory containing spaces. Use an ASCII path without
spaces; the engine explicitly rejects whitespace in a custom Windows path. These
are backend environment constraints, not OCR accuracy scores. The first two
development failures are retained in the experiment results.

For example, from the repository root in a PowerShell session:

```powershell
$week6OcrData = 'D:/week6-ocr/tessdata'
New-Item -ItemType Directory -Path $week6OcrData -Force | Out-Null
Copy-Item -Path 'Document/Week6/resources/tesseract/*.traineddata' -Destination $week6OcrData
$env:TESSDATA_PREFIX = $week6OcrData
```

This changes the current process environment only and does not modify the system
Tesseract installation. Load `configs/week6_accuracy.yaml` for the opt-in screen
profile, or set `perception.ocr.tesseract_tessdata_dir` explicitly to this folder.
The original `native` profile and original default language remain available for
compatibility. The screen profile trades additional CPU time for small-label and
Chinese recognition; it is not an end-to-end speed improvement.
