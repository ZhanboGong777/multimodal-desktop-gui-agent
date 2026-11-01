# Week 6 full requirement report design contract

Reference DOCX: `C:/Users/联想/Documents/ChatGPT/Intern/output/docx/Week6_实验报告_文字修订版_修正.docx`.
SHA256: `f117680176cb238aa18b69b2d39557313a1b5f916ad2bf2106403cb403172218`.
The unchanged reference is the package base. `template_inspection_001.json`
inventories every part, section, paragraph and table. The retained reference
render is `tmp/week6_date_cleanup_20261028/render_001`; the fifteen-page actual
visual review is bound by its `final_delivery_001.json`. Its PDF SHA256 is
`5391301894ad874613bd165b1ef5be96d7720931a13036c08cf476e477d6e6f2`.
The formerly published PDF filename is currently absent; use this retained
render without renaming or recreating the old output.

This new report replaces body content to describe the newly implemented
semantic sequence, recovery, complete-frame perception and monitoring. The old
report and every old experiment remain unchanged. New normal/transient and
complete-regression results require original evidence; pending values remain
in a private draft and cannot produce the final-named output.

## Preserve-only design and package parts

One portrait US Letter section, 612x792pt. Margins left/right79.2pt, top72pt,
bottom62pt; text width453.6pt (6.3in). Header distance36pt, footer27pt. No running
header or cover date/timezone line. Preserve the source section, footer PAGE
field, styles, theme, numbering, font table, settings and opaque package parts
byte for byte. New body XML, new image relationships and image bytes are
intentional changes. The new artifact's core subject/version, keywords,
description and creation/modification timestamps are updated; its author
identity remains blank as in the reference. Existing unrelated media and
relationships are retained.

Body: Times New Roman / 华文宋体12pt, exact23.4pt, after6pt, justified, regular
black. Title and subtitle: source two centered black18pt 黑体 paragraphs, first
after0pt and second after17pt. H1: 黑体15pt, exact21pt, before17/after10pt,
direct bold false; H2: 黑体13pt, exact19pt, before11/after5pt. Keep headings with
their following paragraph. No forced page break before each section.

Tables: source-derived6.3in open three-rule design. Black1pt top/bottom,
0.75pt beneath repeated header, no body grids or shading, 4pt padding,
10.5pt text with exact16pt lines. Deliberate grids by numeric/narrative columns;
rows expand naturally, cannot clip or use exact row heights. Numeric cells
centered, narrative cells left aligned. Captions10.5pt/exact16pt. New perception
chart is a standard quantitative plot; preserve its aspect ratio at6.3in.
Code: Courier New9pt/exact15pt. Hyperlinks use the reference black underlined
appearance and retain the full target. The plain centered numeric footer
inherits12pt and stays below body text, near y753pt.

## Slot map and flow

All old body paragraphs/tables/old workflow drawing are replaced. Source role
paragraph properties and table properties are cloned; old data are summarized
only as explicitly dated history. New flow: purpose and requirement mapping;
semantic plan and fixed postconditions; retry/budget and monitoring; perception
method; new paired accuracy/time tables and figure; real-window results;
independent CPU/full regression; failure history and limits; reproduction paths.
No original code, report, artifact, snapshot or experiment directory is edited.

`report_template.md` is the editable prose source. It uses explicit `{{key}}`
slots for future facts and tables. `facts_manifest.pending.json` is private
and has report_ready=false; pending GUI/regression values must not become
numbers, success flags or final-named deliverables. Final facts must bind the
engineering acceptance bytes, actual raw batch/owner/validation records and
unchanged source hashes. The engineering acceptance may still say report and
package pending; the report does not create final overall completion.

## Fidelity and factual gates

Before final build, verify the reference SHA; refuse existing output paths.
Compare preserve-only package parts and the single section exactly. Check all
generated table cells and all numeric/code/link content against the resolved
Markdown. Keep frozen perception parameters/results byte identical. Final
DOCX/PDF must be rendered through bundled Python/portable LibreOffice with
render_docx.py, then every page actually viewed. Check body/footer gaps, split
tables, repeated headers, all text and URI relationships; never infer layout
quality from XML. Keep each build/render/QA iteration separately. Do not
publish until the parent independently reviews content and final page images.

The docx and pdf create-operation markers were each run successfully exactly
once for this report task before authoring. No desktop Office is used.
