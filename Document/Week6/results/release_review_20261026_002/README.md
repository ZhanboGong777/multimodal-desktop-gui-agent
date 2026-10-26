# Week 6 完整回归与换行字节审查追加材料

本目录是第二份独立新增归档，不修改 `release_review_20261026_001` 或它的清单。`copy_manifest_001.json` 逐项记录原路径、目标路径、字节数及 SHA256。

full006 已由原执行记录确认真实 exit 0，3027 项通过，pytest 535.12 秒、封装 566.027 秒，244 个受检文件前后及当前字节一致。本目录保留该原 validation、stdout 和原 harness，以及合法生成的限定范围 completion、原证据审计和创建 helper。completion 允许生成正式报告，尚不等于 DOCX/PDF 逐页 QA、最终打包及用户交付全部完成。

换行审查发现根 `README.md` 是 244 项中唯一包含 CRLF 的文件（190 对）。其原 SHA256 为 `f3e7fea31e0a587281e11707494d684a54de2fd1e07b623a9972c2696f97d2de`。原 `.gitattributes` 的 `*.md text eol=lf` 会在 Git 入库时改动这些已回归字节。

Root 在属性文件末尾增加精确根路径规则 `/README.md -text -whitespace`；`.gitattributes` 本身不属于上述 244 个回归输入。`-text` 禁止对该文件进行换行转换，`-whitespace` 只影响它的空白诊断。该规则不改变子目录 README 的文本策略、生产代码或任何成功门槛。

`line_endings/before_saved_tool_output_001.json` 如实保存原内联只读命令的工具返回（chunk `9324ca`）。它在 Root 改属性后才落盘，不是当时另存的 harness，也没有补造观察时间。`current_filter_audit_001.json` 则是之后新执行的独立只读审核：244 项当前 SHA 仍等于原回归，Git 过滤也不再改变任何受检字节。它没有比较当前 HEAD 或最终 ZIP。

`line_endings/readonly_audit.py.txt` 是事后保存的等效只读检查程序，检查当前规则；再次运行不能重现旧属性状态。原始受检 README 未被重写，没有用“规范化后的相等”替代精确字节相等。

`provenance/verify_tested_git_archive_001.py.txt` 是 Root 准备的后提交只读审核脚本。归档时尚未运行，因为当时 HEAD 仍是旧提交。Root 后续应在最终提交后单独比较 244 个原 pin 与当前/Git blob，再在生成最终 ZIP 后另用新输出核验同一批成员与 RELEASE commit；已有审核不得覆盖。

只核验本目录副本字节：

```powershell
python verify_snapshot_bytes.py.txt
python verify_snapshot_bytes.py.txt --original-sources
```

以上命令不运行模型、GUI、测试、Git 提交或打包。创建 completion 的 helper 仅作来源保留，不能为了复现而覆盖已有合法 completion。大批原实验仍由原目录保留，这里只复制必要的小型审核材料。
