"""Source-package report builder; unfinished evidence permits private draft only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from lxml import etree as E

WORKSPACE = Path(__file__).resolve().parents[2]
TASK = Path(__file__).resolve().parent
REPO = WORKSPACE / "week6-worktree"
REFERENCE = WORKSPACE / "output/docx/Week6_实验报告_文字修订版_修正.docx"
REFERENCE_SHA = "f117680176cb238aa18b69b2d39557313a1b5f916ad2bf2106403cb403172218"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "r": R, "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}
ARMS = ["native_eng_subprocess", "native_bilingual_subprocess",
        "screen_bilingual_subprocess", "screen_bilingual_persistent"]
LABELS = ["native英文CLI", "native双语CLI", "screen双语CLI", "screen双语C API"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_bytes())


def save(path: Path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))


def table(header, rows):
    clean = lambda value: str(value).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(clean, header)) + " |",
                      "| " + " | ".join("---" for _ in header) + " |"] +
                     ["| " + " | ".join(map(clean, row)) + " |" for row in rows])


def resolve_facts(manifest, final):
    evidence = {}
    perception = manifest["perception"]
    for key in ("primary", "replication", "independent_raw_audit", "source_freeze"):
        path = REPO / perception[key]
        assert sha(path) == perception[key+"_sha256"], key
        evidence[key] = read(path)
    a, b = evidence["primary"], evidence["replication"]
    assert a["all_criteria_passed"] and b["all_criteria_passed"]
    assert evidence["independent_raw_audit"]["both_batches_passed"] is True
    values = {}
    values['semantic_layer_scope'] = '正在实施与独立审查的v5方案让高层只接收完整用户意图和调用方最初的文字判据，不发送当前OCR、控件几何、窗口snapshot或图片。高层负责排列语义目标，低层TaskPlanner仍使用每次输入前的新截图定位控件。初始截图、OCR和窗口信息继续归档，用于范围检查、输入前条件和事后复核；文件留存不代表高层模型看过这些内容。当前源码尚未确认为最终冻结版本，也没有新的窗口验收成功，纯图形高层理解不在已验证范围内。'
    if isinstance(manifest.get('semantic_source_freeze'), dict):
        bound = manifest['semantic_source_freeze']
        semantic_freeze_path = REPO / bound['path']
        assert sha(semantic_freeze_path) == bound['sha256']
        semantic_freeze = read(semantic_freeze_path)
        assert semantic_freeze['protocol'] == 'week6.semantic_subtasks.v5'
        assert semantic_freeze['input_modality'] == 'caller_intent_only'
        assert semantic_freeze['image_attached'] is False
        assert semantic_freeze['schema'] == 'week6.semantic.v5.source_freeze.v1'
        assert semantic_freeze['status'] == 'frozen_pending_root_full003_and_real_acceptance'
        assert all(sha(REPO / relative) == digest for relative, digest in semantic_freeze['code_pins'].items())
        target_path = semantic_freeze_path.parent/'cpu_validation_001/validation.json'
        assert sha(target_path) == semantic_freeze['targeted_validation_sha256']
        target = read(target_path)
        assert target['pins_unchanged'] is True and all(r['wait_exit_code'] == 0 for r in target['receipts'])
        target_text = (target_path.parent/'pytest.stdout.txt').read_text(encoding='utf-8')
        target_match = re.search(r'(?m)^(\d+) passed in ([0-9.]+)s', target_text)
        assert target_match and int(target_match[1]) == semantic_freeze['v5_targeted_passed']
        values['semantic_layer_scope'] = values['semantic_layer_scope'].replace(
            '正在实施与独立审查的v5方案', '已通过定向检查并冻结的v5方案').replace(
            '当前源码尚未确认为最终冻结版本，也没有新的窗口验收成功', '源码已冻结，但尚无新的窗口验收成功')
        evidence['semantic_targeted_validation'] = target
        evidence['semantic_targeted_summary'] = {'passed':int(target_match[1]), 'seconds':float(target_match[2])}
        evidence['semantic_source_freeze'] = semantic_freeze
    values["opening_result"] = (
        "8帧合成新留出上的感知对照已经完成。持久化方案的文字F1为93.6508%，两批稳定运行后的整帧处理中位耗时为108.291和109.888毫秒，均满足预先登记的准确率及速度门槛。"
        "本稿尚未填入本轮真实窗口验收、完整回归和交付检查结果，整体验收结论暂不生成。"
    )
    requirements = [["自动分解与执行", "模型生成有顺序的语义目标，阶段条件输入前确定", "本轮窗口结果待补"],
                    ["错误检测与恢复", "共享预算，保留失败；未知输入停止重放", "本轮恢复结果待补"],
                    ["感知准确率与速度", "8帧新留出，两批各10组四臂配对", "已通过预登记门槛"],
                    ["监测与日志", "目标台账、累计预算、逐输入记录和子退出码", "本轮窗口记录待补"]]
    values["requirements_table"] = table(["要求", "本轮验证内容", "结果"], requirements)
    values["semantic_plan_table"] = "实际模型生成的语义方案及其阶段检查，待本轮正常与恢复试验结束后按原始记录列入。"
    values["monitoring_result"] = "当前实现与离线反例已经覆盖语义状态字段；新窗口中状态与输入计数的一致性，待对应原始日志核对后补入。"
    rows = []
    for arm, label in zip(ARMS, LABELS):
        m = a["unique_frame_accuracy"][arm]
        rows.append([label, f"{m['tp']}/{m['fp']}/{m['fn']}", f"{m['precision']*100:.4f}",
                     f"{m['recall']*100:.4f}", f"{m['f1']*100:.4f}"])
    values["accuracy_table"] = table(["配置", "TP/FP/FN", "P %", "R %", "F1 %"], rows)
    rows = []
    for name, data in [("第一批", a), ("复核批", b)]:
        for arm, label in zip(ARMS, LABELS):
            warm = data["timings"]["warm"][arm]
            rows.append([name, label, f"{warm['median_ms']:.3f}", f"{warm['p95_nearest_rank_ms']:.3f}", f"{warm['max_ms']:.3f}"])
    values["timing_table"] = table(["批次", "配置", "中位 ms", "p95 ms", "最大 ms"], rows)
    values["speed_conclusion"] = (
        "第一批候选相对英文 native、双语 native 和原 screen CLI 的热态中位耗时分别减少45.178%、66.624%和70.119%；复核批分别减少44.349%、66.494%和70.148%。"
        "各帧的全部配对差值另存原记录，两批均满足至少10%的总中位改善，且没有一帧的候选中位数比原 screen 慢超过5%。"
    )
    values["latency_figure"] = "![图 1 两批热态整帧处理中位耗时](assets/perception_warm_latency.png)"
    values["live_acceptance_table"] = table(["批次", "目标", "输入/规划/HTTP", "墙钟秒", "终态"], [["本轮正常", "ALPHA/GREEN", "待补", "待补", "待补"], ["本轮恢复", "BETA/BLUE", "待补", "待补", "待补"]])
    values["live_acceptance_analysis"] = "本轮真实运行尚未结束，不能以模型方案已解析、单个动作已派发或进程退出码为0代替任务成功。"
    values["external_perception_result"] = "另已在OCR前冻结一张2560×1600真实初始画面的5个中心文字标签和2个按钮。右下系统通知不在完整标注范围内，因此整帧精确率和召回率仍缺失。保存图四臂像素验证结果待补，它与桌面执行成功分别记录。"
    if isinstance(manifest.get("external_perception"),dict):
        external_path = REPO/manifest['external_perception']['path']
        assert sha(external_path) == manifest['external_perception']['sha256']
        external = read(external_path)
        evidence['external_perception'] = external
        assert external['records'] == 56 and not external['all_errors']
        assert external['source_unchanged'] and external['runtime_unchanged'] and external['inputs_unchanged']
        assert external['screen_full_predictions_exact'] is True
        external_rows = []
        for arm,label in zip(ARMS,LABELS):
            values_warm = external['timings']['warm'][arm]
            external_rows.append([label,f"{external['timings']['cold'][arm]['median_ms']:.3f}",
                                  f"{values_warm['median_ms']:.3f}",f"{values_warm['p95_nearest_rank_ms']:.3f}"])
        values['external_perception_result'] = (
            "另使用normal001初始保存的2560×1600画面做四臂像素验证，输入始终是未裁切的整帧。该轮任务随后在语义请求阶段超时且没有输入，保存图本身不表示桌面任务成功。"
            "OCR前人工冻结中心区域的5个英文标签和2个按钮，四臂在该区域均全部识别正确。右下系统通知未完整标注，因此整帧精确率和召回率仍缺失。\n\n"+
            table(['配置','冷中位 ms','热中位 ms','热p95 ms'],external_rows)+"\n\n"
            "这张易英文画面没有准确率增益。持久化方案比原screen CLI快36.64%，但比英文native慢22.98%；相对双语native仅约1.37%的差距，不据此断言单帧改善稳定。"
            "它保留了screen的全部有序输出，也说明合成集上的速度优势不能推及每个真实桌面画面。"
        )
    values["cpu_validation_table"] = table(["范围", "通过", "pytest秒", "说明"], [["感知定向", "59", "1.37", "C API寿命、异常及原OCR兼容"], ["v4语义定向（历史）", "235", "67.03", "不能代替当前v5验证"], ["早期独立CPU反例（历史）", "23", "见原记录", "验证002，另8项审计边界在验证004单列"], ["v5定向与独立审查", "待补", "待补", "当前实现仍在合流"], ["v5完整回归", "待补", "待补", "最终冻结版本另行运行"]])
    if 'semantic_targeted_summary' in evidence:
        target = evidence['semantic_targeted_summary']
        values['cpu_validation_table'] = values['cpu_validation_table'].replace(
            '| v5定向与独立审查 | 待补 | 待补 | 当前实现仍在合流 |',
            f'| v5定向与独立审查 | {target["passed"]} | {target["seconds"]:.2f} | pytest与Ruff退出码0，包含独立审计用例，不重复相加 |')
    values["full_validation_result"] = "第一次完整回归保存了1项失败、1225项通过和1958项设置错误，pytest耗时158.35秒、退出码1，Ruff退出码0，255份代码文件前后相同。设置错误由验证脚本未建立新basetemp父目录引起；另1项失败是README遗漏既有测试要求的数量声明。修正仅涉及验证脚本的目录创建和README说明，没有修改或放宽测试。第二次完整回归尚未结束，3184是待验证数量，不是通过数量。历史3027项通过只属于上一版，不能代替新源码回归。"
    if isinstance(manifest.get('historical_full_validation_002'), dict):
        bound = manifest['historical_full_validation_002']
        history_path = REPO / bound['path']
        assert sha(history_path) == bound['sha256']
        history = read(history_path)
        assert history['checks_passed'] is True and history['code_unchanged'] is True
        assert read(history_path.parent/'tested_code_pins_before.json') == read(history_path.parent/'tested_code_pins_after.json')
        assert read(history_path.parent/'pytest.receipt.json')['actual_child_exit_code'] == 0
        assert read(history_path.parent/'ruff.receipt.json')['actual_child_exit_code'] == 0
        history_stdout = (history_path.parent/'pytest.stdout.txt').read_text(encoding='utf-8')
        history_passed = re.search(r'(?m)^(\d+) passed in ([0-9.]+)s', history_stdout)
        assert history_passed is not None
        values['cpu_validation_table'] += f'\n| v4完整回归002（历史） | {history_passed[1]} | {history_passed[2]} | pytest与Ruff退出码0，不代表修改后的v5通过 |'
        values['full_validation_result'] = (
            'v4第一次完整回归保存了1项失败、1225项通过和1958项设置错误，pytest耗时158.35秒、退出码1，Ruff退出码0，255份代码文件前后相同。'
            '设置错误由验证脚本未建立新basetemp父目录引起；另1项失败是README遗漏既有测试要求的数量声明。修正仅涉及目录创建和README，没有修改或放宽测试。'
            f'v4第二次完整回归为{history_passed[1]}项通过，pytest耗时{history_passed[2]}秒，pytest与Ruff实际退出码均0。该结果及历史3027项通过分别保留，均不能转移为修改后v5源码的通过结论。')
        evidence['historical_full_validation_002'] = history
    if isinstance(manifest.get('historical_full_validation_003'), dict):
        bound = manifest['historical_full_validation_003']
        old_path = REPO / bound['path']
        assert sha(old_path) == bound['sha256']
        old = read(old_path)
        old_freeze_ref = manifest['historical_v5_source_freeze_001']
        old_freeze_path = REPO / old_freeze_ref['path']
        assert sha(old_freeze_path) == old_freeze_ref['sha256']
        old_freeze = read(old_freeze_path)
        assert read(old_path.parent/'tested_code_pins_before.json') == read(old_path.parent/'tested_code_pins_after.json') == old_freeze['code_pins']
        assert old['checks_passed'] is True and old['code_unchanged'] is True
        assert read(old_path.parent/'pytest.receipt.json')['actual_child_exit_code'] == 0
        assert read(old_path.parent/'ruff.receipt.json')['actual_child_exit_code'] == 0
        old_match = re.search(r'(?m)^(\d+) passed in ([0-9.]+)s', (old_path.parent/'pytest.stdout.txt').read_text(encoding='utf-8'))
        assert old_match
        values['cpu_validation_table'] += f'\n| v5首轮完整回归003（历史） | {old_match[1]} | {old_match[2]} | 对应normal007失败所用源，不转移到后续提示修订 |'
        values['full_validation_result'] += (
            f' v5首轮full003为{old_match[1]}项通过，pytest耗时{old_match[2]}秒，pytest与Ruff实际退出码0。'
            f'{old["code_pin_count"]}份代码在该轮测试前后与首轮冻结相同。该源随后用于normal007，窗口方案在所有输入前因重复完成条件被拒绝；后续提示修订仍需新的同源完整回归。')
        if not manifest.get('full_validation'):
            values['opening_result'] = values['opening_result'].replace(
                '本稿尚未填入本轮真实窗口验收、完整回归和交付检查结果，整体验收结论暂不生成。',
                f'v5首轮完整回归{old_match[1]}项通过已保留为历史，后续提示修订仍需重新回归。本稿尚无新的窗口验收与交付结果，整体验收结论暂不生成。')
        evidence['historical_full_validation_003'] = old
    if isinstance(manifest.get('full_validation'), dict):
        bound = manifest['full_validation']
        full_path = Path(bound['path'])
        if not full_path.is_absolute():
            full_path = REPO / full_path
        assert sha(full_path) == bound['sha256']
        full = read(full_path)
        assert full['schema'] == 'week6.full_regression.actual.v1'
        assert full['checks_passed'] is True and full['code_unchanged'] is True
        before = read(full_path.parent/'tested_code_pins_before.json')
        after = read(full_path.parent/'tested_code_pins_after.json')
        assert before == after and len(before) == full['code_pin_count']
        if 'semantic_source_freeze' in evidence:
            assert before == evidence['semantic_source_freeze']['code_pins']
        assert all(sha(REPO / relative) == digest for relative, digest in before.items())
        pytest_receipt = read(full_path.parent/'pytest.receipt.json')
        ruff_receipt = read(full_path.parent/'ruff.receipt.json')
        assert pytest_receipt['actual_child_exit_code'] == ruff_receipt['actual_child_exit_code'] == 0
        summary = (full_path.parent/'pytest.stdout.txt').read_text(encoding='utf-8')
        passed = re.search(r'(?m)^(\d+) passed in ([0-9.]+)s', summary)
        assert passed is not None
        count, seconds = int(passed[1]), float(passed[2])
        values['opening_result'] = values['opening_result'].replace(
            '本稿尚未填入本轮真实窗口验收、完整回归和交付检查结果，整体验收结论暂不生成。',
            f'完整回归另已记录{count}项通过。本稿尚未填入新窗口验收与交付检查结果，整体验收结论暂不生成。')
        values['cpu_validation_table'] = values['cpu_validation_table'].replace(
            '| v5完整回归 | 待补 | 待补 | 最终冻结版本另行运行 |',
            f'| v5完整回归{full_path.parent.name.removeprefix("full_validation_")} | {count} | {seconds:.2f} | pytest与Ruff退出码均0，{len(before)}份代码文件前后及当前相同 |')
        values['full_validation_result'] += (
            f' 当前v5的{full_path.parent.name}为{count}项通过，pytest报告耗时{seconds:.2f}秒，拥有该子进程的运行器等待耗时{pytest_receipt["elapsed_seconds"]:.3f}秒，完整验证运行器耗时{full["elapsed_seconds"]:.3f}秒；pytest与Ruff的实际退出码均为0。'
            f'{len(before)}份代码文件在测试前后、v5冻结记录及当前逐字节相同。该回归仍不能替代真实模型窗口验收。')
        evidence['full_validation'] = full
    values["development_failure_table"] = table(["记录", "观察到的问题", "处理与结论"], [["感知pilot001", "C API TSV不含列名", "保留错误后按官方接口解析"], ["感知pilot003", "小数与整数置信度不等", "匹配原包装器数值转换，之后冻结"], ["语义开发CPU", "1失败37通过，错误信息预期不符", "独立保存输入前错误，未放松限制"], ["normal001", "初始拆解请求超时", "0输入、1规划、1HTTP；运行失败，子退出码1"], ["normal002", "首个语义回复理由循环", "4096输出token后length截断，0输入"], ["normal003（v2）", "模型误判缺少项目选择信息", "needs_clarification，0输入，子退出码1；不是用户缺信息"], ["normal004（v3）", "模型把未来页面控件未出现当作需澄清", "0输入、1规划、1HTTP，子退出码1；跨页面语义判断失败"], ["normal005（v4）", "连续两次画面桥接超时，缺少专用窗口激活收据", "0输入、0规划、0HTTP，子退出码1；模型调用前的桥接失败"], ["其余本轮窗口尝试", "待补", "保留每次失败及未知，不拼接成功"]])
    values['development_failure_table'] += "\n| full_validation001设置 | 1958项设置错误，pytest退出码1 | 私有验证脚本漏建basetemp父目录，WinError3；原件和单节点复现均保留 |\n| full_validation001断言 | 1项README数量声明断言失败 | 补回数量说明，未改测试；第二次完整回归结果另列 |"
    values['development_failure_table'] += '\n| normal006（v4） | 模型混淆当前选项与未来完成文字，误称缺少审阅信息 | needs_clarification，0输入、0未知、1规划、1HTTP，子退出码1 |'
    if manifest.get('negative_attempt_007'):
        bound = manifest['negative_attempt_007']
        negative_path = Path(bound['path'])
        assert sha(negative_path) == bound['sha256']
        negative = read(negative_path)
        assert negative['counts_as_success'] is False and negative['known_inputs'] == negative['unknown_inputs'] == 0
        values['development_failure_table'] += '\n| normal007（v5首轮） | 3个相同回复把审阅和提交分成重复完成条件的目标 | 输入前3次拒绝，3规划、3HTTP、0输入、0未知，子退出码1；不计目标完成 |'
        evidence['negative_attempt_007'] = negative
    values['semantic_diagnostic_result'] = '三次模型开发诊断另存原始请求和回复，不作为窗口验收或已完成目标。'
    if isinstance(manifest.get('semantic_diagnostics'), dict):
        bound = manifest['semantic_diagnostics']
        diagnostic_path = REPO / bound['path']
        assert sha(diagnostic_path) == bound['sha256']
        diagnostic = read(diagnostic_path)
        execution = diagnostic['execution']
        assert execution['child_wait_exit_code'] == 0 and execution['actual_http_requests'] == 3
        assert execution['new_gui_capture_input_executor_calls'] == 0
        values['semantic_diagnostic_result'] = (
            '在normal006结束后，按预登记顺序做了三次模型开发诊断，每组只请求一次、不重试，没有新屏幕采集、输入或执行器调用。运行器等待子进程的退出码0只表示诊断脚本完成。'
            'A把相同JSON格式写成可读说明，并保留历史snapshot；模型给出6个目标，但多目标重复检查条件，被旧本地校验拒绝。'
            'B同样提供格式说明而不附当前snapshot，生成了3个结构允许的目标；其中Review和Commit只是未执行的页面文案预测，不能说明完成条件可实现。'
            'C明确缺少项目参数，模型仍给出ready并把未填写标记写进目标，缺参负对照失败。'
            '三组包装计时分别为45.250、15.047和13.438秒，共3次HTTP，均不计目标完成或GUI验收成功。'
            'B/C仍沿用了一句描述屏幕输入的v4说明，因此这些诊断也不能单独确定失败原因，或证明仅去掉snapshot就能解决问题。')
        evidence['semantic_diagnostics'] = diagnostic
    prefix = "https://github.com/ZhanboGong777/multimodal-desktop-gui-agent/blob/"
    values["source_links"] = (
        f"[本轮使用说明]({prefix}{manifest['repository_ref']}/Document/Week6/README.md)\n\n"
        f"[本轮实验记录](https://github.com/ZhanboGong777/multimodal-desktop-gui-agent/tree/{manifest['repository_ref']}/Document/Week6/results/full_requirements_20261028/)\n\n"
        f"[历史留出实验说明]({prefix}codex/week6-delivery/Document/Week6/results/perception_accuracy_20261024/README.md)"
    )
    values["engineering_result"] = "四项工程验收、报告页面检查和源码发布检查分别记录。本稿保留未填结果，不生成整体完成状态。"
    if final:
        assert manifest["report_ready"] is True
        required = ("engineering_acceptance", "normal", "transient", "full_validation", "external_perception", "development_failures")
        for key in required:
            bound = manifest.get(key)
            assert isinstance(bound, dict) and bound.get("path") and bound.get("sha256"), key
            path = Path(bound["path"])
            if not path.is_absolute():
                path = REPO / path
            assert sha(path) == bound["sha256"], key
            evidence[key] = read(path)
        engineering = evidence["engineering_acceptance"]
        assert engineering["engineering_passed"] is True
        assert engineering["delivery_status"] == "report_and_package_pending"
        # Final prose comes from reviewed, explicitly populated slots, while
        # every original engineering/raw result remains independently bound.
        overrides = manifest.get("resolved_slots")
        assert isinstance(overrides, dict)
        for key in ("opening_result", "requirements_table", "semantic_layer_scope", "semantic_plan_table", "monitoring_result",
                    "live_acceptance_table", "live_acceptance_analysis", "external_perception_result",
                    "cpu_validation_table", "full_validation_result", "development_failure_table", "engineering_result"):
            assert isinstance(overrides.get(key), str) and overrides[key].strip(), key
            values[key] = overrides[key]
        assert not any("待补" in value or "尚未填入" in value for value in values.values())
    raw = (TASK / "report_template.md").read_text(encoding="utf-8")
    if not final:
        raw = raw.replace("# 系统 v2.1.0  复杂任务分解与完整验收",
                          "# 系统 v2.1.0  复杂任务分解待验收稿", 1)
    for key, value in values.items():
        raw = raw.replace("{{"+key+"}}", value)
    assert "{{" not in raw
    return raw, evidence


def figure(a, b):
    import pypdfium2 as pdfium
    from reportlab.graphics import renderPDF
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib.colors import Color, black
    folder = TASK / "assets"
    folder.mkdir(exist_ok=True)
    path = folder / "perception_warm_latency.png"
    if path.exists():
        return path
    drawing = Drawing(604.8, 252)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 48, 44, 540, 172
    chart.data = [tuple(data['timings']['warm'][arm]['median_ms'] for arm in ARMS) for data in (a,b)]
    chart.categoryAxis.categoryNames = ["Native English CLI", "Native bilingual CLI", "Screen CLI", "Screen C API"]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 9
    chart.categoryAxis.labels.dy = -7
    chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = 0, 450, 100
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 9
    chart.bars[0].fillColor = Color(.35,.35,.35)
    chart.bars[1].fillColor = Color(.75,.75,.75)
    chart.bars.strokeColor, chart.bars.strokeWidth = black, .5
    chart.barSpacing, chart.groupSpacing = 3, 16
    drawing.add(chart)
    drawing.add(String(8,234,"Full-frame median (ms)",fontName="Helvetica",fontSize=11))
    for x,label,fill in [(385,"Primary",Color(.35,.35,.35)),(480,"Replication",Color(.75,.75,.75))]:
        drawing.add(Rect(x,230,10,10,fillColor=fill,strokeColor=black,strokeWidth=.5))
        drawing.add(String(x+15,230,label,fontName="Helvetica",fontSize=10))
    pdf = pdfium.PdfDocument(renderPDF.drawToString(drawing))
    bitmap = pdf[0].render(scale=240/72)
    bitmap.to_pil().save(path)
    bitmap.close()
    pdf.close()
    return path


def q(name):
    return "{"+W+"}"+name


def content(element):
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


def set_property(parent, name, **attrs):
    element = parent.find("w:"+name, NS)
    if element is None:
        element = E.SubElement(parent, q(name))
    element.attrib.clear()
    for key, value in attrs.items():
        element.set(q(key), str(value))
    return element


def build_docx(markdown, output, figure_path):
    assert sha(REFERENCE) == REFERENCE_SHA
    with zipfile.ZipFile(REFERENCE) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    root = E.fromstring(parts["word/document.xml"])
    body = root.find("w:body", NS)
    paragraphs = body.findall("w:p", NS)
    roles = {"title": paragraphs[0], "subtitle": paragraphs[1],
             "body": next(p for p in paragraphs if p.find("w:pPr/w:pStyle", NS) is None and content(p))}
    for name, style in [("h1", "Heading1"), ("h2", "Heading2")]:
        candidates = [p for p in paragraphs if p.find("w:pPr/w:pStyle", NS) is not None and p.find("w:pPr/w:pStyle", NS).get(q("val")) == style]
        if candidates:
            roles[name] = candidates[0]
        else:
            # The retained Week6 body uses H1 only. Its unused H2 style gets
            # the Week5 direct formatting already recorded in artifact.md.
            roles[name] = deepcopy(roles["h1"])
            set_property(roles[name].find("w:pPr",NS),"pStyle",val="Heading2")
            set_property(roles[name].find("w:pPr",NS),"spacing",before=220,after=100,line=380,lineRule="exact")
            rp=roles[name].find("w:r/w:rPr",NS)
            if rp is not None:
                set_property(rp,"sz",val=26);set_property(rp,"szCs",val=26)
    roles["caption"] = next(p for p in paragraphs if re.match(r"图\s*1", content(p)))
    roles["code"] = next(p for p in paragraphs if content(p).startswith("python scripts/"))
    sample_table = body.find("w:tbl", NS)
    roles["cell"] = sample_table.find("w:tr/w:tc/w:p", NS)
    sect = deepcopy(body.find("w:sectPr", NS))
    relationships = E.fromstring(parts["word/_rels/document.xml.rels"])
    ids = {r.get("Id") for r in relationships}
    next_id = 1

    def relation(kind, target, external=False):
        nonlocal next_id
        while f"rIdFull{next_id}" in ids:
            next_id += 1
        identity = f"rIdFull{next_id}"
        ids.add(identity)
        next_id += 1
        link = E.SubElement(relationships, "{"+P+"}Relationship", Id=identity,
                            Type=R+"/"+kind, Target=target)
        if external:
            link.set("TargetMode", "External")
        return identity

    def run(text, role="body", code=False):
        node = E.Element(q("r"))
        sample = roles["code" if code else role]
        rp = sample.find("w:r/w:rPr", NS)
        if rp is not None:
            node.append(deepcopy(rp))
        value = E.SubElement(node, q("t"))
        value.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        value.text = text
        return node

    def paragraph(text, role="body"):
        node = E.Element(q("p"))
        pp = roles[role].find("w:pPr", NS)
        if pp is not None:
            node.append(deepcopy(pp))
        for token in re.split(r"(`[^`]+`|\[[^\]]+\]\([^)]+\))", text):
            if not token:
                continue
            link = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)
            if link:
                hyperlink = E.SubElement(node, q("hyperlink"))
                hyperlink.set("{"+R+"}id", relation("hyperlink", link[2], True))
                child = run(link[1], role)
                rp = child.find("w:rPr", NS)
                if rp is None:
                    rp = E.Element(q("rPr")); child.insert(0, rp)
                set_property(rp, "u", val="single")
                hyperlink.append(child)
            elif token.startswith("`"):
                node.append(run(token[1:-1], role, True))
            else:
                node.append(run(token, role))
        return node

    def make_table(rows):
        count = len(rows[0])
        node = E.Element(q("tbl"))
        node.append(deepcopy(sample_table.find("w:tblPr", NS)))
        grid = E.SubElement(node, q("tblGrid"))
        weights = {3:[1.5,3.3,1.5],4:[1.5,1.1,1.2,2.5],5:[1.1,1.6,1.15,1.15,1.3]}.get(count, [1]*count)
        if rows[0] == ["记录", "观察到的问题", "处理与结论"]:
            weights = [1.2, 2.2, 2.9]
        elif rows[0] == ["配置", "冷中位 ms", "热中位 ms", "热p95 ms"]:
            weights = [2.4, 1.3, 1.3, 1.3]
        widths = [round(9072*w/sum(weights)) for w in weights]
        widths[-1] = 9072-sum(widths[:-1])
        for width in widths:
            E.SubElement(grid, q("gridCol")).set(q("w"), str(width))
        for index, row in enumerate(rows):
            assert len(row) == count
            tr = E.SubElement(node, q("tr"))
            trpr = E.SubElement(tr, q("trPr"))
            E.SubElement(trpr, q("cantSplit"))
            if index == 0:
                E.SubElement(trpr, q("tblHeader"))
            for col, value in enumerate(row):
                tc = E.SubElement(tr, q("tc"))
                source_row = sample_table.findall("w:tr", NS)[0 if index == 0 else 1]
                tc.append(deepcopy(source_row.find("w:tc/w:tcPr", NS)))
                set_property(tc.find("w:tcPr", NS), "tcW", w=widths[col], type="dxa")
                p = paragraph(value, "cell")
                pp = p.find("w:pPr", NS)
                if pp is None:
                    pp = E.Element(q("pPr")); p.insert(0, pp)
                set_property(pp, "jc", val="center" if index == 0 or len(value) < 15 else "left")
                set_property(pp, "spacing", before=0, after=0, line=320, lineRule="exact")
                keep_next = index < len(rows)-1 if len(rows) <= 10 else index <= 1
                set_property(pp, "keepNext", val=int(keep_next))
                for r in p.findall(".//w:r", NS):
                    rp = r.find("w:rPr", NS)
                    if rp is None:
                        rp = E.Element(q("rPr")); r.insert(0,rp)
                    set_property(rp,"rFonts",ascii="Times New Roman",hAnsi="Times New Roman",eastAsia="华文宋体")
                    set_property(rp,"sz",val=21); set_property(rp,"szCs",val=21)
                    set_property(rp,"b",val=int(index==0)); set_property(rp,"color",val="000000")
                tc.append(p)
        return node

    for node in list(body):
        body.remove(node)
    lines = markdown.splitlines()
    index = 0
    title_count = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1; continue
        if line.startswith("```"):
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                body.append(paragraph(lines[index], "code")); index += 1
        elif line.startswith("| "):
            rows = []
            while index < len(lines) and lines[index].startswith("| "):
                if not re.fullmatch(r"[| :\-]+", lines[index]):
                    rows.append([x.strip() for x in lines[index].strip("|").split("|")])
                index += 1
            body.append(make_table(rows)); continue
        elif line.startswith("!["):
            source = next(p for p in paragraphs if p.find(".//w:drawing", NS) is not None)
            drawing = deepcopy(source)
            drawing.find(".//a:blip", NS).set("{"+R+"}embed", relation("image", "media/week6_full_perception_latency.png"))
            from PIL import Image
            with Image.open(figure_path) as image:
                width, height = image.size
            cx, cy = 5760720, round(5760720*height/width)
            drawing.find(".//wp:extent", NS).set("cx", str(cx)); drawing.find(".//wp:extent", NS).set("cy", str(cy))
            for extent in drawing.findall(".//a:ext", NS):
                extent.set("cx", str(cx)); extent.set("cy", str(cy))
            docpr = drawing.find(".//wp:docPr", NS)
            docpr.set("descr", "两批热态完整帧处理中位耗时，四个预登记OCR配置")
            body.append(drawing)
            body.append(paragraph(re.match(r"!\[([^]]+)\]", line)[1], "caption"))
        elif line.startswith("### "):
            body.append(paragraph(line[4:], "h2"))
        elif line.startswith("## "):
            body.append(paragraph(line[3:], "h1"))
        elif line.startswith("# "):
            body.append(paragraph(line[2:], "title" if title_count == 0 else "subtitle")); title_count += 1
        else:
            body.append(paragraph(line))
        index += 1
    body.append(sect)
    parts["word/document.xml"] = E.tostring(root,encoding="UTF-8",xml_declaration=True,standalone=True)
    parts["word/_rels/document.xml.rels"] = E.tostring(relationships,encoding="UTF-8",xml_declaration=True,standalone=True)
    parts["word/media/week6_full_perception_latency.png"] = figure_path.read_bytes()
    core = E.fromstring(parts["docProps/core.xml"])
    for name,text in [("{http://purl.org/dc/elements/1.1/}subject","系统 v2.1.0 复杂任务分解与完整验收"),
                      ("{http://purl.org/dc/elements/1.1/}description","Week 6 语义目标执行 恢复 感知与运行监测实验"),
                      ("{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}keywords","Week 6 语义分解 恢复 感知 监测")]:
        element=core.find(name)
        if element is not None:
            element.text=text
    for name in ("created","modified"):
        element=core.find("{http://purl.org/dc/terms/}"+name)
        if element is not None:
            element.text=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts["docProps/core.xml"]=E.tostring(core,encoding="UTF-8",xml_declaration=True,standalone=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists()
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name,data)
    with zipfile.ZipFile(REFERENCE) as old, zipfile.ZipFile(output) as new:
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
    assert changed == [name for name in changed if name in {"word/document.xml","word/_rels/document.xml.rels","docProps/core.xml"}]
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts",type=Path,default=TASK/"facts_manifest.pending.json")
    parser.add_argument("--final",action="store_true")
    parser.add_argument("--docx",action="store_true")
    parser.add_argument("--iteration",default="draft_001")
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_]+",args.iteration)
    folder = TASK/args.iteration
    folder.mkdir(exist_ok=False)
    markdown,evidence = resolve_facts(read(args.facts),args.final)
    source = folder/"report.md"
    source.write_bytes(markdown.encode("utf-8"))
    result = {"created_utc":datetime.now(UTC).isoformat(),"facts":str(args.facts),
              "facts_sha256":sha(args.facts),"markdown":str(source),"markdown_sha256":sha(source),
              "reference_sha256":sha(REFERENCE),"final_named_output":args.final,"report_ready":args.final}
    if args.docx:
        name = "Week6_实验报告_完整验收版.docx" if args.final else "Week6_实验报告_验收待完成稿.docx"
        output = folder/name
        result.update(docx=str(output),changed_existing_parts=build_docx(markdown,output,figure(evidence['primary'],evidence['replication'])),docx_sha256=sha(output))
    save(folder/"build_manifest.json",result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
