"""Week5-derived private report builder; final mode requires the strict evidence gate."""
from __future__ import annotations

import argparse
import ast
import json
import re
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import build_report as retained

TASK = Path(__file__).resolve().parent
REPO = retained.REPO


def bound_json(reference):
    path = Path(reference['path'])
    if not path.is_absolute():
        path = REPO/path
    if retained.sha(path) != reference['sha256']:
        raise ValueError(f'changed evidence: {path}')
    return retained.read(path)


def private_markdown(manifest):
    if manifest.get('report_ready') is not False:
        raise ValueError('This preparation builder accepts pending private facts only')
    # The retained builder supplies verified perception/historical material.
    # Do not route v6 source records through its old v5 freeze-specific adapter.
    pending = deepcopy(manifest)
    pending['semantic_source_freeze'] = None
    pending['full_validation'] = None
    markdown, evidence = retained.resolve_facts(pending, False)
    if isinstance(manifest.get('current_v6_source_freeze'), dict):
        freeze_ref = manifest['current_v6_source_freeze']
        freeze = bound_json(freeze_ref)
        if (freeze['protocol'] != 'week6.semantic_subtasks.v6'
                or freeze['input_modality'] != 'caller_intent_only'
                or freeze['postcondition_source'] != 'untrusted_model_prediction'
                or freeze['final_source'] != 'unchanged_caller'):
            raise ValueError('Unexpected current v6 source contract')
        freeze_path = Path(freeze_ref['path'])
        if not freeze_path.is_absolute():
            freeze_path = REPO/freeze_path
        target_path = freeze_path.parent/freeze['targeted_validation']
        if retained.sha(target_path) != freeze['targeted_validation_sha256']:
            raise ValueError('Changed current targeted validation')
        target = retained.read(target_path)
        before = retained.read(target_path.parent/'tested_code_pins_before.json')
        after = retained.read(target_path.parent/'tested_code_pins_after.json')
        if before != after or after != freeze['code_pins'] or len(before) != freeze['code_pin_count']:
            raise ValueError('Current source/target pin disagreement')
        if not all(retained.sha(REPO/name) == digest for name, digest in before.items()):
            raise ValueError('Current source changed after freeze')
        if target['checks_passed'] is not True or target['full_regression'] is not False:
            raise ValueError('Targeted validation must remain distinct from full regression')
        for receipt in ('pytest.receipt.json', 'ruff.receipt.json'):
            if retained.read(target_path.parent/receipt)['actual_child_exit_code'] != 0:
                raise ValueError('Current target receipt did not pass')
        evidence['current_v6_source_freeze'] = freeze
        evidence['current_v6_targeted_validation'] = target
    prose = bound_json(manifest['report_v6_prose'])
    if prose['report_ready'] is not False:
        raise ValueError('Private prose must not assert report readiness')
    for reference in prose['evidence_bindings']:
        bound_json(reference)
    old_sections = re.split(r'(?m)(?=^## )', markdown)
    replaced = set()
    result = []
    for section in old_sections:
        heading = section.splitlines()[0] if section.startswith('## ') else None
        if heading in prose['section_overrides']:
            result.append(prose['section_overrides'][heading].rstrip()+'\n\n')
            replaced.add(heading)
        else:
            result.append(section)
    if replaced != set(prose['section_overrides']):
        raise ValueError('Missing reviewed section override')
    markdown = ''.join(result)
    for original, replacement in prose['exact_replacements'].items():
        if markdown.count(original) != 1:
            raise ValueError('Private prose replacement is not unique')
        markdown = markdown.replace(original, replacement, 1)
    for appendix in prose.get('appendix_sections', []):
        if not isinstance(appendix, str) or not appendix.startswith('## 附录'):
            raise ValueError('Invalid private appendix')
        markdown = markdown.rstrip()+'\n\n'+appendix.rstrip()+'\n'
    for heading in ('## 4 界面识别优化方法', '## 5 新留出上的准确率与完整帧耗时'):
        original = next(section for section in old_sections if section.startswith(heading+'\n'))
        current = next(section for section in re.split(r'(?m)(?=^## )', markdown) if section.startswith(heading+'\n'))
        if original != current:
            raise ValueError('Frozen perception prose/table changed unexpectedly')
    evidence['private_v6_prose'] = prose
    return markdown, evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--facts', type=Path, default=TASK/'facts_manifest.pending.json')
    parser.add_argument('--iteration', default='draft_018')
    parser.add_argument('--docx', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--engineering-gate', type=Path)
    parser.add_argument('--final-prose', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.iteration):
        parser.error('Invalid iteration name')
    if args.final:
        import final_evidence_gate as final_gate
        if args.engineering_gate is None or args.final_prose is None:
            parser.error('--final requires --engineering-gate and --final-prose original files')
        folder = TASK/args.iteration
        if folder.exists():
            parser.error('Exclusive new iteration directory required')
        folder.mkdir()
        audit_work = folder/'gate_replay'
        audit_work.mkdir()
        try:
            evidence = final_gate.validate(args.engineering_gate, audit_work)
            markdown = final_gate.final_markdown(evidence, args.final_prose)
            # Check again after saved-evidence replay/prose assembly, before artifact write.
            final_gate.validate_pins(evidence['current_code_pins'])
            result = {
                'schema':'week6.full_report.final_build.v1',
                'engineering_gate':str(args.engineering_gate),
                'engineering_gate_sha256':final_gate.digest(args.engineering_gate),
                'final_prose':str(args.final_prose), 'final_prose_sha256':final_gate.digest(args.final_prose),
                'builder_sha256':final_gate.digest(Path(__file__)),
                'gate_checker_sha256':final_gate.digest(Path(final_gate.__file__)),
                'retained_layout_builder_sha256':retained.sha(TASK/'build_report.py'),
                'reference_sha256':retained.sha(retained.REFERENCE),
                'report_generation_gate_passed':True, 'overall_delivery_complete':False,
                'report_ready_for_delivery':False,
                'delivery_status':'report_pages_and_package_qa_pending',
                'numerical_facts':{'full':evidence['full'], 'live':evidence['live']},
                'no_cover_date_or_timezone_line':True, 'all_page_visual_qa_required':True,
                'check_only':args.check_only,
            }
            if not args.check_only:
                source=folder/'report.md'
                source.write_bytes(markdown.encode('utf-8'))
                result.update(markdown=str(source), markdown_sha256=retained.sha(source))
                if args.docx:
                    output=folder/'Week6_实验报告_完整验收版.docx'
                    figure=final_gate.bound(evidence['gate']['figure'])
                    result.update(docx=str(output), changed_existing_parts=retained.build_docx(markdown,output,figure),
                                  docx_sha256=retained.sha(output),
                                  expected_pdf_name='Week6_实验报告_完整验收版.pdf', rendered=False)
            retained.save(folder/'build_manifest.json',result)
            print(json.dumps(result,ensure_ascii=True,indent=2))
        except (ValueError,OSError,KeyError,TypeError,IndexError,AssertionError) as exc:
            retained.save(folder/'gate_rejection.json',{'gate_passed':False,'error_type':type(exc).__name__,
                'error':str(exc),'final_artifact_created':any(folder.glob('*.docx')),
                'overall_delivery_complete':False})
            parser.exit(2,f'Final report refused: {exc}\n')
        raise SystemExit(0)
    if args.engineering_gate or args.final_prose or args.check_only:
        parser.error('Final gate arguments require --final')
    facts = retained.read(args.facts)
    markdown, evidence = private_markdown(facts)
    folder = TASK/args.iteration
    folder.mkdir(exist_ok=False)
    source = folder/'report.md'
    source.write_bytes(markdown.encode('utf-8'))
    result = {
        'schema': 'week6.full_report.private_v6_build.v1',
        'created_utc': datetime.now(UTC).isoformat(),
        'facts':str(args.facts), 'facts_sha256':retained.sha(args.facts),
        'markdown':str(source), 'markdown_sha256':retained.sha(source),
        'builder_sha256':retained.sha(Path(__file__)),
        'retained_layout_builder_sha256':retained.sha(TASK/'build_report.py'),
        'reference_sha256':retained.sha(retained.REFERENCE),
        'report_ready':False, 'final_named_output':False, 'rendered':False,
        'no_cover_date_or_timezone_line':True,
    }
    if args.docx:
        output = folder/'Week6_实验报告_验收待完成稿.docx'
        result.update(docx=str(output), changed_existing_parts=retained.build_docx(
            markdown, output, retained.figure(evidence['primary'], evidence['replication'])),
            docx_sha256=retained.sha(output))
    ast.parse(Path(__file__).read_text('utf-8'))
    retained.save(folder/'build_manifest.json', result)
    print(json.dumps(result, ensure_ascii=True, indent=2))
