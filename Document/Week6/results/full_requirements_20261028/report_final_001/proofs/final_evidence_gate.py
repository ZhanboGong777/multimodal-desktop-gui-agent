"""Private, read-only report gate. No model/GUI/OCR/test execution is imported.

Saved live verdicts are independently rerun by the SHA-bound project auditor
only when the eventual final gate is used. A pending specification cannot pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1] / 'week6-worktree'
SCHEMA = 'week6.report.engineering_gate.v1'
REFERENCE_SHA = 'f117680176cb238aa18b69b2d39557313a1b5f916ad2bf2106403cb403172218'
BASE_SHA = 'e6adfe28bcf8ec5781fe5ca8de2e9ea30b9e40cb6589446135f154a1f8fe9440'
FIGURE_SHA = '0461ab0ed4948f3897754c402686894cfb2bd2afe3ca8758bf0600f67ff89353'
ROLES = ('normal', 'transient')
PERCEPTION_SHAS = {
    'primary':'cfa65ae176a1c7d593890520e8032938798c66beb48546d668139d56c80551d2',
    'replication':'36c5e17519516967d5eb7f02ed6d159eb6f6afeccd7eaf4fbe1d42ff76b584d9',
    'raw_audit':'2d31c6da900193374131f03db2ad8aa8b229c17a455230a678b54bf59d2e23df',
    'source_freeze':'3a1d7e69f7508d23e39aee728a2830e452e3194bd7d0c3da0f00e02a2ffda641',
    'external':'8fbece4eba052538c6916225281dbebc3abef02f2940dd95bd32ad9bcbf1fa4a',
    'external_raw_audit':'604d5c537f5fff6d8b3edaa0ef89febe2711b75297b719f14bf67378d975b3da',
}
HEADINGS = ('## 1 工作与验收范围', '## 2 语义任务分解与顺序执行',
            '## 3 错误恢复与运行监测', '## 6 真实窗口上的分解与恢复', '## 7 测试结果与失败记录')


class GateError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise GateError(code)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate_json_key')
        value[key] = item
    return value


def load(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(GateError('nonfinite_json')))


def validate_shape(value, schema, document=None, location='$'):
    """Small strict interpreter for the bundled schema's deliberately limited vocabulary."""
    document = schema if document is None else document
    if '$ref' in schema:
        require(schema['$ref'].startswith('#/$defs/'), 'nonlocal_schema_reference')
        schema = document['$defs'][schema['$ref'].split('/')[-1]]
    if 'const' in schema:
        require(type(value) is type(schema['const']) and value == schema['const'], f'schema_const:{location}')
    if 'enum' in schema:
        require(any(type(value) is type(item) and value == item for item in schema['enum']), f'schema_enum:{location}')
    kind = schema.get('type')
    types = {'object':dict, 'array':list, 'string':str, 'integer':int, 'number':(int,float), 'boolean':bool}
    if kind:
        accepted = types[kind]
        require((type(value) in accepted if type(accepted) is tuple else type(value) is accepted),
                f'schema_type:{location}')
    if kind == 'object':
        require(set(schema.get('required', [])) <= set(value), f'schema_required:{location}')
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            require(set(value) <= set(properties), f'schema_unknown_field:{location}')
        for key, item in value.items():
            if key in properties:
                validate_shape(item, properties[key], document, location+'.'+key)
    if kind == 'array':
        require(len(value) >= schema.get('minItems', 0) and len(value) <= schema.get('maxItems', len(value)),
                f'schema_array_length:{location}')
        for index, item in enumerate(value):
            validate_shape(item, schema.get('items', {}), document, f'{location}[{index}]')
    if kind == 'string':
        require(len(value) >= schema.get('minLength', 0), f'schema_string_empty:{location}')
        if 'pattern' in schema:
            require(re.fullmatch(schema['pattern'], value) is not None, f'schema_string_pattern:{location}')


def stamp(text):
    value = datetime.fromisoformat(text)
    require(value.tzinfo is not None, 'timestamp_has_no_timezone')
    return value


def finite(value, minimum=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def bound(reference, *, root=REPO):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}, 'invalid_file_reference')
    require(type(reference['path']) is str and reference['path'], 'missing_file_path')
    require(type(reference['sha256']) is str and re.fullmatch('[0-9a-f]{64}', reference['sha256']),
            'missing_or_invalid_sha256')
    path = Path(reference['path'])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path.is_file(), f'missing_file:{path.name}')
    require(digest(path) == reference['sha256'], f'file_sha256_changed:{path.name}')
    return path


def bound_json(reference):
    return load(bound(reference))


def inventory(repo):
    return ({p.relative_to(repo).as_posix() for folder in ('src', 'tests', 'scripts')
             for p in (repo / folder).rglob('*.py')}
            | {p.relative_to(repo).as_posix() for p in (repo / 'configs').rglob('*')
               if p.is_file() and p.suffix in ('.yaml', '.yml', '.json')}
            | {'pyproject.toml'})


def validate_pins(pins, *, repo=REPO):
    require(type(pins) is dict and pins and set(pins) == inventory(repo), 'incomplete_code_inventory')
    for name, expected in pins.items():
        target = (repo / name).resolve()
        require(target.is_relative_to(repo.resolve()) and target.is_file(), 'pin_path_outside_repository')
        require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected)
                and digest(target) == expected, f'current_code_changed:{name}')


def receipt(reference, kind):
    path = bound(reference)
    value = load(path)
    exit_key = 'child_exit_code' if kind == 'live' else 'actual_child_exit_code'
    require(type(value.get(exit_key)) is int and value[exit_key] == 0, f'{kind}_actual_wait_not_zero')
    require(stamp(value['finished_utc']) >= stamp(value['started_utc'])
            and finite(value['elapsed_seconds']), f'{kind}_invalid_process_interval')
    launcher = value['launcher']
    require(type(launcher.get('pid')) is int and launcher['pid'] > 0
            and finite(launcher.get('create_time'), 1), f'{kind}_process_identity_missing')
    command = value['command']
    require(type(command) is list and all(type(x) is str for x in command), 'invalid_receipt_command')
    if kind == 'pytest':
        require(len(command) == 6 and command[1:5] == ['-m', 'pytest', '-q', '--basetemp'],
                'pytest_receipt_is_not_full_unfiltered_execution')
    elif kind == 'ruff':
        require(command[1:] == ['-m', 'ruff', 'check', 'src', 'tests', 'scripts'], 'ruff_scope_changed')
    return value


def pytest_summary(text):
    # A single terminal pytest summary with no failures/errors/skips/deselections.
    summaries = re.findall(r'(?m)^\s*(?:=+\s*)?(\d+) passed in ([0-9.]+)s(?:\s*\([^\n]*\))?\s*(?:=+)?\s*$', text)
    require(len(summaries) == 1 and not re.search(
        r'(?m)^.*\b\d+ (?:failed|errors?|skipped|deselected|xfailed|xpassed)\b.*(?:in [0-9.]+s|=+\s*$)', text),
        'pytest_full_summary_missing_or_not_all_passed')
    count, seconds = summaries[0]
    require(int(count) > 0 and finite(float(seconds)), 'invalid_pytest_summary')
    return int(count), float(seconds)


def _full_auxiliary_identity(full, freeze, validation_directory, collection_directory, *, repo=REPO):
    """New freezes bind auxiliary inputs throughout the full run; old records stay historical."""
    if 'requirements_dev_sha256' not in freeze:
        return
    for key, name in (('requirements_dev', 'requirements-dev.txt'), ('readme', 'README.md')):
        expected = freeze.get(key + '_sha256')
        require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected)
                and full.get(key + '_sha256_before') == full.get(key + '_sha256_after') == expected
                and digest(repo / name) == expected, f'full_{key}_before_after_freeze_current_disagree')
        stem, suffix = ('requirements-dev', '.txt') if key == 'requirements_dev' else ('README', '.md')
        for phase in ('before', 'after'):
            original = validation_directory / (stem + '.' + phase + suffix)
            require(original.is_file() and digest(original) == expected,
                    f'full_auxiliary_original_copy_changed:{original.name}')
    before = collection_directory / 'tested_code_pins_before.json'
    after = collection_directory / 'tested_code_pins_after.json'
    require(before.is_file() and after.is_file(), 'collection_source_pins_missing')
    require(load(before) == load(after) == freeze['code_pins'], 'collection_sources_differ_from_full_freeze')


def full_regression(refs, freeze):
    required = {'validation', 'before', 'after', 'pytest_receipt', 'ruff_receipt', 'pytest_stdout',
                'ruff_stdout', 'collection_receipt', 'collection_stdout', 'tested_readme'}
    require(type(refs) is dict and set(refs) == required, 'full_regression_reference_set_incomplete')
    paths = {key: bound(value) for key, value in refs.items()}
    full, before, after = (load(paths[key]) for key in ('validation', 'before', 'after'))
    require(full.get('schema') == 'week6.full_regression.actual.v1'
            and full.get('checks_passed') is True and full.get('code_unchanged') is True,
            'not_actual_full_regression')
    require(before == after == freeze['code_pins'] and full['code_pin_count'] == len(before),
            'full_regression_not_bound_to_current_freeze')
    validate_pins(before)
    py, ruff = receipt(refs['pytest_receipt'], 'pytest'), receipt(refs['ruff_receipt'], 'ruff')
    require(full.get('results') == [py, ruff], 'full_validation_embedded_receipts_changed')
    for record in (py, ruff):
        require(stamp(full['started_utc']) <= stamp(record['started_utc'])
                <= stamp(record['finished_utc']) <= stamp(full['finished_utc']), 'receipt_outside_full_owner')
    passed, seconds = pytest_summary(paths['pytest_stdout'].read_text('utf-8'))
    require(paths['ruff_stdout'].read_text('utf-8').strip() == 'All checks passed!', 'ruff_stdout_not_clean')
    coll = load(paths['collection_receipt'])
    require(type(coll.get('actual_child_exit_code')) is int and coll['actual_child_exit_code'] == 0
            and '--collect-only' in coll['command'], 'collection_receipt_invalid')
    matches = re.findall(r'(?m)^(\d+) tests? collected in [0-9.]+s', paths['collection_stdout'].read_text('utf-8'))
    require(len(matches) == 1 and int(matches[0]) == passed, 'collection_count_not_actual_passed_count')
    require(digest(paths['tested_readme']) == freeze['readme_sha256'] == digest(REPO / 'README.md'),
            'tested_readme_differs_from_frozen_current')
    if 'requirements_dev_sha256' in freeze:
        require(paths['collection_receipt'].parent == paths['collection_stdout'].parent,
                'collection_receipt_stdout_directories_disagree')
        _full_auxiliary_identity(full, freeze, paths['validation'].parent,
                                 paths['collection_receipt'].parent)
    return {'passed': passed, 'pytest_seconds': seconds, 'pytest_owned_seconds': py['elapsed_seconds'],
            'owner_seconds': full['elapsed_seconds'], 'pins': len(before), 'pytest_exit': 0, 'ruff_exit': 0}


def replay_auditor(live, auditor_ref, python_ref, work):
    """Rerun only the pinned saved-evidence checker; never the experiment/harness."""
    script, python = bound(auditor_ref), bound(python_ref)
    require(script == (REPO / 'scripts/week6_audit_semantic_acceptance.py').resolve(), 'wrong_live_auditor')
    container = Path(live['container']).resolve()
    fixture = Path(live['fixture_directory']).resolve()
    require(container.is_relative_to(REPO) and fixture.is_relative_to(REPO), 'live_directory_outside_repository')
    owner_harness = bound(live['owner_harness'])
    saved = bound_json(live['independent_audit'])
    require(saved.get('passed') is True and saved.get('auditor_sha256') == auditor_ref['sha256']
            and Path(saved['container']).resolve() == container, 'saved_live_audit_not_bound')
    audit_wait = receipt(live['audit_receipt'], 'audit')
    actual = audit_wait['command']
    require(len(actual) == 10 and Path(actual[0]).resolve() == python and Path(actual[1]).resolve() == script
            and actual[2::2] == ['--container', '--owner-harness', '--fixture-directory', '--out']
            and Path(actual[3]).resolve() == container and Path(actual[5]).resolve() == owner_harness
            and Path(actual[7]).resolve() == fixture
            and Path(actual[9]).resolve() == bound(live['independent_audit']), 'saved_audit_command_not_pinned')
    output = work / f"{live['role']}_replayed_audit.json"
    command = [str(python), str(script), '--container', str(container), '--owner-harness', str(owner_harness),
               '--fixture-directory', str(fixture), '--out', str(output)]
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, check=False, timeout=180)
    except subprocess.TimeoutExpired as exc:
        (work / f"{live['role']}_auditor.stdout").write_bytes(exc.stdout or b'')
        (work / f"{live['role']}_auditor.stderr").write_bytes(exc.stderr or b'')
        raise GateError('saved_live_auditor_replay_timed_out') from exc
    (work / f"{live['role']}_auditor.stdout").write_bytes(result.stdout)
    (work / f"{live['role']}_auditor.stderr").write_bytes(result.stderr)
    require(result.returncode == 0 and output.is_file(), 'saved_live_auditor_replay_failed')
    replayed = load(output)
    require(replayed == saved, 'independent_live_audit_no_longer_reproduces')
    return replayed


def manual_meaning_review(reference, live, audit, freeze_ref):
    review = bound_json(reference)
    require(review.get('schema') == 'week6.report.semantic_operation_review.v1'
            and review.get('reviewer') == 'Root' and review.get('approved') is True
            and review.get('source_freeze') == freeze_ref
            and review.get('independent_audit') == live['independent_audit']
            and review.get('runtime') == live['runtime']
            and Path(review['container']).resolve() == Path(live['container']).resolve(),
            'missing_bound_root_semantic_review')
    stages = audit['native_semantic_stages']
    rows = review.get('stages')
    require(type(rows) is list and len(rows) == len(stages) and len(rows) >= 2, 'incomplete_semantic_operation_review')
    for row, stage in zip(rows, stages, strict=True):
        require(row.get('stage_key') == stage['stage_key'] and row.get('native_instruction') == stage['instruction']
                and row.get('independent_transitions') == stage['independent_transitions']
                and row.get('semantic_alignment_passed') is True
                and type(row.get('rationale')) is str and bool(row['rationale'].strip()),
                'native_instruction_event_meaning_not_reviewed')
    return review


def live_monitor(reference, live, owner, runtime):
    proof = bound_json(reference)
    require(proof.get('schema') == 'week6.report.live_monitor_review.v1'
            and proof.get('root_reviewed') is True and proof.get('runtime') == live['runtime'], 'monitor_proof_unbound')
    root_path, child_path = bound(proof['parent_events']), bound(proof['child_events'])
    container = Path(live['container']).resolve()
    require(root_path == (container / 'run/automatic/status_events.jsonl').resolve()
            and child_path.is_relative_to(container / 'run/automatic'), 'monitor_proof_wrong_scope')
    parent = next((load_line for load_line in map(json.loads, root_path.read_text('utf-8').splitlines())
                   if load_line['seq'] == proof['parent_seq']), None)
    child = next((load_line for load_line in map(json.loads, child_path.read_text('utf-8').splitlines())
                  if load_line['seq'] == proof['child_seq']), None)
    index = proof['goal_outcome_index']
    require(type(index) is int and 0 <= index < len(runtime['goal_outcomes']), 'monitor_goal_index_invalid')
    goal = runtime['goal_outcomes'][index]
    require(child_path.is_relative_to(Path(goal['evidence_directory']).resolve()), 'monitor_child_is_not_reviewed_goal')
    actions_offset = sum(row['total_actions'] for row in runtime['workflow_results'][:index])
    planning_offset = (sum(row['planning_calls'] for row in runtime['workflow_results'][:index])
                       + sum(row['planning_calls'] for row in runtime['plans'][:goal['round']]))
    completed_before = sum(row['status'] == 'succeeded' for row in runtime['goal_outcomes'][:index])
    require(proof['actions_offset'] == actions_offset and proof['planning_offset'] == planning_offset
            and proof['verified_completed_before_child'] == completed_before, 'monitor_offsets_not_original_ledger')
    require(parent and child and parent['stage'] == 'executing' and parent['task_status'] is None
            and parent['subgoal_id'] == goal['goal_id']
            and parent['detail_code'] == 'child_' + child['stage']
            and child['actions_used'] > 0 and parent['actions_used'] < runtime['total_actions']
            and parent['actions_used'] == proof['actions_offset'] + child['actions_used']
            and parent['planning_pass'] == proof['planning_offset'] + child['planning_pass']
            and parent['subgoals_completed'] == proof['verified_completed_before_child'], 'no_midrun_global_monitor_evidence')
    require(stamp(owner['started_utc']) <= stamp(child['updated_at_utc'])
            <= stamp(parent['updated_at_utc']) <= stamp(owner['finished_utc']), 'monitor_event_outside_real_run')
    return {'parent_seq': parent['seq'], 'child_seq': child['seq'], 'actions': parent['actions_used'],
            'planning': parent['planning_pass'], 'completed': parent['subgoals_completed']}


def live_run(live, gate, freeze, work):
    require(set(live) == {'role', 'container', 'fixture_directory', 'owner_harness', 'owner_receipt', 'runtime',
                         'independent_audit', 'audit_receipt', 'root_semantic_review', 'midrun_monitor'},
            'incomplete_live_reference_set')
    require(live['role'] in ROLES, 'wrong_live_role')
    base = Path(live['container']).resolve()
    require(bound(live['owner_receipt']) == base / 'owner_receipt.json'
            and bound(live['runtime']) == base / 'run/runtime_result.json', 'live_primary_paths_not_bound')
    owner, runtime = receipt(live['owner_receipt'], 'live'), bound_json(live['runtime'])
    require(runtime['status'] == 'succeeded' and runtime['accounting_complete'] is True
            and runtime['unconfirmed_action_attempts'] == 0 and runtime['total_actions'] > 0
            and runtime['final_verification']['outcome'] == 'passed', 'runtime_not_confirmed_final_success')
    command = owner['command']
    require(command[command.index('--mode') + 1] == live['role'], 'live_role_differs_from_actual_command')
    contract = load(base / 'run/contract.json')
    if freeze.get('runtime_and_configuration_unchanged_from_gui') is True:
        original = bound_json(freeze['original_gui_freeze'])
        changed = [name for name, value in freeze['code_pins'].items()
                   if original['code_pins'].get(name) != value]
        require(changed == ['scripts/week6_audit_semantic_acceptance.py']
                and set(original['code_pins']) == set(freeze['code_pins'])
                and original['readme_sha256'] == freeze['readme_sha256']
                and original['runtime_model_digest'] == freeze['runtime_model_digest']
                and 'scripts/week6_audit_semantic_acceptance.py' not in contract['source_sha256']
                and all(original['code_pins'].get(name) == value
                        for name, value in contract['source_sha256'].items()),
                'auditor_maintenance_changed_executed_runtime_or_model')
    require(contract['model'] == freeze['runtime_model'], 'live_model_not_current_candidate')
    if 'runtime_text_model' in freeze:
        require(contract.get('text_model') == freeze['runtime_text_model'], 'live_text_model_not_current_candidate')
    for name, expected in contract['source_sha256'].items():
        require(freeze['code_pins'].get(name) == expected, 'live_sources_differ_from_full_freeze')
    audit = replay_auditor(live, gate['auditor'], gate['audit_python'], work)
    require(audit['actual_owner_child_exit_code'] == 0 and audit['unconfirmed_action_attempts'] == 0
            and audit['known_actions'] == runtime['total_actions']
            and audit['planning_calls'] == runtime['planning_calls']
            and audit['actual_model_requests'] == runtime['model_requests'], 'audit_runtime_count_disagreement')
    manual_meaning_review(live['root_semantic_review'], live, audit, gate['source_freeze'])
    monitor = live_monitor(live['midrun_monitor'], live, owner, runtime)
    return {'container': str(base), 'name': base.name, 'actions': audit['known_actions'],
            'planning': audit['planning_calls'], 'http': audit['actual_model_requests'],
            'owner_seconds': owner['elapsed_seconds'], 'stages': len(audit['native_semantic_stages']),
            'native_stages': audit['native_semantic_stages'], 'transitions': audit['transition_attribution'],
            'monitor': monitor, 'model': contract['model'],
            'text_model': contract.get('text_model', contract['model'])}


def perception(refs):
    require(set(refs) == {'primary', 'replication', 'raw_audit', 'source_freeze', 'external', 'external_raw_audit'},
            'perception_references_incomplete')
    require(all(refs[key]['sha256'] == expected for key, expected in PERCEPTION_SHAS.items()),
            'frozen_perception_originals_replaced')
    values = {key: bound_json(value) for key, value in refs.items()}
    audit, freeze = values['raw_audit'], values['source_freeze']
    require(audit['both_batches_passed'] is True
            and audit['source_freeze_sha256'] == refs['source_freeze']['sha256'], 'perception_audit_not_bound')
    for key in ('primary', 'replication'):
        summary = values[key]
        require(summary['all_criteria_passed'] is True, 'perception_frozen_batch_failed')
        entry = audit['batches'][bound(refs[key]).parent.name]
        require(entry['summary_sha256'] == refs[key]['sha256'] and entry['genuine_exit'] == 0
                and entry['records_rebuilt'] == 364 and entry['screen_equivalent_pairs'] == 91,
                'perception_original_batch_not_audited')
    compatibility = []
    for name, expected in freeze['source_pins'].items():
        current = REPO / name
        if digest(current) == expected:
            continue
        require(name.replace(chr(92), '/') == 'src/gui_agent/config.py', 'perception_source_changed_since_measurement')
        archived = REPO / 'Document/Week6/results/full_requirements_20261028/candidate_visual_details_001/source_before/src/gui_agent/config.py'
        require(digest(archived) == expected, 'perception_original_config_archive_not_bound')
        original = archived.read_bytes()
        needle = b'    allow_real_execution: bool = False\n\n\nclass ExecutionConfig'
        addition = b'    allow_real_execution: bool = False\n    visual_candidate_details: bool = False\n\n\nclass ExecutionConfig'
        require(original.count(needle) == 1 and current.read_bytes() == original.replace(needle, addition, 1),
                'perception_config_changed_beyond_exact_non_ocr_planning_flag')
        compatibility.append({'path': name, 'measured_sha256': expected,
            'current_sha256': digest(current), 'archive_sha256': digest(archived),
            'scope': 'Only one optional PlanningConfig visual_candidate_details flag was inserted. OCR config and all other bytes are unchanged. Original perception timings exclude the later model image-detail attachment cost.'})
    values['source_compatibility'] = compatibility
    # Runtime dependencies and external audit remain bound original evidence.
    for name, expected in freeze['runtime_pins'].items():
        require(digest(Path(name)) == expected, 'perception_runtime_changed_since_measurement')
    external = values['external_raw_audit']
    require(external['summary_sha256'] == refs['external']['sha256']
            and external['actual_child_wait_returncode'] == 0 and external['all_assertions_passed'] is True
            and external['records_rebuilt'] == 56 and external['full_frame_precision'] is None
            and external['full_frame_recall'] is None, 'external_perception_claim_exceeds_audited_scope')
    return values


def validate_text_model_identity(freeze, metadata, tags):
    """Additional identity checks for an explicitly paired candidate."""
    paired = 'runtime_text_model' in freeze or 'runtime_text_model_digest' in freeze
    if paired:
        require(isinstance(freeze.get('runtime_text_model'), str) and freeze['runtime_text_model']
                and isinstance(freeze.get('runtime_text_model_digest'), str)
                and re.fullmatch('[0-9a-f]{64}', freeze['runtime_text_model_digest']),
                'incomplete_text_model_freeze_identity')
        require(metadata.get('text_model') == freeze['runtime_text_model']
                and metadata.get('text_digest') == freeze['runtime_text_model_digest'],
                'text_model_candidate_identity_changed')
        require(any(row.get('name') == metadata['text_model']
                    and row.get('digest', '').removeprefix('sha256:') == metadata['text_digest']
                    for row in tags['models']), 'text_model_digest_not_in_saved_actual_metadata')


def validate(gate_path, work):
    gate = load(gate_path)
    schema_path = TASK/'engineering_gate.schema.json'
    validate_shape(gate, load(schema_path))
    require(gate['gate_spec_sha256'] == digest(schema_path), 'gate_specification_hash_changed')
    require(gate.get('schema') == SCHEMA and gate.get('engineering_accepted') is True
            and gate.get('report_generation_allowed') is True
            and gate.get('overall_delivery_complete') is False
            and gate.get('delivery_status') == 'report_and_package_pending', 'engineering_gate_is_pending_or_not_report_only')
    require(gate.get('requirements') == {'semantic_decomposition': True, 'bounded_recovery': True,
                                       'perception_accuracy_and_fullframe_time': True, 'live_monitoring': True},
            'not_all_four_requirements_accepted')
    freeze = bound_json(gate['source_freeze'])
    require(freeze['protocol'] == 'week6.semantic_subtasks.v6' and freeze['input_modality'] == 'caller_intent_only'
            and freeze['postcondition_source'] == 'untrusted_model_prediction'
            and freeze['final_source'] == 'unchanged_caller', 'unsupported_final_semantic_contract')
    validate_pins(freeze['code_pins'])
    require(freeze['code_pin_count'] == len(freeze['code_pins']), 'freeze_pin_count_incorrect')
    require(bound(gate['readme']) == (REPO / 'README.md').resolve()
            and gate['readme']['sha256'] == freeze['readme_sha256'], 'root_readme_not_tested_version')
    metadata = bound_json(gate['model_metadata'])
    require(metadata['schema'] == 'week6.report.model_identity.v1'
            and metadata['model'] == freeze['runtime_model']
            and metadata['digest'] == freeze['runtime_model_digest'], 'model_candidate_identity_changed')
    tags = bound_json(metadata['tags'])
    require(any(row.get('name') == metadata['model'] and row.get('digest', '').removeprefix('sha256:') == metadata['digest']
                for row in tags['models']), 'model_digest_not_in_saved_actual_metadata')
    validate_text_model_identity(freeze, metadata, tags)
    full = full_regression(gate['full_regression'], freeze)
    perception_values = perception(gate['perception'])
    require(type(gate['live']) is list and sorted(x['role'] for x in gate['live']) == sorted(ROLES), 'need_unique_normal_and_transient')
    runs = {item['role']: live_run(item, gate, freeze, work) for item in gate['live']}
    require(runs['normal']['container'] != runs['transient']['container'], 'same_live_batch_used_twice')
    base = bound(gate['base_markdown'])
    require(digest(base) == BASE_SHA, 'report_history_base_changed')
    require(gate['reference_docx']['sha256'] == REFERENCE_SHA, 'reference_docx_is_not_user_revision')
    require(bound(gate['reference_docx']) == (TASK.parents[1]/'output/docx/Week6_实验报告_文字修订版_修正.docx').resolve(),
            'reference_docx_not_original_user_file')
    require(gate['figure']['sha256'] == FIGURE_SHA, 'original_perception_figure_replaced')
    bound(gate['figure'])
    links = re.findall(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', base.read_text('utf-8'))
    require([(row['label'], row['url']) for row in gate['document_links']] == links, 'report_links_changed')
    for row in gate['document_links']:
        bound(row['local_target'])
    return {'gate': gate, 'freeze': freeze, 'full': full, 'live': runs, 'perception': perception_values,
            'base_markdown': base.read_text('utf-8'),
            'gate_sha256': digest(gate_path), 'current_code_pins': freeze['code_pins']}


def sections(markdown):
    return re.split(r'(?m)(?=^## )', markdown)


def protected_components(markdown):
    real_image_section = next(s for s in sections(markdown) if s.startswith(HEADINGS[3]+'\n'))
    external_index = real_image_section.find('另使用normal001初始保存的2560×1600画面')
    require(external_index >= 0, 'external_perception_prose_missing')
    return {
        'perception_sections': [s for s in sections(markdown) if s.startswith((
            '## 4 界面识别优化方法\n', '## 5 新留出上的准确率与完整帧耗时\n'))],
        'code': re.findall(r'```[^\n]*\n.*?```', markdown, re.S),
        'links': re.findall(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', markdown),
        'figures': re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', markdown),
        'history': [s for s in sections(markdown) if s.startswith('## 附录')],
        'external_perception': real_image_section[external_index:],
    }


def final_markdown(evidence, prose_path):
    prose = load(prose_path)
    require(prose.get('schema') == 'week6.report.final_prose.v1' and prose.get('root_reviewed') is True
            and prose.get('engineering_gate_sha256') == evidence['gate_sha256'], 'final_prose_not_reviewed_against_same_gate')
    require(set(prose['architecture_sections']) == set(HEADINGS[1:3]), 'final_architecture_sections_missing')
    base = evidence['base_markdown']
    normal, transient = evidence['live']['normal'], evidence['live']['transient']
    full = evidence['full']
    def grid(head, rows):
        clean = lambda x: str(x).replace('|', '/') .replace('\n', ' ')
        return '\n'.join(['| '+' | '.join(map(clean, head))+' |',
                          '| '+' | '.join('---' for _ in head)+' |'] +
                         ['| '+' | '.join(map(clean, row))+' |' for row in rows])
    intro = (
        f'{HEADINGS[0]}\n\n系统接收完整任务后，模型先提出有顺序的语义目标，再依据当前画面逐项执行。'
        '目标条件在输入前固定，任务结束时另取新画面检查用户最初的要求。本文分别报告任务分解、错误恢复、界面识别及运行监测。\n\n'
        f'同一冻结源码的正常和恢复试验均通过。正常试验有{normal["actions"]}次输入、{normal["planning"]}次规划和{normal["http"]}次HTTP；'
        f'恢复试验分别为{transient["actions"]}、{transient["planning"]}和{transient["http"]}次。'
        f'完整回归为{full["passed"]}项通过，pytest耗时{full["pytest_seconds"]:.2f}秒，pytest与Ruff退出码均为0，'
        f'{full["pins"]}份代码在测试前后、冻结记录和报告生成前相同。\n\n'
        '8帧完全标注的合成新留出上，持久化方案的文字F1为93.6508%，两批热态整帧处理中位耗时为108.291和109.888毫秒。'
        '冷启动、一个英文标签的损失和真实单帧的速度限制另列。窗口试验只覆盖专用本地四阶段界面与两组目标，不推算任意应用成功率；'
        'Week 7的跨应用、跨分辨率综合比较不在本报告范围内。\n\n' +
        grid(['要求', '验收依据', '结果'], [
            ['自动分解与执行', '模型原生有序目标、固定阶段条件及独立操作审查', '通过受控窗口验收'],
            ['错误检测与恢复', '一次应用审阅拒绝、重新观察与预算内重规划', '通过同源恢复验收'],
            ['感知准确率与速度', '8帧合成新留出、两批交错配对及原输出等价', '通过预登记门槛，真实单帧限制另列'],
            ['监测与日志', '运行中父子计数、完成台账、终态和所属进程退出记录', '通过原记录核对']])+'\n\n')
    live_rows = []
    for role, result in [('正常', normal), ('恢复', transient)]:
        live_rows.append([role+' / '+result['name'], result['model'],
                          f'{result["actions"]}/{result["planning"]}/{result["http"]}',
                          f'{result["owner_seconds"]:.3f}', result['stages'], '通过'])
    semantic_rows = [[role, stage['stage_key'], stage['status'], stage['instruction'], '、'.join(stage['independent_transitions'])]
                     for role, result in [('正常', normal), ('恢复', transient)]
                     for stage in result['native_stages']]
    # The original external image paragraph/table remains byte-for-byte inside section 6.
    original_live = next(s for s in sections(base) if s.startswith(HEADINGS[3]+'\n'))
    external_start = original_live.find('另使用normal001初始保存的2560×1600画面')
    require(external_start >= 0, 'frozen_external_perception_paragraph_missing')
    external = original_live[external_start:]
    setting = original_live.split('\n\n', 2)[1]
    if 'runtime_text_model' in evidence['freeze']:
        setting += (f' 两组均使用同一模型组合：无图像的任务分解采用{evidence["freeze"]["runtime_text_model"]}，'
                    f'带图像的动作规划采用{evidence["freeze"]["runtime_model"]}。两层共享预算，逐次记录实际模型名；'
                    '下表列出动作模型，两个完整digest另存于冻结记录。')
    live_section = (HEADINGS[3]+'\n\n'+setting+'\n\n'+grid(['试验', '动作模型' if 'runtime_text_model' in evidence['freeze'] else '模型', '输入/规划/HTTP', '墙钟秒', '目标尝试数', '结果'], live_rows)+'\n\n'
        '墙钟时间包含模型推理、桥接等待、截图、识别和日志开销。两次运行分别列出，不作正常与恢复速度的因果比较。'
        '应用层审阅拒绝与执行器失败分开统计；原动作检查、阶段条件及最终条件各自保留，任何单项通过都不能替代整项任务验收。\n\n'
        +grid(['试验', '阶段', '原始状态', '模型原生指令', '独立应用事件'], semantic_rows)+'\n\n'
        +'阶段表保留每次尝试的原始状态；失败尝试发生过的输入和应用事件不因此改记为目标完成。\n\n'+prose['live_analysis'].strip()+'\n\n'+external)
    test_section = (HEADINGS[4]+'\n\n'+grid(['范围', '通过', 'pytest秒', '说明'], [
        ['感知定向', 59, '1.37', 'C API寿命、异常及原OCR兼容'],
        ['最终源码完整回归', full['passed'], f'{full["pytest_seconds"]:.2f}',
         f'pytest与Ruff实际退出码0；{full["pins"]}份代码前后与当前相同']])+'\n\n'
        f'pytest所属运行器等待{full["pytest_owned_seconds"]:.3f}秒，全验证运行器{full["owner_seconds"]:.3f}秒。'
        '通过数来自执行后的pytest终端汇总，并与同源收集数核对；定向测试与完整回归有重叠，不相加。\n\n'
        '开发失败、桥接中断、格式和定位拒绝保留在附录，未改记为成功，也不作为模型质量零分。历史结果只对应各自源码和模型候选，不合并计算最终成功率。\n\n')
    overrides = {HEADINGS[0]: intro, HEADINGS[3]: live_section, HEADINGS[4]: test_section}
    for heading, text in prose['architecture_sections'].items():
        require(type(text) is str and text.startswith(heading+'\n'), 'architecture_heading_changed')
        overrides[heading] = text.rstrip()+'\n\n'
    output = ''.join(overrides.get(s.splitlines()[0], s) if s.startswith('## ') else s for s in sections(base))
    output = output.replace('# 系统 v2.1.0  复杂任务分解待验收稿', '# 系统 v2.1.0  完整验收版', 1)
    old = '四项工程验收、报告页面检查和源码发布检查分别记录。本稿保留未填结果，不生成整体完成状态。'
    require(output.count(old) == 1, 'old_delivery_sentence_not_unique')
    output = output.replace(old, '四项工程验收、报告页面检查和源码发布检查分别记录。报告生成依据与页图、数字、链接及分发文件的检查结果由校验清单记录。', 1)
    require(protected_components(output) == protected_components(base), 'protected_perception_code_links_figure_or_history_changed')
    require(not any(line.startswith(('# 日期', '# 时区', '日期：', '时区：')) for line in output.splitlines()[:5]), 'cover_date_added')
    return output
