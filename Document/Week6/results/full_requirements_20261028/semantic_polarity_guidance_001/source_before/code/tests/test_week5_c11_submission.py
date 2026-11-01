"""CPU checks for the C11 publication and the preserved numerical contract."""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from gui_agent.finetuning import resized_contract as contract
from gui_agent.finetuning import structured_ce as objective
from gui_agent.finetuning.c11_training import epoch_groups, recipe
from gui_agent.finetuning.training import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dimensions", [(883, 641), (1707, 1019), (780, 600)])
def test_native_resize_mapping_preserves_bounds_and_does_not_mutate(dimensions):
    width, height = dimensions
    geometry = contract.expected_geometry(width, height)
    original = [{"action_type": "click", "x": width - 1, "y": height - 1}]
    before = contract.canonical(original)
    resized = contract.to_resized(original, geometry)
    mapped = contract.from_resized(resized, geometry)
    assert contract.canonical(original) == before
    assert 0 <= mapped[0]["x"] < width and 0 <= mapped[0]["y"] < height
    assert abs(mapped[0]["x"] - original[0]["x"]) <= 2
    assert abs(mapped[0]["y"] - original[0]["y"]) <= 2


@pytest.mark.parametrize("bad", [True, -1, 780, 1.2])
def test_coordinate_errors_are_rejected_without_clamping(bad):
    with pytest.raises(ValueError):
        contract.map_axis(bad, 780, 560)


def test_labels_and_private_metadata_never_enter_prompt():
    geometry = contract.expected_geometry(780, 600)
    row = {"instruction": "Click the search control", "image_width": 780,
           "image_height": 600, "actions": [{"action_type": "click", "x": 123, "y": 456}],
           "private_metadata": "never-visible-PRIVATE"}
    prompt = contract.build_prompt(row, geometry)
    assert "123" not in prompt and "456" not in prompt and "PRIVATE" not in prompt
    assert '"actions"' not in prompt


def test_complete_eight_epoch_budget_and_partial_group():
    groups = [epoch_groups(2482, epoch) for epoch in range(1, 9)]
    assert sum(len(g) for epoch in groups for g in epoch) == 19856
    assert sum(len(epoch) for epoch in groups) == 2488
    assert all(len(epoch[-1]) == 2 for epoch in groups)
    assert all(sorted(i for g in epoch for i in g) == list(range(2482)) for epoch in groups)
    assert groups[0] == epoch_groups(2482, 1) and groups[0] != groups[1]


def test_structure_beats_coordinate_for_a_mixed_token_and_eos_stays_supervised():
    text = '[{"action_type":"click","x":12,"y":34}]'
    # A deliberately mixed token touches both numeric value and structural comma.
    pieces = [text[:text.index('12')].encode(), b'12,', text[text.index('12') + 3:].encode(),
              b'<|im_end|>', b'\n']
    decoder = objective.TokenBytes(MappingProxyType(dict(enumerate(pieces, 2))), 5, '0' * 64)
    full, prefix = [1, 2, 3, 4, 5, 6], [1]
    plan = objective.value_weights(full_ids=full, prefix_ids=prefix,
        labels=[-100, 2, 3, 4, 5, 6], target_json=text, decoder=decoder)
    assert plan.weights[0] == 0 and plan.weights[2] == 6 and plan.weights[4] == 2
    assert 2 in plan.structure_numeric_overlap_tokens


def test_weighted_ce_uses_active_count_instead_of_weight_mass():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 4, 3), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2]], dtype=torch.long)
    weights = torch.tensor([[0.0, 0.0, 6.0, 2.0]], dtype=torch.float64)
    result = objective.causal_ce(logits, labels, weights)
    assert result['supervised_tokens'] == 2
    assert float(result['weighted_ce'].detach()) == pytest.approx(4 * float(result['raw_ce'].detach()))
    result['weighted_ce'].backward()
    assert torch.equal(logits.grad[0, 0], torch.zeros(3, dtype=torch.float64))
    assert logits.grad[0, 1].abs().sum() > logits.grad[0, 2].abs().sum()


def test_published_weights_and_relocated_config_match_manifest():
    summary = json.loads((ROOT / 'Document/Week5/results/summary.json').read_text('utf-8'))
    adapter = ROOT / 'artifacts/week5/c11'
    assert sha256_file(adapter / 'adapter_model.safetensors') == summary['weight_sha256']
    assert sha256_file(adapter / 'adapter_config.json') == summary['public_adapter_config_sha256']
    assert summary['original_adapter_config_sha256'] != summary['public_adapter_config_sha256']
    config = json.loads((adapter / 'adapter_config.json').read_text('utf-8'))
    assert config['base_model_name_or_path'] == recipe('local-model').base_model_id
    assert config['revision'] == recipe('local-model').base_model_revision
    assert summary['offline_original_gates'] == summary['offline_passed_gates'] == 16
    assert sum(len(s['gates']) for s in summary['offline_stages'].values()) == 16
    cases = [i for segment in summary['desktop']['segments'] for i in segment['cases']]
    assert sorted(cases) == list(range(1, 13)) and summary['single_prompt_12of12'] is False


def test_published_numerical_sources_have_original_bytes():
    provenance = json.loads((ROOT / 'Document/Week5/results/provenance.json').read_text('utf-8'))
    for relative, record in provenance['numerical_sources_copied_byte_identically'].items():
        assert sha256_file(ROOT / relative) == record['sha256']


def test_final_prompt_preserves_literal_and_explicit_type_schema():
    import subprocess
    import sys

    from gui_agent.finetuning.t04_prompts import final_prompt

    marker = 'DEMO_' + 'a' * 39  # Synthetic fixture, never an original message.
    instruction, step = final_prompt(marker, 'type')
    assert json.dumps(marker) in instruction and json.dumps(marker) in step
    assert 'text field' in step and 'no x or y' in step
    with pytest.raises(ValueError):
        final_prompt(marker + '.', 'type')
    planned = subprocess.run([sys.executable, 'scripts/week5_c11.py', 'train',
        '--prepared', 'does-not-exist', '--output', 'must-not-be-created'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(planned.stdout)['status'] == 'plan_only'
    assert not (ROOT / 'must-not-be-created').exists()
