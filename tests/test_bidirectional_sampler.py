from __future__ import annotations

from pathlib import Path

from cmg.config import load_yaml
from cmg.data.sampler import DirectionAwareObjectBatchSampler
from cmg.training import collect_sampler_epoch_metrics


ROOT = Path(__file__).resolve().parents[1]


class FakeDataset:
    def __init__(self, sample_records: list[dict[str, object]]) -> None:
        self.sample_records = sample_records

    def __len__(self) -> int:
        return len(self.sample_records)


def _fake_dataset() -> FakeDataset:
    records: list[dict[str, object]] = []
    counts = {
        'OBJ001': {'W2A': 3, 'A2W': 2},
        'OBJ002': {'W2A': 2, 'A2W': 3},
    }
    for object_id, direction_counts in counts.items():
        for direction, count in direction_counts.items():
            for trial_index in range(count):
                records.append({
                    'sample_id': f'{object_id}_{direction}_{trial_index}',
                    'object_id': object_id,
                    'direction': direction,
                    'interface_expert_count': trial_index + 1,
                    'interface_window_ratio': 0.1,
                })
    return FakeDataset(records)


def test_direction_sampler_pairs_each_object_and_balances_draws() -> None:
    dataset = _fake_dataset()
    sampler = DirectionAwareObjectBatchSampler(
        dataset,
        batch_size=4,
        interface_alpha=1.0,
        seed=7,
    )
    batches = list(iter(sampler))
    assert len(batches) == len(sampler) == 3
    for batch in batches:
        assert len(batch) == 4
        directions = [dataset.sample_records[index]['direction'] for index in batch]
        assert directions.count('W2A') == directions.count('A2W') == 2
        for pair_start in range(0, len(batch), 2):
            left = dataset.sample_records[batch[pair_start]]
            right = dataset.sample_records[batch[pair_start + 1]]
            assert left['object_id'] == right['object_id']
            assert (left['direction'], right['direction']) == ('W2A', 'A2W')

    summary = sampler.last_epoch_summary
    assert summary['direction_draw_counts'] == {'W2A': 6, 'A2W': 6}
    assert summary['total_draws'] == 12
    assert summary['unique_draws'] == 10
    assert summary['repeated_draws'] == 2

    loader = type('FakeLoader', (), {'batch_sampler': sampler})()
    metrics = collect_sampler_epoch_metrics(loader)
    assert metrics['sampler_balance_mode'] == 'paired_cycle'
    assert metrics['sampler_repeat_rate'] == 2 / 12
    assert metrics['sampler_direction_draw_counts'] == {'W2A': 6, 'A2W': 6}


def test_direction_sampler_is_deterministic_per_seed() -> None:
    dataset = _fake_dataset()
    left = list(DirectionAwareObjectBatchSampler(dataset, batch_size=4, seed=11))
    right = list(DirectionAwareObjectBatchSampler(dataset, batch_size=4, seed=11))
    assert left == right


def test_direction_sampler_rejects_missing_direction() -> None:
    dataset = FakeDataset([
        {'sample_id': 'only_w2a', 'object_id': 'OBJ001', 'direction': 'W2A'},
    ])
    try:
        DirectionAwareObjectBatchSampler(dataset, batch_size=2)
    except ValueError as exc:
        assert 'both directions' in str(exc)
    else:
        raise AssertionError('Sampler must reject objects without both directions.')


def test_bidirectional_train_config_enables_paired_cycle() -> None:
    config = load_yaml(ROOT / 'configs' / 'train' / 'bidirectional_v1.yaml')
    assert config['sampling_mode'] == 'direction_object_aware'
    assert config['direction_balance_mode'] == 'paired_cycle'
    assert config['default_allowed_trial_results']['train'] == ['stable']
