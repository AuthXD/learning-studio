import json

import httpx
import pytest

from learning_app.training_cli import prepare, read_records, submit


def dataset(path, count=15):
    records = [{'messages': [{'role': 'system', 'content': 'Use the notes.'},
                            {'role': 'user', 'content': f'Question {i}: explain concept {i}.'},
                            {'role': 'assistant', 'content': f'Explanation {i}.'}]} for i in range(count)]
    path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')


def test_prepare_splits_without_overlap_and_preserves_files(tmp_path):
    source = tmp_path / 'reviewed.jsonl'
    dataset(source)
    target = tmp_path / 'split'
    result = prepare(source, target)
    assert result['training_examples'] == 12
    assert result['validation_examples'] == 3
    train = read_records(target / 'train.jsonl')
    validation = read_records(target / 'validation.jsonl')
    assert not {r['messages'][1]['content'] for r in train} & {r['messages'][1]['content'] for r in validation}
    with pytest.raises(ValueError, match='already exists'):
        prepare(source, target)


def test_tiny_and_duplicate_datasets_rejected(tmp_path):
    source = tmp_path / 'tiny.jsonl'
    dataset(source, 2)
    with pytest.raises(ValueError, match='at least 12'):
        prepare(source, tmp_path / 'out')
    source.write_text(source.read_text() * 2)
    with pytest.raises(ValueError, match='repeats'):
        read_records(source)


def test_real_submission_contract_with_fake_network(tmp_path):
    source = tmp_path / 'reviewed.jsonl'
    dataset(source)
    prepare(source, tmp_path / 'split')
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path == '/v1/files':
            return httpx.Response(200, json={'id': 'file-' + str(len(calls))})
        data = json.loads(request.content)
        assert data['training_file'] == 'file-1'
        assert data['validation_file'] == 'file-2'
        assert data['model'] == 'account-supported-model'
        return httpx.Response(200, json={'id': 'ftjob-test', 'status': 'validating_files'})

    with httpx.Client(base_url='https://api.openai.com', transport=httpx.MockTransport(respond)) as client:
        result = submit(tmp_path / 'split/train.jsonl', tmp_path / 'split/validation.jsonl', 'account-supported-model', client)
    assert result['id'] == 'ftjob-test'
    assert len(calls) == 3


def test_overlap_fails_before_upload(tmp_path):
    source = tmp_path / 'reviewed.jsonl'
    dataset(source)
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request))) as client:
        with pytest.raises(ValueError, match='overlap'):
            submit(source, source, 'model', client)
    assert not calls
