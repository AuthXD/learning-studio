"""Prepare reviewed training data locally; optionally submit and inspect a real fine-tuning job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import httpx
from dotenv import load_dotenv


def read_records(path: Path) -> list[dict]:
    records = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Line {number} is not valid JSON.') from exc
        messages = record.get('messages') if isinstance(record, dict) else None
        if (not isinstance(messages, list) or len(messages) != 3 or not all(isinstance(m, dict) for m in messages)
                or [m.get('role') for m in messages] != ['system', 'user', 'assistant']):
            raise ValueError(f'Line {number} must contain system, user, and assistant messages in that order.')
        if any(not isinstance(m.get('content'), str) or not m['content'].strip() or len(m['content']) > 30000 for m in messages):
            raise ValueError(f'Line {number} contains empty or oversized message text.')
        question = ' '.join(messages[1]['content'].lower().split())
        if question in seen:
            raise ValueError(f'Line {number} repeats an input. Deduplicate examples before splitting.')
        seen.add(question)
        records.append({'messages': [{'role': m['role'], 'content': m['content']} for m in messages]})
    return records


def prepare(source: Path, directory: Path) -> dict:
    records = read_records(source)
    if len(records) < 12:
        raise ValueError('Provide at least 12 distinct reviewed examples: at least 10 training and 2 validation.')
    records.sort(key=lambda record: hashlib.sha256(record['messages'][1]['content'].encode()).hexdigest())
    validation_count = max(2, len(records) // 5)
    validation, train = records[:validation_count], records[validation_count:]
    directory.mkdir(parents=True, exist_ok=True)
    for name, group in [('train.jsonl', train), ('validation.jsonl', validation)]:
        target = directory / name
        if target.exists():
            raise ValueError(f'{target} already exists. Choose a new output directory to preserve the existing split.')
    for name, group in [('train.jsonl', train), ('validation.jsonl', validation)]:
        (directory / name).write_text(''.join(json.dumps(r) + '\n' for r in group), encoding='utf-8')
    return {'training_examples': len(train), 'validation_examples': len(validation), 'directory': str(directory)}


def submit(train_path: Path, validation_path: Path, model: str, client: httpx.Client) -> dict:
    train, validation = read_records(train_path), read_records(validation_path)
    if len(train) < 10 or len(validation) < 2:
        raise ValueError('Require at least 10 training and 2 validation examples.')
    inputs = {' '.join(r['messages'][1]['content'].lower().split()) for r in train}
    if any(' '.join(r['messages'][1]['content'].lower().split()) in inputs for r in validation):
        raise ValueError('Training and validation inputs overlap. Prepare a new split.')
    uploaded = []
    for path in (train_path, validation_path):
        with path.open('rb') as handle:
            response = client.post('/v1/files', data={'purpose': 'fine-tune'},
                                   files={'file': (path.name, handle, 'application/jsonl')})
        response.raise_for_status()
        uploaded.append(response.json()['id'])
    response = client.post('/v1/fine_tuning/jobs', json={
        'training_file': uploaded[0], 'validation_file': uploaded[1], 'model': model,
        'method': {'type': 'supervised', 'supervised': {'hyperparameters': {'n_epochs': 3}}},
    })
    response.raise_for_status()
    value = response.json()
    return {key: value.get(key) for key in ('id', 'status', 'model', 'fine_tuned_model', 'training_file', 'validation_file')}


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preparation = commands.add_parser('prepare', help='Validate and split a reviewed JSONL export. Offline.')
    preparation.add_argument('source', type=Path)
    preparation.add_argument('--output-dir', type=Path, required=True)
    training = commands.add_parser('submit', help='Upload the supplied data to OpenAI and start a paid training job.')
    training.add_argument('--train-file', type=Path, required=True)
    training.add_argument('--validation-file', type=Path, required=True)
    training.add_argument('--model', required=True, help='A fine-tuning-capable model available to your account.')
    status = commands.add_parser('status', help='Read the actual state of an existing training job.')
    status.add_argument('job_id')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            value = prepare(args.source, args.output_dir)
        else:
            key = os.getenv('OPENAI_API_KEY')
            if not key:
                parser.error('Set OPENAI_API_KEY in your environment or project26/.env.')
            with httpx.Client(base_url='https://api.openai.com', headers={'Authorization': 'Bearer ' + key}, timeout=60) as client:
                if args.command == 'submit':
                    value = submit(args.train_file, args.validation_file, args.model, client)
                else:
                    if not re.fullmatch(r'ftjob-[A-Za-z0-9_-]+', args.job_id):
                        parser.error('Provide a fine-tuning job ID beginning ftjob-.')
                    response = client.get('/v1/fine_tuning/jobs/' + args.job_id)
                    response.raise_for_status()
                    body = response.json()
                    value = {key: body.get(key) for key in ('id', 'status', 'model', 'fine_tuned_model', 'trained_tokens', 'result_files', 'error')}
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    except httpx.HTTPError as exc:
        parser.error(f'OpenAI request failed: {type(exc).__name__}. Check account access and job status before retrying a submission.')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
