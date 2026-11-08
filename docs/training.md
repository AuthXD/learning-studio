# Prepare and train a tutor from reviewed examples

Learning Studio does not train on conversations automatically. Its training export includes only answers you marked helpful, requires an explicit export choice, and includes the cited course excerpts. Review the examples for accuracy and personal information before training.

## Prepare a split without external calls

Export reviewed examples from the browser. Collect at least 12 distinct reviewed inputs, then run from the repository root:

```powershell
.venv\Scripts\python.exe -m learning_app.training_cli prepare reviewed-training-examples.jsonl --output-dir training-data/run-01
```

The command validates message structure, rejects duplicate inputs, and creates deterministic disjoint `train.jsonl` and `validation.jsonl` files. It requires at least 10 training examples and 2 validation examples and refuses to overwrite existing splits. These minimums are for a functioning pipeline, not evidence of a useful dataset.

## Submit a real training job

This step uploads the files to OpenAI and starts a paid job. It was not run during this build. Configure `OPENAI_API_KEY`, then choose a fine-tuning-capable model available to your account:

```powershell
.venv\Scripts\python.exe -m learning_app.training_cli submit --train-file training-data/run-01/train.jsonl --validation-file training-data/run-01/validation.jsonl --model YOUR_SUPPORTED_MODEL
```

Record the returned job ID. Check its actual state with:

```powershell
.venv\Scripts\python.exe -m learning_app.training_cli status ftjob-YOUR_JOB_ID
```

An upload or job error stops the command. It does not fall back to a demo job or claim training succeeded. If a submission times out, inspect your OpenAI job list before submitting again because the server may already have created the job.

## Use a trained model only after evaluation

When the job actually succeeds, configure `LLM_PROVIDER=openai` and `OPENAI_MODEL` with its returned `fine_tuned_model` ID, then restart Learning Studio. Compare its answers with the base model on held-out questions and human-reviewed source support. The small retrieval benchmark is independent of generation quality and does not prove fine-tuning improved the tutor.

The implementation follows OpenAI's [supervised fine-tuning guide](https://developers.openai.com/api/docs/guides/supervised-fine-tuning) and [fine-tuning job API](https://developers.openai.com/api/reference/resources/fine_tuning/subresources/jobs/methods/create). Model availability and training cost depend on the account. The build includes no LoRA training or trained weights.
