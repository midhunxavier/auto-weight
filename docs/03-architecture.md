# Architecture

```mermaid
flowchart TD
    Owner["Model owner"] --> Issue["Freeze base, examples, gates and budget"]
    Owner --> LLM["External LLM proposer"]
    LLM --> Proposal["Structured tensor edit"]
    Issue --> Trial["Apply to temporary checkpoint"]
    Proposal --> Trial
    Trial --> Dev["Evaluate base and candidate on development examples"]
    Dev --> Report["Record scores and delete temporary weights"]
    Report --> LLM
    Report --> Selection["Select one passing attempt"]
    Selection --> Replay["Rebuild and verify candidate bytes"]
    Replay --> Confirm["Evaluate on fresh confirmation examples"]
    Confirm -->|Every gate passes| Export["Publish checkpoint and report"]
    Confirm -->|A gate fails or evaluation errors| Discard["Discard weights and retain report"]
```

The CLI owns the workspace, freezes input hashes and runs one supervised evaluator process at a time. The worker selects an adapter from the frozen issue. It applies the declarative operations and scores both checkpoints with the same token or text evaluator. The Hugging Face adapter carries the original tokenizer and inference configuration into the export. It evaluates the candidate from the serialized file so the export contains the evaluated bytes.

The runner reserves confirmation before starting inference. Result directories become visible by rename once complete. An interrupted reservation consumes its slot. A workspace lock serializes operations and prevents two callers from publishing competing results.
