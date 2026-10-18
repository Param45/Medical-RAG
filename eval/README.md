# Extraction evaluation

Measures triple-extraction quality for the local model against a hand-annotated
gold standard. Every number the extraction work claims comes from here.

## Why a hand-annotated gold set

The first version of this harness scored against the Gemini-built Neo4j graph.
That is not ground truth. Gemini extracts `Diagnosis: Ascites` from
`report_0001`, whose text reads **"Free fluid: No ascites."** A model reference
therefore cannot measure the two errors that matter most in a medical record —
negation and hallucination — because it commits them itself.

`gold_patient_a.json` is annotated by reading the raw chunk text. It records
both:

- `expected` — facts a careful reader can verify in the text
- `forbidden` — facts the text explicitly **negates** or does not contain

The `forbidden` list is what makes hallucination measurable rather than assumed.

## Running

    python eval/run_extraction_eval.py --label mylabel --n 8
    python eval/run_extraction_eval.py --label qwen7b --model Qwen2.5-7B-Instruct-Q4_K_M.gguf

    bash eval/run_matrix.sh      # the full model comparison, sequentially

Results land in `eval/results/<label>.json`. Each file records the model,
thread count, context size, grammar and few-shot flags, sampling temperature and
a **SHA of the exact prompt used**, so a result reconstructs its own conditions
even after the prompt has since changed.

Runs are sequential by design: 16 GB of RAM cannot hold two of these models.

## Metrics

| metric | question it answers |
|---|---|
| `entity_recall` | did it find the facts that are there |
| `entity_precision` | are its facts real facts |
| `relation_accuracy` | of facts it found, is the relation right |
| `negated_rate` | **does it assert facts the text denies** |
| `unsupported_rate` | does it invent names absent from the source |
| `duplicate_predictions` | does it emit the same fact repeatedly |
| `json_valid_rate` | did the reply parse |

`json_valid_rate` is reported **separately from accuracy and never as a proxy for
it**. A run can score `json_valid_rate = 1.0` with `entity_recall = 0.0`, and one
did: an unbounded array in the schema let the model emit 72 perfectly
schema-conformant "facts" for a chunk containing 3, of which 68 were invented.
Valid output is not correct output.

## Known limits of this gold set

- One patient, chunks 0-7. Report types unique to patient_b are not covered.
- Chunks 1, 5 and 6 are OCR'd handwritten flowsheets whose text is close to
  unreadable. Recall there is bounded by OCR quality, not model quality, and the
  annotations themselves are least certain. Treat the typed reports (0, 2, 4, 7)
  as the trustworthy signal.
- 26 expected facts is small. Differences of one or two facts between runs are
  noise, not progress.
