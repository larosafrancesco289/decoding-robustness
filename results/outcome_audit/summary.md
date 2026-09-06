# Blinded outcome audit, 2026-09-06

Sample: 152 plain-temperature Q8 generations (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B, Qwen3-8B; T 0.7 and 1.3; both tasks),
stratified by parser path with the uncertain paths oversampled (scripts/outcome_audit_sample.py). Labelled by a Claude Fable
5.1 annotator that saw only task and raw output (no model, temperature, gold, or parser fields): completed / misformatted /
cutoff-coherent / degenerate / other. Key: tmp/outcome_audit/key.jsonl; labels: labels.jsonl; 34 marked borderline.

Crosstab parser path x label (all 152):
  strict   27 completed, 4 misformatted (decorated answer sentence), 1 degenerate (A152: garble, then recovers to a strict answer)
  flexible 23 completed, 9 misformatted, 1 cutoff, 3 degenerate (2 borderline), 7 other (Mistral refusals / 'not among the options')
  failed   15 misformatted (mostly 'The answer is **H**.' without parentheses on Qwen3-8B; Llama giving a value, not a letter), 6 degenerate, 1 cutoff, 4 other
  cap      30 cutoff-coherent, 16 degenerate, 4 completed, 1 misformatted

Cap hits by condition: Llama T1.3: 8/8 degenerate. Llama T0.7: 8/8 cutoff-coherent (budget censoring, not degeneration).
Robust models T1.3 cap: 8 degenerate / 9 cutoff / 3 completed / 1 misformatted (n=21). Robust T0.7 cap: 13 cutoff / 1 completed.
Reading: the run-to-cap CHANGE from T0.7 to T1.3 is a degeneration signal on Llama; a raw cap rate at T0.7 is not.
The parse-failed path at T0.7 is formatting, which the markdown-tolerant regrade (App B) already covers.
Disagreements worth a human look (parser says answered, annotator says degenerate): A152 (strict, graded correct), A085, A091, A148.
Parser missed a stated answer: the 15 misformatted 'failed' records (all graded incorrect; regrade check says no drop moves >1pp).
