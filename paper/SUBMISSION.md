# OpenReview submission — copy-paste pack

Portal: https://openreview.net/group?id=EMNLP/2026/Workshop/BlackboxNLP
Track: **Full paper (archival)** — 8 pages + references
Deadline: July 17, 2026, 11:59PM UTC-12 (AoE). You can edit the submission until then.
PDF to upload: `paper/main.pdf` (the current `[review]` build, commit bb8b2e5)

---

## Title (plain text, one line, no LaTeX)

```
Temperature Robustness Is a Model Property, Not a Sampler Choice
```

## Abstract (plain Unicode, no LaTeX)

```
Truncation samplers such as top-p, min-p, and top-nσ exist to prevent the accuracy collapse of high-temperature sampling. Sampler evaluations compare one rule against another and take the collapse for granted. We ask whether it occurs in every model at temperatures used in deployment (T ≤ 1.3). In a single controlled pipeline, we run seven instruction-tuned models from five families across 8 sampler arms, 4 quantization levels, and 2 tasks. In that range, sampler choice barely matters in every family except Llama. Under plain temperature sampling at T = 1.3, Llama-3.1-8B and Llama-3.2-3B lose 34 to 41 accuracy points on GSM8K and MMLU-Pro, and Llama-3-8B loses 17 on MMLU-Pro; no model outside the family loses more than 8. The collapse is a termination failure. Generations derail midway and never reach an answer, but when a model does produce a well-formed answer, it is as accurate as under greedy decoding. A temperature ladder up to T = 2.0 shows that every model collapses at some temperature, and truncation samplers begin to matter only past it. Forcing one off-distribution token into the context multiplies a fragile model's chance of drawing another, so errors compound. Practitioners tuning a current model can leave the sampler alone.
```

## TL;DR (if the form asks)

```
At deployment temperatures, high-temperature accuracy collapse happens only in the Llama family: sampler choice barely matters, so temperature robustness is a property of the model, not the sampler.
```

## Keywords (if the form asks; pick what fits, comma-separated)

```
decoding, sampling temperature, truncation sampling, top-p, min-p, robustness, language model evaluation, GSM8K, MMLU-Pro, quantization
```

---

## Submission steps

1. Log in at the portal link above with your activated OpenReview profile.
2. Click the workshop's submission button (labeled like "BlackboxNLP 2026 Submission").
3. Paste title, abstract, TL;DR, keywords from above — plain text only, no `\\`, no `$...$`.
4. Author: you, pulled from your profile. Check the name and affiliation are the ones you want at camera-ready (reviewers never see them; the PDF stays anonymous).
5. Select the full paper / archival track.
6. Upload `paper/main.pdf` as-is. No supplementary material needed — the code link (anonymous.4open.science mirror) is already in Appendix K.
7. Submit, then open the PDF **from OpenReview** and glance at page 1 to confirm the upload is intact (title, "Anonymous ACL submission", ruler in the margins).

## Notes

- The `\\` in the tex title is a PDF line break only — never type it into a form field.
- If a field rejects σ or ≤, fall back to "top-n-sigma" and "T <= 1.3".
- This file is deliberately untracked — delete it after submission or leave it; don't merge it into the `anon` branch.
