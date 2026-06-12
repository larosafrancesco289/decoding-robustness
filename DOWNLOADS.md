# Model downloads — full roster (verified on HF, 2026-06-01)

All files are Bartowski imatrix GGUFs. **Save every file into `models/`** with its exact
filename (the runner resolves filenames from `quant_files` in the config). Sizes are
approximate. After each download, record SHA256 + extract the chat template with:

```bash
uv run python scripts/fetch_models.py --local models/<file>.gguf --repo <repo>
```

Direct-download link pattern: `https://huggingface.co/<repo>/resolve/main/<file>?download=true`

---

## Priority 1 — canary ladder ✅ DONE (all 4 downloaded, hashed, templates extracted 2026-06-01)

Repo: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`. SHA256 recorded in each `<file>.meta.json`
sidecar and pinned here for pre-registration (SPEC §12):

| File | ~Size | SHA256 |
|---|---|---|
| Meta-Llama-3.1-8B-Instruct-Q8_0.gguf | 8.5 GB | `9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283` |
| Meta-Llama-3.1-8B-Instruct-Q6_K.gguf | 6.6 GB | `33981adf6bae52c503fb5c24f72539010632f7ed290a56c1315a8cd50adca587` |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 4.9 GB | `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c` |
| Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf | 4.0 GB | `6be122b5c8f2a33974953e58e0ffd2be505661acc6f4caf733e4bca130e89fea` |

## Priority 2 — family axis (needed at M4)

Repo: `bartowski/Mistral-7B-Instruct-v0.3-GGUF`

| File | ~Size | Link |
|---|---|---|
| Mistral-7B-Instruct-v0.3-Q8_0.gguf | 7.7 GB | https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q8_0.gguf?download=true |
| Mistral-7B-Instruct-v0.3-Q6_K.gguf | 5.9 GB | https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q6_K.gguf?download=true |
| Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | 4.4 GB | https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true |
| Mistral-7B-Instruct-v0.3-Q3_K_M.gguf | 3.5 GB | https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q3_K_M.gguf?download=true |

Repo: `bartowski/Qwen_Qwen3-8B-GGUF`

| File | ~Size | Link |
|---|---|---|
| Qwen_Qwen3-8B-Q8_0.gguf | 8.7 GB | https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q8_0.gguf?download=true |
| Qwen_Qwen3-8B-Q6_K.gguf | 6.7 GB | https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q6_K.gguf?download=true |
| Qwen_Qwen3-8B-Q4_K_M.gguf | 5.0 GB | https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q4_K_M.gguf?download=true |
| Qwen_Qwen3-8B-Q3_K_M.gguf | 4.0 GB | https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF/resolve/main/Qwen_Qwen3-8B-Q3_K_M.gguf?download=true |

## Priority 3 — Qwen3 size axis (needed at M4)

Repo: `bartowski/Qwen_Qwen3-4B-GGUF`  (includes the **BF16 anchor**)

| File | ~Size | Link |
|---|---|---|
| Qwen_Qwen3-4B-Q8_0.gguf | 4.3 GB | https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q8_0.gguf?download=true |
| Qwen_Qwen3-4B-Q6_K.gguf | 3.3 GB | https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q6_K.gguf?download=true |
| Qwen_Qwen3-4B-Q4_K_M.gguf | 2.5 GB | https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf?download=true |
| Qwen_Qwen3-4B-Q3_K_M.gguf | 2.0 GB | https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q3_K_M.gguf?download=true |
| Qwen_Qwen3-4B-bf16.gguf | 8.0 GB | https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-bf16.gguf?download=true |

Repo: `bartowski/Qwen_Qwen3-1.7B-GGUF`

| File | ~Size | Link |
|---|---|---|
| Qwen_Qwen3-1.7B-Q8_0.gguf | 1.8 GB | https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q8_0.gguf?download=true |
| Qwen_Qwen3-1.7B-Q6_K.gguf | 1.4 GB | https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q6_K.gguf?download=true |
| Qwen_Qwen3-1.7B-Q4_K_M.gguf | 1.1 GB | https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf?download=true |
| Qwen_Qwen3-1.7B-Q3_K_M.gguf | 0.9 GB | https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q3_K_M.gguf?download=true |

---

Total ≈ 21 files, ~100 GB. Disk has ~800 GB free, so capacity is not a concern.
Q4_K_M is the only level needed for the M1 smoke; the rest fill in for M3 (Llama ladder)
and M4 (other checkpoints).
