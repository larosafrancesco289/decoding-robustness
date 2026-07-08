# Model checkpoints

All quantized checkpoints are Bartowski imatrix GGUF conversions downloaded from
Hugging Face (one conversion provider, one recipe, so the quantization axis is not
confounded by conversion recipe). Save every file into `models/` under its exact
filename; the runner resolves filenames from `quant_files` in the config. Download
and verify everything for a config in one step with

```bash
uv run python scripts/fetch_models.py --config configs/full_matrix.yaml
```

or fetch a single file and record its checksum and chat template with

```bash
uv run python scripts/fetch_models.py --repo <repo> --file <file>.gguf
```

Direct link pattern: `https://huggingface.co/<repo>/resolve/main/<file>?download=true`

## SHA256 manifest

31 files in total: 7 models at 4 quantization levels each, the BF16 anchor on
Qwen3-4B, and the two Q8 checkpoints of the Llama lineage cell.

`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`

| File | SHA256 |
|---|---|
| Meta-Llama-3.1-8B-Instruct-Q8_0.gguf | `9da71c45c90a821809821244d4971e5e5dfad7eb091f0b8ff0546392393b6283` |
| Meta-Llama-3.1-8B-Instruct-Q6_K.gguf | `33981adf6bae52c503fb5c24f72539010632f7ed290a56c1315a8cd50adca587` |
| Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c` |
| Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf | `6be122b5c8f2a33974953e58e0ffd2be505661acc6f4caf733e4bca130e89fea` |

`bartowski/Mistral-7B-Instruct-v0.3-GGUF`

| File | SHA256 |
|---|---|
| Mistral-7B-Instruct-v0.3-Q8_0.gguf | `404857e776114baada71a08ebd3bba79d721ec7fca99705e7e7b892ae8bc583f` |
| Mistral-7B-Instruct-v0.3-Q6_K.gguf | `757d75df89821a94841f341f3270b0709d36891dd8fbc41dfd059d213b54c3f5` |
| Mistral-7B-Instruct-v0.3-Q4_K_M.gguf | `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6` |
| Mistral-7B-Instruct-v0.3-Q3_K_M.gguf | `145b11df092f04bbecb9fd99e83d1f3d8a6d1e53805682bfba1fe314f306e1d5` |

`bartowski/Qwen_Qwen3-8B-GGUF`

| File | SHA256 |
|---|---|
| Qwen_Qwen3-8B-Q8_0.gguf | `edbd7ae01df991a2d4061e61451ad3d6829d8ca19f7cc846f7177499ac280c33` |
| Qwen_Qwen3-8B-Q6_K.gguf | `69a45d3c366bab1736b201fdb21eb9d58160999a0f330a79d2c096f50de8985d` |
| Qwen_Qwen3-8B-Q4_K_M.gguf | `54fffa050078e984116639c83dfb64b5aa6d4cd474e018b076777c632bbccccd` |
| Qwen_Qwen3-8B-Q3_K_M.gguf | `c2c61b55d39ec6fc43f84b4cbbed4505fde386d4e03ed005087bfd8c69c502c3` |

`bartowski/Qwen_Qwen3-4B-GGUF`

| File | SHA256 |
|---|---|
| Qwen_Qwen3-4B-bf16.gguf | `52486602bdca589fd1507962537b4fc7fa2f1fc57222fa36a960cf691d4960c8` |
| Qwen_Qwen3-4B-Q8_0.gguf | `4050871d20fa939e88e83b1a86060061bdeff01e0d9e5d000d374766f0caf7d7` |
| Qwen_Qwen3-4B-Q6_K.gguf | `12b0913feed83232737d860e299ca410d3b2bbf8d33e156ba594f3b44ba17ef1` |
| Qwen_Qwen3-4B-Q4_K_M.gguf | `fbe1d5edd4ce802ae3ae7c7e4ab7d09789d697fdac1fc7929f8df4ca3c41bae3` |
| Qwen_Qwen3-4B-Q3_K_M.gguf | `29183d72b1b4e9666287d82acb9aea189512c99cb3bf6d9c77c4a43ebcb30414` |

`bartowski/Qwen2.5-7B-Instruct-GGUF`

| File | SHA256 |
|---|---|
| Qwen2.5-7B-Instruct-Q8_0.gguf | `9c6a6e61664446321d9c0dd7ee28a0d03914277609e21bc0e1fce4abe780ce1b` |
| Qwen2.5-7B-Instruct-Q6_K.gguf | `489138dfed4f04cd6dea56e5a8423e4aa05a0318cce2a4a72250fe1278e97cf8` |
| Qwen2.5-7B-Instruct-Q4_K_M.gguf | `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` |
| Qwen2.5-7B-Instruct-Q3_K_M.gguf | `6738a2d4f9b280c55b2a19a7ab27334a75d2cafc6ef82a11a075f4b5613eb736` |

`bartowski/Qwen_Qwen3-1.7B-GGUF`

| File | SHA256 |
|---|---|
| Qwen_Qwen3-1.7B-Q8_0.gguf | `74bb7c53538ab2cc81b93f0c64da14a503159de68cff3c6770428d3850479db3` |
| Qwen_Qwen3-1.7B-Q6_K.gguf | `95aec3c8e76caf949b5a7a3b02adbb7e307eb0aa55880f6a6f9fb5f46abe6d4f` |
| Qwen_Qwen3-1.7B-Q4_K_M.gguf | `72c5c3cb38fa32d5256e2fe30d03e7a64c6c79e668ad84057e3bd66e250b24fb` |
| Qwen_Qwen3-1.7B-Q3_K_M.gguf | `d544d30dbb7b2f608775d69c939a9b8d89c45df2be1a5ad1390ddc1bd7f8d6f5` |

`bartowski/google_gemma-3-12b-it-GGUF`

| File | SHA256 |
|---|---|
| google_gemma-3-12b-it-Q8_0.gguf | `da78c6801f4fae0061780cdd0c12cf7b1deb4459587dca1c2d3c1c11528b60e1` |
| google_gemma-3-12b-it-Q6_K.gguf | `a1385c0fd73b8fcb560bf55f9e1870346292a096476a2b9372aa04bd0aeae1d0` |
| google_gemma-3-12b-it-Q4_K_M.gguf | `fc57f67efa46d711c346e587cbef7d049e95f3df8db2eb2271153343ef0acc7b` |
| google_gemma-3-12b-it-Q3_K_M.gguf | `b566ae4850815706bbaa9105efa0e722421544aa41999424924d61f915ea868e` |

`bartowski/Meta-Llama-3-8B-Instruct-GGUF`

| File | SHA256 |
|---|---|
| Meta-Llama-3-8B-Instruct-Q8_0.gguf | `583c616da14b82930f887f991ab446711da0b029166200b67892d7c9f8f45958` |

`bartowski/Llama-3.2-3B-Instruct-GGUF`

| File | SHA256 |
|---|---|
| Llama-3.2-3B-Instruct-Q8_0.gguf | `b5607b5090a8280063fff2d706bb3408ca6542341b06aab39c3eca0a28575921` |
