# Cloud-GPU runbook — getting your first real result

DecodeBound needs an NVIDIA GPU. This walks you from zero to committed, real numbers
in about **one evening for ~\$2**. No prior cloud experience assumed.

The plan:

```
rent a GPU pod  →  put the code on it  →  install  →  run the sweep  →
save the results  →  TEAR DOWN the pod (so you stop paying)
```

> 💡 The default model below is **Qwen2.5-7B-Instruct** because it is *ungated* —
> it downloads with no account or token. Llama-3.1-8B needs a Hugging Face license
> click + token (see the optional step at the end). Start with Qwen; switch later.

---

## 0 · Before you start

You need two things:

1. **The code on GitHub.** The pod will `git clone` it. If you haven't pushed this
   repo yet, create an empty repo on github.com and run, from your laptop:
   ```bash
   git remote add origin https://github.com/<you>/decodebound.git
   git push -u origin master
   ```
   (Ask Claude to do this for you if you want — it won't push without you saying so.)

2. **~\$5–10 of credit** on a GPU provider. You'll use under \$2; the rest is buffer.

---

## 1 · Rent a GPU pod

Any of these work. **RunPod** is the most beginner-friendly (web terminal, no SSH keys).

| Provider | GPU to pick | ~Price | Notes |
|----------|-------------|--------|-------|
| **RunPod** | RTX 4090 (24 GB) | ~\$0.45/hr | Easiest UI; pick a **PyTorch** template |
| Vast.ai | RTX 4090 | ~\$0.30/hr | Cheapest; slightly more fiddly |
| Lambda | A100 40 GB | ~\$1.10/hr | Clean, sometimes capacity-limited |

A **24 GB RTX 4090 is plenty** for a 7–8B model. You do **not** need an A100.

**On RunPod specifically:**
1. Sign up, add credit.
2. **Deploy → Pods → pick "RTX 4090".**
3. For the template, choose one with PyTorch + CUDA (e.g. "RunPod PyTorch 2.x").
4. Set **Container Disk / Volume to ≥ 40 GB** (model weights are ~16 GB).
5. Deploy, then open the pod's **web terminal** (or "Connect → Start Web Terminal").

---

## 2 · Put the code on the pod + install

In the pod's terminal:

```bash
# 1. get the code
git clone https://github.com/<you>/decodebound.git
cd decodebound

# 2. install vLLM (the serving backend) — it pulls a matching torch
pip install vllm

# 3. install this harness + its analysis deps (GPU-free deps; won't touch torch)
pip install -e .

# 4. sanity-check the GPU is visible
decodebound check-gpu
```

`check-gpu` should now print your GPU, VRAM, and driver — **not** the "no GPU" halt
you saw on the Mac. If it still halts, the pod has no GPU attached; fix that before
continuing.

> Confirm the vLLM entry point matches your installed version (AGENTS.md rule):
> ```bash
> vllm serve --help | head      # current CLI form
> ```
> The harness launches the server via `python -m vllm.entrypoints.openai.api_server`.
> If your vLLM version only ships `vllm serve`, use **Path B** below (it calls the
> server yourself) and `--no-launch`.

---

## 3 · Run it — Path A (one command, recommended)

`reproduce.sh` launches vLLM, runs the sweep, makes the figures, and prints the table.
The first launch downloads the model (~16 GB, a couple of minutes).

```bash
./reproduce.sh --model Qwen/Qwen2.5-7B-Instruct --concurrency 1,2,4,8,16,32
```

> First run tip: it issues 256 requests per concurrency point by default. That's fine,
> but if you want a faster first pass, use Path B with `--n-requests 128`.

When it finishes you'll have:
- `results/raw/raw_c*.parquet` + `results/raw/run_meta.json`  ← the proof
- `results/figures/{prefill_decode,pareto,tail_latency}.png`  ← the graphs
- a printed table with the **knee** (your honest operating point)

Skip to **step 4**.

---

## 3b · Run it — Path B (two terminals, robust fallback)

Use this if the model download is slow, if you want to watch server logs, or if your
vLLM only has `vllm serve`. It separates "start the server" from "measure it".

**Terminal 1 — start the server and leave it running:**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.90
# wait for:  "Application startup complete."  /  "Uvicorn running on ... :8000"
```

**Terminal 2 — measure the already-running server (`--no-launch`):**
```bash
cd decodebound
decodebound sweep --model Qwen/Qwen2.5-7B-Instruct \
  --concurrency 1,2,4,8,16,32 --n-requests 128 --no-launch --yes
decodebound plot
decodebound analyze
```

(Open a second web terminal on RunPod, or run Terminal 1 inside `tmux`.)

---

## 4 · Look at the result

```bash
decodebound analyze        # the derived table + the knee
```

Read the p99 ITL column climb as concurrency rises while p50 barely moves — **that's the
whole finding.** Open `results/figures/tail_latency.png` to see it.

---

## 5 · Save the results (so the work survives the pod)

The cleanest option — commit the real data from the pod and push it back. This is the
"committed real results" the project is designed around:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
git add results/ && git commit -m "Add first real sweep: Qwen2.5-7B on RTX 4090"
git push
```

Then on your laptop: `git pull`. Your figures and `run_meta.json` are now local.

(Alternative: `tar czf results.tgz results/` and download `results.tgz` via the pod's
file browser.)

---

## 6 · ⚠️ TEAR DOWN THE POD

**Do this or you keep paying by the hour.**

- RunPod: **Pods → your pod → Terminate** (not just "Stop" — Stop on some plans still
  bills for the disk).
- Confirm it's gone from the dashboard.

A forgotten 4090 pod is ~\$11/day. Don't.

---

## 7 · After the run — make it count

1. **Replace the README placeholders** with your real numbers (the tables in
   `README.md` and the `‹measured›` blanks in `report.md`).
2. Commit that: `git commit -am "Replace placeholders with measured Qwen2.5-7B numbers"`.
3. Now the repo reads as a finished measurement project, not a sketch.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `check-gpu` still halts | Pod has no GPU attached; redeploy picking a GPU type. |
| Server never becomes healthy (10 min) | Model still downloading on a slow link, or OOM. Use Path B and watch Terminal 1's logs. |
| CUDA out of memory at high concurrency | Lower `--gpu-memory-utilization` to 0.85, or drop the top concurrency (use `--concurrency 1,2,4,8,16`). |
| `vllm: command not found` | `pip install vllm` didn't finish, or wrong env. Re-run it; check `python -c "import vllm"`. |
| Want Llama-3.1-8B instead | Accept the license on its HF model page, then `pip install huggingface_hub && huggingface-cli login` (paste your HF token), then use `--model meta-llama/Llama-3.1-8B-Instruct`. |

---

## What a good first run looks like (so you know it worked)

- `run_meta.json` records your real GPU name + driver + vLLM version.
- p99 ITL at the highest concurrency is clearly **several times** the p50.
- The knee printed by `analyze` is somewhere in the middle of your sweep, not at the ends.
- Three PNGs exist and are non-empty.

If all four are true, you have a real, defensible result. Go update the README.
