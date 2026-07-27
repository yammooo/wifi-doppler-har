# DEI SHARP preprocessing

This directory contains the complete CPU-only deployment for generating
aligned AR and PC Doppler traces on the DEI cluster.

## 1. Transfer this source snapshot

Run locally from the repository root:

```bash
tar -czf /tmp/wifi-doppler-cluster.tar.gz \
    cluster \
    src/preprocessing/preprocess_sharp.py \
    external/sharp/optimization_utility.py

scp /tmp/wifi-doppler-cluster.tar.gz \
    frigogianm@login.dei.unipd.it:
```

On the DEI login host:

```bash
mkdir -p ~/wifi-doppler-har
tar -xzf ~/wifi-doppler-cluster.tar.gz -C ~/wifi-doppler-har

du -sh ~/CSI-80Mhz
find ~/CSI-80Mhz -type f -name '*.mat' | wc -l
find ~/CSI-80Mhz/AR-* ~/CSI-80Mhz/PC-* -maxdepth 1 -type f -name '*.mat' | wc -l
quota -s
```

`~/CSI-80Mhz` is a valid persistent input location because home is mounted
inside Singularity automatically. It should contain the `AR-*`, `PC-*`, and
`PI-*` directories directly. The expected counts for the current archive are
248 total MAT files and 204 AR+PC MAT files. The job uses `/ext` for temporary
phase, H-estimation, and reconstruction files.

## 2. Build the image once

Do not build on the login host. Start an interactive allocation:

```bash
sinteractive --mem 8G --time 01:00:00
cd ~/wifi-doppler-har
mkdir -p /ext/$USER/apptainer-build
export APPTAINER_TMPDIR=/ext/$USER/apptainer-build
apptainer build --fakeroot \
    ~/wifi-doppler-preprocess.sif \
    cluster/wifi-doppler-preprocess.def
apptainer test ~/wifi-doppler-preprocess.sif
exit
```

If the installed command is named `singularity`, use it in place of
`apptainer`. If DEI has not enabled `--fakeroot` for the account, build the
same definition on a Linux machine with `sudo singularity build`, then upload
the resulting SIF.

## 3. Run a bounded smoke job

From `~/wifi-doppler-har`:

```bash
sbatch \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --export=ALL,SHARP_SUBSETS=AR-1a,SHARP_LIMIT=1,SHARP_H_END=512,SHARP_DOPPLER_START=0,SHARP_DOPPLER_END=0,SHARP_OUTPUT_ROOT=$HOME/doppler-smoke \
    cluster/preprocess_ar_pc.slurm
```

Monitor it using the job ID printed by `sbatch`:

```bash
squeue -j JOB_ID
tail -f sharp-ar-pc-JOB_ID.out
seff JOB_ID
```

The smoke output should contain one recording and `manifest.json`. Remove
`~/doppler-smoke` after inspection.

## 4. Submit the full AR+PC computation

First ensure that home or the chosen persistent group/NAS path has at least
20 GiB free for final Doppler traces. Then submit:

```bash
sbatch cluster/preprocess_ar_pc.slurm
```

The default request is one task, 48 CPUs, 96 GiB RAM, and 12 hours. Override
resources or paths without editing the file:

```bash
sbatch \
    --cpus-per-task=96 \
    --mem=160G \
    --export=ALL,SHARP_OUTPUT_ROOT=/path/to/group/storage/doppler_traces_recomputed \
    cluster/preprocess_ar_pc.slurm
```

During execution:

```bash
squeue -j JOB_ID
sstat --jobs=JOB_ID
srun --pty --jobid JOB_ID /bin/bash
myjobinfo JOB_ID
```

After completion:

```bash
seff JOB_ID
sacct -j JOB_ID -o JobID,State,Elapsed,ReqCPUS,ReqMem,MaxRSS,ExitCode
```

On failure, the log prints the retained `/ext` scratch path. On success,
`manifest.json` is validated and scratch is removed. Set
`SHARP_KEEP_SCRATCH=1` only when the intermediates are intentionally needed.

## Recovery after output quota failure

If all H estimation and reconstruction completed but home filled while writing
Doppler pickles, preserve the failed job's `/ext` directory. Copy the raw input
there before freeing home quota, then submit `resume_postprocess.slurm` on the
same node:

```bash
sbatch \
    --nodelist=runner-11 \
    --export=ALL,SHARP_RESUME_SCRATCH=/ext/$USER/wifi-doppler-har/FAILED_JOB_ID \
    cluster/resume_postprocess.slurm
```

The recovery job verifies all intermediates, removes only a truncated target
pickle left by the quota failure, keeps valid partial targets, and reruns the
pipeline with every expensive completed stage skipped. Scratch is intentionally
retained after recovery.
