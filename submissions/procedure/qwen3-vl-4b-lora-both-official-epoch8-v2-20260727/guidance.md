# Docker-host handoff

This folder is the complete x86_64 Docker build context for the Procedure
submission `qwen3-vl-4b-lora-both-official-epoch8-v2-20260727`. Source-cluster
Apptainer evidence and Docker-host evidence are separate. Do not mark Docker as
tested until the Docker steps below pass.

## Recorded source-cluster status

- Training job 1096: completed on 2026-07-27 with exit code 0, epoch 8, step
  3440/3440.
- v2 static validation passed with both the bundled and `make-submission`
  validators: 58 regular files, no symlinks or cache directories. All five
  resource checksums passed.
- v2 Apptainer build job 1338 produced the exact SIF after its guarded build
  retry. It then exposed and allowed repair of runtime tmp-directory ordering.
- v2 A100 smoke job 1341 passed in BF16 with 96 frames: 9/9 non-empty
  responses, zero failures, 100.69 s model-ready time, 139.67 s inference
  total versus the 390 s pooled allowance, and 12.89 GiB peak GPU memory. The
  validator passed, and the temporary SIF and 4 GB build cache were removed.
- Diagnostic jobs 1334, 1336, and 1337 established that shared-scratch and
  Slurm `/tmp` are NFS and unsuitable for rootless Apptainer unpacking; the
  build now uses tmpfs with one guarded fresh-directory retry.
- The predecessor bundle passed A100 job 1333, but that evidence is not claimed
  for v2 because its inference and fixture code changed.
- Docker build/test/save: pending; Docker is unavailable on the source cluster.

The synthetic fixture contains nine 16-minute H.264 clips at 5 fps and three
different resolutions. Every question reaches the final 96-frame cap. It covers
binary, number, percentage, FO-class, single-select, multi-select, and
multi-timestamp handling; one request starts at a non-zero absolute time and the
metadata adds a synthetic `Marker Band` class. It checks the bounded video
decoder, runtime definition parsing, prompt construction, model path, and
response contract; it does not measure model quality or production throughput.

## Why the V100 test differs from the final path

The challenge runs Procedure containers on one 80 GB H100 and allows a pooled
`120 + 30 × batch_size` seconds. The final image therefore defaults to BF16 and
96 frames. The Docker handoff machine has a 32 GB V100, which does not natively
support BF16. Use FP16 only as a `docker run` test override. First try all 96
frames. If that OOMs, repeat at 48 frames and preserve both logs. A reduced-frame
V100 pass validates packaging and compatibility only; it does not validate the
final H100/BF16/96-frame path.

## Docker-host procedure for Codex

Use the x86_64 Linux Docker machine with its 32 GB NVIDIA V100, NVIDIA Container
Toolkit, and at least 65 GB free disk. Start in this bundle directory.

1. Verify the transfer and prerequisites:

   ```bash
   uname -m
   docker version
   nvidia-smi
   sha256sum --check resources/weights.sha256
   python3 validate_bundle.py . --track procedure
   df -h .
   ```

   Require `x86_64`, five passing checksums, a visible V100, and a validator
   PASS. If Docker cannot expose the GPU, fix NVIDIA Container Toolkit; do not
   substitute a CPU-only model test.

2. Build and inspect the Docker image:

   ```bash
   /usr/bin/time -v ./do_build.sh 2>&1 | tee docker-build.log
   docker image inspect focus-procedure-qwen3-vl-4b-epoch8-v2 \
     --format '{{.Id}} {{.Architecture}} {{.Size}}'
   ```

   Confirm architecture `amd64`. Record image ID, size, duration, base-image
   digest, host driver, and warnings.

3. Confirm the image can see the V100:

   ```bash
   docker run --rm --gpus=all \
     --entrypoint python focus-procedure-qwen3-vl-4b-epoch8-v2 \
     -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list()); assert torch.cuda.is_available()'
   ```

   A CUDA/image compatibility failure here is a Docker-host limitation, not
   evidence that the challenge H100 path is invalid. Preserve the exact output.

4. Run the full 96-frame fixture in test-only FP16:

   ```bash
   FOCUS_TEST_MODEL_DTYPE=float16 FOCUS_TEST_MAX_FRAMES=96 \
     /usr/bin/time -v ./do_test_run.sh 2>&1 | tee docker-smoke-v100-fp16-96f.log
   python3 validate_output.py \
     --requests test/input/interface_1/request.json \
     --answers test/output/interface_1/answer.json \
     --require-nonempty
   ```

   Require nine unique non-empty responses, no per-question traceback, and validator
   PASS. Record model-ready time, total wall time, selected frames, and peak GPU
   memory from the inference log.

5. Only if step 4 fails specifically with CUDA OOM, repeat with 48 frames:

   ```bash
   FOCUS_TEST_MODEL_DTYPE=float16 FOCUS_TEST_MAX_FRAMES=48 \
     /usr/bin/time -v ./do_test_run.sh 2>&1 | tee docker-smoke-v100-fp16-48f.log
   python3 validate_output.py \
     --requests test/input/interface_1/request.json \
     --answers test/output/interface_1/answer.json \
     --require-nonempty
   ```

   Do not edit or rebuild the image to make FP16 or 48 frames permanent. Record
   the 96-frame OOM and label the 48-frame result as a reduced test path.

6. Prove the unchanged image still defaults to BF16 and 96 frames:

   ```bash
   docker image inspect focus-procedure-qwen3-vl-4b-epoch8-v2 \
     --format '{{range .Config.Env}}{{println .}}{{end}}' \
     | grep -Fx 'FOCUS_MODEL_DTYPE=bfloat16'
   docker image inspect focus-procedure-qwen3-vl-4b-epoch8-v2 \
     --format '{{range .Config.Env}}{{println .}}{{end}}' \
     | grep -Fx 'FOCUS_MAX_FRAMES=96'
   docker run --rm --platform=linux/amd64 \
     --entrypoint python focus-procedure-qwen3-vl-4b-epoch8-v2 \
     -c 'import sys,tarfile; tarfile.open("/opt/app/resources/python-vendor.tar").extractall("/tmp/vendor-probe"); sys.path.insert(0,"/tmp/vendor-probe"); import decord,peft,torch,transformers; print(decord.__version__,peft.__version__,torch.__version__,transformers.__version__)'
   docker history --no-trunc focus-procedure-qwen3-vl-4b-epoch8-v2
   ```

   Confirm the image has no dataset, optimizer state, cache, credentials, SIF,
   source-cluster absolute path, or duplicate checkpoint.

7. Save and verify the upload archive:

   ```bash
   ./do_save.sh 2>&1 | tee docker-save.log
   archive=$(find . -maxdepth 1 -name 'focus-procedure-qwen3-vl-4b-epoch8-v2_*.tar.gz' -print | sort | tail -1)
   gzip --test "${archive}"
   sha256sum "${archive}" | tee "${archive}.sha256"
   ls -lh "${archive}" "${archive}.sha256"
   ```

   `do_save.sh` rebuilds and refuses to save unless BF16 and 96 frames remain the
   image defaults. Never pass test overrides while saving.

8. Update this status section and `provenance.json` with actual Docker evidence.
   Keep the final archive, checksum, and logs for upload. Remove superseded images
   and archives only after confirming which archive will be submitted.

If any step fails, preserve the failing evidence, diagnose the exact cause, and
repeat. Never claim Docker validation from the source-cluster Apptainer test.
