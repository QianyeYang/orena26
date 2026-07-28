# Docker-host handoff

This folder is the complete x86_64 Docker build context for the Frame submission
`qwen3-vl-4b-lora-both-official-epoch30-20260725`. The source-cluster Apptainer
stage and the Docker-host stage are separate evidence; do not mark Docker as tested
until every check below passes on the Docker machine.

## Recorded source-cluster status

- Static bundle validation: passed on 2026-07-25.
- Local Apptainer build and offline GPU smoke test: passed on 2026-07-25.
- Temporary SIF, probe image, duplicate vendor archive, and build-cache cleanup:
  passed.
- Docker build/test/save: pending; Docker is unavailable on the source cluster.

Local evidence is in `test/apptainer-smoke-1037.log` and
`test/output/interface_1/answer.json`. The real epoch-30 model ran on an NVIDIA
A100-SXM4-80GB with Apptainer 1.5.3 and networking disabled. It produced 3/3
structurally valid responses with zero inference errors. Total process time was
104.93 seconds, model-ready time was 99.60 seconds, and peak allocated GPU memory
was 9.62 GiB. This three-question synthetic fixture validates packaging and I/O,
not accuracy or production-batch throughput.

## Docker-host procedure for Codex

Use the x86_64 Linux Docker machine with its 32 GB NVIDIA V100 GPU, NVIDIA
Container Toolkit, and at least 60 GB free disk. Start in this bundle directory.
The challenge image must remain BF16 by default. Because V100/Volta does not
natively support BF16 Tensor Cores, use the bundle's test-only FP16 runtime
override for the Docker-host smoke test. The override affects only that
`docker run`; it does not alter the image or its BF16 default.

1. Verify the transfer and prerequisites:

   ```bash
   uname -m
   docker version
   nvidia-smi
   sha256sum --check resources/weights.sha256
   python3 validate_bundle.py . --track frame
   df -h .
   ```

   `uname -m` must report `x86_64`; all four SHA-256 checks and static validation
   must pass. The V100 must be visible in `nvidia-smi`. If Docker cannot expose
   it, fix NVIDIA Container Toolkit rather than substituting a CPU-only test.

2. Build the Docker image:

   ```bash
   ./do_build.sh 2>&1 | tee docker-build.log
   docker image inspect focus-frame-qwen3-vl-4b-epoch30 \
     --format '{{.Id}} {{.Architecture}} {{.Size}}'
   ```

   Confirm architecture `amd64`. Record the image ID, compressed/uncompressed
   size, build duration, base-image digest, and any warnings in this file.

3. Confirm Docker can expose the V100:

   ```bash
   docker run --rm --gpus=all \
     --entrypoint python focus-frame-qwen3-vl-4b-epoch30 \
     -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list()); assert torch.cuda.is_available()'
   ```

   Require the V100 to be named and CUDA availability to be true. Preserve any
   failure output; do not mark Docker GPU testing as passed when this probe fails.

4. Run the supplied synthetic batch on the V100, with networking disabled and
   the test-only FP16 override:

   ```bash
   FOCUS_TEST_MODEL_DTYPE=float16 FOCUS_TEST_BATCH_SIZE=3 \
     ./do_test_run.sh 2>&1 | tee docker-smoke-v100-fp16.log
   python3 validate_output.py \
     --requests test/input/interface_1/request.json \
     --answers test/output/interface_1/answer.json
   ```

   Require three unique responses for `q001`, `q002`, and `q003`, no traceback,
   and a validator PASS. The inference log must say `dtype=float16` and name the
   V100. Also record total wall time and peak GPU memory. This fixture proves
   packaging, I/O, and V100/FP16 execution only; it does not validate accuracy or
   the final BF16 numerical path. The source-cluster A100 test above is the BF16
   runtime evidence.

5. Verify that the image itself still defaults to BF16, then inspect its
   offline/runtime properties:

   ```bash
   docker image inspect focus-frame-qwen3-vl-4b-epoch30 \
     --format '{{range .Config.Env}}{{println .}}{{end}}' \
     | grep -Fx 'FOCUS_MODEL_DTYPE=bfloat16'
   docker run --rm --platform=linux/amd64 \
     --entrypoint python focus-frame-qwen3-vl-4b-epoch30 \
     -c 'import sys,tarfile; tarfile.open("/opt/app/resources/python-vendor.tar").extractall("/tmp/vendor-probe"); sys.path.insert(0,"/tmp/vendor-probe"); import peft,torch,transformers; print(peft.__version__,torch.__version__,transformers.__version__)'
   docker history --no-trunc focus-frame-qwen3-vl-4b-epoch30
   ```

   Confirm no source dataset, optimizer state, checkpoint cache, credential, or
   absolute source path is present. Do not change the model or dependency versions
   merely to silence non-fatal warnings. Do not edit `Dockerfile` or
   `inference.py` to make FP16 permanent.

6. Save and verify the upload archive:

   ```bash
   ./do_save.sh 2>&1 | tee docker-save.log
   archive=$(find . -maxdepth 1 -name 'focus-frame-qwen3-vl-4b-epoch30_*.tar.gz' -print | sort | tail -1)
   gzip --test "${archive}"
   sha256sum "${archive}" | tee "${archive}.sha256"
   ls -lh "${archive}" "${archive}.sha256"
   ```

   `do_save.sh` rebuilds from the unchanged context and refuses to save unless the
   image default is BF16. Never pass the FP16 test override while saving.

7. Update the status section above and `provenance.json` with actual Docker
   evidence. Record the Docker smoke as **V100 32 GB / FP16 test override** and
   the saved image as **BF16 default**; do not conflate the two. Keep the final
   `.tar.gz`, its checksum, and logs for upload. Remove superseded images and
   archives only after confirming which archive will be submitted.

If any step fails, preserve the failing logs and image, diagnose the precise cause,
and repeat the build/test. Never label the package Docker-tested based only on the
source-cluster Apptainer result.
