# Cross-Platform ResNet-50 Benchmarking: Alveo U280 (Vitis AI DPU) vs NVIDIA V100

**Project Update 2.** Author: Chirag. Submission for grading: this repository, branch `project-update2`. The full report is in `docs/REPORT.md`.

## 1. Project description

This project compares ResNet-50 inference on two accelerators in the Open Cloud Testbed (CloudLab):

- **FPGA:** AMD Alveo U280 running AMD's DPUCAHX8H deep-learning processor (INT8, Vitis AI flow, VART host runtime)
- **GPU:** NVIDIA Tesla V100-PCIE-32GB running PyTorch (FP32 baseline) and TensorRT (FP32, FP16, INT8, INT8+FP16)

The questions are throughput, latency, performance per watt, and the accuracy cost of INT8 quantization. This is a measurement and systems-evaluation project: the DPU IP comes from AMD, and the work is getting it built and running, verifying correctness against an FP32 golden reference, and benchmarking fairly.

Planned evaluation (full project):

- Accuracy: Top-1 and Top-5 on a 5,000-image ImageNet validation subset; the INT8 path should be within 2% absolute of the FP32 reference.
- Performance: batch sizes 1 to 32, end-to-end and kernel-only latency, throughput, and power (`xbutil` vs `nvidia-smi`).

## 2. What is implemented in this update

1. **FP32 GPU golden reference** (`gpu_ref/fp32_reference.py`): torchvision ResNet-50 (`IMAGENET1K_V1`) on the V100, writing per-image Top-1/Top-5 labels with softmax scores and an aggregate summary. Two runs on the same image list gave a byte-identical predictions file.
2. **Golden outputs** (`data/`): 100-image and 5,000-image lists, predictions and summaries.
3. **GPU batch-size benchmark** (`gpu_ref/gpu_benchmark.py`): batch 1 to 32, pinned and pageable host memory, host-to-device, compute and device-to-host time measured separately, power from `nvidia-smi`. Results are in `data/bench/`.
4. **TensorRT path** (`trt/`): ONNX export, engine builds for FP32, FP16, INT8 and INT8+FP16 at batch 1 to 32, INT8 calibration on 500 validation images disjoint from the test list, a benchmark script, an accuracy script, and a layer-precision inspector.
5. **DPU build work for the U280** (`fpga/retarget_and_build.sh`, `docs/environment.md` section 5): retargets AMD's DPUCAHX8H package from shell `xilinx_u280_xdma_201920_3` to the OCT shell `xilinx_u280_gen3x16_xdma_1_202211_1`.

**Not yet done:** a working DPU `.xclbin` on the U280, the VART host application, quantization and compilation of the Model Zoo model, and the FPGA-vs-GPU comparison.

## 3. Results so far (V100)

Accuracy on the 5,000-image list:

| Path | Top-1 | Top-5 | Top-1 agreement with PyTorch FP32 |
|---|---|---|---|
| PyTorch FP32 | 75.02% | 92.92% | n/a |
| TensorRT FP32 | 75.02% | 92.92% | 100% |
| TensorRT FP16 | 75.00% | 92.94% | 99.9% |
| TensorRT INT8 | 74.98% | 93.00% | 96.8% |
| TensorRT INT8+FP16 | 75.10% | 92.84% | 98.0% |

On 100 images, PyTorch FP32 scores 81% Top-1 and 95% Top-5; that sample is too small to mean much. With 5,000 images, one image is 0.02%, so differences of about 0.1% are within sampling noise. Per-image agreement is the more sensitive check.

Performance at batch 32 (pinned memory, mean of 3 runs of 15 s, raw `nvidia-smi` board power):

| Path | Images/s | Images/s/W | End-to-end ms |
|---|---|---|---|
| PyTorch FP32 | 1,017 | 4.72 | 31.5 |
| TensorRT FP32 | 1,441 | 5.97 | 22.2 |
| TensorRT FP16 | 4,431 | 22.16 | 7.2 |
| TensorRT INT8 | 3,854 | 20.29 | 8.3 |
| TensorRT INT8+FP16 | 4,434 | 22.18 | 7.2 |

- Host-to-device copy at batch 32 is 1.49 ms, about 21% of end-to-end time for the TensorRT engines.
- At batch 1, PyTorch FP32 end-to-end latency is 2.91 ms and TensorRT INT8+FP16 is 1.02 ms.
- INT8 is slower than FP16 and INT8+FP16 matches FP16. Volta tensor cores have no INT8 mode, so no INT8 speedup is claimed on the V100.
- Idle power differed between sessions (36.5 W for the PyTorch runs, 24.7 W for the TensorRT runs). Reported images/s/W uses raw power, not idle-subtracted.

The torchvision weights differ from the Vitis AI Model Zoo's TensorFlow ResNet-50 (`tf_resnetv1_50_imagenet_224_224_6.97G_3.0`) used for the FPGA path, so the accuracy comparison is cross-framework. INT8 calibration uses validation images (disjoint from the test list) rather than training images.

**DPU build status:** not completed. `fpga/retarget_and_build.sh` has not been run successfully end to end; the first build VM setup failed on the cluster and the build must be rerun.

## 4. Files

```
README.md
gpu_ref/
  fp32_reference.py        FP32 ResNet-50 reference (GPU)
  gpu_benchmark.py         PyTorch batch-size benchmark with power
  requirements-gpu.txt     pip freeze of the working GPU environment
trt/
  export_onnx.py           PyTorch to ONNX export
  build_engines.py         TensorRT engine builder and INT8 calibration
  bench_trt.py             TensorRT batch-size benchmark with power
  eval_accuracy.py         Top-1/Top-5 and agreement for TensorRT engines
  inspect_layers.py        per-layer precision of a built engine
data/
  image_list.txt           the 100 images used (fixed sample, seed 0)
  image_list_5000.txt      the 5,000 images used (fixed sample, seed 0)
  calib_list_500.txt       INT8 calibration images, disjoint from the test lists
  labels_100.csv           ground-truth labels for the 100 images (filename,wnid)
  fp32_predictions.csv     golden output, 100 images: Top-1/Top-5 index and probability
  fp32_summary.json        accuracy and software versions, 100 images
  fp32_predictions_5000.csv   golden output, 5,000 images
  fp32_summary_5000.json      accuracy and software versions, 5,000 images
  bench/pinned, bench/pageable   GPU benchmark CSVs and metadata
docs/
  REPORT.md                Update 2 report
  environment.md           nodes, versions, setup problems and fixes, DPU build notes
  index.html               project site
fpga/
  retarget_and_build.sh    DPUCAHX8H build for the OCT U280 shell
```

The ImageNet validation images are not included. They are read from the shared OCT copy at `/proj/octfpga-PG0/validation`. Generated files (engines, ONNX, calibration cache, logs, benchmark and result folders) are not committed.

## 5. How to run

All commands run from the repository root on the V100 node.

### Environment

```
python3 -m venv ~/venv-gpu && source ~/venv-gpu/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
```

The cu126 wheel is required; the default PyTorch wheel has no V100 kernels. Or use `pip install -r gpu_ref/requirements-gpu.txt` with the same index URL. The TensorRT path uses TensorRT 10.4.0.

### Labels and synsets

```
LC_ALL=C ls /proj/octfpga-PG0/imagenet/val | sort > synsets.txt
find /proj/octfpga-PG0/imagenet/val \( -type f -o -type l \) -printf '%f,%h\n' | sed 's#,.*/#,#' > labels.csv
```

### GPU reference and golden check

```
python gpu_ref/fp32_reference.py --images /proj/octfpga-PG0/validation \
    --labels labels.csv --synsets synsets.txt --list data/image_list.txt --out results_100
diff results_100/fp32_predictions.csv data/fp32_predictions.csv && echo IDENTICAL
```

Expected: `IDENTICAL`; Top-1 81%, Top-5 95%. Use `--num 5000` instead of `--list` for the 5,000-image baseline (Top-1 75.02%, Top-5 92.92%).

### GPU benchmark

```
python gpu_ref/gpu_benchmark.py --help
```

Settings used: FP32, batches 1, 2, 4, 8, 16, 32, 15 s per run after 50 warm-up iterations, 3 repeats, pinned and pageable host memory, power from `nvidia-smi` every 100 ms.

### TensorRT engines, accuracy and benchmark

```
python trt/export_onnx.py
python trt/build_engines.py --precisions fp32 fp16 int8 int8fp16 --batches 1 2 4 8 16 32
python trt/eval_accuracy.py --help
python trt/bench_trt.py --help
python trt/inspect_layers.py trt/resnet50_int8fp16_b1.engine
```

`build_engines.py` expects `trt/resnet50_fp32.onnx` (from `export_onnx.py`), `data/calib_list_500.txt`, and the images at `/proj/octfpga-PG0/validation`. INT8 calibration runs once and is cached in `trt/int8_calib.cache`; pass `--recalibrate` to redo it.

### FPGA path (in progress)

1. Build on an OCT build machine (profile `oct-build`, Vitis 2023.2) with `fpga/retarget_and_build.sh` run from the root of AMD's DPUCAHX8H package. Target: Alveo U280, shell `xilinx_u280_gen3x16_xdma_1_202211_1`, 3 engines per SLR, 250 MHz. Copy the xclbin to the shared drive.
2. On the U280 node, load the xclbin and run the VART host application (not yet implemented).

## 6. Risks and blockers

1. **No DPU binary for the OCT shell.** The U280 run node has no Vitis AI runtime and no prebuilt DPU. AMD's package targets an older shell than the OCT card's, so it must be rebuilt and may fail timing.
2. **No Docker or VART on the run node.** The runtime must be built or provided. The Vitis AI quantizer and compiler environment is also unresolved.
3. **Cross-framework weights** (see section 3).
4. **Power measurement** differs between tools (`xbutil` vs `nvidia-smi`) and idle baselines; the sampling method is documented and repeated runs are averaged.

## 7. Plan for the next update

1. Finish the DPU build, or document why it cannot be done and agree on a fallback with the instructor.
2. Take `xbutil` baselines on the U280 (PCIe bandwidth, idle and loaded power).
3. Get the Vitis AI runtime working on the U280 node and run AMD's stock ResNet-50 example.
4. Quantize and compile the Model Zoo ResNet-50 for the built DPU, then run the 100-image check against the FP32 golden output.
5. Run the 5,000-image accuracy comparison and XRT trace checks, then the batch 1 to 32 sweep with PCIe DMA overhead quantified separately.
6. Final write-up and comparison.

## 8. Code not written by me

- AMD/Xilinx DPUCAHX8H package (Vitis AI 2.5 `reference_design` download): used for the build; only the Makefile device name and an empty cleanup file were changed.
- torchvision ResNet-50 `IMAGENET1K_V1` weights and preprocessing transform.
- TensorRT `IInt8EntropyCalibrator2` interface and the ONNX-parse and engine-build pattern follow NVIDIA's TensorRT documentation and samples.
- All other scripts in `gpu_ref/` and `trt/` were drafted with Claude (section 9) and then run and edited by me.

## 9. Use of AI

Claude (Anthropic) was used throughout this project:

- To help structure the project proposal (see the original proposal for that statement).
- To troubleshoot the cluster setup (the NVIDIA driver and nouveau conflict, the PyTorch CUDA wheel, the DPU build flow and its shell mismatch), by reading the command output I pasted.
- To draft `gpu_ref/` and `trt/` scripts, `docs/environment.md`, `docs/REPORT.md` and this README, and to review the repository against the assignment checklist.

I ran every command on the cluster myself, checked the results, and reviewed and edited the files before submitting. Claude could not access the nodes. The project idea, platform choices and technical decisions are mine.
