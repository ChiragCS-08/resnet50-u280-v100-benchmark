# Cross-Platform ResNet-50 Benchmarking: Alveo U280 (Vitis AI DPU) vs NVIDIA V100

**Project Update 1.** Author: Chirag. Submission for grading: this repository, `main` branch at the commit tagged `project-update-1` (**TODO**: confirm the tag or fork the course wants).

## 1. Project description

This project compares ResNet-50 inference on two accelerators in the Open Cloud Testbed (CloudLab):

- **FPGA:** AMD Alveo U280 running AMD's DPUCAHX8H deep-learning processor (INT8, Vitis AI flow, VART host runtime)
- **GPU:** NVIDIA Tesla V100-PCIE-32GB running PyTorch (FP32 baseline; optional INT8 TensorRT)

The questions are throughput, latency, performance per watt, and the accuracy cost of INT8 quantization. This is a measurement and systems-evaluation project: the DPU IP comes from AMD, and the work is getting it built and running, verifying correctness against an FP32 golden reference, and benchmarking fairly.

Planned evaluation (full project):

- Accuracy: Top-1 and Top-5 on a 5,000-image ImageNet validation subset; INT8 FPGA path should be within 2% absolute of the FP32 reference.
- Performance: batch sizes 1 to 32, end-to-end and kernel-only latency, throughput, and power (`xbutil` vs `nvidia-smi`).

## 2. What is implemented in this update

1. **Environment bring-up** on all three machine types (U280 run node, V100 node, OCT build machine), with the problems found and fixes documented in `docs/environment.md`.
2. **FP32 GPU golden reference** (`gpu_ref/fp32_reference.py`): torchvision ResNet-50 (`IMAGENET1K_V1`) on the V100, writing per-image Top-1/Top-5 labels with softmax scores and an aggregate summary. Verified reproducible: two runs on the same image list gave a byte-identical predictions file.
3. **100-image golden output** (`data/`): sample list, labels, predictions and summary.
4. **5,000-image FP32 baseline** run on the same script (summary reported below; large files are not committed).
5. **DPU build flow work** for the U280. Findings and the changes needed to retarget AMD's package to the OCT shell are in `docs/environment.md`, section 5.

**Not yet done:** a working DPU `.xclbin` on the U280 (build status below), the VART host application, quantization and compilation of the model, and the FPGA-vs-GPU comparison.

## 3. Results so far

| Run | Images | Top-1 | Top-5 |
|---|---|---|---|
| FP32 V100, 100 images (seed 0) | 100 | 81% | 95% |
| FP32 V100, 5,000 images (seed 0) | 5,000 | **TODO** | **TODO** |

The 100-image numbers are noisy because the sample is small; the weights are known to score about 76% Top-1 on the full validation set.

The torchvision weights differ from the Vitis AI Model Zoo's TensorFlow ResNet-50 (`tf_resnetv1_50_imagenet_224_224_6.97G_3.0`) used for the FPGA path, so the accuracy comparison is cross-framework. This is a known limitation.

**DPU build status:** **TODO** (running / failed with ... / finished; link the log or summary under `fpga/`).

## 4. Files

```
README.md
gpu_ref/
  fp32_reference.py        FP32 ResNet-50 reference (GPU)
  requirements-gpu.txt     pip freeze of the working GPU environment
data/
  image_list.txt           the 100 images used (fixed sample, seed 0)
  labels_100.csv           ground-truth labels for those images (filename,wnid)
  fp32_predictions.csv     golden output: Top-1/Top-5 index and probability per image
  fp32_summary.json        accuracy and software versions for the run
docs/
  environment.md           nodes, versions, setup problems and fixes, DPU build notes
fpga/
  (DPU build notes and the Makefile change; TODO)
```

The ImageNet validation images are not included. They are read from the shared OCT copy at `/proj/octfpga-PG0/validation`.

## 5. How to run

### GPU reference (V100 node)

1. Create the environment (the cu126 wheel is required; the default PyTorch wheel has no V100 kernels):
   ```
   python3 -m venv ~/venv-gpu && source ~/venv-gpu/bin/activate
   pip install --upgrade pip
   pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
   Or `pip install -r gpu_ref/requirements-gpu.txt` with the same index URL.
2. Make a synsets file (1,000 sorted class ids) and a labels file (`filename,wnid`) from the class-folder copy of ImageNet:
   ```
   LC_ALL=C ls /proj/octfpga-PG0/imagenet/val | sort > synsets.txt
   find /proj/octfpga-PG0/imagenet/val \( -type f -o -type l \) -printf '%f,%h\n' | sed 's#,.*/#,#' > labels.csv
   ```
3. Reproduce the 100-image golden output:
   ```
   python gpu_ref/fp32_reference.py --images /proj/octfpga-PG0/validation \
       --labels labels.csv --synsets synsets.txt --list data/image_list.txt --out results_100
   diff results_100/fp32_predictions.csv data/fp32_predictions.csv && echo IDENTICAL
   ```
   Use `--num 5000` instead of `--list` for the 5,000-image baseline.

Expected: `IDENTICAL`; Top-1 81%, Top-5 95% for the 100-image list.

### FPGA path (in progress)

Steps, once the DPU xclbin exists (see `docs/environment.md` section 5 for the build):

1. Build on an OCT build machine (profile `oct-build`, Vitis 2023.2) and copy the xclbin to the shared drive.
2. On the U280 node, load the xclbin and run the VART host application. **TODO**: not yet implemented.

## 6. Risks and blockers

1. **No DPU binary for the OCT shell.** The U280 run node has no Vitis AI runtime and no prebuilt DPU. AMD's package targets an older shell (`xilinx_u280_xdma_201920_3`) than the OCT card's (`xilinx_u280_gen3x16_xdma_1_202211_1`), so it must be rebuilt and may fail timing.
2. **No Docker or VART on the run node.** The runtime must be built or provided. The Vitis AI quantizer and compiler environment is also unresolved.
3. **Cross-framework weights** (see section 3).
4. **Power measurement** differs between tools (`xbutil` vs `nvidia-smi`); sampling method must be documented.

## 7. Plan for the rest of the semester

Dates are placeholders: **TODO** (fill in from the course schedule).

1. Finish the DPU build, or document why it cannot be done and agree on a fallback with the instructor.
2. Get the Vitis AI runtime working on the U280 node; run AMD's stock example to validate.
3. Quantize and compile the Model Zoo ResNet-50 for the built DPU; run the 100-image check against the FP32 golden output.
4. Layer 2: full 5,000-image accuracy comparison, and XRT trace checks before trusting timings.
5. Layer 3: batch-size sweep 1 to 32 on both platforms with latency, throughput and power; quantify PCIe DMA overhead separately.
6. Optional: INT8 TensorRT on the V100 for a quantized-vs-quantized comparison.
7. Final write-up and comparison.

## 8. Use of AI

Claude (Anthropic) was used throughout this update:

- To help structure the project proposal (see the original proposal for that statement).
- To troubleshoot the cluster setup (the NVIDIA driver and nouveau conflict, the PyTorch CUDA wheel, the DPU build flow and its shell mismatch), by reading the command output I pasted.
- To write the first draft of `gpu_ref/fp32_reference.py`, `docs/environment.md` and this README.

I ran every command on the cluster myself, checked the results, and reviewed and edited the files before submitting. Claude could not access the nodes. The project idea, platform choices and technical decisions are mine. **TODO**: edit this section so it accurately describes your own use.
