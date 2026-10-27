# Environment and Setup Log

Everything here was observed on the nodes during bring-up. Items marked **TODO** still need to be filled in.

## 1. Nodes used

| Role | Node | Hardware | OS / kernel |
|---|---|---|---|
| FPGA run node | `pc168` (CloudLab/OCT) | Alveo U280, HPE ProLiant DL380 Gen10, 64 cores, ~580 GB RAM | Ubuntu 22.04.2, 5.15.0-187-generic |
| GPU node | `pc179` (CloudLab) | Tesla V100-PCIE-32GB (PCI `b5:00.0`) | Ubuntu 22.04, 5.15.0-187-generic |
| Build machine | `fpga-tools` (profile `oct-build`) | 4 vCPU, 62 GB RAM, 63 GB disk | Linux |

CloudLab profile used for `pc168` and `pc179`: **TODO** (record exact profile names and node-type names).

## 2. FPGA node (`pc168`)

- XRT 2.16.204 (branch 2023.2)
- Shell: `xilinx_u280_gen3x16_xdma_base_1` (platform name `xilinx_u280_gen3x16_xdma_1_202211_1`), device BDF `0000:37:00.1`, Device Ready = Yes
- Check command: `xbutil examine`
- Not installed on this node: Vitis, Vivado, Docker, Vitis AI runtime (VART/XIR packages), `/opt/xilinx/overlaybins`, any `dpu.xclbin`
- Only platform present: `/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1`
- The shared project drive `/proj/octfpga-PG0` holds other users' Vitis AI source trees (U280 build flow, no prebuilt U280 DPU) and unrelated `.xclbin` files. Nothing there was modified.

## 3. GPU node (`pc179`)

- Driver 580.178.04 (`nvidia-smi` header shows CUDA 13.0), idle power about 23 W, power cap 250 W
- Python 3.10.12, virtual environment `~/venv-gpu`
- PyTorch 2.14.1+cu126 (see `gpu_ref/requirements-gpu.txt`)

Problems hit and fixes:

1. `nvidia-smi` failed although the driver packages were installed. `dmesg` showed the GPU was already bound to **nouveau**. Fix:
   ```
   printf 'blacklist nouveau\noptions nouveau modeset=0\n' | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
   sudo update-initramfs -u
   sudo reboot
   ```
2. The default `pip install torch` wheel (cu130) has no kernels for the V100 (compute capability 7.0), even though `torch.cuda.is_available()` returns True. Fix:
   ```
   pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
   Verified with a real GPU matrix multiply and `get_device_capability(0) == (7, 0)`.

## 4. Build machine (`fpga-tools`)

- Profile `oct-build`, Vitis and Vivado 2023.2 (`/fpga/Xilinx/...`)
- Platform present: `xilinx_u280_gen3x16_xdma_1_202211_1` (same name as the shell on `pc168`)
- Login prints harmless warnings about missing `r.sh` and `Model_Composer` settings scripts.
- Bitstreams are copied to `/proj/octfpga-PG0/chiragcs/` so the FPGA node can reach them before the build machine expires.

## 5. DPU build (DPUCAHX8H, U280)

Source: the DPUCAHX8H download linked from the Vitis AI 2.5 `reference_design` README (`.xo` kernels plus a Makefile-driven link flow). Its README says the XOs were generated with 2021.2.

Findings:

1. The package contains source and `.xo` files, not a ready bitstream.
2. Its Makefile builds U280 against `xilinx_u280_xdma_201920_3`, an older shell than the OCT card's.
3. The link script needs a per-shell file `sys_link_post_cleanup_Mike.<DEVICE>.tcl`. None exists for `202211_1`. The `201920_3` file is 0 bytes and the U55C one contains only a license header.

Changes made (originals untouched):

```
# Makefile: retarget the U280 platform
sed -i 's/override DEVICE = xilinx_u280_xdma_201920_3/override DEVICE = xilinx_u280_gen3x16_xdma_1_202211_1/' Makefile
# empty cleanup template for the new shell
touch bit_gen/constraints/sys_link_post_cleanup_Mike.xilinx_u280_gen3x16_xdma_1_202211_1.tcl
```

Build command (3 engines per SLR, 250 MHz):

```
make build ALVEO=u280 USE_2LV=false SLR0=3ENGINE SLR1=3ENGINE SLR2=3ENGINE ACLK_FREQ=250 DWC_EN=0 2>&1 | tee build_333_try2.log
```

Notes:

- The first attempt (`build_333.log`) stopped at the missing cleanup file. The second run is in progress.
- The last lines of the `build` recipe are leftover `u50lv` steps that do not apply to U280 and may report an error after `u280.xclbin` is produced. Check for the file rather than trusting the exit status.
- Placement constraints in `bit_gen/constraints` were tuned for the older shell, so timing closure is not guaranteed.

Status: **TODO** (record result, timing summary and the xclbin location).

## 6. Golden reference (GPU, FP32)

- Script: `gpu_ref/fp32_reference.py`, torchvision ResNet-50 `IMAGENET1K_V1` weights, FP32, batch 1
- Data: ImageNet validation images from the shared copy at `/proj/octfpga-PG0/validation` (50,000 flat files); labels built from the class folders under `/proj/octfpga-PG0/imagenet/val` (folder name = wnid, 1,000 classes, sorted order)
- Sample: 100 images, seed 0 (list saved in `data/image_list.txt`)
- Command:
  ```
  python fp32_reference.py --images /proj/octfpga-PG0/validation --labels ~/gpu_ref_data/labels.csv --synsets ~/gpu_ref_data/synsets.txt --num 100 --out results_100
  ```
- Result on the 100-image sample: Top-1 81%, Top-5 95% (small sample, so noisy; the weights score about 76% Top-1 on the full validation set)
- A second run with the same image list produced a byte-identical `fp32_predictions.csv`.

The torchvision weights differ from the Model Zoo's TensorFlow ResNet-50, so the comparison with the INT8 DPU model is cross-framework. State this in the write-up.

## 7. Open items

- Vitis AI version for the runtime, quantizer and compiler: **TODO**
- VART/XIR installation on `pc168` (no Docker): **TODO**
- Quantizer and compiler environment: **TODO**
- Exact CloudLab profile names and reservation details: **TODO**
