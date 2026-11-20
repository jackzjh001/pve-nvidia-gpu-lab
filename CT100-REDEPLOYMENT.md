# CT100 / Qwen3.8 vLLM 部署与恢复记录

本文补全 [PVE NVIDIA GPU 主机重部署说明](./README.md) 中 host 侧以外的部分，记录从 PVE host 到 CT100 当前模型服务状态的顺序。host 的 USTC 源、驱动、GRUB、PEX8796 ACS、P2P/NCCL 工具及 gpu-lab 安装步骤仍以主 README 为准。

**重要：重装 PVE 会清除原来的 LVM-thin、CT100、CT101、Fresh 快照和本地模型。** 必须先把模型和需要保留的配置复制到不在这台 PVE 本机磁盘上的备份目标。不要把密码或 GitHub token 写进备份文档。

## PVE 重装后的当前进度（2026-11-20）

PVE host 已按主 README 重装和配置完成，NVIDIA 驱动、GRUB 参数、P2P、NCCL 和显存保持下 P8 均已在 host 验证。CT 100 尚未重建，vLLM 服务也尚未恢复。

| PVE 存储 | 当前实际配置 | 用途 |
|---|---|---|
| `local` | `/var/lib/vz`，约 79.8 GiB 可用；内容为 import、vztmpl、iso | CT template；Ubuntu 26.04 template 已下载到此处 |
| `local-lvm` | `pve/data` LVM-thin，约 293 GiB 可用 | CT rootfs、`/mnt/vllm` 与 `/mnt/kv-cache` |
| `xg3` | `/mnt/pve/xg3` 上的独立 ext4 盘，约 466 GB（444 GiB）可用；PVE 内容设为 backup | `/mnt/pve/xg3/models` 模型目录与 PVE dump 备份 |

当前磁盘没有 RAID。`xg3` 盘上尚无模型权重；原来由用户上传的 NVFP4/FP8 两个模型需要重新提供或从其他备份恢复，DFlash2 可以从 ModelScope 重新下载。

当前 `local-lvm` 约 293 GiB，不足以安全创建原来的 16 GiB CT rootfs、80 GiB vLLM 卷和至少 400 GiB KV 卷（合计至少 496 GiB）。不要在 thin pool 中超额分配一个标称 400 GiB 的稀疏卷：底层 thin pool 会先耗尽，实际 KV 上限达不到 400 GiB。创建 CT 与 KV 卷前需要先调整目标容量或重新确认可接受的 cache 大小/磁盘角色。

记录时间：2026-11-20。最终配置通过上游 [vLLM 2080 Ti Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/93c9013e375b1ecbf2da88179d2a31175a0d5c99) 的 commit 93c9013e375b1ecbf2da88179d2a31175a0d5c99 构建；未使用个人 fork，也未修改上游源码。

## 1. PVE 重装前最后一次已验证的 CT 状态

| 项目 | 当前状态 |
|---|---|
| PVE host（重装前 CT 状态） | Debian 13 trixie / Proxmox VE 9.2 |
| host 地址 | 10.10.2.254 |
| GPU | 4× Tesla T10 16 GiB，SM75；host 驱动分支 620.70.06 |
| GPU P2P/省电（重装前） | PEX8796 ACS 动态处理；host nvidia-pstated active；空闲卡 P8 |
| CT（重装前） | ID 100，名称 vllm，Ubuntu 26.04，CT 地址 10.10.2.100；PVE 重装后已不存在 |
| CT rootfs | 16 GiB |
| CT 挂载 | /mnt/models 200 GiB；/mnt/vllm 80 GiB；/mnt/kv-cache 450 GiB（文件系统约 442 GiB） |
| NVIDIA/CUDA | host driver 620.70.06-1；CT APT cuda-toolkit-13-5 版本 13.5.0-1，nvcc 13.5 |
| vLLM runtime | vLLM 0.2.2.post3；PyTorch 2.13.0+cu130；CUDA runtime 13.0；4 GPU TP4 |
| 服务端点 | CT http://10.10.2.100:8000/v1；LAN 监听；最后验证后服务已停止 |
| API 模型 | Qwen3.8-27B，max_model_len=200000 |
| 磁盘 KV | 文件系统独立挂载，442 GiB 上限；当前测试缓存已清空，约 420 GiB 可用 |

### 模型目录

    /mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4  # 本地上传，NVFP4 W4A16，约 19 GiB
    /mnt/models/orcarouter/Qwen3.8-27B-Uncensored-FP8   # 本地上传，FP8，约 29 GiB
    /mnt/models/preetpatel/Qwen3.8-27B-DFlash2          # ModelScope incoai/Qwen3.8-27B-DFlash2，约 3.6 GiB

NVFP4、FP8 模型权重不是 GitHub 仓库的一部分。PVE 重装前必须另行备份这三个目录，或者确认有可重复访问的原始上传源。

### 活跃服务配置

当前默认 profile /etc/vllm-lab/profiles/nvfp4-dflash2-200k-3-image.env：

    MODEL_FAMILY=qwen35
    MODEL_VARIANT=nvfp4
    QUANTIZATION=compressed-tensors
    KV_CACHE_DTYPE=fp8
    MAX_MODEL_LEN=200000
    GPU_UTIL=0.96
    MAX_BATCHED_TOKENS=2048
    MAX_NUM_SEQS=3
    SPECULATIVE_METHOD=dflash
    SPECULATIVE_TOKENS=7
    MESSAGE_TYPE=text+image

服务启动时的其他关键参数：

    target:             /mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4
    draft:              /mnt/models/preetpatel/Qwen3.8-27B-DFlash2
    served name:        Qwen3.8-27B
    GPU indices:        0,1,2,3 (TP4；按 nvidia-smi 索引选择，不固定 PCI BDF)
    context:            200000
    max sequences:      3
    max batched tokens: 2048
    KV:                 FP8
    disk KV path:       /mnt/kv-cache
    CPU staging:        8589934592 bytes (8 GiB)
    image limit:        4 images; MM_LIMIT_JSON={"image":4,"video":0,"audio":0}
    port/scope:         8000 / LAN
    mode:               fast

最初 MAX_BATCHED_TOKENS=16384 时 GPU KV 池只有 506,557 tokens，启动器按 3 × 200000 的保守配置校验后拒绝保持服务。改为上游 4×T10 基准使用的 2048 后，GPU KV 池达到 **670,491 tokens**，校验通过；约 70,491 tokens 是三路满上下文之外的剩余 GPU KV。磁盘 KV 通过 OffloadingConnector 写到专用文件系统；vLLM 文件系统 tier 没有自身的驱逐或硬配额，442 GiB 专用卷的容量才是磁盘使用上限。没有设置定时删除任务。

图像输入已启用；此前一次图像请求处理约 3,003 个图像 tokens 并给出描述。DFlash2 draft 不接收目标模型的外部多模态 embedding，所以图片由目标模型处理，草稿侧使用 text-only 输入。

## 2. PVE 重装之前：复制数据到外部备份

将 /mnt/backup/... 替换成外接硬盘、NAS 或另一台机器上**不会被 PVE 安装器清除**的路径。先检查目标可用空间至少覆盖模型约 52 GiB、vLLM/runtime 约 10 GiB，另加 CT 根盘和配置备份所需空间。

~~~bash
BACKUP=/mnt/backup/pve-gpu-rebuild-2026-11
mkdir -p "$BACKUP"
set -o pipefail

# 记录 CT 的精确网络、CPU、内存、特权/映射和 mountpoint 配置。
pct config 100 > "$BACKUP/ct100.conf"
pct status 100 > "$BACKUP/ct100.status.txt"
pvesm status > "$BACKUP/pve-storage.txt"

# PVE host 不一定直接挂载 CT 的 /mnt/models 和 /mnt/vllm；从运行中的 CT
# 打 tar 流到 host 上的外部备份盘。vzdump 是否包含 mpX 由 backup= 决定，
# 因此仍然单独复制这两块数据卷。
pct exec 100 -- tar --numeric-owner --xattrs --acls --sparse -C /mnt -cf - models vllm \
  | tar --numeric-owner --xattrs --acls --sparse -C "$BACKUP" -xf -
mkdir -p "$BACKUP/ct-config/etc-vllm-lab" "$BACKUP/ct-config/bin"
pct exec 100 -- tar --numeric-owner --xattrs --acls -C / -cf - \
  etc/vllm-lab/profiles usr/local/bin/vllm-lab \
  | tar --numeric-owner --xattrs --acls -C "$BACKUP/ct-config" -xf -

# CT 及挂载卷数据都已单独导出，再归档 CT rootfs 到外部备份目标。
pct shutdown 100 --timeout 120 || true
vzdump 100 --mode stop --compress zstd --dumpdir "$BACKUP"

# host 配置可帮助恢复；其中不包含 SSH 密码或任何 token。
cp -a /etc/pve/lxc/100.conf "$BACKUP/ct-config/"
cp -a /etc/default/grub "$BACKUP/ct-config/" 2>/dev/null || true
cp -a /usr/local/bin/gpu-lab "$BACKUP/ct-config/" 2>/dev/null || true
sync
~~~

重装前 `/mnt/kv-cache` 是空的测试缓存卷，不需要备份其内容；旧环境使用独立 450 GiB 卷。新 PVE 上尚无 CT 或 KV 卷。旧 CT 的 Fresh snapshot 和保留用 CT101 都只在旧 PVE 存储中，除非另行导出，否则重装后不可恢复。

## 3. 装好全新 PVE host

1. 安装 Debian 13 / Proxmox VE 9.x，配置 host 网络为原来的 10.10.2.254。模板放 `local`，CT 系统盘、vLLM 与 KV cache 放 `local-lvm`；模型和备份放独立的 `xg3`。当前 `local-lvm` 容量不足以满足 400 GiB KV 目标，先按本节开头解决容量差额。
2. 按 [主 README 的 host 自动部署步骤](./README.md#自动部署) 运行 deploy-pve-gpu-lab.sh。这会切换 USTC 源、执行 apt update、full-upgrade 和清理、安装当时可用的 NVIDIA host driver 与 P2P/NCCL 工具、设置 GRUB 参数及启用 nvidia-pstated。
3. 重启 PVE host，检查 /proc/cmdline、nvidia-smi、pex8796-gpu-p2p.service、nvidia-pstated.service 和 gpu-lab power。host 的详细手动命令及 PCIe P2P 背景见主 README。

在本次记录中 host 是 driver 620.70.06-1、CUDA compiler 13.5、NCCL；这些是 2026-11-19/20 实测值。重新部署时 NVIDIA APT 仓库的最新驱动可能更高，CT helper 会按 host driver 匹配容器用户态库。

## 4. 恢复或重建 CT100

### 推荐：从外部 vzdump 归档恢复

先在外部备份目录确认只有一个 CT100 归档，然后将路径填入变量。恢复 rootfs 到 `local-lvm`：

~~~bash
BACKUP=/mnt/backup/pve-gpu-rebuild-2026-11
ls -lh "$BACKUP"/vzdump-lxc-100-*.tar.zst
ARCHIVE=/mnt/backup/pve-gpu-rebuild-2026-11/vzdump-lxc-100-YYYY_MM_DD-HH_MM_SS.tar.zst
pct restore 100 "$ARCHIVE" --storage local-lvm
pct config 100
~~~

核对 rootfs、net0、unprivileged、cores、memory 和所有 mpX。若 vzdump 没包含独立挂载卷，按下面的重建步骤添加新卷并从外部备份恢复文件。

### 无可用 CT 归档时：新建 CT 与挂载卷

用 PVE UI 创建 ID 100、名称 vllm、Ubuntu 26.04 amd64 CT；模板从 `local` 选择，rootfs 选 `local-lvm`。CPU/内存、网关、unprivileged 和桥接配置照外部备份中的 ct100.conf 恢复。rootfs 使用 16 GiB。不要给 rootfs 设置 idmap=passthrough：旧配置曾因此触发 LXC pre-start EOVERFLOW；GPU helper 会在发现该项时先备份并询问是否移除。

先创建 xg3 模型目录并作为 host bind mount 加进 CT。模型目录不作为 vzdump 的内容重复备份；xg3 上的模型和备份分别管理：

~~~bash
mkdir -p /mnt/pve/xg3/models /mnt/pve/xg3/dump
pvesm set xg3 --content backup
pct set 100 --mp0 /mnt/pve/xg3/models,mp=/mnt/models,backup=0
~~~

确认 `local-lvm` 的真实 thin pool 容量足够后，再创建 vLLM 与 KV 卷。以下 80/400 GiB 是原计划值；在当前 293 GiB pool 上**不要直接执行**：

~~~bash
pct set 100 --mp1 local-lvm:80,mp=/mnt/vllm
pct set 100 --mp2 local-lvm:400,mp=/mnt/kv-cache
pct config 100
~~~

挂载配置避免把 GPU PCI 地址写死。启动 CT 后，在 PVE host 运行：

~~~bash
gpu-lab ct
~~~

交互选择 CT 100 和 all（或四个当前 nvidia-smi GPU 索引），同意更新 CT 的 Ubuntu APT 源为 USTC、安装与 host driver 匹配的 NVIDIA 用户态库/CUDA runtime/NCCL，并执行 CT GPU 可见性、P2P 和显存保持下 P8 检查。通过 PVE pct exec 在 CT 里以 root 执行安装命令，不需要 CT 用户名或密码。此命令映射 /dev/nvidia* 字符设备，按运行时 GPU 索引选卡；它不是 VM 的 VFIO hostpci 设备直通。

从外部备份恢复 vLLM 数据；模型文件要恢复或重新上传到 host 的 `/mnt/pve/xg3/models`，再由 `/mnt/models` 映射进 CT：

~~~bash
BACKUP=/mnt/backup/pve-gpu-rebuild-2026-11
set -o pipefail
tar --numeric-owner --xattrs --acls --sparse -C "$BACKUP" -cf - vllm \
  | pct exec 100 -- tar --no-same-owner --xattrs --acls --sparse -C /mnt -xf -
~~~

检查 /mnt/models、/mnt/vllm 的目录映射及文件对 CT 内 vLLM 用户可读。然后安装/确认 CT 的完整 CUDA Toolkit（gpu-lab ct 默认装 runtime，不会装完整 toolkit）：

~~~bash
pct exec 100 -- apt-get update
pct exec 100 -- apt-get install -y cuda-toolkit-13-5
pct exec 100 -- /usr/local/cuda-13.5/bin/nvcc --version
pct exec 100 -- nvidia-smi
~~~

这是 Ubuntu 26.04 上本次通过 NVIDIA APT 仓库安装的 Toolkit 13.5.0-1。vLLM runtime 仍是上游定义的 CUDA 13.0 / PyTorch 2.13.0+cu130；Toolkit 13.5 提供本机编译器，并不把 PyTorch runtime 改成 cu135。

## 5. 恢复上游 vLLM runtime

如果 /mnt/vllm/ 是从外部备份完整恢复的，先核对 clone 和环境；否则重新 clone 固定版本并构建。不要改成个人 fork：

~~~bash
mkdir -p /mnt/vllm
cd /mnt/vllm
git clone https://github.com/weicj/vLLM-2080Ti-Definitive.git
cd /mnt/vllm/vLLM-2080Ti-Definitive
git checkout 93c9013e375b1ecbf2da88179d2a31175a0d5c99
~~~

GitHub 连接慢时再使用代理 clone：

~~~bash
git clone https://ghproxy.net/https://github.com/weicj/vLLM-2080Ti-Definitive.git
~~~

准备 Python/build 工具并编译 SM75 runtime。VENV_DIR、FLASHQLA_DIR 和 CUDA compiler 路径要保持在持久的 /mnt/vllm 和 /usr/local/cuda-13.5 上：

~~~bash
apt-get update
apt-get install -y build-essential cmake git curl jq python3.12-dev python3.12-venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/usr/local/cuda-13.5/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.5
export VENV_DIR=/mnt/vllm/runtime/.venv
export FLASHQLA_DIR=/mnt/vllm/deps/FlashQLA-SM70-SM75
cd /mnt/vllm/vLLM-2080Ti-Definitive
./build.sh
~~~

此 commit 的上游 release metadata 固定 runtime CUDA 13.0、Torch 2.13.0 和 Python 3.12，并面向 SM75 架构；build 结束应确认 CUDA 可用、4 张卡 capability 均为 7.5。build.sh 可能需要数十分钟，日志在构建输出中给出的 build-logs 路径。

验证 runtime：

~~~bash
/mnt/vllm/runtime/.venv/bin/python - <<'PY'
import torch, vllm
print("vLLM:", vllm.__version__)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
~~~

预期 Torch 为 2.13.0+cu130，CUDA runtime 为 13.0，4 卡 capability 为 (7, 5)。

## 6. 恢复 DFlash2 和启动配置

如果备份里没有草稿模型，使用与 runtime 同一个 venv 安装 ModelScope SDK 并重新下载：

~~~bash
/mnt/vllm/runtime/.venv/bin/python -m pip install modelscope==1.40.3
mkdir -p /mnt/models/preetpatel
/mnt/vllm/runtime/.venv/bin/python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "incoai/Qwen3.8-27B-DFlash2",
    local_dir="/mnt/models/preetpatel/Qwen3.8-27B-DFlash2",
)
PY
~~~

ModelScope 模型页：[incoai/Qwen3.8-27B-DFlash2](https://www.modelscope.cn/models/incoai/Qwen3.8-27B-DFlash2)。FP8 与 NVFP4 是用户本地上传的权重，按前述外部备份还原到各自目录。

建立本地 profile：

~~~bash
mkdir -p /etc/vllm-lab/profiles
cat > /etc/vllm-lab/profiles/nvfp4-dflash2-200k-3-image.env <<'EOF'
MODEL_FAMILY=qwen35
MODEL_VARIANT=nvfp4
QUANTIZATION=compressed-tensors
KV_CACHE_DTYPE=fp8
MAX_MODEL_LEN=200000
GPU_UTIL=0.96
MAX_BATCHED_TOKENS=2048
MAX_NUM_SEQS=3
SPECULATIVE_METHOD=dflash
SPECULATIVE_TOKENS=7
MESSAGE_TYPE=text+image
EOF
~~~

创建 /usr/local/bin/vllm-lab 环境 wrapper。上游 launcher.sh 会把 Triton/Inductor 编译缓存固定到 checkout 的 triton-cache/ 和 torchinductor-cache/；这两个由 launcher 管理的缓存目录留在 /mnt/vllm 内，不要挪走或重定向到旧 checkout 的绝对路径：

~~~bash
cat > /usr/local/bin/vllm-lab <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /mnt/vllm/cache/vllm /mnt/vllm/cache/xdg /mnt/vllm/cache/torch \
  /mnt/vllm/cache/inductor /mnt/vllm/cache/triton /mnt/vllm/cache/flashinfer \
  /mnt/vllm/cache/pycache /mnt/vllm/tmp /mnt/vllm/logs/launcher
export RUNTIME_ROOT=/mnt/vllm/runtime
export PROFILE_DIR=/etc/vllm-lab/profiles
export LOG_DIR=/mnt/vllm/logs/launcher
export STATE_FILE=/mnt/vllm/runtime/launcher-state.env
export VLLM_CACHE_ROOT=/mnt/vllm/cache/vllm
export FLASHQLA_ROOT=/mnt/vllm/deps/FlashQLA-SM70-SM75
export TORCH_EXTENSIONS_DIR=/mnt/vllm/cache/torch-extensions
export XDG_CACHE_HOME=/mnt/vllm/cache/xdg
export TORCH_HOME=/mnt/vllm/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/mnt/vllm/cache/inductor
export TRITON_CACHE_DIR=/mnt/vllm/cache/triton
export FLASHINFER_WORKSPACE_BASE=/mnt/vllm/cache/flashinfer
export PYTHONPYCACHEPREFIX=/mnt/vllm/cache/pycache
export TMPDIR=/mnt/vllm/tmp
export CUDA_HOME=/usr/local/cuda-13.5
export PATH=/usr/local/cuda-13.5/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONHASHSEED=0
exec /mnt/vllm/vLLM-2080Ti-Definitive/launcher.sh "$@"
EOF
chmod 0755 /usr/local/bin/vllm-lab
~~~

启动当前 NVFP4 + DFlash2 路由。选卡填 0,1,2,3 是当前 nvidia-smi 索引，不要把 PCI 地址写进配置：

~~~bash
/usr/local/bin/vllm-lab \
  --profile nvfp4-dflash2-200k-3-image \
  --set MODEL_DIR=/mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4 \
  --set SPECULATIVE_MODEL=/mnt/models/preetpatel/Qwen3.8-27B-DFlash2 \
  --set SERVED_NAME=Qwen3.8-27B \
  --set TP_SIZE=4 \
  --set GPU_DEVICES=0,1,2,3 \
  --set KV_DISK_CACHE_DIR=/mnt/kv-cache \
  --set KV_DISK_CPU_BYTES=8589934592 \
  --set MM_IMAGE_LIMIT=4 \
  --set 'MM_LIMIT_JSON={"image":4,"video":0,"audio":0}' \
  --set SERVICE_SCOPE=lan \
  --set PORT=8000 \
  --non-interactive
~~~

启动器会先计算目标和 draft 模型 fingerprint，再把 disk KV 写入 /mnt/kv-cache/checkpoint-<fingerprint>。该专用分区不与模型或 vLLM runtime 共用空间，因此缓存达到分区容量时不会填满模型盘或 rootfs。vLLM 本身不清除旧 fingerprint 目录；更换模型/不兼容 runtime 后，服务停止时查看 du -sh /mnt/kv-cache/checkpoint-*，确认后手动删除相应旧目录。

## 7. 验证和停服

启动器应输出 GPU KV cache tokens ≥ 600,000，并给出 KV concurrency check: OK。本机 2048 batch-token 配置实测 670,491。随后检查 API 与显卡：

~~~bash
curl -fsS http://127.0.0.1:8000/v1/models
nvidia-smi
~~~

/v1/models 必须报告 served model Qwen3.8-27B 和 max_model_len: 200000。启动器的 smoke check 和 4K/128、32K/512 基准可以验证基本推理；对图像及三路并发请求还应单独发请求验证。先前验证记录：smoke OK，三路短请求均返回，4K/128 平均 prefill/decode 约 1085/218 tok/s，32K/512 约 1037/213 tok/s。

验收结束后，使用 wrapper 菜单 9. Stop service 停止服务并确认。确认 host 上四张卡回到 P8、显存 0，再清理测试产生的 disk KV checkpoint 和 /dev/shm/vllm_offload_*.mmap 临时文件。JIT 编译缓存留在上游 checkout 中。当前服务默认保持停止；LAN API 地址为 http://10.10.2.100:8000/v1。

## 8. 重装前服务验证记录与重装后的待办

重装前，vLLM smoke test、三路并发、图像输入与上下文检查均通过，随后停止了 API 服务；测试 KV 文件和 `/dev/shm` mmap 已清理。配置完成后的新快照当时仍待确认，所以未创建。旧 CT101 和 LVM-thin snapshot 未导出，重装后不再存在。

重装后，PVE host 的 GPU 驱动、P2P 和省电状态已重新验证，但 CT100、vLLM runtime 和模型文件尚未恢复。继续前需要解决 `local-lvm` 400 GiB KV 容量不足，并恢复本地上传的 NVFP4/FP8 模型权重。

## 参考链接

- [上游项目固定 commit](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/93c9013e375b1ecbf2da88179d2a31175a0d5c99)
- [上游 build 和 launch 说明](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/93c9013e375b1ecbf2da88179d2a31175a0d5c99#build-and-launch)
- [上游非交互 launcher / disk KV 说明](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/93c9013e375b1ecbf2da88179d2a31175a0d5c99/docs/non-interactive-launch.md)
- [NVIDIA CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [ModelScope DFlash2 模型页](https://www.modelscope.cn/models/incoai/Qwen3.8-27B-DFlash2)

