# CT100 / Qwen3.8 vLLM 部署与恢复记录

本文补全 [PVE NVIDIA GPU 主机重部署说明](./README.md) 中 host 侧以外的部分，记录从 PVE host 到 CT100 当前模型服务状态的顺序。host 的 USTC 源、驱动、GRUB、PEX8796 ACS、P2P/NCCL 工具及 gpu-lab 安装步骤仍以主 README 为准。

下文的磁盘与内存容量均按 GiB 记录；少数参数若使用 bytes，会明确写出单位。

**重要：重装 PVE 会清除原来的 LVM-thin、CT100、CT101、Fresh 快照和本地模型。** 必须先把模型和需要保留的配置复制到不在这台 PVE 本机磁盘上的备份目标。不要把密码或 GitHub token 写进备份文档。

## PVE 重装后的当前状态（2026-11-20）

PVE host、CT100、GPU 映射、CUDA Toolkit 与上游 vLLM runtime 均已重建。P2P、NCCL、显存占用时回到 P8 的省电行为已在 host 和 CT 验证。NVFP4/FP8/DFlash2 模型均已在 CT `/mnt/models`；启动 profile 完成 Health、smoke、三路并发和图像请求验收后，vLLM 服务已停止。

| PVE 存储 | 当前实际配置 | 用途 |
|---|---|---|
| `local` | `/var/lib/vz`；Ubuntu 26.04 template 与 CT100 的 32 GiB 系统盘 | 模板、CT rootfs |
| `samsung` | 独立 ext4，挂载 `/mnt/pve/samsung` | CT `/mnt/vllm` 32 GiB、`/mnt/kv-cache` 400 GiB（`df` 显示约 393 GiB 可用）；原有 swap 保留 |
| `toshiba` | 独立 ext4，挂载 `/mnt/pve/toshiba` | CT `/mnt/models` 200 GiB |

CT100 为 Ubuntu 26.04、12 vCPU、48 GiB RAM、8 GiB swap，地址 `10.10.2.100/16`，网关 `10.10.0.100`，已设为 host 开机自启。`gpu-lab ct` 通过当前 GPU 枚举映射四张 Tesla T10；配置不写死 PCI BDF。PVE host 没有 `local-lvm` thin pool，也没有 RAID。

KV 卷是 400 GiB；ext4 已扩展到整个卷，root 保留块设为 0%，`df` 显示约 393 GiB 可用，为 384 GiB 清理线留下约 9 GiB 余量。CT 的 `vllm-kv-cache-prune.timer` 每 15 秒统计磁盘 KV `.bin` 块，超过 384 GiB 时按最旧优先删除到 90% 水位（约 345.6 GiB）。这是独立清理器，不修改上游 vLLM 仓库。

host 的 `/`、Samsung、Toshiba 以及 CT 的 `/`、`/mnt/vllm`、`/mnt/kv-cache`、`/mnt/models` 均以 `noatime` 挂载。没有启用持续 `discard`：host 的 SSD 已由 `fstrim.timer` 每周 trim；CT 内的 Ubuntu fstrim 服务因容器条件不会运行，所以另设 host `pve-ct100-fstrim.timer` 每周调用 `pct fstrim 100`。首次 trim 释放了约 2.8 GiB 的 KV raw 文件宿主机分配空间；后续运行仍会安全检查并 trim 新释放的块。host 根盘是无 discard 能力的闪存设备，不配置 discard。

vLLM 由上游 [vLLM 2080 Ti Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/93c9013e375b1ecbf2da88179d2a31175a0d5c99) 的 commit `93c9013e375b1ecbf2da88179d2a31175a0d5c99` 构建；未使用个人 fork，也未修改上游源码。运行时为 vLLM 0.2.2.post3、PyTorch 2.13.0+cu130、4×SM75；CUDA Toolkit 13.5.0 与 FlashQLA SM75 扩展已通过构建自检。

## 1. 当前 CT100 与服务预设

| 项目 | 当前状态 |
|---|---|
| PVE host | Debian 13 trixie / Proxmox VE 9.2.2 |
| host 地址 | 10.10.2.254 |
| GPU | 4× Tesla T10 16 GiB，SM75；host 驱动分支 620.70.06 |
| GPU P2P/省电 | PEX8796 ACS 动态处理；host `nvidia-pstated` active；host/CT P2P 验证通过，空闲卡 P8 |
| CT100 | ID 100，名称 vllm，Ubuntu 26.04，地址 10.10.2.100；当前运行 |
| CT rootfs | local 上 32 GiB |
| CT 挂载 | /mnt/vllm Samsung 32 GiB；/mnt/kv-cache Samsung 400 GiB；/mnt/models Toshiba 200 GiB |
| NVIDIA/CUDA | host driver 620.70.06-1；CT `cuda-toolkit-13-5` 13.5.0-1，nvcc 13.5 |
| vLLM runtime | 上游 vLLM 0.2.2.post3；PyTorch 2.13.0+cu130；CUDA runtime 13.0；4 GPU TP4 构建自检通过 |
| 服务端点 | 预设为 CT `10.10.2.100:8000` LAN；目前没有运行 API 服务 |
| 启动预设 | Qwen3.8-27B NVFP4 + DFlash2，max_model_len=200000；launcher KV 容量检查通过，3 × 200k tokens |
| 磁盘 KV | 400 GiB 文件系统；独立定时清理器将 `.bin` KV 块限制在 384 GiB |
| CT root SSH | 公钥、密码登录均启用；`PermitRootLogin yes`、`PasswordAuthentication yes`、`KbdInteractiveAuthentication no`；密码不写入仓库 |
| ext4 挂载 | host 与 CT 的 ext4 均为 `noatime`；不启用连续 `discard`，host 每周 trim，CT100 通过 host 定时器每周 `pct fstrim 100` |

### 模型目录

    /mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4  # 本地上传，NVFP4 W4A16，约 19 GiB
    /mnt/models/orcarouter/Qwen3.8-27B-Uncensored-FP8   # 本地上传，FP8，约 29 GiB
    /mnt/models/preetpatel/Qwen3.8-27B-DFlash2          # ModelScope incoai/Qwen3.8-27B-DFlash2，约 3.6 GiB

NVFP4、FP8 模型权重不是 GitHub 仓库的一部分。两份本地模型现已上传到 CT 的 `/mnt/models`，DFlash2 位于 `/mnt/models/preetpatel/Qwen3.8-27B-DFlash2`（`model.safetensors` 约 3.59 GiB）。上传后核对了两份主模型每个文件的大小与源目录一致。

### 已准备的服务配置（API 当前停止）

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

此参数组合实测 GPU KV 池为 670,491 tokens，启动器通过 3 × 200000 并发容量校验；`/v1/models` 报告 `max_model_len=200000`，三路并发请求均成功。图像请求实际使用 3,003 个图像 tokens 并生成 256 个输出 tokens。验收完成后已停止 vLLM API。上游文件系统 KV tier 本身不限制磁盘增长；独立清理器每 15 秒按 384 GiB 限值检查 `.bin` 块，超限后按最旧块回收到 90% 水位。

图像输入在服务预设中已启用，限制为每次请求最多 4 张图。DFlash2 draft 不接收目标模型的外部多模态 embedding，所以图片由目标模型处理，草稿侧使用 text-only 输入。

### Profile 字段说明

`nvfp4-dflash2-200k-3-image.env` 是上游 launcher 的 route profile，只保留 launcher 白名单内的路由和推理字段：

| 字段 | 当前值 | 作用 |
|---|---|---|
| `MODEL_FAMILY` | `qwen35` | 选择 Qwen 3.5 系列的模型路由规则 |
| `MODEL_VARIANT` | `nvfp4` | 选择 NVFP4 主模型变体 |
| `QUANTIZATION` | `compressed-tensors` | 告诉 vLLM 使用 compressed-tensors 量化格式 |
| `KV_CACHE_DTYPE` | `fp8` | 将 GPU KV cache 存为 FP8，降低其显存占用 |
| `MAX_MODEL_LEN` | `200000` | 每个请求的最大上下文 token 数 |
| `GPU_UTIL` | `0.96` | vLLM 的 GPU 显存利用率目标 |
| `MAX_BATCHED_TOKENS` | `2048` | 单次调度批次的 token 上限 |
| `MAX_NUM_SEQS` | `3` | 最多并行调度 3 个请求序列 |
| `SPECULATIVE_METHOD` | `dflash` | 启用 DFlash speculative decoding |
| `SPECULATIVE_TOKENS` | `7` | 每轮由 draft 提议 7 个 token |
| `MESSAGE_TYPE` | `text+image` | 允许文本和图像消息输入 |

模型路径、draft 路径、服务名、TP/GPU 索引、磁盘 KV 目录、CPU staging、图片数量、端口和 LAN 绑定属于运行环境参数，不写进 route profile，而在启动时通过 `--set KEY=VALUE` 传入：`MODEL_DIR=/mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4`、`SPECULATIVE_MODEL=/mnt/models/preetpatel/Qwen3.8-27B-DFlash2`、`SERVED_NAME=Qwen3.8-27B`、`TP_SIZE=4`、`GPU_DEVICES=0,1,2,3`、`KV_DISK_CACHE_DIR=/mnt/kv-cache`、`KV_DISK_CPU_BYTES=8589934592`、`MM_IMAGE_LIMIT=4`、`SERVICE_SCOPE=lan`、`PORT=8000`。图片限制被 launcher 转成 `MM_LIMIT_JSON={"image":4,"video":0,"audio":0}`。此前曾把这些运行环境字段混放进 route profile；该 launcher 会拒绝白名单之外的 key，因此现已拆分并从 profile 移除，启动命令单独提供这些值。

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

重装前 `/mnt/kv-cache` 是空的测试缓存卷，不需要备份其内容；旧环境使用独立 450 GiB 卷。旧 CT 的 Fresh snapshot 和保留用 CT101 都只在旧 PVE 存储中，除非另行导出，否则重装后不可恢复。

## 3. 装好全新 PVE host

1. 安装 Debian 13 / Proxmox VE 9.x，配置 host 网络为 `10.10.2.254`。模板与 32 GiB CT 系统盘放 `local`；Samsung 的 `samsung` storage 放 32 GiB vLLM 卷和 400 GiB KV 卷；Toshiba 的 `toshiba` storage 放 200 GiB models 卷。磁盘为独立 ext4，无 RAID 或 local-lvm thin pool。
2. 按 [主 README 的 host 自动部署步骤](./README.md#自动部署) 运行 deploy-pve-gpu-lab.sh。这会切换 USTC 源、执行 apt update、full-upgrade 和清理、安装当时可用的 NVIDIA host driver 与 P2P/NCCL 工具、设置 GRUB 参数及启用 nvidia-pstated。
3. 重启 PVE host，检查 /proc/cmdline、nvidia-smi、pex8796-gpu-p2p.service、nvidia-pstated.service 和 gpu-lab power。host 的详细手动命令及 PCIe P2P 背景见主 README。

### 挂载 Samsung/Toshiba 并注册 PVE storage

先按型号查看磁盘与 UUID；使用 UUID 挂载，避免重装后 `/dev/nvme*` 编号变化导致挂错盘。以下 UUID 是本机 Samsung/Toshiba 分区的实测值；重新分区或格式化后请以 `blkid` 输出为准。不要格式化已有数据盘。

~~~bash
lsblk -o NAME,SIZE,MODEL,SERIAL,FSTYPE,UUID,MOUNTPOINTS
blkid
mkdir -p /mnt/pve/samsung /mnt/pve/toshiba
grep -q '20d7b846-1ee5-42ce-9f5f-fb7ef01e5bfc' /etc/fstab || \
  echo 'UUID=20d7b846-1ee5-42ce-9f5f-fb7ef01e5bfc /mnt/pve/samsung ext4 defaults 0 2' >> /etc/fstab
grep -q '8b0399bb-3673-46b8-a021-30768652e66e' /etc/fstab || \
  echo 'UUID=8b0399bb-3673-46b8-a021-30768652e66e /mnt/pve/toshiba ext4 defaults 0 2' >> /etc/fstab
mount -a
findmnt /mnt/pve/samsung /mnt/pve/toshiba
pvesm add dir samsung --path /mnt/pve/samsung --content images,rootdir
pvesm add dir toshiba --path /mnt/pve/toshiba --content images,rootdir
pvesm status
~~~

Samsung 上现有的 `swap.img` 与 fstab 启用配置要保留。若 `pvesm status` 已经显示 `samsung`/`toshiba`，不要重复执行对应的 `pvesm add`。

在本次记录中 host 是 driver 620.70.06-1、CUDA compiler 13.5、NCCL；这些是 2026-11-19/20 实测值。重新部署时 NVIDIA APT 仓库的最新驱动可能更高，CT helper 会按 host driver 匹配容器用户态库。

## 4. 恢复或重建 CT100

### 无 CT 归档时：按当前磁盘角色新建

先确认 PVE storage ID 为 `local`、`samsung`、`toshiba`，模板已下载至 `local`。下列数值均为 GiB：CT rootfs 32、vLLM 卷 32、KV 卷 400（缓存限额 384）、models 卷 200。CT 为 unprivileged LXC，不设置 `idmap=passthrough`。

~~~bash
pvesm status
pct create 100 local:vztmpl/ubuntu-26.04-standard_26.04-1_amd64.tar.zst \
  --hostname vllm --cores 12 --memory 49152 --swap 8192 \
  --rootfs local:32 \
  --mp0 samsung:32,mp=/mnt/vllm,backup=0 \
  --mp1 samsung:400,mp=/mnt/kv-cache,backup=0 \
  --mp2 toshiba:200,mp=/mnt/models,backup=0 \
  --net0 name=eth0,bridge=vmbr0,ip=10.10.2.100/16,gw=10.10.0.100 \
  --unprivileged 1 --onboot 1 --features nesting=1,keyctl=1
pct config 100
~~~

PVE 建立 KV 卷后，停止 CT 并把 ext4 扩展到完整 400 GiB，关闭默认 5% root 保留块。这样文件系统可用空间大于 384 GiB 清理阈值：

~~~bash
pct stop 100
KV_VOLID=$(pct config 100 | awk -F'[ ,]+' '$1 == "mp1:" {print $2}')
KV_IMAGE=$(pvesm path "$KV_VOLID")
LOOPDEV=$(losetup --find --show "$KV_IMAGE")
trap 'losetup -d "$LOOPDEV" 2>/dev/null || true' EXIT
e2fsck -f "$LOOPDEV"
resize2fs "$LOOPDEV"
tune2fs -m 0 "$LOOPDEV"
e2fsck -f "$LOOPDEV"
losetup -d "$LOOPDEV"
trap - EXIT
pct start 100
pct exec 100 -- df -h /mnt/vllm /mnt/kv-cache /mnt/models
~~~

启动后在 PVE host 运行 `gpu-lab ct`，交互选择 CT 100 和 `all`，同意更新 Ubuntu APT 源为 USTC、安装与 host driver 匹配的 NVIDIA 用户态库/CUDA runtime/NCCL，并执行 CT GPU 可见性、P2P 与显存保持下 P8 检查。通过 `pct exec` 在 CT 内以 root 执行命令，不需要 CT 用户名或密码。此方法映射 `/dev/nvidia*` 字符设备，按当前枚举使用 GPU，不写死 PCI 地址；它不是 VM VFIO `hostpci` 直通。

安装完整 CUDA Toolkit（`gpu-lab ct` 只装运行时）：

~~~bash
pct exec 100 -- apt-get update
pct exec 100 -- apt-get install -y cuda-toolkit-13-5
pct exec 100 -- /usr/local/cuda-13.5/bin/nvcc --version
pct exec 100 -- nvidia-smi
~~~

本次 Toolkit 为 13.5.0-1；上游 vLLM runtime 使用 CUDA 13.0 / PyTorch 2.13.0+cu130。Toolkit 13.5 提供本机编译器；构建脚本自检确认四张 GPU 的 capability 都是 7.5。

### 打开 CT root 的 SSH 公钥与密码登录

当前 CT 同时允许 root 公钥与密码登录。配置文件 `/etc/ssh/sshd_config.d/99-ct-root-password.conf` 内容如下；`KbdInteractiveAuthentication no` 保持键盘交互式认证关闭，密码认证由 `PasswordAuthentication yes` 提供。私钥和 root 密码都不要放入 Git 仓库：

~~~text
PermitRootLogin yes
PasswordAuthentication yes
KbdInteractiveAuthentication no
~~~

在 CT 内创建/修改配置，并交互式设置 root 密码：

~~~bash
cat > /etc/ssh/sshd_config.d/99-ct-root-password.conf <<'EOF'
PermitRootLogin yes
PasswordAuthentication yes
KbdInteractiveAuthentication no
EOF
passwd root
sshd -t
systemctl reload ssh
sshd -T | grep -E '^(permitrootlogin|passwordauthentication|kbdinteractiveauthentication) '
~~~

`passwd root` 会交互提示两次，不要把密码作为命令行参数或写进脚本。检查 `passwd -S root` 状态为 `P`，并从另一台机器通过密码认证实际登录一次。

如果只需要公钥登录，可改回 `PermitRootLogin prohibit-password`、`PasswordAuthentication no`，然后执行 `sshd -t && systemctl reload ssh`。私钥只保存在工作站，不放进 Git 仓库。重装后可重新创建密钥并把公钥放入 CT：

在 Windows PowerShell：

~~~powershell
$Key = Join-Path $env:USERPROFILE '.ssh\ct100-upload-ed25519'
ssh-keygen -t ed25519 -a 100 -f $Key -C ct100-upload
ssh root@PVE_IP 'install -d -m 700 /root/model-upload'
scp "$Key.pub" root@PVE_IP:/root/model-upload/ct100-upload-ed25519.pub
~~~

在 PVE host：

~~~bash
pct exec 100 -- systemctl enable --now ssh
pct exec 100 -- install -d -m 700 /root/.ssh
pct push 100 /root/model-upload/ct100-upload-ed25519.pub /root/ct100-upload.pub
pct exec 100 -- sh -c 'cat /root/ct100-upload.pub >> /root/.ssh/authorized_keys; chown root:root /root/.ssh /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; rm -f /root/ct100-upload.pub'
pct exec 100 -- sshd -T | grep -E '^(permitrootlogin|pubkeyauthentication) '
~~~

Windows 工作站验证 root SSH：

~~~powershell
ssh -i $Key root@10.10.2.100 'id -u; hostname'
~~~

### ext4 挂载参数与 CT 定期 TRIM

当前将 host 根文件系统、host 的 Samsung/Toshiba 文件系统，以及 CT rootfs 和三个 ext4 数据卷都设为 `noatime`。它避免每次读取文件都写访问时间，模型权重与 KV 文件这类读密集工作负载可少一些元数据写入。未启用连续 `discard`：它会把 discard 操作放进日常文件删除/释放路径，可能增加写入延迟；每周 fstrim 通常已足够。host 自带的 `fstrim.timer` 管理 host 文件系统，但 CT 内的 Ubuntu 定时器被容器运行条件跳过，所以由 host 增加 `pve-ct100-fstrim.timer`，每周调用一次 `pct fstrim 100`。CT 停止时 helper 安全跳过。

host 根盘不支持 discard；Samsung 和 Toshiba SSD 支持，且仍由 host 的周度 fstrim 使用。当前配置与安装命令如下（适用于本文记录的这台 PVE 主机）：

~~~bash
# root 文件系统：fstab 加 noatime 并立即重挂载
sed -i 's#^/dev/pve/root / ext4 errors=remount-ro 0 1$#/dev/pve/root / ext4 errors=remount-ro,noatime 0 1#' /etc/fstab
mount -o remount,noatime /

# systemd 管理的 Samsung/Toshiba mount 单元使用 drop-in，不改 PVE 原始 unit
mkdir -p /etc/systemd/system/mnt-pve-samsung.mount.d /etc/systemd/system/mnt-pve-toshiba.mount.d
printf '[Mount]\nOptions=defaults,noatime\n' > /etc/systemd/system/mnt-pve-samsung.mount.d/override.conf
printf '[Mount]\nOptions=defaults,noatime\n' > /etc/systemd/system/mnt-pve-toshiba.mount.d/override.conf
systemctl daemon-reload
mount -o remount,noatime /mnt/pve/samsung
mount -o remount,noatime /mnt/pve/toshiba

# PVE CT rootfs 与 mpX 均用 mountoptions 保存；停止/启动 CT 以重新挂载生效
pct set 100 --rootfs local:100/vm-100-disk-0.raw,size=32G,mountoptions=noatime
pct set 100 --mp0 samsung:100/vm-100-disk-0.raw,mp=/mnt/vllm,backup=0,size=32G,mountoptions=noatime
pct set 100 --mp1 samsung:100/vm-100-disk-1.raw,mp=/mnt/kv-cache,backup=0,size=400G,mountoptions=noatime
pct set 100 --mp2 toshiba:100/vm-100-disk-0.raw,mp=/mnt/models,backup=0,size=200G,mountoptions=noatime
pct shutdown 100 --timeout 60
pct start 100

# host timer 定期 trim CT 的 rootfs 和挂载卷；执行一次可立即检查
cat > /usr/local/sbin/pve-ct100-fstrim <<'EOF'
#!/bin/sh
set -eu
if /usr/sbin/pct status 100 | /usr/bin/grep -qx 'status: running'; then
    exec /usr/sbin/pct fstrim 100
fi
/usr/bin/logger -t pve-ct100-fstrim 'CT 100 is stopped; skipping weekly trim'
EOF
chmod 0755 /usr/local/sbin/pve-ct100-fstrim
cat > /etc/systemd/system/pve-ct100-fstrim.service <<'EOF'
[Unit]
Description=Trim CT 100 rootfs and mounted volumes
After=pve-guests.service
RequiresMountsFor=/mnt/pve/samsung /mnt/pve/toshiba

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/pve-ct100-fstrim
Nice=19
IOSchedulingClass=idle
EOF
cat > /etc/systemd/system/pve-ct100-fstrim.timer <<'EOF'
[Unit]
Description=Weekly TRIM for CT 100 sparse volumes

[Timer]
OnCalendar=weekly
Persistent=true
RandomizedDelaySec=30m
Unit=pve-ct100-fstrim.service

[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now pve-ct100-fstrim.timer
systemctl start pve-ct100-fstrim.service

# verify noatime on the host and inside CT
findmnt -no SOURCE,FSTYPE,OPTIONS / /mnt/pve/samsung /mnt/pve/toshiba
pct exec 100 -- findmnt -no SOURCE,FSTYPE,OPTIONS / /mnt/vllm /mnt/kv-cache /mnt/models
~~~

`pct fstrim 100` 会把 guest/CT 已释放的块通知到底层文件系统。首次运行从 `/mnt/kv-cache` 对应的 raw 文件释放了约 2.8 GiB 宿主机分配空间；后续若没有新增删除数据，trim 报告 0 B 也是正常结果。Debian [`fstrim(8)` 手册](https://manpages.debian.org/trixie/util-linux/fstrim.8.en.html)说明周度 trim 通常足够，并指出频繁 discard 可能有性能与设备寿命代价。

### 从工作站直接上传模型到 CT

模型盘挂载在 CT 的 `/mnt/models`。下面的 PowerShell 命令只传模型目录顶层文件，不重复传 Hugging Face `.cache`；先传入同一文件系统的临时目录，核对后再移动到正式目录：

~~~powershell
$Key = Join-Path $env:USERPROFILE '.ssh\ct100-upload-ed25519'
$NVFP4 = 'D:\LLM\preetpatel\Qwen3.8-27B-Uncensored-NVFP4'
$FP8 = 'D:\LLM\orcarouter\Qwen3.8-27B-Uncensored-FP8'
ssh -i $Key root@10.10.2.100 'mkdir -p /mnt/models/.incoming/preetpatel/Qwen3.8-27B-Uncensored-NVFP4 /mnt/models/.incoming/orcarouter/Qwen3.8-27B-Uncensored-FP8'
$NVFiles = Get-ChildItem -LiteralPath $NVFP4 -Force -File | ForEach-Object FullName
$FPFiles = Get-ChildItem -LiteralPath $FP8 -Force -File | ForEach-Object FullName
scp -i $Key @NVFiles root@10.10.2.100:/mnt/models/.incoming/preetpatel/Qwen3.8-27B-Uncensored-NVFP4/
scp -i $Key @FPFiles root@10.10.2.100:/mnt/models/.incoming/orcarouter/Qwen3.8-27B-Uncensored-FP8/
~~~

文件数与源目录一致后，在 CT 完成同文件系统内移动：

~~~powershell
$Key = Join-Path $env:USERPROFILE '.ssh\ct100-upload-ed25519'
ssh -i $Key root@10.10.2.100 'set -e; mkdir -p /mnt/models/orcarouter; test ! -e /mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4; test ! -e /mnt/models/orcarouter/Qwen3.8-27B-Uncensored-FP8; mv /mnt/models/.incoming/preetpatel/Qwen3.8-27B-Uncensored-NVFP4 /mnt/models/preetpatel/; mv /mnt/models/.incoming/orcarouter/Qwen3.8-27B-Uncensored-FP8 /mnt/models/orcarouter/; rmdir /mnt/models/.incoming/preetpatel /mnt/models/.incoming/orcarouter /mnt/models/.incoming; df -h /mnt/models'
~~~

### 安装 384 GiB KV 缓存自动清理

将仓库里的清理器与 systemd unit 复制到 PVE host，再用 `pct push` 放进 CT。清理器每 15 秒扫描 `.bin` KV 块；超过 384 GiB 时按最旧优先删除到 90% 水位（约 345.6 GiB）。它只作用于 `/mnt/kv-cache` 挂载，并拒绝在挂载缺失时运行。

在工作站执行：

~~~bash
ssh root@PVE_IP 'mkdir -p /root/gpu-lab-ct-assets/systemd /root/gpu-lab-ct-assets/profiles'
scp -r tools/vllm-kv-cache-prune.py tools/vllm-kv-cache-prune.env.example \
  tools/systemd tools/vllm-lab tools/profiles root@PVE_IP:/root/gpu-lab-ct-assets/
~~~

在 PVE host 将文件部署到 CT：

~~~bash
pct exec 100 -- mkdir -p /etc/default /etc/systemd/system /etc/vllm-lab/profiles
pct push 100 /root/gpu-lab-ct-assets/vllm-kv-cache-prune.py /usr/local/sbin/vllm-kv-cache-prune
pct push 100 /root/gpu-lab-ct-assets/vllm-kv-cache-prune.env.example /etc/default/vllm-kv-cache-prune
pct push 100 /root/gpu-lab-ct-assets/systemd/vllm-kv-cache-prune.service /etc/systemd/system/vllm-kv-cache-prune.service
pct push 100 /root/gpu-lab-ct-assets/systemd/vllm-kv-cache-prune.timer /etc/systemd/system/vllm-kv-cache-prune.timer
pct push 100 /root/gpu-lab-ct-assets/vllm-lab /usr/local/bin/vllm-lab
pct push 100 /root/gpu-lab-ct-assets/profiles/nvfp4-dflash2-200k-3-image.env /etc/vllm-lab/profiles/nvfp4-dflash2-200k-3-image.env
pct exec 100 -- chmod 0755 /usr/local/sbin/vllm-kv-cache-prune /usr/local/bin/vllm-lab
pct exec 100 -- systemctl daemon-reload
pct exec 100 -- systemctl enable --now vllm-kv-cache-prune.timer
pct exec 100 -- systemctl start vllm-kv-cache-prune.service
~~~

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

准备 Python/build 工具并编译 SM75 runtime。构建会使用 `/mnt/vllm` 保存虚拟环境、缓存与日志；以下将并发限制为 4，避免 CT 的编译峰值过高。Ubuntu 26.04 的系统 Python 不参与此环境，`uv` 会自行安装 Python 3.12：

~~~bash
apt-get update
apt-get install -y build-essential cmake git curl jq pkg-config
curl -fL --retry 5 --retry-all-errors --retry-delay 2 \
  https://releases.astral.sh/installers/uv/latest/uv-installer.sh \
  -o /tmp/uv-installer.sh
sh /tmp/uv-installer.sh
rm -f /tmp/uv-installer.sh
export PATH="$HOME/.local/bin:/usr/local/cuda-13.5/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.5
export VENV_DIR=/mnt/vllm/runtime/.venv
export FLASHQLA_DIR=/mnt/vllm/deps/FlashQLA-SM70-SM75
export TORCH_EXTENSIONS_DIR=/mnt/vllm/cache/torch-extensions
export UV_CACHE_DIR=/mnt/vllm/cache/uv
export BUILD_LOG_DIR=/mnt/vllm/logs/build
export BUILD_MAX_JOBS=4
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
cd /mnt/vllm/vLLM-2080Ti-Definitive
./build.sh
~~~

本次 CT 无法直连 PyTorch 官方 cu130 索引；PVE host 可以访问，所以临时通过 host `apt-cacher-ng` 只转发 PyTorch HTTPS 请求。若 CT 可直连 `download.pytorch.org`，跳过这段临时代理设置即可。

在 PVE host 暂时安装并配置代理：

~~~bash
apt-get install -y apt-cacher-ng
printf '%s\n' 'Port: 3142' \
  'PassThroughPattern: ^download(-r2)?\.pytorch\.org:443$' \
  >> /etc/apt-cacher-ng/acng.conf
systemctl restart apt-cacher-ng
~~~

将下面的代理变量加在上面构建命令的 `./build.sh` 前。USTC、Aliyun、SJTU 和 GitHub 流量仍直连；只有 `download*.pytorch.org` 走临时代理：

~~~bash
export HTTPS_PROXY=http://10.10.2.254:3142
export https_proxy="$HTTPS_PROXY"
export NO_PROXY=10.10.2.254,localhost,127.0.0.1,mirrors.aliyun.com,mirror.sjtu.edu.cn,github.com,githubusercontent.com,raw.githubusercontent.com,gh-proxy.com,ghfast.top
export no_proxy="$NO_PROXY"
~~~

构建完成后在 host 删除临时代理：

~~~bash
systemctl disable --now apt-cacher-ng
apt-get purge -y apt-cacher-ng
rm -rf /etc/apt-cacher-ng
~~~

此 commit 固定 runtime CUDA 13.0、Torch 2.13.0 和 Python 3.12，并面向 SM75。构建脚本确认 `torch.cuda.is_available()`、四张卡 capability 都是 7.5，并成功编译 FlashQLA legacy 扩展。由于本机 `nvcc` 是 Toolkit 13.5，构建包的本地版本标签为 `+cu135`，Torch runtime 仍为 `2.13.0+cu130`。完整模型的加载和推理要等本地权重恢复后验证。

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

预期 Torch 为 2.13.0+cu130、CUDA runtime 为 13.0，4 卡 capability 为 (7, 5)。

## 6. 恢复 DFlash2 和启动配置

如果备份里没有草稿模型，建立独立 ModelScope 工具环境并重新下载。独立环境避免改动 vLLM runtime 的 Python 依赖：

~~~bash
export UV_CACHE_DIR=/mnt/vllm/cache/uv
/root/.local/bin/uv venv --python 3.12 /mnt/vllm/runtime/modelscope-env
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
  /root/.local/bin/uv pip install \
  --python /mnt/vllm/runtime/modelscope-env/bin/python modelscope==1.40.3
mkdir -p /mnt/models/preetpatel
MODELSCOPE_CACHE=/mnt/models/.modelscope-cache \
  /mnt/vllm/runtime/modelscope-env/bin/python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "incoai/Qwen3.8-27B-DFlash2",
    local_dir="/mnt/models/preetpatel/Qwen3.8-27B-DFlash2",
)
PY
~~~

ModelScope 模型页：[incoai/Qwen3.8-27B-DFlash2](https://www.modelscope.cn/models/incoai/Qwen3.8-27B-DFlash2)。FP8 与 NVFP4 是用户本地上传的权重，按前述外部备份还原到各自目录。

建立本地 profile：

上游 route profile 只接受下列路由字段。模型目录、GPU 选择、服务端口和缓存目录等运行参数要在启动时通过 `--set` 传入，不要追加到这个 profile 文件，否则 launcher 会拒绝不支持的 profile key。

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

安装仓库提供的 `/usr/local/bin/vllm-lab` 环境 wrapper。它将 Triton、Inductor、FlashInfer 与 Python 编译缓存固定放在 `/mnt/vllm` 卷：

~~~bash
cat > /usr/local/bin/vllm-lab <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /mnt/vllm/cache/vllm /mnt/vllm/cache/xdg /mnt/vllm/cache/torch \
  /mnt/vllm/cache/inductor /mnt/vllm/cache/triton /mnt/vllm/cache/flashinfer \
  /mnt/vllm/cache/pycache /mnt/vllm/cache/torch-extensions \
  /mnt/vllm/tmp /mnt/vllm/logs/launcher
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
export PATH=/usr/local/cuda-13.5/bin:/mnt/vllm/runtime/.venv/bin:/usr/local/bin:/usr/bin:/bin
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
  --set SERVICE_SCOPE=lan \
  --set PORT=8000 \
  --non-interactive
~~~

启动器会先计算目标和 draft 模型 fingerprint，再把 disk KV 写入 `/mnt/kv-cache/checkpoint-<fingerprint>`。独立的 `vllm-kv-cache-prune.timer` 每 15 秒统计 `.bin` 块文件；超过 384 GiB 时按文件修改时间从旧到新删除，直到低于 90% 水位（约 345.6 GiB）。这会丢弃可重建的 KV 缓存块，不删除模型权重或其他文件。缓存写入突增仍可能短暂超过阈值；`df` 报告的约 393 GiB 可用空间为该清理器提供约 9 GiB 余量。

## 7. 验证和停服

模型权重恢复到 `/mnt/models` 后，确认 NVFP4 主模型和 DFlash2 文件齐全，再启动服务：

~~~bash
test -f /mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4/config.json
test -f /mnt/models/preetpatel/Qwen3.8-27B-Uncensored-NVFP4/model.safetensors
test -f /mnt/models/orcarouter/Qwen3.8-27B-Uncensored-FP8/model.safetensors.index.json
test -f /mnt/models/preetpatel/Qwen3.8-27B-DFlash2/model.safetensors
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
  --set SERVICE_SCOPE=lan \
  --set PORT=8000 \
  --non-interactive
~~~

启动器应输出 GPU KV cache tokens ≥ 600,000，并给出 KV concurrency check: OK。本次首次启动得到 670,491 tokens（≥ 3 × 200,000），Health check 和 smoke test 通过。随后检查 API 与显卡：

~~~bash
curl -fsS http://127.0.0.1:8000/v1/models
nvidia-smi
~~~

/v1/models 报告 served model `Qwen3.8-27B` 和 `max_model_len: 200000`。本次三路并发短请求均返回成功；图像请求实际处理约 3,003 个 image tokens，关闭本次请求的 thinking 后在 256 输出 tokens 内返回了描述。完整模型首启会编译 CUDA/FlashInfer 内核，可能需要数分钟。

验收结束后，使用 wrapper 菜单 9. Stop service 停止服务并确认。确认 host 上四张卡回到 P8、显存 0，再清理测试产生的 disk KV checkpoint 和 /dev/shm/vllm_offload_*.mmap 临时文件。JIT 编译缓存留在上游 checkout 中。当前服务默认保持停止；LAN API 地址为 http://10.10.2.100:8000/v1。

## 8. 当前待完成项

重装前的旧部署曾通过 smoke test、三路并发、图像输入与上下文检查，验收后 API 服务停止、测试 KV 文件清理。旧 CT101 与旧快照未导出，重装后不存在。

本次 host/CT/P2P/P8、Toolkit、vLLM build、三个模型、384 GiB KV 清理器和实际模型启动验收均已完成。Health、smoke、三路并发、200k 上下文报告与图片请求都通过。API 已按要求停止；需要提供服务时，从 CT 菜单选择 8 启动。

## 参考链接

- [上游项目固定 commit](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/93c9013e375b1ecbf2da88179d2a31175a0d5c99)
- [上游 build 和 launch 说明](https://github.com/weicj/vLLM-2080Ti-Definitive/tree/93c9013e375b1ecbf2da88179d2a31175a0d5c99#build-and-launch)
- [上游非交互 launcher / disk KV 说明](https://github.com/weicj/vLLM-2080Ti-Definitive/blob/93c9013e375b1ecbf2da88179d2a31175a0d5c99/docs/non-interactive-launch.md)
- [NVIDIA CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [ModelScope DFlash2 模型页](https://www.modelscope.cn/models/incoai/Qwen3.8-27B-DFlash2)

