# PVE NVIDIA GPU 主机重部署说明

本文记录这台主机重装后的 Proxmox VE、NVIDIA 驱动、GPU P2P、低功耗和 CT 部署配置，并附有一份自动部署脚本。

配套脚本：[deploy-pve-gpu-lab.sh](./deploy-pve-gpu-lab.sh)

CT100 的模型、磁盘布局、vLLM 构建和启动配置另见[完整恢复记录](./CT100-REDEPLOYMENT.md)。PVE host、CT100、GPU 映射、CUDA Toolkit 和上游 vLLM runtime 已重建；vLLM 服务保持停止，等待重新提供本地 NVFP4/FP8 主模型权重。

## 当前状态快照

以下是 2026-11-20 PVE 重装后的实际状态。脚本会从 NVIDIA 仓库安装执行时可用的最新驱动。

| 项目 | 已验证状态 |
|---|---|
| 系统 | Debian 13 `trixie`，Proxmox VE 9.2 |
| 最近运行内核 | `7.0.14-30-pve` |
| GPU | 4 张 NVIDIA Tesla T10，每张 16 GB |
| PCIe | GPU 具备 PCIe Gen3 x8 链路能力，连接到 Broadcom PEX8796 |
| NVIDIA 驱动 | `620.70.06-1`，NVIDIA 仓库当前最新候选；DKMS 模块已为运行内核构建 |
| CUDA / NCCL | CUDA nvcc 13.5、NVML/NVAPI 开发与运行库、NCCL 2.32.6 |
| 软件源 | Debian、PVE、可用的 Ceph 源和 CT 模板元数据切换到 USTC |
| P2P 修正 | `pex8796-gpu-p2p.service` 开机时沿 GPU 的 sysfs 父级关系发现 PEX8796，清除对应 ACS 控制位 |
| 省电 | `nvidia-pstated` 正常运行；持有约 263 MiB 显存、无计算时 GPU0 回到 P8；四卡空闲均为 P8 |
| 命令 | `/usr/local/bin/gpu-lab`，提供 `power`、`bench`、`ct` 子命令 |
| CT | CT100 Ubuntu 26.04 正在运行并设为 host 开机自启；rootfs 32 GiB 在 `local`，静态地址 `10.10.2.100`；四卡 NVIDIA 设备已映射 |
| 存储 | `/mnt/vllm` Samsung 32 GiB；`/mnt/kv-cache` Samsung 400 GiB 卷（`df` 可用约 393 GiB）、384 GiB 缓存上限；`/mnt/models` Toshiba 200 GiB |
| vLLM | 上游 `vLLM-2080Ti-Definitive` 固定 commit 已构建；PyTorch 2.13.0+cu130，4×SM75，FlashQLA 扩展通过构建自检 |
| 模型与服务 | NVFP4/FP8/DFlash2 均已在 CT `/mnt/models`；Health、smoke、三路并发和图片请求验证通过；API 验收后已停止 |
| KV 清理 | CT 内 `vllm-kv-cache-prune.timer` 每 15 秒检查；超过 384 GiB 后按最旧 `.bin` 块回收到 90% 水位 |

本次在 host 验证：`nvidia-smi topo -p2p p` 的所有 GPU 对均为 `OK`；nvbandwidth 的 GPU-to-GPU 单向 copy-engine 带宽约 6.57–6.67 GB/s，双向合计约 12.5 GB/s；SM 读/写约 6.67/5.53 GB/s。四卡 NCCL all-reduce 在 8 MiB–1 GiB 测试范围内完成，平均 bus bandwidth 5.45 GB/s，数据校验错误为 0。PCIe 测试期间服务会暂停，测试后已恢复；四卡空闲重新回到 P8。

Samsung 与 Toshiba 是独立 ext4 文件系统，无 RAID。PVE `local` 存放 CT rootfs 和模板；`samsung` 存放 vLLM、KV 卷与现存 swap；`toshiba` 存放模型。KV 卷创建为 400 GiB，ext4 扩展到完整卷并关闭 5% root 保留块，给 384 GiB 清理阈值留出文件系统余量。KV 定时清理器单独作为系统服务部署，未改动上游 vLLM 仓库。

## 自动部署

### 适用条件

- 在 Debian 13、Proxmox VE 9 的 x86_64 主机上以 root 执行。
- 执行前确保 PVE 配置、CT 和重要数据已有可恢复的备份。
- 部署过程会更改 APT 源、完整升级系统、安装驱动与编译工具、修改启动参数并安装 systemd 服务。
- 脚本不指定任何 GPU PCI 地址。ACS 服务通过 sysfs 动态追踪 GPU 的父设备，并且只处理识别为 Broadcom PEX8796 的上游端口。CT 设备也从当前 `nvidia-smi` 索引动态选择。

在本地工作区根目录把两个文件复制到 PVE 主机后运行。下面用 `PVE_IP` 代表主机地址，请替换成实际地址：

```bash
scp outputs/deploy-pve-gpu-lab.sh outputs/README.md root@PVE_IP:/root/
ssh root@PVE_IP
cd /root
chmod 0755 deploy-pve-gpu-lab.sh
./deploy-pve-gpu-lab.sh
```

脚本会要求输入 `YES`。完成部署后，交互式执行时会询问是否立即重启。需要无人值守运行时，可显式确认并要求脚本自动重启：

```bash
CONFIRM=YES AUTO_REBOOT=1 bash deploy-pve-gpu-lab.sh
```

如果没有设置 `AUTO_REBOOT=1`，非交互执行不会自动重启。重启会中断当前 SSH 会话；主机重启后可重新连接并继续验证。

脚本会在 `/root/pve-gpu-lab-backup-时间戳/` 保存修改前的 APT、GRUB、内核命令行和 PVE 模板元数据备份；CT 配置变更前则保存到 `/root/gpu-lab-ct-backups/`。

### 脚本所做的配置

1. 将 Debian、PVE no-subscription 和原先已配置的 Ceph 仓库切换到 USTC。
2. 将 PVE 和 TurnKey CT 模板元数据地址替换为 USTC；并设置 `pve-daily-update.service` 的 drop-in，在更新 TurnKey 模板列表后改写其中的旧镜像 URL。
3. 运行 `apt update`、`apt full-upgrade` 和 `apt autoremove --purge`。
4. 添加 NVIDIA Debian 13 仓库，安装仓库当前版本的 `nvidia-driver-cuda`、`nvidia-kernel-open-dkms`、`nvidia-modprobe`、`libnvidia-api1`、`cuda-nvcc-13-5`、`cuda-nvml-dev-13-5` 和 NCCL 开发包。
5. 屏蔽 nouveau，并写入下列启动参数：

   ```text
   intel_iommu=off iommu=pt pcie_acs_override=downstream,override pci=realloc=off
   ```

6. 安装 `pex8796-gpu-p2p.service`，开机时动态查找 GPU 上游 PEX8796 并关闭 ACS 控制。无需在脚本或配置中维护 GPU 的 PCI 地址。
7. 构建 NVIDIA 官方 `nvbandwidth` 和 `nccl-tests`，下载最新的 `nvidia-pstated` Linux x86_64 发布版，并启用其 systemd 服务。
8. 构建显存保持测试程序，并安装 `gpu-lab` 命令。
9. `gpu-lab ct` 会把 Ubuntu/Debian 的传统 `.list` 源和 deb822 `.sources` 源切换到 USTC；可为 Ubuntu 22.04/24.04/26.04 和 Debian 12/13 amd64 安装与主机驱动版本匹配的用户态库、CUDA runtime、NCCL，并在 CT 内测试 P2P 和显存保持下的 P8。

## 详细手动部署步骤

以下步骤适用于已经安装好 Debian 13 / Proxmox VE 9 的主机。请在 PVE 主机本地控制台或 root SSH 会话中逐步执行，并保持在同一个 root shell 里，以便后续代码块继续使用 `STAMP`、`BACKUP` 和 `CUDA_HOME`。不要把下面的配置直接用于其他 Debian/PVE 大版本。

自动部署与手动部署会得到同一套主要配置。手动步骤中最后安装的 `gpu-lab` 命令正文保存在配套自动脚本的 heredoc 中；文末提供了从该脚本提取正文的命令，避免两份实现发生偏差。

### 1. 备份当前配置并准备 USTC 软件源

```bash
test -x /usr/bin/pveversion
. /etc/os-release
test "$VERSION_CODENAME" = trixie

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="/root/pve-gpu-lab-backup-$STAMP"
mkdir -p "$BACKUP"
cp -a /etc/apt "$BACKUP/apt"
cp -a /etc/default/grub "$BACKUP/grub" 2>/dev/null || true
cp -a /etc/kernel/cmdline "$BACKUP/kernel-cmdline" 2>/dev/null || true
cp -a /usr/share/perl5/PVE/APLInfo.pm "$BACKUP/APLInfo.pm" 2>/dev/null || true
```

如果原来启用了 Ceph 仓库，先从旧源配置中记下它的发行代号：

```bash
CEPH_SUITE="$(grep -h -Eo 'ceph-(squid|reef|quincy|pacific)' \
  /etc/apt/sources.list.d/ceph.* 2>/dev/null | head -n1 || true)"
for f in pve-enterprise.list pve-enterprise.sources ceph.list ceph.sources; do
  if [ -e "/etc/apt/sources.list.d/$f" ]; then
    mv "/etc/apt/sources.list.d/$f" "/etc/apt/sources.list.d/$f.disabled-$STAMP"
  fi
done
```

写入 Debian 13 和 PVE 9 的 DEB822 源文件：

```bash
mkdir -p /etc/apt/sources.list.d
cat > /etc/apt/sources.list.d/debian.sources <<'EOF'
Types: deb
URIs: https://mirrors.ustc.edu.cn/debian
Suites: trixie trixie-updates
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: https://mirrors.ustc.edu.cn/debian-security
Suites: trixie-security
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF

cat > /etc/apt/sources.list.d/pve-no-subscription.sources <<'EOF'
Types: deb
URIs: https://mirrors.ustc.edu.cn/proxmox/debian/pve
Suites: trixie
Components: pve-no-subscription
Signed-By: /usr/share/keyrings/proxmox-archive-keyring.gpg
EOF

if [ -n "$CEPH_SUITE" ]; then
  cat > /etc/apt/sources.list.d/ceph.sources <<EOF
Types: deb
URIs: https://mirrors.ustc.edu.cn/proxmox/debian/$CEPH_SUITE
Suites: trixie
Components: no-subscription
Signed-By: /usr/share/keyrings/proxmox-archive-keyring.gpg
EOF
fi

if [ -f /etc/apt/sources.list ]; then
  cp -a /etc/apt/sources.list "$BACKUP/sources.list"
  : > /etc/apt/sources.list
fi
```

如果没有配置 Ceph，跳过 Ceph 源即可。安装 NVIDIA 驱动后还会添加 NVIDIA 官方仓库；Debian/PVE 源使用 USTC，NVIDIA 驱动仓库保留为 NVIDIA 官方地址。

### 2. 切换 PVE 与 TurnKey CT 模板镜像

PVE appliance 模板和 TurnKey 模板分别使用 Proxmox 与 TurnKey 的元数据地址。修改 PVE 文件并建立 TurnKey 缓存 URL 改写规则：

```bash
sed -i \
  -e 's|http://download.proxmox.com|https://mirrors.ustc.edu.cn/proxmox|g' \
  -e 's|https://releases.turnkeylinux.org|https://mirrors.ustc.edu.cn/turnkeylinux/metadata|g' \
  /usr/share/perl5/PVE/APLInfo.pm
systemctl restart pvedaemon

mkdir -p /etc/systemd/system/pve-daily-update.service.d
cat > /etc/systemd/system/pve-daily-update.service.d/update-turnkey-mirror.conf <<'EOF'
[Service]
ExecStopPost=/bin/sed -i 's|http://mirror.turnkeylinux.org|https://mirrors.ustc.edu.cn|g' /var/lib/pve-manager/apl-info/releases.turnkeylinux.org
EOF
systemctl daemon-reload
systemctl start pve-daily-update.service || true
if [ -f /var/lib/pve-manager/apl-info/releases.turnkeylinux.org ]; then
  sed -i 's|http://mirror.turnkeylinux.org|https://mirrors.ustc.edu.cn|g' \
    /var/lib/pve-manager/apl-info/releases.turnkeylinux.org
fi
```

PVE 更新 `pve-manager` 后，`APLInfo.pm` 可能被软件包覆盖；届时需要重新应用上面的两条 `sed` 替换。USTC 的 [Proxmox 镜像说明](https://mirrors.ustc.edu.cn/help/proxmox.html) 和 [TurnKey 模板镜像说明](https://mirrors.ustc.edu.cn/help/turnkeylinux.html) 描述了这些地址。

### 3. 更新系统并安装构建依赖

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y full-upgrade
DEBIAN_FRONTEND=noninteractive apt-get -y autoremove --purge
sed -i \
  -e 's|http://download.proxmox.com|https://mirrors.ustc.edu.cn/proxmox|g' \
  -e 's|https://releases.turnkeylinux.org|https://mirrors.ustc.edu.cn/turnkeylinux/metadata|g' \
  /usr/share/perl5/PVE/APLInfo.pm
systemctl restart pvedaemon
systemctl start pve-daily-update.service || true
if [ -f /var/lib/pve-manager/apl-info/releases.turnkeylinux.org ]; then
  sed -i 's|http://mirror.turnkeylinux.org|https://mirrors.ustc.edu.cn|g' \
    /var/lib/pve-manager/apl-info/releases.turnkeylinux.org
fi
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl git jq pciutils kmod build-essential dkms cmake pve-headers
```

升级 `pve-manager` 可能覆盖前一步对 `APLInfo.pm` 的修改，所以这里在升级之后重新写入模板镜像地址。

### 4. 添加 NVIDIA 仓库并安装最新可用驱动

这里安装仓库当前的 `nvidia-driver-cuda` 与开放内核模块包，不把驱动分支固定为 615。最近一次配置时最新候选版本为 `620.70.06-1`；以后执行时以 NVIDIA 仓库当时的候选版本为准。

```bash
curl -fL --retry 3 \
  https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
  -o /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb
apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install -y \
  nvidia-driver-cuda nvidia-kernel-open-dkms nvidia-modprobe \
  libnvidia-api1 cuda-nvcc-13-5 cuda-nvml-dev-13-5 libnccl2 libnccl-dev

apt-cache policy nvidia-driver-cuda nvidia-kernel-open-dkms
dpkg-query -W -f='${Package} ${Version}\n' \
  nvidia-driver-cuda nvidia-kernel-open-dkms cuda-nvcc-13-5 libnccl2 libnccl-dev
```

安装完成后，先不要急着测 P2P：还需要设置启动参数和 PEX8796 ACS。

### 5. 屏蔽 nouveau、更新 initramfs 并设置启动参数

写入 nouveau 屏蔽规则并更新 initramfs：

```bash
cat > /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
update-initramfs -u -k all
```

在 GRUB 主机上，把四个参数逐个补进 `GRUB_CMDLINE_LINUX_DEFAULT`，然后生成 GRUB 配置：

```bash
for arg in intel_iommu=off iommu=pt pcie_acs_override=downstream,override pci=realloc=off; do
  if ! grep -Fq "$arg" /etc/default/grub; then
    sed -i "/^[[:space:]]*GRUB_CMDLINE_LINUX_DEFAULT=/ s/\\\"$/ $arg\\\"/" /etc/default/grub
  fi
done
update-grub
```

若主机使用 Proxmox boot tool 而不是 GRUB，则把这四个参数补到 `/etc/kernel/cmdline` 同一行末尾，并刷新启动分区：

```bash
for arg in intel_iommu=off iommu=pt pcie_acs_override=downstream,override pci=realloc=off; do
  if ! grep -qw "$arg" /etc/kernel/cmdline; then
    sed -i "s/$/ $arg/" /etc/kernel/cmdline
  fi
done
proxmox-boot-tool refresh
```

确认开机后 `cat /proc/cmdline` 中出现上述参数。

### 6. 创建动态 PEX8796 ACS 启动服务

下列程序不包含任何 GPU PCI 地址。它从 `/sys/bus/pci/devices` 找到 NVIDIA 显示设备，沿设备树向上查找 Broadcom PEX8796（vendor/device ID `10b5:8796`），再对当时发现的父级端口关闭 ACS control：

```bash
cat > /usr/local/sbin/disable-pex8796-gpu-acs <<'EOF'
#!/usr/bin/env bash
set -eo pipefail
command -v setpci >/dev/null
found=0
for gpu in /sys/bus/pci/devices/*; do
  [ -r "$gpu/vendor" ] || continue
  [ "$(cat "$gpu/vendor")" = "0x10de" ] || continue
  class="$(cat "$gpu/class" 2>/dev/null || true)"
  case "$class" in
    0x030000|0x030200) ;;
    *) continue ;;
  esac
  node="$(readlink -f "$gpu")"
  while [ "$node" != / ]; do
    node="$(dirname "$node")"
    [ -r "$node/vendor" ] && [ -r "$node/device" ] || continue
    [ "$(cat "$node/vendor")" = "0x10b5" ] || continue
    [ "$(cat "$node/device")" = "0x8796" ] || continue
    bdf="$(basename "$node")"
    printf 'GPU %s is below dynamically found PEX8796 %s; clearing ACS control\n' \
      "$(basename "$gpu")" "$bdf"
    setpci -s "$bdf" ECAP_ACS+0x6.w=0000
    found=1
    break
  done
done
if [ "$found" -eq 0 ]; then
  echo "No NVIDIA GPU below a PEX8796 was found; no ACS register was changed."
fi
EOF
chmod 0755 /usr/local/sbin/disable-pex8796-gpu-acs

cat > /etc/systemd/system/pex8796-gpu-p2p.service <<'EOF'
[Unit]
Description=Clear ACS redirect on discovered PEX8796 ports below NVIDIA GPUs
After=systemd-udev-settle.service
Before=nvidia-pstated.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/disable-pex8796-gpu-acs
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable pex8796-gpu-p2p.service
```

关闭 ACS 会减弱 PCIe 设备隔离能力；此项是为当前裸机 GPU P2P 路径配置。不要把它当成多租户隔离配置。

### 7. 构建 P2P 带宽和 NCCL 测试工具

创建 CUDA 标准路径链接，再构建 NVIDIA 官方 `nvbandwidth`：

```bash
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
  NVCC="$(find /usr/local -type f -path '*/bin/nvcc' -print -quit)"
fi
NVCC="$(readlink -f "$NVCC")"
CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
if [ -e /usr/local/cuda ] && [ ! -L /usr/local/cuda ]; then
  mv /usr/local/cuda "/root/pve-gpu-lab-backup-$STAMP/local-cuda-directory"
fi
ln -sfn "$CUDA_HOME" /usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p /usr/local/src

git clone --depth 1 https://github.com/NVIDIA/nvbandwidth.git /usr/local/src/nvbandwidth
cmake -S /usr/local/src/nvbandwidth -B /usr/local/src/nvbandwidth/build \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build /usr/local/src/nvbandwidth/build --parallel "$(nproc)"
NVBANDWIDTH_BIN="$(find /usr/local/src/nvbandwidth/build -type f -name nvbandwidth -print -quit)"
test -n "$NVBANDWIDTH_BIN"
install -m 0755 "$NVBANDWIDTH_BIN" /usr/local/bin/nvbandwidth
```

再构建 NCCL tests：

```bash
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /usr/local/src/nccl-tests
make -C /usr/local/src/nccl-tests -j"$(nproc)" CUDA_HOME="$CUDA_HOME" NCCL_HOME=/usr \
  NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"
```

如果 `/usr/local/src/nvbandwidth` 或 `/usr/local/src/nccl-tests` 已存在，就进入原目录执行 `git pull --ff-only`，再运行对应的构建命令。NVIDIA 官方构建说明见 [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) 与 [nccl-tests](https://github.com/NVIDIA/nccl-tests)。

### 8. 安装 nvidia-pstated 并启用功耗管理

下载 GitHub 最新版发布的 Linux 二进制（当前资产名为 `nvidia-pstated`）：

```bash
PSTATED_URL="$(curl -fsSL https://api.github.com/repos/sasha0552/nvidia-pstated/releases/latest |
  jq -r '[.assets[] | select(.name == "nvidia-pstated" or
    ((.name | ascii_downcase | test("linux")) and
    (.name | ascii_downcase | test("amd64|x86_64|x86-64"))))] |
    .[0].browser_download_url // empty')"
test -n "$PSTATED_URL"
PSTATED_TMP="$(mktemp -d)"
curl -fL --retry 3 "$PSTATED_URL" -o "$PSTATED_TMP/release"
if tar -tzf "$PSTATED_TMP/release" >/dev/null 2>&1; then
  tar -xzf "$PSTATED_TMP/release" -C "$PSTATED_TMP"
  PSTATED_BIN="$(find "$PSTATED_TMP" -type f -name nvidia-pstated -print -quit)"
else
  PSTATED_BIN="$PSTATED_TMP/release"
fi
test -n "$PSTATED_BIN"
install -m 0755 "$PSTATED_BIN" /usr/local/bin/nvidia-pstated
rm -rf "$PSTATED_TMP"
```

创建 systemd 服务：

```bash
cat > /etc/systemd/system/nvidia-pstated.service <<'EOF'
[Unit]
Description=Automatically manage NVIDIA GPU performance states
After=pex8796-gpu-p2p.service
Wants=pex8796-gpu-p2p.service

[Service]
DynamicUser=yes
ExecStart=/usr/local/bin/nvidia-pstated
Restart=on-failure
RestartSec=1s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable nvidia-pstated.service
```

当前内核已能识别 GPU 时，可以立即启动服务：

```bash
systemctl start pex8796-gpu-p2p.service
systemctl start nvidia-pstated.service
```

如果 NVIDIA 模块要等重启后才能正常加载，此时只需确保服务已 enable，重启后再检查其状态。

### 9. 安装 gpu-lab 命令

`gpu-lab` 的三个子命令分别是 `power`、`bench` 和 `ct`。为保证手动部署与自动脚本中的实现完全一致，可把完整命令正文从配套部署脚本提取出来：

```bash
awk '
  $0 == "cat > /usr/local/bin/gpu-lab <<" "\\047" "GPU_LAB" "\\047" { copy=1; next }
  copy && $0 == "GPU_LAB" { exit }
  copy { print }
' ./deploy-pve-gpu-lab.sh > /usr/local/bin/gpu-lab
chmod 0755 /usr/local/bin/gpu-lab
```

如果部署脚本不在当前目录，请把命令最后的 `./deploy-pve-gpu-lab.sh` 换成它在主机上的实际路径。此命令只提取并安装子命令文件，不会运行整套部署。

### 10. 重启、验收与 CT 配置

更新引导并重启：

```bash
update-grub
systemctl reboot
```

重连主机后验收：

```bash
cat /proc/cmdline
nvidia-smi
systemctl status pex8796-gpu-p2p.service nvidia-pstated.service
gpu-lab power
nvidia-smi topo -p2p p
```

运行完整 P2P 带宽和 NCCL all-reduce 测试：

```bash
gpu-lab bench
```

给 CT 配置并测试 GPU 时运行：

```bash
gpu-lab ct
```

CT 子命令会交互询问 CT ID、GPU 索引、是否重启 CT、是否替换 APT 源以及是否安装容器用户态工具。它支持 Ubuntu 22.04/24.04/26.04 和 Debian 12/13 amd64，可为旧 `.list` 和 deb822 `.sources` 格式备份并替换发行版归档源；NVIDIA 官方仓库仍保留官方地址。通过 `pct exec` 安装时使用容器 root，不需要容器登录用户名或密码。P2P 测试后可运行 36 秒显存保持测试，观察主机 P-state、功耗和显存占用。

## 日常使用

在 PVE 主机的 root shell 中执行。

### 查看省电状态

```bash
gpu-lab power
```

命令会询问 GPU 索引、要保持的显存大小和测试时长，默认分配 128 MiB、运行 36 秒；期间每两秒采样 P-state、功耗、利用率和显存占用，并确认显存分配成功。空闲状态应最终进入 `P8`。本机在显存保持约 263 MiB 时观察到 GPU 回到 P8，功耗约为每张 10–11 W。`nvidia-pstated` 需要空闲观察时间，若尚未进入 P8，应延长等待并检查是否有其他 GPU 任务。

### 测量 P2P 和 NCCL

```bash
gpu-lab bench
```

此命令会暂时停止 `nvidia-pstated`，避免只进行 DMA 的 P2P 测试被当成“空闲”并降频；测试结束会恢复服务。测试内容包括：

- `nvidia-smi topo -p2p p` 的 P2P 能力矩阵。
- `nvbandwidth` 的 GPU 间复制带宽和传输验证。
- 所有可用 GPU 上的 NCCL all-reduce。

2026-11-20 重装后重新验证：四卡所有配对均可用，P2P copy-engine 单向读/写分别约 6.57/6.59 GB/s；四卡 all-reduce 平均 bus bandwidth 5.45 GB/s，数据校验错误为 0。负载运行时 `LnkSta` 为 PCIe 8 GT/s x8；GPU 空闲到 P8 时协商速率降至 2.5 GT/s x8，再做传输会升回 8 GT/s x8，所以空闲时看到 2.5 GT/s 不代表带宽测试也被锁在 Gen1。P2P 吞吐约为 Gen3 x8 编码后理论线速 7.88 GB/s 的 83%；拓扑矩阵全为 `PIX`、P2P 状态均 `OK`，这组实测没有显示 ACS 未关闭或 PCIe 降速问题。协议和复制实现仍会让有效负载低于线速。

补充说明：Broadcom 标注 PEX8796 为 PCIe Gen3、8 GT/s；PCI-SIG 说明 Gen3 使用 128b/130b 编码，因此 x8 单方向编码后理论线速约 7.88 GB/s，尚未扣除协议开销。此前 6.57–6.67 GB/s 的结果表示 P2P 路径已经工作，但还没有达到该理论上限；要继续追峰值，应核对当前协商的链路宽度/速率，并看 `nvbandwidth` 的具体 GPU 对结果。[Broadcom PEX8796](https://www.broadcom.com/products/pcie-switches-retimers/pcie-switches/pex8796), [PCI-SIG PCIe 3.0 rate and encoding](https://pcisig.com/what-bit-rate-pcie-30-and-how-does-it-compare-prior-generations-pcie)

### 为 CT 配置 GPU

```bash
gpu-lab ct
```

交互过程会：

1. 显示 CT 列表并询问 CT ID。
2. 显示当前主机 GPU 索引和型号，允许选择单卡、多卡或全部 GPU。
3. 备份 CT 配置，动态映射 `/dev/nvidia*` 设备；重复运行时会替换脚本管理的 NVIDIA 设备映射。
4. 对已停止的 CT 自动启动；对运行中的 CT 询问是否重启以应用设备映射。
5. 将常见 Ubuntu/Debian APT 归档源切换到 USTC，涵盖 `.list` 和 `.sources` 两种格式，并在改动文件旁保存备份。
6. 可选地为 Ubuntu 22.04/24.04/26.04 或 Debian 12/13 amd64 CT 安装与主机驱动版本匹配的 NVIDIA 用户态库、CUDA runtime 和 NCCL；不在 CT 内安装内核驱动。
7. 在 CT 内执行 `nvidia-smi`、P2P 矩阵和 `nvbandwidth` 测试；通过在 CT 内保持显存分配，检查主机是否仍能让空闲 GPU 进入 P8。

脚本通过 `pct exec` 以容器 root 身份运行安装命令，所以此流程不需要 CT 登录用户名或密码。其他发行版仍可配置设备映射，但用户态驱动需要按该容器发行版自行安装。

P2P 性能测试期间，脚本会暂时停止主机的 `nvidia-pstated`；退出后会恢复。CT 显存保持测试会在主机侧观察所选 GPU 的状态，确认显存占用上升后继续采样；预期在没有活跃计算时回到 P8。功耗管理由主机上的 `nvidia-pstated` 执行，不是在 CT 内运行守护进程。测试期间如果有其他任务持续使用这些 GPU，状态可能不会降到 P8。

### 手动切换 CT 软件源并安装 NVIDIA 用户态库

先选定 CT 并转换它正在使用的 Debian/Ubuntu 源文件。命令会备份有改动的文件，支持 `/etc/apt/sources.list`、`*.list` 和 deb822 `*.sources`：

```bash
CT_ID=100
pct exec "$CT_ID" -- sh -c '
set -eu
stamp=$(date +%Y%m%d-%H%M%S)
for file in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
  [ -f "$file" ] || continue
  if grep -Eiq "archive\\.ubuntu\\.com|security\\.ubuntu\\.com|ports\\.ubuntu\\.com|((deb|ftp|cdn-fastly)\\.)?debian\\.org|security\\.debian\\.org" "$file"; then
    cp -a "$file" "$file.gpu-lab.$stamp.bak"
    sed -E -i \\
      -e "s#https?://([[:alnum:]-]+\\.)*(archive|security|ports)\\.ubuntu\\.com/ubuntu(-ports)?/?#https://mirrors.ustc.edu.cn/ubuntu#g" \\
      -e "s#https?://security\\.debian\\.org(/debian-security)?/?#https://mirrors.ustc.edu.cn/debian-security#g" \\
      -e "s#https?://((deb|ftp|cdn-fastly)\\.)?debian\\.org/debian-security/?#https://mirrors.ustc.edu.cn/debian-security#g" \\
      -e "s#https?://((deb|ftp|cdn-fastly)\\.)?debian\\.org/debian/?#https://mirrors.ustc.edu.cn/debian#g" "$file"
  fi
done
'
pct exec "$CT_ID" -- apt-get update
```

如需在 CT 内手动安装用户态组件，按容器发行版选择 NVIDIA CUDA 仓库，然后安装与主机驱动版本前缀相同的 `libnvidia-compute`。Ubuntu 22.04/24.04/26.04 对应 `ubuntu2204`/`ubuntu2404`/`ubuntu2604`；Debian 12/13 对应 `debian12`/`debian13`：

```bash
CT_ID=100
REPO=ubuntu2604                 # 按上表修改
KEYRING="/tmp/cuda-keyring-ct-${CT_ID}.deb"
curl -fL --retry 3 \
  "https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/cuda-keyring_1.1-1_all.deb" \
  -o "$KEYRING"
pct push "$CT_ID" "$KEYRING" /tmp/cuda-keyring.deb
pct exec "$CT_ID" -- dpkg -i /tmp/cuda-keyring.deb
pct exec "$CT_ID" -- sh -c "printf '%s\\n' 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$REPO/x86_64/ /' > /etc/apt/sources.list.d/cuda-$REPO-x86_64.list"
pct exec "$CT_ID" -- apt-get -o APT::Update::Error-Mode=any update

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
BRANCH="${DRIVER%%.*}"
PKG=libnvidia-compute
PKG_VERSION="$(pct exec "$CT_ID" -- sh -c "apt-cache madison $PKG | awk '\$3 ~ /^${DRIVER}-/ {print \$3; exit}'" || true)"
if [ -z "$PKG_VERSION" ]; then
  PKG="libnvidia-compute-$BRANCH"
  PKG_VERSION="$(pct exec "$CT_ID" -- sh -c "apt-cache madison $PKG | awk '\$3 ~ /^${DRIVER}-/ {print \$3; exit}'")"
fi
test -n "$PKG_VERSION"
pct exec "$CT_ID" -- apt-get install -y "$PKG=$PKG_VERSION" cuda-cudart-13-5 libnccl2
```

这只安装 NVIDIA 用户态库、CUDA runtime 和 NCCL，不会在容器内安装内核驱动。完成后可运行 `pct exec "$CT_ID" -- nvidia-smi`、`pct exec "$CT_ID" -- nvidia-smi topo -p2p p`，再用 `gpu-lab ct` 执行带宽和显存保持功耗测试。

## 重启后的核对

```bash
cat /proc/cmdline
nvidia-smi
systemctl status pex8796-gpu-p2p.service nvidia-pstated.service
gpu-lab power
```

确认 `/proc/cmdline` 包含四个指定参数，`nvidia-smi` 能看到所有 GPU，两个服务均正常。然后运行：

```bash
nvidia-smi topo -p2p p
gpu-lab bench
```

`nvidia-smi topo -p2p p` 检查驱动报告的 P2P 能力；`gpu-lab bench` 进一步检查实际带宽与 all-reduce。

## P2P 故障背景与安全影响

最初的瓶颈是 PEX8796 下游端口的 ACS redirect 配置。仅添加 `pcie_acs_override` 不会修改交换芯片的 ACS 寄存器，因此内核分组变化并不等于 GPU P2P 路径已经修好。启动服务会在每次开机时根据 sysfs 设备树找到当前 GPU 上游端口，并动态清除 ACS 控制位。NVIDIA 的 NCCL 故障排查也将 ACS redirect 和裸机 IOMMU 列为 GPU Direct / PCIe P2P 性能问题的常见原因，并建议用实际复制测试确认链路。[NVIDIA NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)

当前配置关闭了 IOMMU，并关闭了 PEX8796 上相关 ACS 控制。这样做是为了此裸机多 GPU P2P 用途；它会降低 PCIe 设备隔离能力，不适合作为多租户安全边界。此配置用于 PVE CT 的设备映射，不等于把整张 PCI 设备以 VFIO 方式直通给虚拟机。

## 参考文档

- [USTC Ubuntu 镜像配置](https://mirrors.ustc.edu.cn/help/ubuntu.html)
- [USTC Debian 镜像配置](https://mirrors.ustc.edu.cn/help/debian.html)
- [USTC Proxmox 镜像配置和 CT 模板镜像说明](https://mirrors.ustc.edu.cn/help/proxmox.html)
- [USTC TurnKey 镜像及 PVE CT 模板配置说明](https://mirrors.ustc.edu.cn/help/turnkeylinux.html)
- [NVIDIA CUDA Linux 安装指南（受支持的 Ubuntu/Debian 版本）](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [NVIDIA Debian 驱动安装指南](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/debian.html)
- [NVIDIA NCCL GPU troubleshooting：P2P、IOMMU 和 ACS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- [NVIDIA nvbandwidth：构建、测试选项和数据验证](https://github.com/NVIDIA/nvbandwidth)
- [NVIDIA nccl-tests：NCCL 测试构建说明](https://github.com/NVIDIA/nccl-tests)
- [nvidia-pstated：GPU P-state 守护进程及 systemd 示例](https://github.com/sasha0552/nvidia-pstated)
