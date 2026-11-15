#!/usr/bin/env bash
#
# Rebuild the PVE 9 / Debian 13 NVIDIA GPU host setup described in the
# companion Markdown. Review the script before running it on a host.
#
set -eo pipefail

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [ "$(id -u)" -ne 0 ]; then
  die "Run as root: bash deploy-pve-gpu-lab.sh"
fi
if [ ! -x /usr/bin/pveversion ]; then
  die "This script targets Proxmox VE 9 on Debian 13 (trixie)."
fi
. /etc/os-release
if [ "$VERSION_CODENAME" != "trixie" ]; then
  die "Detected $VERSION_CODENAME; expected Debian trixie."
fi

cat <<'NOTICE'
This will update APT sources, fully upgrade PVE, install the current NVIDIA
driver/CUDA test tools, change the kernel command line, disable ACS redirect
on discovered PEX8796 parent ports, and install system services and gpu-lab.
It creates backups of files it changes. The final reboot is opt-in.
NOTICE
if [ -t 0 ]; then
  read -r -p "Type YES to continue: " CONFIRM
else
  CONFIRM="$CONFIRM"
fi
[ "$CONFIRM" = "YES" ] || die "Confirmation not received."

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="/root/pve-gpu-lab-backup-$STAMP"
mkdir -p "$BACKUP"
log "Backing up current package and boot configuration to $BACKUP"
cp -a /etc/apt "$BACKUP/apt"
cp -a /etc/default/grub "$BACKUP/grub" 2>/dev/null || true
cp -a /etc/kernel/cmdline "$BACKUP/kernel-cmdline" 2>/dev/null || true
cp -a /usr/share/perl5/PVE/APLInfo.pm "$BACKUP/APLInfo.pm" 2>/dev/null || true
cp -a /etc/systemd/system/pve-daily-update.service.d "$BACKUP/pve-daily-update.service.d" 2>/dev/null || true

log "Configuring Debian 13 and Proxmox repositories at USTC"
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
for f in pve-enterprise.list pve-enterprise.sources ceph.list ceph.sources; do
  if [ -e "/etc/apt/sources.list.d/$f" ]; then
    mv "/etc/apt/sources.list.d/$f" "/etc/apt/sources.list.d/$f.disabled-$STAMP"
  fi
done
CEPH_SUITE="$(grep -h -Eo 'ceph-(squid|reef|quincy|pacific)' \
  "$BACKUP"/apt/sources.list.d/ceph.* 2>/dev/null | head -n1 || true)"
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

log "Switching Proxmox appliance/template metadata to USTC"
if [ -f /usr/share/perl5/PVE/APLInfo.pm ]; then
  sed -i \
    -e 's|http://download.proxmox.com|https://mirrors.ustc.edu.cn/proxmox|g' \
    -e 's|https://releases.turnkeylinux.org|https://mirrors.ustc.edu.cn/turnkeylinux/metadata|g' \
    /usr/share/perl5/PVE/APLInfo.pm
  systemctl restart pvedaemon || true
fi
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

log "Updating packages and applying full upgrade"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y full-upgrade
DEBIAN_FRONTEND=noninteractive apt-get -y autoremove --purge

log "Reapplying template mirror URLs after PVE packages were upgraded"
if [ -f /usr/share/perl5/PVE/APLInfo.pm ]; then
  sed -i \
    -e 's|http://download.proxmox.com|https://mirrors.ustc.edu.cn/proxmox|g' \
    -e 's|https://releases.turnkeylinux.org|https://mirrors.ustc.edu.cn/turnkeylinux/metadata|g' \
    /usr/share/perl5/PVE/APLInfo.pm
  systemctl restart pvedaemon || true
fi
systemctl start pve-daily-update.service || true
if [ -f /var/lib/pve-manager/apl-info/releases.turnkeylinux.org ]; then
  sed -i 's|http://mirror.turnkeylinux.org|https://mirrors.ustc.edu.cn|g' \
    /var/lib/pve-manager/apl-info/releases.turnkeylinux.org
fi

log "Installing kernel headers and build tools"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl git jq pciutils kmod build-essential dkms cmake \
  pve-headers

log "Adding NVIDIA's Debian 13 CUDA repository"
KEYRING_DEB="/tmp/cuda-keyring-$STAMP.deb"
curl -fL --retry 3 \
  https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
  -o "$KEYRING_DEB"
dpkg -i "$KEYRING_DEB"
rm -f "$KEYRING_DEB"
apt-get update

log "Installing latest repository driver plus CUDA/NCCL test dependencies"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  nvidia-driver-cuda nvidia-kernel-open-dkms nvidia-modprobe \
  cuda-nvcc-13-5 libnccl2 libnccl-dev

log "Blocking nouveau and adding the requested GRUB kernel arguments"
cat > /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
update-initramfs -u -k all
if [ -f /etc/default/grub ]; then
  cp -a /etc/default/grub "$BACKUP/grub-before-args"
  for arg in intel_iommu=off iommu=pt pcie_acs_override=downstream,override pci=realloc=off; do
    if ! grep -Fq "$arg" /etc/default/grub; then
      sed -i "/^[[:space:]]*GRUB_CMDLINE_LINUX_DEFAULT=/ s/\"$/ $arg\"/" /etc/default/grub
    fi
  done
  update-grub
elif [ -f /etc/kernel/cmdline ]; then
  cp -a /etc/kernel/cmdline "$BACKUP/kernel-cmdline-before-args"
  for arg in intel_iommu=off iommu=pt pcie_acs_override=downstream,override pci=realloc=off; do
    grep -qw "$arg" /etc/kernel/cmdline || sed -i "s/$/ $arg/" /etc/kernel/cmdline
  done
  proxmox-boot-tool refresh
else
  die "Could not find GRUB or proxmox-boot-tool kernel command line configuration."
fi

log "Installing dynamic PEX8796 ACS workaround"
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

log "Building NVIDIA nvbandwidth and NCCL tests"
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then
  NVCC="$(find /usr/local -type f -path '*/bin/nvcc' -print -quit 2>/dev/null || true)"
fi
[ -x "$NVCC" ] || die "nvcc was not found after installing cuda-nvcc-13-5."
NVCC="$(readlink -f "$NVCC")"
CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
if [ -e /usr/local/cuda ] && [ ! -L /usr/local/cuda ]; then
  mv /usr/local/cuda "$BACKUP/local-cuda-directory"
fi
ln -sfn "$CUDA_HOME" /usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p /usr/local/src
if [ -d /usr/local/src/nvbandwidth/.git ]; then
  git -C /usr/local/src/nvbandwidth pull --ff-only
else
  rm -rf /usr/local/src/nvbandwidth
  git clone --depth 1 https://github.com/NVIDIA/nvbandwidth.git /usr/local/src/nvbandwidth
fi
cmake -S /usr/local/src/nvbandwidth -B /usr/local/src/nvbandwidth/build \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build /usr/local/src/nvbandwidth/build --parallel "$(nproc)"
NVBANDWIDTH_BIN="$(find /usr/local/src/nvbandwidth/build -type f -name nvbandwidth -print -quit)"
[ -n "$NVBANDWIDTH_BIN" ] || die "nvbandwidth build did not produce an executable."
install -m 0755 "$NVBANDWIDTH_BIN" /usr/local/bin/nvbandwidth
if [ -d /usr/local/src/nccl-tests/.git ]; then
  git -C /usr/local/src/nccl-tests pull --ff-only
else
  rm -rf /usr/local/src/nccl-tests
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /usr/local/src/nccl-tests
fi
make -C /usr/local/src/nccl-tests -j"$(nproc)" CUDA_HOME="$CUDA_HOME" NCCL_HOME=/usr

log "Installing the latest nvidia-pstated Linux x86_64 release"
PSTATED_URL="$(curl -fsSL https://api.github.com/repos/sasha0552/nvidia-pstated/releases/latest |
  jq -r '[.assets[] | select((.name | ascii_downcase | test("linux")) and
    (.name | ascii_downcase | test("amd64|x86_64|x86-64")))] |
    .[0].browser_download_url // empty')"
[ -n "$PSTATED_URL" ] || die "No Linux x86_64 nvidia-pstated release asset was found."
PSTATED_TMP="$(mktemp -d)"
curl -fL --retry 3 "$PSTATED_URL" -o "$PSTATED_TMP/release"
if tar -tzf "$PSTATED_TMP/release" >/dev/null 2>&1; then
  tar -xzf "$PSTATED_TMP/release" -C "$PSTATED_TMP"
  PSTATED_BIN="$(find "$PSTATED_TMP" -type f -name nvidia-pstated -print -quit)"
else
  PSTATED_BIN="$PSTATED_TMP/release"
fi
[ -n "$PSTATED_BIN" ] && [ -f "$PSTATED_BIN" ] || die "Could not extract nvidia-pstated."
install -m 0755 "$PSTATED_BIN" /usr/local/bin/nvidia-pstated
rm -rf "$PSTATED_TMP"
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

log "Installing gpu-lab user command"
cat > /usr/local/bin/gpu-lab <<'GPU_LAB'
#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'USAGE'
Usage: gpu-lab <command>

  power       Show each discovered NVIDIA GPU's P-state, power, and VRAM
  bench       Pause nvidia-pstated, test P2P bandwidth, then NCCL all-reduce
  ct          Interactively pass selected GPUs into a Proxmox CT and verify it
USAGE
}

need_root() {
  [ "$(id -u)" -eq 0 ] || { echo "Run as root." >&2; exit 1; }
}

gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader | wc -l
}

power_report() {
  nvidia-smi --query-gpu=index,name,pstate,power.draw,memory.used,memory.total \
    --format=csv
}

pause_pstated() {
  if systemctl is-active --quiet nvidia-pstated.service; then
    systemctl stop nvidia-pstated.service
    PSTATED_WAS_ACTIVE=1
  else
    PSTATED_WAS_ACTIVE=0
  fi
}

resume_pstated() {
  if [ "$PSTATED_WAS_ACTIVE" = "1" ]; then
    systemctl start nvidia-pstated.service || true
  fi
}

add_ct_device() {
  local node="$1"
  if printf '%s\n' "$config" | grep -Fq "$node"; then
    return 0
  fi
  while printf '%s\n' "$config" | grep -q "^dev$slot:"; do
    slot=$((slot + 1))
  done
  pct set "$id" "--dev$slot" "$node,mode=0666"
  config="$(pct config "$id")"
  slot=$((slot + 1))
}

bench_host() {
  local count
  need_root
  command -v nvidia-smi >/dev/null || { echo "NVIDIA driver is not loaded." >&2; exit 1; }
  count="$(gpu_count)"
  [ "$count" -gt 0 ] || { echo "No NVIDIA GPUs detected." >&2; exit 1; }
  trap resume_pstated EXIT INT TERM
  pause_pstated
  echo "=== PCIe P2P availability ==="
  nvidia-smi topo -p2p p
  echo "=== Pairwise GPU bandwidth and transfer verification ==="
  /usr/local/bin/nvbandwidth -p device_to_device -b 512 -i 5
  echo "=== NCCL all-reduce ($count GPUs) ==="
  /usr/local/src/nccl-tests/build/all_reduce_perf -b 8M -e 1G -f 2 -g "$count"
}

ct_command() {
  local id config state choice selected all_gpu slot node branch keyring host_driver_version
  local install_userspace ct_arch ct_codename
  need_root
  command -v pct >/dev/null || { echo "pct is required on the PVE host." >&2; exit 1; }
  command -v nvidia-smi >/dev/null || { echo "NVIDIA driver is not loaded." >&2; exit 1; }
  echo "=== Available containers ==="
  pct list
  read -r -p "CT ID: " id
  pct config "$id" >/dev/null
  config="$(pct config "$id")"
  state="$(pct status "$id" | awk '{print $2}')"
  echo "=== Host GPUs (indexes are discovered at runtime) ==="
  nvidia-smi --query-gpu=index,name --format=csv
  all_gpu="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  read -r -p "GPU indexes to pass through (comma list, or all) [$all_gpu]: " selected
  if [ -z "$selected" ]; then selected="all"; fi
  if [ "$selected" = "all" ]; then selected="$all_gpu"; fi
  for item in $(echo "$selected" | tr ',' ' '); do
    case "$item" in *[!0-9]*|'') echo "Invalid GPU index: $item" >&2; return 1 ;; esac
    nvidia-smi --query-gpu=index --format=csv,noheader | grep -Fxq "$item" || {
      echo "GPU index $item was not found on the host." >&2
      return 1
    }
  done
  read -r -p "Continue with CT $id and GPUs $selected? [y/N] " choice
  case "$choice" in y|Y|yes|YES) ;; *) echo "Cancelled."; return 0 ;; esac

  nvidia-modprobe -u -c=0 || true
  for item in $(echo "$selected" | tr ',' ' '); do
    [ -c "/dev/nvidia$item" ] || { echo "Missing device node /dev/nvidia$item" >&2; return 1; }
  done
  mkdir -p /root/gpu-lab-ct-backups
  pct config "$id" > "/root/gpu-lab-ct-backups/$id-$(date +%Y%m%d-%H%M%S).conf"
  for olddev in $(printf '%s\n' "$config" | awk -F: '$1 ~ /^dev[0-9]+$/ && $2 ~ /\/dev\/nvidia/ {print $1}'); do
    pct set "$id" --delete "$olddev"
  done
  config="$(pct config "$id")"
  slot=0
  for node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
    [ -c "$node" ] || continue
    add_ct_device "$node"
  done
  for item in $(echo "$selected" | tr ',' ' '); do
    node="/dev/nvidia$item"
    add_ct_device "$node"
  done
  for node in /dev/nvidia-caps/*; do
    [ -c "$node" ] || continue
    add_ct_device "$node"
  done
  echo "Added NVIDIA device nodes. A stopped/restarted CT is required to apply them."
  if [ "$state" = "running" ]; then
    read -r -p "Restart CT $id now? [y/N] " choice
    case "$choice" in
      y|Y|yes|YES) pct reboot "$id" || { pct stop "$id"; pct start "$id"; } ;;
      *) echo "Device configuration saved. Restart the CT later, then run gpu-lab ct again to verify."; return 0 ;;
    esac
  else
    pct start "$id"
  fi
  sleep 3
  ct_codename="$(pct exec "$id" -- sh -c '. /etc/os-release 2>/dev/null; echo "$VERSION_CODENAME"')"
  ct_arch="$(pct exec "$id" -- dpkg --print-architecture 2>/dev/null || true)"
  read -r -p "Install NVIDIA user-space utilities in Debian 13 CT if supported? [y/N] " install_userspace
  case "$install_userspace" in
    y|Y|yes|YES)
      if [ "$ct_codename" = "trixie" ] && [ "$ct_arch" = "amd64" ]; then
        keyring="/tmp/cuda-keyring-ct-$id.deb"
        curl -fL --retry 3 \
          https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
          -o "$keyring"
        pct push "$id" "$keyring" /tmp/cuda-keyring.deb
        pct exec "$id" -- dpkg -i /tmp/cuda-keyring.deb
        rm -f "$keyring"
        pct exec "$id" -- apt-get update
        branch="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | awk -F. '{print $1}')"
        host_driver_version="$(dpkg-query -W -f='${Version}' nvidia-kernel-open-dkms)"
        if pct exec "$id" -- apt-cache show "nvidia-utils-$branch=$host_driver_version" >/dev/null 2>&1 &&
          pct exec "$id" -- apt-cache show "libnvidia-compute-$branch=$host_driver_version" >/dev/null 2>&1 &&
          pct exec "$id" -- apt-cache show cuda-cudart-13-5 >/dev/null 2>&1; then
          pct exec "$id" -- apt-get install -y "nvidia-utils-$branch=$host_driver_version" "libnvidia-compute-$branch=$host_driver_version" cuda-cudart-13-5
        else
          echo "Exact host driver user-space packages are unavailable in this CT; device mapping remains configured."
        fi
      else
        echo "Automatic user-space install supports Debian 13 amd64 only; device mapping is complete."
      fi
      ;;
  esac
  if ! pct exec "$id" -- sh -c 'command -v nvidia-smi >/dev/null 2>&1'; then
    echo "Device mapping is configured, but nvidia-smi is unavailable inside the CT."
    echo "Install the NVIDIA driver user-space package in the CT, then rerun gpu-lab ct."
    return 0
  fi
  echo "=== NVIDIA visibility inside CT ==="
  pct exec "$id" -- nvidia-smi
  echo "=== CT PCIe P2P capability matrix ==="
  pct exec "$id" -- nvidia-smi topo -p2p p
  if [ -x /usr/local/bin/nvbandwidth ]; then
    pct push "$id" /usr/local/bin/nvbandwidth /usr/local/bin/nvbandwidth
    pct exec "$id" -- chmod 0755 /usr/local/bin/nvbandwidth
    echo "=== CT P2P bandwidth ==="
    trap resume_pstated EXIT INT TERM
    pause_pstated
    pct exec "$id" -- /usr/local/bin/nvbandwidth -p device_to_device -b 512 -i 5
    resume_pstated
    trap - EXIT INT TERM
  else
    echo "nvbandwidth was not found on the host; installed host script may need repair."
  fi
  read -r -p "Wait up to 15 minutes for the CT idle-P8 power test? [Y/n] " choice
  case "$choice" in
    n|N|no|NO) ;;
    *) echo "Leave the selected GPUs idle for up to 15 minutes..."
       sleep 900 ;;
  esac
  echo "=== Host GPU power state ==="
  power_report
  echo "=== CT-visible GPU power state ==="
  pct exec "$id" -- nvidia-smi --query-gpu=index,pstate,power.draw,memory.used --format=csv
  echo "To sample power again later, run:"
  echo "  pct exec $id -- nvidia-smi --query-gpu=index,pstate,power.draw,memory.used --format=csv"
  echo "The host nvidia-pstated daemon must remain enabled for idle P8 management."
}

case "${1:-}" in
  power) power_report ;;
  bench|p2p) bench_host ;;
  ct) ct_command ;;
  -h|--help|"") usage ;;
  *) usage >&2; exit 2 ;;
esac
GPU_LAB
chmod 0755 /usr/local/bin/gpu-lab

log "Enabling host services and checking the deployment"
systemctl enable nvidia-pstated.service
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  systemctl start nvidia-pstated.service
else
  echo "NVIDIA is not ready in the running kernel; nvidia-pstated is enabled and will start after reboot."
fi
systemctl start pex8796-gpu-p2p.service
apt-get -y autoremove --purge
echo
echo "Driver package candidate installed: $(dpkg-query -W -f='${Version}\n' nvidia-driver-cuda)"
echo "GRUB arguments are active after reboot; current command line:"
cat /proc/cmdline
echo "Installed command: gpu-lab {power|bench|ct}"
echo "Backup directory: $BACKUP"
if [ "$AUTO_REBOOT" = "1" ]; then
  systemctl reboot
elif [ -t 0 ]; then
  read -r -p "Reboot now to activate the new kernel command line and driver? [y/N] " REBOOT_NOW
  case "$REBOOT_NOW" in y|Y|yes|YES) systemctl reboot ;;
    *) echo "Reboot when convenient, then run: systemctl status nvidia-pstated pex8796-gpu-p2p" ;;
  esac
else
  echo "Reboot manually, or set AUTO_REBOOT=1 to reboot automatically."
fi
