#!/usr/bin/env bash
# setup.sh — Install dependencies and configure iptables for the active warden.
#
# Usage:
#   sudo ./setup.sh install   # One-time setup: install deps, create config
#   sudo ./setup.sh start     # Add iptables NFQUEUE rules
#   sudo ./setup.sh stop      # Remove iptables NFQUEUE rules
#   sudo ./setup.sh status    # Show current rules and warden status

set -euo pipefail

QUEUE_NUM="${WARDEN_QUEUE_NUM:-0}"
CHAIN="${WARDEN_CHAIN:-FORWARD}"
CONFIG_DIR="/etc/warden"
CONFIG_FILE="${CONFIG_DIR}/warden.yaml"

log() { echo "[warden-setup] $*"; }

cmd_install() {
    log "Installing system dependencies..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq
        apt-get install -y -qq python3-pip libnetfilter-queue-dev iptables
    elif command -v yum &>/dev/null; then
        yum install -y python3-pip libnetfilter_queue-devel iptables
    else
        log "WARNING: Unknown package manager, install deps manually:"
        log "  - python3-pip, libnetfilter-queue-dev, iptables"
    fi

    log "Installing Python dependencies..."
    pip3 install NetfilterQueue PyYAML

    log "Creating config directory..."
    mkdir -p "${CONFIG_DIR}"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log "Creating default config at ${CONFIG_FILE}..."
        cat > "${CONFIG_FILE}" <<'YAML'
# Active Warden Configuration
# See cmd/warden/config.py for all options.

secret: "change-me-to-a-real-secret"
ttl: 64
queue_num: 0
max_queue_len: 4096
stats_interval: 30
log_level: INFO
chain: FORWARD
YAML
        chmod 600 "${CONFIG_FILE}"
        log "IMPORTANT: Edit ${CONFIG_FILE} and set a real secret key!"
    fi

    log "Install complete."
}

cmd_start() {
    log "Adding iptables NFQUEUE rule: ${CHAIN} -> queue ${QUEUE_NUM}"

    # Avoid duplicate rules: check if it already exists.
    if iptables -C "${CHAIN}" -p tcp -j NFQUEUE --queue-num "${QUEUE_NUM}" 2>/dev/null; then
        log "Rule already exists, skipping."
    else
        iptables -I "${CHAIN}" 1 -p tcp -j NFQUEUE --queue-num "${QUEUE_NUM}"
        log "Rule added."
    fi

    # Also queue UDP/ICMP for IP-level normalization.
    if ! iptables -C "${CHAIN}" -p udp -j NFQUEUE --queue-num "${QUEUE_NUM}" 2>/dev/null; then
        iptables -I "${CHAIN}" 2 -p udp -j NFQUEUE --queue-num "${QUEUE_NUM}"
    fi
    if ! iptables -C "${CHAIN}" -p icmp -j NFQUEUE --queue-num "${QUEUE_NUM}" 2>/dev/null; then
        iptables -I "${CHAIN}" 3 -p icmp -j NFQUEUE --queue-num "${QUEUE_NUM}"
    fi

    log "iptables rules active."
}

cmd_stop() {
    log "Removing iptables NFQUEUE rules for queue ${QUEUE_NUM}..."

    # Remove all matching rules (may be multiple).
    for proto in tcp udp icmp; do
        while iptables -D "${CHAIN}" -p "${proto}" -j NFQUEUE --queue-num "${QUEUE_NUM}" 2>/dev/null; do
            log "Removed ${proto} rule."
        done
    done

    log "iptables rules removed."
}

cmd_status() {
    log "Current iptables rules for chain ${CHAIN}:"
    iptables -L "${CHAIN}" -n -v --line-numbers 2>/dev/null || echo "(no rules or chain not found)"
    echo ""
    log "Warden service status:"
    systemctl status warden 2>/dev/null || echo "(service not installed)"
}

# --- Main ---
case "${1:-}" in
    install) cmd_install ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    *)
        echo "Usage: $0 {install|start|stop|status}"
        exit 1
        ;;
esac
