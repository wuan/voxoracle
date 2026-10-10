#!/usr/bin/env bash
#
# VoxOracle installer for a headless Raspberry Pi (64-bit aarch64 Raspberry Pi
# OS, Debian trixie). Installs uv if missing, syncs the project, writes default
# config.yaml + .env from the examples WITHOUT overwriting existing files, and
# installs + enables the systemd service.
#
# It can install from a git remote (default) or from this local checkout:
#
#   ./deploy/install.sh                 # clone/pull https://github.com/wuan/voxoracle
#   VOXORACLE_SOURCE=$(pwd) ./deploy/install.sh   # install this checkout
#
# Everything is scoped to the invoking user: no paths are hardcoded to root.
set -euo pipefail

REPO_URL="${VOXORACLE_REPO_URL:-https://github.com/wuan/voxoracle.git}"
BRANCH="${VOXORACLE_BRANCH:-main}"
INSTALL_DIR="${VOXORACLE_INSTALL_DIR:-$HOME/voxoracle}"
SOURCE="${VOXORACLE_SOURCE:-}"
SERVICE_NAME="voxoracle"

log() { printf '\n== %s\n' "$*"; }

# --- 1. uv ------------------------------------------------------------------
UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="$HOME/.local/bin/uv"
fi
export PATH="$(dirname "$UV_BIN"):$PATH"
log "Using uv: $("$UV_BIN" --version)"

# --- 2. Fetch the source ----------------------------------------------------
if [ -n "$SOURCE" ]; then
  log "Installing from local checkout $SOURCE"
  if [ "$(readlink -f "$SOURCE")" != "$(readlink -f "$INSTALL_DIR" 2>/dev/null || echo)" ]; then
    mkdir -p "$INSTALL_DIR"
    rsync -a --delete --exclude '.venv' --exclude '.git' "$SOURCE"/ "$INSTALL_DIR"/
  fi
elif [ -d "$INSTALL_DIR/.git" ]; then
  log "Updating existing checkout in $INSTALL_DIR"
  git -C "$INSTALL_DIR" fetch --prune origin
  git -C "$INSTALL_DIR" checkout "$BRANCH"
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
else
  log "Cloning $REPO_URL ($BRANCH) into $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

# --- 3. Sync dependencies (uses the lockfile) -------------------------------
log "Syncing dependencies (uv sync)"
"$UV_BIN" sync --frozen

# --- 4. Default config + .env (never overwrite) -----------------------------
if [ ! -f config.yaml ]; then
  log "Writing config.yaml from config.example.yaml"
  cp config.example.yaml config.yaml
else
  log "Keeping existing config.yaml"
fi
if [ ! -f .env ]; then
  log "Writing .env from .env.example"
  cp deploy/.env.example .env
  chmod 600 .env
else
  log "Keeping existing .env"
fi

# --- 5. systemd unit --------------------------------------------------------
log "Installing systemd unit ($SERVICE_NAME.service)"
if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  SCOPE="system"
  UNIT_DIR="/etc/systemd/system"
  SUDO="sudo"
else
  SCOPE="user"
  UNIT_DIR="$HOME/.config/systemd/user"
  SUDO=""
fi

# Render the unit with the invoking user's paths, then install it.
rendered="$(mktemp)"
trap 'rm -f "$rendered"' EXIT
sed -e "s|__USER__|$(id -un)|g" \
    -e "s|__GROUP__|$(id -gn)|g" \
    -e "s|__HOME__|$HOME|g" \
    -e "s|__INSTALL_DIR__|$INSTALL_DIR|g" \
    deploy/systemd/voxoracle.service > "$rendered"

if [ "$SCOPE" = "system" ]; then
  $SUDO install -m 0644 "$rendered" "$UNIT_DIR/$SERVICE_NAME.service"
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now "$SERVICE_NAME"
else
  mkdir -p "$UNIT_DIR"
  install -m 0644 "$rendered" "$UNIT_DIR/$SERVICE_NAME.service"
  systemctl --user daemon-reload
  systemctl --user enable --now "$SERVICE_NAME"
fi

# --- 6. Next steps ----------------------------------------------------------
cat <<EOF

== Done

Installed to : $INSTALL_DIR
Service      : $SERVICE_NAME.service ($SCOPE)

Note: \`voxoracle run\` is a placeholder until WP6 lands. The service is installed
and enabled, but it exits non-zero and retries (Restart=on-failure) until the
voice loop exists. This is expected.

Next steps:
  1. Edit $INSTALL_DIR/config.yaml set docoracle.url to your DocOracle server.
  2. Put a Mistral key in $INSTALL_DIR/.env (MISTRAL_API_KEY or LLM_API_KEY).
  3. Run:  $INSTALL_DIR/.venv/bin/voxoracle doctor
  4. Logs: $([ "$SCOPE" = system ] && echo "sudo journalctl -u $SERVICE_NAME -f" || echo "journalctl --user -u $SERVICE_NAME -f")
EOF
