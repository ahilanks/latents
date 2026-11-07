#!/usr/bin/env bash
# First-time setup for the H-Net latent training environment.
# Run once on a new pod: bash setup.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# .env holds WANDB_/HF_/GH_ creds; the training code also auto-loads it via python-dotenv.
load_env() {
    if [[ -f "${PROJECT_DIR}/.env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source "${PROJECT_DIR}/.env"
        set +a
    fi
}
load_env

mkdir -p runs "${WANDB_DIR:-runs/wandb}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

require_env() {
    local missing=0
    for key in WANDB_API_KEY HF_TOKEN GH_TOKEN; do
        if [[ -z "${!key:-}" ]]; then
            echo "  ERROR: required env var $key is not set (see .env)"
            missing=1
        fi
    done
    if [[ $missing -ne 0 ]]; then
        echo "  Fill in .env before running setup."
        exit 1
    fi
}

ensure_node_tooling() {
    export NVM_DIR="${HOME}/.nvm"
    if [[ ! -s "${NVM_DIR}/nvm.sh" ]]; then
        echo "Installing nvm"
        curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
    fi
    set +u
    # shellcheck disable=SC1091
    . "${NVM_DIR}/nvm.sh"
    [[ -s "${NVM_DIR}/bash_completion" ]] && . "${NVM_DIR}/bash_completion"
    nvm install --lts
    nvm use --lts
    nvm alias default 'lts/*'
    set -u
    echo "Installing Codex CLI"
    npm install -g @openai/codex
}

ensure_gh_cli() {
    if have_cmd gh; then return; fi
    echo "Installing GitHub CLI"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y gh
}

ensure_tmux() {
    if have_cmd tmux; then return; fi
    echo "Installing tmux"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y tmux
}

ensure_bwrap() {
    if have_cmd bwrap; then return; fi
    echo "Installing bwrap"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y bubblewrap
}

ensure_hf_cli() {
    if have_cmd hf; then return; fi
    echo "Installing Hugging Face CLI"
    python3 -m pip install -U "huggingface_hub[cli]"
}

ensure_claude_cli() {
    export PATH="${HOME}/.local/bin:${PATH}"
    if have_cmd claude; then return; fi
    echo "Installing Claude CLI"
    curl -fsSL https://claude.ai/install.sh | bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
}

ensure_pushover_hook() {
    # Pushover phone/laptop notifications for Claude Code (done responding / needs input).
    # Credentials come from .env (loaded above): the app token as PUSHOVER_APPLICATION_KEY or
    # PUSHOVER_TOKEN, the user key as PUSHOVER_KEY or PUSHOVER_USER. The hook is always
    # wired; notifications stay off until both are set.
    PUSHOVER_TOKEN="${PUSHOVER_TOKEN:-${PUSHOVER_APPLICATION_KEY:-}}"
    PUSHOVER_USER="${PUSHOVER_USER:-${PUSHOVER_KEY:-}}"
    local cdir="${HOME}/.claude"
    mkdir -p "${cdir}/hooks"

    if ! have_cmd jq; then
        echo "Installing jq (needed by the notification hook)"
        export DEBIAN_FRONTEND=noninteractive
        apt-get update && apt-get install -y jq || true
    fi

    cat > "${cdir}/hooks/pushover-notify.sh" <<'NOTIFY_EOF'
#!/usr/bin/env bash
# Claude Code -> Pushover notification hook.
# Usage:  pushover-notify.sh stop | notification   (hook JSON on stdin)
EVENT="${1:-stop}"
INPUT="$(cat)"

if [ -z "${PUSHOVER_TOKEN:-}" ] || [ -z "${PUSHOVER_USER:-}" ]; then
  [ -f "$HOME/.claude/pushover.env" ] && . "$HOME/.claude/pushover.env"
fi
if [ -z "${PUSHOVER_TOKEN:-}" ] || [ -z "${PUSHOVER_USER:-}" ]; then
  exit 0
fi

host="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo host)"
cwd="$(printf '%s' "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)"
[ -z "$cwd" ] && cwd="$PWD"
project="$(basename "$cwd" 2>/dev/null || echo session)"

sid="$(printf '%s' "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)"
[ -z "$sid" ] && sid="default"
STATE_DIR="$HOME/.claude/.notif-state"
marker="$STATE_DIR/$(printf '%s' "$sid" | tr -c 'A-Za-z0-9._-' '_').notified"

tp="$(printf '%s' "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)"
if [ -z "$tp" ] || [ ! -f "$tp" ]; then
  [ "$sid" != "default" ] && tp="$(find "$HOME/.claude/projects" -type f -name "${sid}.jsonl" 2>/dev/null | head -1)"
fi

JQ_ACTION='
  def short($s): if ($s|length) > 160 then ($s[0:159] + "…") else $s end;
  def detail:
    .name as $n | (.input // {}) as $i |
    if   $n=="Bash" then "Run: " + ($i.command // $i.description // "command")
    elif ($n=="Edit" or $n=="MultiEdit") then "Edit " + ($i.file_path // "")
    elif $n=="Write" then "Write " + ($i.file_path // "")
    elif $n=="Read" then "Read " + ($i.file_path // "")
    elif $n=="NotebookEdit" then "Edit " + ($i.notebook_path // "")
    elif $n=="WebFetch" then "Fetch " + ($i.url // "")
    elif $n=="WebSearch" then "Search: " + ($i.query // "")
    elif $n=="Grep" then "Grep: " + ($i.pattern // "")
    elif $n=="Glob" then "Glob: " + ($i.pattern // "")
    elif $n=="Task" then "Task: " + ($i.description // "")
    elif $n=="AskUserQuestion" then "Question: " + ((try $i.questions[0].question catch "") // "")
    else $n + ": " + (([$i[]? | select(type=="string")][0]) // "") end;
  select(.type=="assistant") | .message.content[]? | select(.type=="tool_use") | short(detail)
'

if [ "$EVENT" = "notification" ]; then
  msg="$(printf '%s' "$INPUT" | jq -r '.message // empty' 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
  case "$msg" in
    *[Ww]aiting*|*[Ii]dle*) exit 0 ;;
  esac
  [ -f "$marker" ] && exit 0
  mkdir -p "$STATE_DIR" 2>/dev/null
  : > "$marker"
  action=""
  [ -n "$tp" ] && [ -f "$tp" ] && action="$(jq -r "$JQ_ACTION" "$tp" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -1)"
  if [ -n "$action" ]; then
    msg="$action"
  else
    msg="$(printf '%s' "$msg" | sed 's/^Claude //')"
    [ -z "$msg" ] && msg="needs your input"
  fi
  title="⏳ ${project}@${host}"
  priority=1
else
  rm -f "$marker" 2>/dev/null
  find "$STATE_DIR" -type f -mtime +7 -delete 2>/dev/null
  msg=""
  if [ -n "$tp" ] && [ -f "$tp" ]; then
    msg="$(jq -r 'select(.type=="assistant") | .message.content[]? | select(.type=="text") | (.text | gsub("\n";" "))' "$tp" 2>/dev/null | sed 's/`//g; s/#\{1,6\} //g; s/  */ /g' | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-200)"
  fi
  [ -z "$msg" ] && msg="finished responding"
  title="✅ ${project}@${host}"
  priority=0
fi

msg="${msg//"$cwd"\//}"
msg="${msg//"$HOME"/~}"

curl -sf --max-time 10 \
  --form-string "token=${PUSHOVER_TOKEN}" \
  --form-string "user=${PUSHOVER_USER}" \
  --form-string "title=${title}" \
  --form-string "message=${msg}" \
  --form-string "priority=${priority}" \
  https://api.pushover.net/1/messages.json >/dev/null 2>&1
exit 0
NOTIFY_EOF
    chmod +x "${cdir}/hooks/pushover-notify.sh"

    if [[ -n "${PUSHOVER_TOKEN:-}" && -n "${PUSHOVER_USER:-}" ]]; then
        touch "${cdir}/pushover.env"; chmod 600 "${cdir}/pushover.env"
        cat > "${cdir}/pushover.env" <<ENV_EOF
PUSHOVER_TOKEN=${PUSHOVER_TOKEN}
PUSHOVER_USER=${PUSHOVER_USER}
export PUSHOVER_TOKEN PUSHOVER_USER
ENV_EOF
        chmod 600 "${cdir}/pushover.env"
        echo "  Pushover credentials -> ~/.claude/pushover.env"
    else
        echo "  PUSHOVER_APPLICATION_KEY/PUSHOVER_KEY not in .env -- hook installed, notifications off"
    fi

    local settings="${cdir}/settings.json"
    [[ -f "$settings" ]] || echo '{}' > "$settings"
    if have_cmd jq; then
        local tmp; tmp="$(mktemp)"
        if jq '.hooks.Stop = [{"hooks":[{"type":"command","command":"bash \"$HOME/.claude/hooks/pushover-notify.sh\" stop","async":true}]}]
               | .hooks.Notification = [{"hooks":[{"type":"command","command":"bash \"$HOME/.claude/hooks/pushover-notify.sh\" notification","async":true}]}]' \
               "$settings" > "$tmp" 2>/dev/null; then
            mv "$tmp" "$settings"
            echo "  Claude Code Stop/Notification hooks -> Pushover"
        else
            rm -f "$tmp"
            echo "  WARNING: could not update $settings (left as-is)"
        fi
    fi
}

configure_optional_auth() {
    if have_cmd gh; then
        if [[ -n "${GH_TOKEN:-}" ]]; then
            printf '%s' "$GH_TOKEN" | gh auth login --with-token >/dev/null 2>&1 || true
        fi
        gh auth setup-git || true
        gh auth status || true
    fi
    if have_cmd hf; then
        if [[ -n "${HF_TOKEN:-}" ]]; then
            hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
        fi
        hf auth whoami || true
    fi
    if have_cmd wandb; then
        if [[ -n "${WANDB_API_KEY:-}" ]]; then
            wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
        fi
    fi
}

echo "=== H-Net Latents Setup ==="

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $python_version"
if python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "  OK (>= 3.10)"
else
    echo "  ERROR: Python >= 3.10 required"
    exit 1
fi

require_env

echo ""
echo "=== Installing dependencies ==="
python3 -m pip install --upgrade pip setuptools wheel
# torch is preinstalled on the pod; don't touch it.
if [[ -f "${PROJECT_DIR}/pyproject.toml" ]]; then
    python3 -m pip install -e ".[dev]"
else
    python3 -m pip install datasets einops wandb python-dotenv
fi

echo ""
echo "=== Installing CLI tooling ==="
ensure_node_tooling
ensure_tmux
ensure_gh_cli
ensure_bwrap
ensure_hf_cli
ensure_claude_cli
ensure_pushover_hook

echo ""
echo "=== Auth Check ==="
configure_optional_auth

echo ""
echo "=== GPU Check ==="
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU memory: {props.total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA GPU detected')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Notes:"
echo "  - .env is auto-loaded by setup (WANDB_API_KEY, HF_TOKEN, GH_TOKEN, optional PUSHOVER_*)"
