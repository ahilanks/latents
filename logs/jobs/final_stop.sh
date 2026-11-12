#!/bin/bash
# Safety net for the pod (user, 2026-11-12: "after that result, whatever it is, just commit, push and then stop the pod").
# The interactive session does the real wrap-up and then removes this script's job. If that never happens, this waits for
# the four last runs (or 16:30 UTC = 8:30 AM PST), commits and pushes anything left, and stops the pod ONLY if the
# remote main equals the local main. It never terminates (deletes) the pod.
cd /workspace/latents && set -a && source .env && set +a
deadline=$(date -u -d "2026-11-12 16:30" +%s)
while true; do
  n=$(cat logs/think/long_rep_s1.out logs/think/long_rep_s0t1.out logs/think/e2e32_s0.out logs/think/e2e32_s1.out 2>/dev/null | grep -c "REP_DONE\|E2E32_DONE\|FAILED")
  if [ "$n" -ge 4 ]; then sleep 2400; break; fi   # everything finished: 40 min for the interactive wrap-up first
  if [ "$(date +%s)" -ge "$deadline" ]; then break; fi
  sleep 60
done
git add logistics.md latents scripts tests configs CLAUDE.md 2>/dev/null; git add -f logs/jobs/*.sh 2>/dev/null
if ! git diff --cached --quiet; then
  git -c user.name="$(git log -1 --format=%an)" -c user.email="$(git log -1 --format=%ae)" commit -q -m "Safety-net commit before stopping the pod (unattended)

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BxuibtSMdjAaM2REZenEM8"
fi
git -c credential.helper='!gh auth git-credential' push -q origin main
git -c credential.helper='!gh auth git-credential' fetch -q origin
if [ "$(git rev-parse main)" = "$(git rev-parse origin/main)" ]; then
  echo "[$(date -u +%H:%M)] pushed and verified ($(git rev-parse --short main)); stopping pod $RUNPOD_POD_ID"
  curl -s -m 60 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"
  sleep 60; runpodctl stop pod "$RUNPOD_POD_ID"
else
  echo "[$(date -u +%H:%M)] remote main differs from local main: NOT stopping the pod"
fi
