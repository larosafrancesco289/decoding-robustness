#!/usr/bin/env bash
# RETIRED 2026-09-05 night: he powers the box off instead of suspending (board ignores RTC alarms). Kept for reference.
# Usage: scripts/sicily-sleep-until.sh 09:00     (default 09:00). NOTE: this board ignores RTC alarms in deep sleep (tested 2026-09-05); wake it with the power button. The alarm is kept in case firmware settings change.
# The systemd sleep hook /etc/systemd/system-sleep/90-engine-check.sh consumes the flag on
# resume and launches scripts/engine_check_20260906.sh in tmux session `engine`.
set -euo pipefail
WAKE="${1:-09:00}"
touch ~/.engine_check_armed
echo "$(date -Is) armed; suspending until $WAKE" >> ~/logs/resume-hook.log
echo deep | sudo tee /sys/power/mem_sleep >/dev/null
sudo rtcwake -m mem --date "$WAKE"
