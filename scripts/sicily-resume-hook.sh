#!/usr/bin/env bash
# RETIRED 2026-09-05 night (uninstalled from Sicily; he powers off instead of suspending). Was: /etc/systemd/system-sleep/90-engine-check.sh. After a resume from suspend, if the
# user armed it (touch ~/.engine_check_armed before `rtcwake`), launch the engine check in a
# tmux session as flr. The flag is consumed so a later manual suspend does nothing.
case "$1" in
  post)
    FLAG=/home/flr/.engine_check_armed
    if [ -f "$FLAG" ]; then
      rm -f "$FLAG"
      sleep 20  # let the GPU and network come back
      su - flr -c 'tmux has-session -t engine 2>/dev/null || tmux new-session -d -s engine "cd ~/decoding-robustness && scripts/engine_check_20260906.sh; bash"'
      echo "$(date -Is) engine check launched after resume" >> /home/flr/logs/resume-hook.log
    fi
    ;;
esac
