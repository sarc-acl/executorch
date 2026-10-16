#!/bin/bash
# session.sh: control vs candidate prefill session on the Radeon 780M, run-local copy of
# sarc-1.5-780m-prefill-refine/tools/e2e5.sh (round 3: page cache fill, R6 predicates, thermal mask 0xFFFFE670,
# monitor of guard-pattern processes and own-user DRM clients). Arms keep the historical names: parent = control
# (780m-final), cand = candidate. Changes for rdna3-780m (owner decisions 2026-10-15 03:57 and 23:30 UTC):
#   - 7 valid runs per arm and cell (--reps 7), at most 3 extra pairs; --clkmin is the A/A floor (0 in the A/A);
#   - before every run: wait until a watcher started with the session has seen 60 s without foreign GPU activity (a guard-pattern process outside this session,
#     or an own-user DRM client outside this session that appears or whose engine time grows); freewait_s recorded;
#   - cooling ends at idle+5 C, or when the temperature has not fallen for three 5 s readings, or at 120 s;
#     cool_end records which;
#   - the gpu-lab lock is waited for up to 2 h;
#   - extra columns per row: session, candidate, commit, binary sha256, driver, monitoring_visibility,
#     freewait_s, cool_end, own_procs_after (children of this session after the run's samplers stopped; 1 = the watcher).
# Monitoring: own-user DRM clients and device-wide busy only; other users' clients are not readable
# (monitoring_visibility=own_user_plus_device_busy_other_users_unobservable, owner decision 2026-10-15 03:57 UTC).
#
# usage: session.sh --stage DIR --out NAME --lock UUID [--reps 7] [--extra 3] [--models 1b,3b,8b] [--schemes 4w,8da4w]
#                   [--prompt prompt_2048.txt] [--tokens 2048] [--clkmin MHZ] [--no-check] [--session S] [--candidate C]
#   DIR/{parent,cand}/{llama_main,libllama_runner.so,env,COMMIT}, DIR/prompt_*.txt, DIR/r1304.txt; output in DIR/NAME/
set -uo pipefail
[[ -n ${SARC_HOLD_UNIT:-} ]] || exec env SARC_HOLD_UNIT=1 "$(dirname "$(readlink -f "$0")")/hold.sh" run "timed session e2e5.sh $*" "$0" "$@"  # coordinator hold: one unit
STAGE=""; OUTN=raw; LOCK=""; REPS=7; EXTRA=3; SESSION=""; CAND=""; MODELS=1b,3b,8b; SCHEMES=4w,8da4w
PROMPT=prompt_2048.txt; TOKENS=2048; CLKMIN=-1; CHECK=1; COOLMAX=120; MROOT=/mnt/linux-share/models
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift ;; --out) OUTN=$2; shift ;; --lock) LOCK=$2; shift ;;
    --reps) REPS=$2; shift ;; --extra) EXTRA=$2; shift ;; --models) MODELS=$2; shift ;;
    --schemes) SCHEMES=$2; shift ;; --prompt) PROMPT=$2; shift ;; --tokens) TOKENS=$2; shift ;;
    --clkmin) CLKMIN=$2; shift ;; --no-check) CHECK=0 ;;
    --session) SESSION=$2; shift ;; --candidate) CAND=$2; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $STAGE && -n $LOCK && $CLKMIN != -1 ]] || { sed -n '2,20p' "$0"; exit 2; }
D=$(cd "$STAGE" && pwd); cd "$D" || exit 2
O=$D/$OUTN; mkdir -p "$O/logs"
exec 9>>"$HOME/.cache/gpu-lab/lock-$LOCK"; flock -w 7200 9 || { echo "gpu-lab lock busy"; exit 75; }
export ETVK_DEVICE_INDEX=0
HW=""; for h in /sys/class/hwmon/hwmon*; do [[ $(cat $h/name 2>/dev/null) == amdgpu ]] && HW=$h; done
BUSY=$(ls /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)
gtemp() { echo $(( $(<$HW/temp1_input) / 1000 )); }
GPAT='llama-server|ComfyUI|comfyui|ollama|vllm|llama_main|test_llama_microbench|Runner.Worker|custom_ops|nvtop'
GM=${BUSY%/*}/gpu_metrics
[[ $(od -An -tu1 -j2 -N2 "$GM" | tr -s ' ') == " 2 1" ]] || { echo "gpu_metrics is not format 2.1"; exit 2; }
others() { pgrep -af "$GPAT" | grep -v pgrep | awk '{print $1":"$2}' | tr '\n' ';'; }
sampler() {  # sampler <file>: epoch_us sclk_hz busy% power_uW temp_mC throttle_status temperature_gfx, every 0.1 s until killed
  local gm
  while :; do
    read -ra gm < <(od -An -v -tu2 -N120 "$GM" | tr '\n' ' ')
    printf '%s %s %s %s %s %s %s\n' "${EPOCHREALTIME/./}" "$(<$HW/freq1_input)" "$(<$BUSY)" "$(<$HW/power1_average)" "$(<$HW/temp1_input)" \
      "$(( ${#gm[@]} == 60 ? gm[54] | gm[55] << 16 : -1 ))" "${gm[2]:--1}"
    sleep 0.1
  done > "$1" 2>/dev/null
}
read -r -d '' MON <<'PY'
import os, re, sys, time
out, root, pat = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
uid, kthread = os.getuid(), set()
def mine(pid):  # does pid descend from this session's script?
    while pid not in ("0", "1"):
        if pid == root: return True
        try: pid = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1]
        except OSError: return False
    return False
with open(out, "w", buffering=1) as f:
    while True:
        t = time.time_ns() // 1000; procs = []; clients = []
        for pid in os.listdir("/proc"):
            if not pid.isdigit() or pid in kthread: continue
            try: cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError: continue
            if not cmd: kthread.add(pid); continue
            own = None
            if pat.search(cmd):
                own = mine(pid)
                if not own: procs.append(f"{pid}:{cmd.split()[0]}")
            try:
                if os.stat(f"/proc/{pid}").st_uid != uid: continue
                fds = os.listdir(f"/proc/{pid}/fdinfo")
            except OSError: continue
            done = set()
            for fd in fds:
                try: info = open(f"/proc/{pid}/fdinfo/{fd}").read()
                except OSError: continue
                if "drm-client-id" not in info: continue
                kv = dict(l.split(":", 1) for l in info.splitlines() if ":" in l)
                cid = kv["drm-client-id"].strip()
                if cid in done: continue
                done.add(cid)
                eng = sum(int(v.split()[0]) for k, v in kv.items() if k.startswith("drm-engine-"))
                if own is None: own = mine(pid)
                clients.append(f"{pid}/{cid}/{'own' if own else 'foreign'}/{eng}")
        f.write(f"{t} {';'.join(procs) or '-'} {';'.join(clients) or '-'}\n")
        time.sleep(0.25)
PY
{
  date -u; hostname; uname -r; echo "lock=$LOCK reps=$REPS extra=$EXTRA prompt=$PROMPT tokens=$TOKENS clkmin=$CLKMIN"
  for b in parent cand; do sha256sum $b/llama_main $b/libllama_runner.so; echo "$b env: $(cat $b/env 2>/dev/null | tr '\n' ' ')"; cat $b/COMMIT 2>/dev/null; done
  sha256sum prompt_*.txt; cat STAGE.md 2>/dev/null
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion'
  echo "dpm: $(cat /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null) sclk levels: $(tr '\n' ' ' < /sys/class/drm/card0/device/pp_dpm_sclk)"
  echo "others: $(others)"
  echo "guard pattern: $GPAT"; echo "gpu_metrics: $GM $(od -An -tu2 -N2 "$GM" | tr -d ' ') bytes, format $(od -An -tu1 -j2 -N2 "$GM" | tr -s ' '), throttle_status now $(od -An -tx4 -j108 -N4 "$GM" | tr -d ' ')"
  echo "DRM clients readable to uid $(id -u) now: $(grep -ls drm-client-id /proc/[0-9]*/fdinfo/* 2>/dev/null | cut -d/ -f3 | sort -u | tr '\n' ' ')"
} > "$O/env.txt" 2>&1
declare -A STEM=([1b]=llama-3.2-1b:llama3_2-1b [3b]=llama-3.2-3b:llama3_2-3b [8b]=llama-3.1-8b:llama3_1-8b)
IFS=, read -ra MS <<< "$MODELS"; IFS=, read -ra QS <<< "$SCHEMES"
pte()  { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/exported/${ST}_vulkan_$2.pte"; }
tokz() { local MD ST; IFS=: read -r MD ST <<< "${STEM[$1]}"; echo "$MROOT/$MD/original/tokenizer.model"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  echo "model $m $q $(sha256sum "$(pte $m $q)" | cut -c1-16) $(pte $m $q)" >> "$O/env.txt"; done; done
declare -A BSHA=([parent]=$(sha256sum parent/llama_main | cut -c1-16) [cand]=$(sha256sum cand/llama_main | cut -c1-16))
DRIVER=$(vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | awk -F= '{gsub(/ /,"",$2); print $2}'); DRIVER=${DRIVER:-unknown}
read -r -d '' FREE <<'PY'
import os, re, sys, time
root, out, pat = sys.argv[1], sys.argv[2], re.compile(os.environ["WATCH_PATTERN"])  # not in argv: others() would match it
uid = os.getuid()
def mine(pid):
    while pid not in ("0", "1"):
        if pid == root: return True
        try: pid = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1]
        except OSError: return False
    return False
def scan():
    busy, eng = [], {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try: cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError: continue
        if cmd and pat.search(cmd) and not mine(pid): busy.append(f"{pid}:{cmd.split()[0]}")
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid: continue
            fds = os.listdir(f"/proc/{pid}/fdinfo")
        except OSError: continue
        for fd in fds:
            try: info = open(f"/proc/{pid}/fdinfo/{fd}").read()
            except OSError: continue
            if "drm-client-id" not in info or mine(pid): continue
            kv = dict(l.split(":", 1) for l in info.splitlines() if ":" in l)
            eng[(pid, kv["drm-client-id"].strip())] = sum(int(v.split()[0]) for k, v in kv.items() if k.startswith("drm-engine-"))
    return busy, eng
last, prev = time.time(), None
with open(out + ".log", "a", buffering=1) as log:
    while True:
        busy, eng = scan(); now = time.time()
        grown = [f"{k[0]}/{k[1]}" for k, v in eng.items() if prev is not None and (k not in prev or v > prev[k])]
        if busy or grown:
            last = now; log.write(f"{now:.0f} {';'.join(busy) or '-'} {';'.join(grown) or '-'}\n")
        prev = eng
        open(out + ".tmp", "w").write(f"{last:.3f}\n"); os.replace(out + ".tmp", out)
        time.sleep(1)
PY
WATCH_PATTERN=$GPAT python3 -c "$FREE" $$ "$O/lastforeign" 9>&- & WP=$!; trap 'kill $WP 2>/dev/null' EXIT
freewait() {  # seconds waited until the watcher has seen 60 s without foreign GPU activity (+timeout after 3 h)
  local t0=$EPOCHSECONDS l
  while :; do
    kill -0 $WP 2>/dev/null || { echo dead; return; }
    l=$(cut -d. -f1 "$O/lastforeign" 2>/dev/null || echo $EPOCHSECONDS)
    (( EPOCHSECONDS - l >= 60 )) && { echo $((EPOCHSECONDS - t0)); return; }
    (( EPOCHSECONDS - t0 >= 10800 )) && { echo "$((EPOCHSECONDS - t0))+timeout"; return; }
    sleep 1
  done
}
sleep 60; IDLE=$(gtemp); echo "idle_temp=$IDLE" >> "$O/env.txt"
cool() {  # sets COOLEND: idle (<= idle+5 C), flat (no fall in three 5 s readings) or max (120 s)
  local t0=$SECONDS t prev=999 flat=0
  while :; do
    t=$(gtemp)
    if (( t <= IDLE + 5 )); then COOLEND=idle; break; fi
    if (( t < prev )); then flat=0; else flat=$((flat + 1)); fi; prev=$t
    if (( flat >= 3 )); then COOLEND=flat; break; fi
    if (( SECONDS - t0 >= COOLMAX )); then COOLEND=max; break; fi
    sleep 5
  done
}
CSV=$O/runs.csv
[[ -f $CSV ]] || echo "gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log,prompt_tokens,generated_tokens,prefill_ms,clk_n,clk_med_mhz,clk_min_mhz,busy_med,power_med_w,temp_max,valid,reason,resident_pct,load_ms,others_post,thr_n,thr_or,thr_run_or,gm_temp_max,mon_n,others_during,foreign_clients,own_engine_ms,thr_bits,session,candidate,commit,binary_sha256,driver,monitoring_visibility,freewait_s,cool_end,own_procs_after" > "$CSV"
run1() {  # run1 <model> <scheme> <build> <rep> <slot> <prompt> <tag> <expected tokens>
  local m=$1 q=$2 b=$3 r=$4 s=$5 p=$6 tag=$7 want=$8 log t0 tp tq rc oth cs sp mp res oth2
  log="logs/$tag-$m-$q-$b-r$r.log"
  t0=$SECONDS; cool; cs=$((SECONDS - t0))
  fw=$(freewait); [[ $fw == dead ]] && { echo "foreign-activity watcher died; stopping"; exit 3; }; tp=$(gtemp); oth=$(others | tr ',' ';')
  local benv=(); [[ -f $D/$b/env ]] && mapfile -t benv < "$D/$b/env"
  res=$(fincore -nbo RES,SIZE "$(pte $m $q)" | awk '{printf "%.2f", 100 * $1 / $2}')
  python3 -c "$MON" "$O/${log%.log}.mon" $$ "$GPAT" 9>&- & mp=$!
  sampler "$O/${log%.log}.clk" 9>&- & sp=$!
  env "${benv[@]}" LD_LIBRARY_PATH=$D/$b timeout 1800 "$D/$b/llama_main" --model_path "$(pte $m $q)" \
    --tokenizer_path "$(tokz $m)" --prompt_file "$p" --max_new_tokens 1 --temperature 0 \
    $([[ $tag == prefill ]] && echo --warmup) < /dev/null > "$O/$log" 2>&1 9>&-
  rc=$?; kill $sp $mp 2>/dev/null; wait $sp $mp 2>/dev/null; tq=$(gtemp); oth2=$(others | tr ',' ';')
  local nown; nown=$(pgrep -cP $$)
  python3 - "$O/$log" "$O/${log%.log}.clk" "$want" "$CLKMIN" "$rc" "$oth$oth2" "$tag" "$O/${log%.log}.mon" <<'PY' > "$O/.row"
import json, re, statistics as st, sys
log, clk, want, clkmin, rc, oth, tag, mon = sys.argv[1:9]
obs = None
for line in open(log, errors="replace"):
    i = line.find("PyTorchObserver")
    if i >= 0:
        try: obs = json.loads(line[line.index("{", i):])
        except Exception: pass
tok = pt = gt = ms = load = ""
rows = []; allrows = [[int(x) for x in l.split()] for l in open(clk) if len(l.split()) == 7]
if obs:
    tok = obs.get("prefill_token_per_sec", ""); pt = obs.get("prompt_tokens", ""); gt = obs.get("generated_tokens", "")
    a, b = obs.get("model_execution_start_ms"), obs.get("model_execution_end_ms")
    if obs.get("model_load_start_ms") and obs.get("model_load_end_ms"): load = obs["model_load_end_ms"] - obs["model_load_start_ms"]
    # prefill window: from inference start to prompt-eval end when available, else the execution window
    a = obs.get("inference_start_ms", a); b2 = obs.get("prompt_eval_end_ms", b)
    if a and b2:
        ms = b2 - a
        rows = [r for r in allrows if a * 1000 <= r[0] <= b2 * 1000]
n = len(rows)
med = lambda k, d: (round(st.median(r[k] for r in rows) / d, 1) if rows else "")
cm = med(1, 1e6); cmin = round(min(r[1] for r in rows) / 1e6, 1) if rows else ""
reason = []
if rc != "0": reason.append("rc")
if tok == "": reason.append("no_tok_s")
if str(pt) != want: reason.append("prompt_tokens")
if tag == "prefill" and str(gt) != "0": reason.append("generated_tokens")
if oth: reason.append("other_gpu_process")
if n < 5: reason.append("clock_unsampled")
elif cm < float(clkmin): reason.append("clock_low")
thr = [r[5] for r in rows if r[5] >= 0]; thr_or = 0
for x in thr: thr_or |= x
run_or = 0
for r in allrows: run_or |= max(r[5], 0)
if len(thr) < 5: reason.append("throttle_unsampled")
elif thr_or & 0xFFFFE670: reason.append("throttle")
bits = ";".join(f"{i}:{c}" for i in range(32) if (c := sum(1 for x in thr if x >> i & 1)))
# monitor: processes of the guard's pattern outside this session, and DRM clients with their summed engine time
during, eng, mon_n = set(), {}, 0
for l in open(mon):
    f = l.split(" ", 2)
    if len(f) != 3 or not l.endswith("\n"): continue
    mon_n += 1
    if f[1] != "-": during.update(f[1].split(";"))
    c = f[2].strip()
    for x in ([] if c == "-" else c.split(";")):
        pid, cid, who, e = x.split("/"); eng.setdefault((pid, cid, who), [mon_n, []])[1].append(int(e))
grow = lambda k: max(eng[k][1]) - (min(eng[k][1]) if eng[k][0] == 1 else 0)  # a client that appears during the run counts in full
foreign = [f"{k[0]}/{k[1]}:{grow(k) / 1e6:.1f}ms" for k in eng if k[2] == "foreign"]
own_ms = round(sum(max(eng[k][1]) for k in eng if k[2] == "own") / 1e6, 1)
if mon_n < 1: reason.append("monitor_unsampled")
if during: reason.append("other_gpu_process_during")
if any(grow(k) > 0 for k in eng if k[2] == "foreign"): reason.append("foreign_gpu_client")
valid = 0 if reason else 1
print(",".join(str(x) for x in [tok, pt, gt, ms, n, cm, cmin, med(2, 1), med(3, 1e6), round(max(r[4] for r in rows) / 1000) if rows else "", valid, "+".join(reason), load,
    len(thr), f"0x{thr_or:08x}", f"0x{run_or:08x}", max((r[6] for r in rows), default=""), mon_n, ";".join(sorted(during)), ";".join(foreign), own_ms, bits]))
PY
  IFS=, read -r tok pt gt ms n cm cmin bm pw tmax valid reason load mcols < "$O/.row"
  [[ $fw == *timeout ]] && { valid=0; reason=${reason:+$reason+}gpu_not_free; }
  echo "780m,$(hostname),$m,$q,$b,$r,$s,$tok,$rc,$tp,$tq,$cs,sclk_med=${cm}MHz,$oth,$(date -u +%FT%TZ),$log,$pt,$gt,$ms,$n,$cm,$cmin,$bm,$pw,$tmax,$valid,$reason,$res,$load,$oth2,$mcols,$SESSION,$( [[ $b == cand ]] && echo "$CAND" || echo control ),$(cat $D/$b/COMMIT),${BSHA[$b]},$DRIVER,own_user_plus_device_busy_other_users_unobservable,$fw,$COOLEND,$nown" >> "$CSV"
  echo "$tag $m $q $b r$r slot$s tok_s=$tok ms=$ms rc=$rc T=$tp->$tq cool=${cs}s clk=$cm/$cmin MHz n=$n busy=$bm resident=$res% load=${load}ms thr/mon=$mcols valid=$valid $reason"
}
nvalid() { awk -F, -v m=$1 -v q=$2 -v b=$3 'NR > 1 && $3 == m && $4 == q && $5 == b && $16 ~ /^logs\/prefill/ && $26 == 1 {n++} END {print n + 0}' "$CSV"; }
gen() { grep -v 'PyTorchObserver\|^[IWE] \|^\[sarc_dev\]' "$O/$1"; }
for m in "${MS[@]}"; do for q in "${QS[@]}"; do
  [[ -f $(pte $m $q) ]] || { echo "missing $(pte $m $q)"; continue; }
  cat "$(pte $m $q)" > /dev/null
  r=1
  while :; do
    if (( r % 2 )); then order=(parent cand); else order=(cand parent); fi
    run1 $m $q ${order[0]} $r 1 $PROMPT prefill $TOKENS
    run1 $m $q ${order[1]} $r 2 $PROMPT prefill $TOKENS
    r=$((r + 1))
    (( r <= REPS )) && continue
    (( $(nvalid $m $q parent) >= REPS && $(nvalid $m $q cand) >= REPS )) && break
    (( r > REPS + EXTRA )) && { echo "CELL $m $q: fewer than $REPS valid runs after $EXTRA extra pairs"; break; }
  done
  if [[ $CHECK == 1 ]]; then
    run1 $m $q parent 0 0 prompt_check.txt check 1972
    run1 $m $q cand 0 0 prompt_check.txt check 1972
    if cmp -s <(gen "logs/check-$m-$q-parent-r0.log") <(gen "logs/check-$m-$q-cand-r0.log"); then x=SAME; else x=DIFFER; fi
    if cmp -s <(gen "logs/prefill-$m-$q-parent-r1.log") <(gen "logs/prefill-$m-$q-cand-r1.log"); then y=SAME; else y=DIFFER; fi
    run1 $m $q parent 0 0 r1304.txt unaligned 1792
    run1 $m $q cand 0 0 r1304.txt unaligned 1792
    if cmp -s <(gen "logs/unaligned-$m-$q-parent-r0.log") <(gen "logs/unaligned-$m-$q-cand-r0.log"); then z=SAME; else z=DIFFER; fi
    echo "$m,$q,$PROMPT:$y,prompt_check.txt:$x,r1304.txt:$z" >> "$O/nexttoken.csv"; echo "nexttoken $m $q $PROMPT=$y check=$x unaligned=$z"
  fi
done; done
echo "others_end: $(others)" >> "$O/env.txt"; date -u > "$O/done.txt"; echo E2E5_DONE
