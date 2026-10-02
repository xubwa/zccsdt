#!/bin/bash
# Check how many cores a benchmark command really uses.
# usage: bench/audit_threads.sh LABEL CMD...
# Runs CMD in its own session and samples every thread of every process in
# it: the CPUs each thread is allowed on, the CPUs threads were seen running
# on, and the largest number of runnable threads in one sample.  Wrap CMD in
# `/usr/bin/time -v` to get its CPU time / wall time as well.
label=$1; shift
out=${TMPDIR:-/tmp}/zc_audit_${label}_$$
t0=$(date +%s.%N)
setsid "$@" > $out.log 2>&1 &
sid=$!
maxrun=0
while kill -0 $sid 2>/dev/null; do
  ps -s $sid -L -o pid=,tid=,psr=,stat=,comm= 2>/dev/null > $out.snap
  nrun=$(awk '$4 ~ /^R/' $out.snap | wc -l)
  [ $nrun -gt $maxrun ] && maxrun=$nrun
  awk '$4 ~ /^R/ {print $3, $5}' $out.snap >> $out.samples
  while read pid tid rest; do
    grep -h Cpus_allowed_list /proc/$pid/task/$tid/status 2>/dev/null
  done < $out.snap | sort -u >> $out.allowed
  sleep 0.2
done
t1=$(date +%s.%N)
echo "== $label: wall $(echo "$t1 - $t0" | bc) s   (output of the command: $out.log)"
echo "allowed CPU lists over all threads: $(sort -u $out.allowed | awk '{print $2}' | tr '\n' ' ')"
echo "CPUs on which a thread was seen running: $(awk '{print $1}' $out.samples | sort -n | uniq | tr '\n' ' ')"
echo "running-thread samples per program: $(awk '{print $2}' $out.samples | sort | uniq -c | sort -rn | head -4 | awk '{printf "%s x%s  ", $2, $1}')"
echo "most runnable threads in one sample: $maxrun"
rm -f $out.snap $out.samples $out.allowed
