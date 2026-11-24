#!/bin/zsh
# usage: detach.sh <logfile> <command...>
# Run a command in a new session (perl forks, setsid, exec) so it survives the exit of the shell or
# application that started it.
LOG=$1; shift
perl -MPOSIX -e 'my $pid = fork(); exit 0 if $pid; POSIX::setsid(); open(STDIN, "<", "/dev/null"); exec @ARGV or die "exec failed: $!"' "$@" >> "$LOG" 2>&1
