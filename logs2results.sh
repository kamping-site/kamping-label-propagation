#!/usr/bin/env bash
log_dir="${1:-"."}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

configs_for_algo() {
    cat $log_dir/config.json | \
         jq ' to_entries | group_by(.value.preset) | map({key: .[0].value.preset, value: [.[] | .key]}) | from_entries' | jq -r " .[\"$1\"] | join (\" \")"
}


parse_logs() {
    gawk -f "$SCRIPT_DIR/parse_logfile.awk" <(
    for x in $(configs_for_algo $1); do
        for file in $log_dir/*-c$x-log.txt; do
        # echo $file
        echo "__BEGIN_FILE__" $(basename "$file")
        cat "$file" | sed "s,\x1B\[[0-9;]*[a-zA-Z],,g"
        echo "__END_FILE__"
        done
    done
    )
}

parse_logs "mpi" > PlainMPI.csv
parse_logs "kamping" > KaMPIngWrapper.csv
parse_logs "kaminpar" > dKaMinParWrapper.csv

exit 0

