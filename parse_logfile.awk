# -*-awk-*-
BEGIN {
    header()
    reset()
}

/__BEGIN_FILE__/ {
    sub(/__BEGIN_FILE__ /, "", $0)
    split($0, parts, "___")
    data["Graph"] = parts[1]

    split(parts[2], parts, "_")
    for (i in parts) {
        if (match(parts[i], /k([0-9]+)/, m)) {
            data["K"] = m[1]
        } else if (match(parts[i], /P([0-9]+)x([0-9]+)x([0-9]+)/, m)) {
            data["NumNodes"] = m[1]
            data["NumMPIsPerNode"] = m[2]
            data["NumThreadsPerMPI"] = m[3]
        } else if (match(parts[i], /seed([0-9]+)/, m)) {
            data["Seed"] = m[1]
        } else if (match(parts[i], /eps([0-9\.\-e]+)/, m)) {
            data["Epsilon"] = m[1]
        }
    }

    level = 0
}

/__END_FILE__/ {
    yield()
}

match($0, /Number of global nodes: *([0-9]+)/, m) {
    data["N"] = m[1]
}

match($0, /Number of global edges: *([0-9]+)/, m) {
    data["M"] = m[1]
}

match($0, /Seed: *([0-9]+)/, m) {
    data["Seed"] = m[1]
}

match($0, /Partition summary:/, m) {
    Summary = 1
}

match($0, /Imbalance: *([0-9\.\-e]+)/, m) {
    data["Balance"] = m[1]
}

match($0, /Edge cut: *([0-9]+)/, m) {
    data["Cut"] = m[1]
}

match($0, /\|- Partitioning: \.* ([0-9\.\-e]+) s/, m) {
    data["Time"] = m[1]
}
match($0, /`- Partitioning: \.* ([0-9\.\-e]+) s/, m) {
    data["Time"] = m[1]
}

match($0, /\|- Label propagation clustering: \.* ([0-9\.\-e]+) s/, m) {
    data["LPTime"] = m[1]
}

END {
    yield()
}

function header() {
    printf "Graph,"
    printf "K,"
    printf "Seed,"
    printf "Cut,"
    printf "Epsilon,"
    printf "Balance,"
    printf "TotalTime,"
    printf "LabelProbTime,"
    printf "NumNodes"
    printf "\n"
}

function yield() {
    if (length(data) == 0) {
        return
    }

    printf "%s,", data["Graph"]
    printf "%d,", data["K"]
    printf "%d,", data["Seed"]
    printf "%d,", data["Cut"]
    printf "%f,", data["Epsilon"]
    printf "%f,", data["Balance"]
    printf "%f,", data["Time"]
    printf "%f,", data["LPTime"]
    printf "%d", data["NumNodes"]
    printf "\n"

    reset()
}

function reset() {
    split("", data)
}
