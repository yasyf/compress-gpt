#!/usr/bin/env python

import json
import re
import subprocess
import tempfile

from rich import print


def run(cmd):
    print(" ".join(cmd))
    return subprocess.run(" ".join(cmd), shell=True, check=True)


def edit(original, start, end, dest):
    run(
        [
            "asciinema-edit",
            "cut",
            "--start",
            start,
            "--end",
            end,
            "--out",
            dest,
            original,
        ],
    )
    lines = open(dest).read().splitlines()
    header = json.loads(lines[0])
    del header["env"], header["theme"]
    lines[0] = json.dumps(header)
    open(dest, "w").write("\n".join(lines) + "\n")


def main(argv):
    original, start, end, dest = argv[0:4]

    lines = open(original).read().splitlines()
    global_start = re.search(r"\[(\d+\.\d+),", lines[1]).group(1)
    global_end = re.search(r"\[(\d+\.\d+),", lines[-1]).group(1)

    temp = tempfile.NamedTemporaryFile(delete=False).name
    temp2 = tempfile.NamedTemporaryFile(delete=False).name

    edit(original, end, global_end, temp)
    edit(temp, global_start, start, temp2)

    run(
        [
            "agg",
            "--font-size",
            "20",
            "--speed",
            "3.5",
            "--rows",
            "10",
            "--idle-time-limit",
            "0.5",
            temp2,
            temp2 + ".gif",
        ]
    )
    run(
        [
            "gifsicle",
            "-j8",
            temp2 + ".gif",
            "-i",
            "--lossy=50",
            "-k",
            "64",
            "'#0--2'",
            "-d200",
            "'#-1'",
            "-O3",
            "-Okeep-empty",
            "--no-conserve-memory",
            "-o",
            temp2 + "-opt.gif",
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            temp2 + "-opt.gif",
            "-movflags",
            "faststart",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "'crop=trunc(iw/2)*2:trunc(ih/2)*2'",
            "-crf",
            "18",
            dest,
        ]
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
