r"""Explain peak live bytes from Triton's shared-memory allocation debug log.

Capture a log with an assertions-enabled triton-opt, for example:

  triton-opt kernel.ttgir --mlir-disable-threading \
    --allocate-shared-memory-nv='compute-capability=120 ptx-version=93' \
    -debug-only=allocation-shared-memory -o /dev/null 2> allocation.log
  python utils/report-shared-memory.py allocation.log

Use the target and PTX version of the original compilation. Each completed
function allocation produces one report, in log order. Operation indices are
local to the allocator's postorder traversal.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import re
import sys


@dataclass(frozen=True)
class Buffer:
    id: int
    size: int
    offset: int
    start: int
    end: int


def format_report(number, frame, buffers, owners):
    if max((b.offset + b.size for b in buffers), default=0) != frame:
        raise ValueError("final offsets disagree with reported frame size")
    changes = defaultdict(int)
    for b in buffers:
        if b.end < b.start:
            raise ValueError(f"reversed interval for buffer {b.id}")
        changes[b.start] += b.size
        changes[b.end] -= b.size

    live = peak = 0
    peak_op = None
    for op, delta in sorted(changes.items()):
        live += delta
        if live > peak:
            peak, peak_op = live, op
    if peak > frame:
        raise ValueError("live-byte peak exceeds reported frame size")

    lines = [
        f"Allocation {number}   frame {frame} B   peak live {peak} B   "
        f"upper bound on packing savings {frame - peak} B"
    ]
    if peak_op is not None:
        lines.extend([f"First peak at operation {peak_op}", "  ID    BYTES   OFFSET   LIFETIME   OWNER"])
        for b in sorted(buffers, key=lambda b: (-b.size, b.id)):
            if b.start <= peak_op < b.end and b.size:
                lines.append(f"{b.id:4} {b.size:8} {b.offset:8}   [{b.start}, {b.end})   "
                             f"{owners.get(b.id, '(owner absent from log)')}")
    return "\n".join(lines)


def reports(lines):
    owners = {}
    buffers = {}
    collecting = pending = False
    count = 0
    for line in lines:
        line = line.rstrip()
        if match := re.fullmatch(r"-- buffer (\d+); value: (.*)", line):
            owners[int(match[1])] = match[2]
            pending = True
        elif line.startswith("[allocation-shared-memory]: Dump bufferRange:"):
            # Placement can take multiple rounds. Only the final snapshot counts.
            buffers.clear()
            collecting = pending = True
        elif collecting and line.startswith("-- "):
            match = re.fullmatch(r"-- (\d+) (\d+) (\d+) interval (\d+) (\d+)", line)
            if not match:
                raise ValueError(f"malformed buffer row: {line}")
            buffer = Buffer(*map(int, match.groups()))
            if buffer.id in buffers:
                raise ValueError(f"duplicate buffer ID {buffer.id}")
            buffers[buffer.id] = buffer
        elif match := re.fullmatch(r"Allocated: (\d+), analyzed: \d+", line):
            if not pending:
                raise ValueError("allocation total without a buffer dump")
            count += 1
            yield format_report(count, int(match[1]), buffers.values(), owners)
            buffers.clear()
            owners.clear()
            collecting = pending = False
        else:
            collecting = False
    if pending:
        raise ValueError("incomplete allocation log")
    if not count:
        raise ValueError("no allocation dump found; enable -debug-only=allocation-shared-memory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    args = parser.parse_args()
    try:
        for report in reports(args.log):
            print(report, end="\n\n")
    except ValueError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
