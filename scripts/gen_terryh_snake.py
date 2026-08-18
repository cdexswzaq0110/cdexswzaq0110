#!/usr/bin/env python3
"""Generate ``assets/terryh-snake.svg``.

A GitHub-contribution-style grid where a snake eats every ordinary green cell
and finally leaves the letters ``TERRYH`` glowing on the board.

The output is a single self-contained SVG animated with pure CSS keyframes
(no JavaScript, no SMIL), so it renders correctly when GitHub embeds it with
``<img src="...svg">``.

Usage::

    python scripts/gen_terryh_snake.py
    python scripts/gen_terryh_snake.py --preview   # also print an ASCII mask
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260818

COLS = 56
ROWS = 9

CELL = 11.0          # square size in user units
GAP = 3.0            # gap between squares
PITCH = CELL + GAP   # distance between two cell origins
RADIUS = 2.6         # square corner radius
MARGIN_X = 16.0
MARGIN_Y = 18.0

SNAKE_LEN = 9                    # head + 8 body segments
OFFSCREEN = SNAKE_LEN + 5        # how far outside the board the snake parks

GREEN_DENSITY = 0.52             # share of free cells that start green
MAX_PELLETS = 160                # hard cap so the animation stays short

IDLE_SEC = 1.0                   # spawn pause before the snake enters
HOLD_SEC = 3.4                   # TERRYH-only pause before the loop restarts
TARGET_EAT_SEC = 20.5            # desired length of the hunting phase
MIN_STEP = 0.045                 # seconds per cell (fastest)
MAX_STEP = 0.075                 # seconds per cell (slowest)
REGROW_SEC = 0.85                # cross-fade back to the full grid at loop end
MIN_TAIL_SEC = 5.0               # quiet time reserved after the last pellet

BG = "#0d1117"
EMPTY = "#161b22"
GREENS = ("#0e4429", "#006d32", "#26a641", "#39d353")
GREEN_WEIGHTS = (0.34, 0.30, 0.22, 0.14)

# head first, tail last: (size, corner radius, fill)
SNAKE_SEGMENTS = (
    (11.6, 3.4, "#eafff5"),
    (11.2, 3.3, "#8affc0"),
    (10.9, 3.2, "#5cf79b"),
    (10.6, 3.1, "#3fee85"),
    (10.3, 3.0, "#2fe875"),
    (10.0, 2.9, "#1fc964"),
    (9.6, 2.8, "#19b072"),
    (9.2, 2.7, "#16a085"),
    (8.7, 2.6, "#14907a"),
)

# 5 x 7 pixel font. Every glyph is 5 columns wide and 7 rows tall.
FONT = {
    "T": ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    "E": ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
    "R": ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
    "Y": ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."],
    "H": ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
}
WORD = "TERRYH"
GLYPH_W = 5
GLYPH_H = 7
LETTER_SPACING = 2
TEXT_TOP = 1                     # leaves row 0 and row ROWS-1 as free corridors

OUT_PATH = Path(__file__).resolve().parent.parent / "assets" / "terryh-snake.svg"

STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def fmt(value: float) -> str:
    """Format a number for CSS/SVG with at most three decimals, no padding."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def cell_center(col: int, row: int) -> tuple:
    """Return the pixel centre of a grid cell (works outside the board too)."""
    return (MARGIN_X + col * PITCH + CELL / 2, MARGIN_Y + row * PITCH + CELL / 2)


def neighbours(cell: tuple, blocked: set) -> list:
    """Four-way neighbours that stay on the board and avoid blocked cells."""
    col, row = cell
    out = []
    for dc, dr in STEPS:
        nxt = (col + dc, row + dr)
        if 0 <= nxt[0] < COLS and 0 <= nxt[1] < ROWS and nxt not in blocked:
            out.append(nxt)
    return out


# --------------------------------------------------------------------------
# Board construction
# --------------------------------------------------------------------------

def build_text_cells() -> set:
    """Rasterise TERRYH into grid coordinates, horizontally centred."""
    total_w = len(WORD) * GLYPH_W + (len(WORD) - 1) * LETTER_SPACING
    if total_w > COLS - 4:
        raise ValueError("TERRYH needs %d columns, only %d available" % (total_w, COLS))
    if TEXT_TOP + GLYPH_H > ROWS:
        raise ValueError("TERRYH does not fit vertically")

    cells = set()
    left = (COLS - total_w) // 2
    for index, char in enumerate(WORD):
        origin = left + index * (GLYPH_W + LETTER_SPACING)
        for row, line in enumerate(FONT[char]):
            for col, pixel in enumerate(line):
                if pixel == "#":
                    cells.add((origin + col, TEXT_TOP + row))
    return cells


def build_initial_grid(text_cells: set, start: tuple) -> set:
    """Return every free cell the snake can actually reach from ``start``.

    Enclosed pockets (for example the counter inside an R) stay dark so the
    snake is never asked to eat something it cannot get to.
    """
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbours(current, text_cells):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def find_food_cells(reachable: set, rng: random.Random) -> dict:
    """Pick which reachable cells start green, and at which contribution level."""
    ordered = sorted(reachable)
    chosen = [cell for cell in ordered if rng.random() < GREEN_DENSITY]
    if len(chosen) > MAX_PELLETS:
        chosen = sorted(rng.sample(chosen, MAX_PELLETS))
    levels = rng.choices(range(len(GREENS)), weights=GREEN_WEIGHTS, k=len(chosen))
    return dict(zip(chosen, levels))


# --------------------------------------------------------------------------
# Route planning
# --------------------------------------------------------------------------

def bfs_distances(start: tuple, blocked: set) -> dict:
    """Breadth-first distance from ``start`` to every reachable free cell."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for nxt in neighbours(current, blocked):
            if nxt not in dist:
                dist[nxt] = dist[current] + 1
                queue.append(nxt)
    return dist


def choose_next_target(dist: dict, candidates: list, rng: random.Random) -> tuple:
    """Pick one of the closest pellets, biased towards - but not locked to - the nearest."""
    ranked = sorted(candidates, key=lambda cell: (dist[cell], cell[0], cell[1]))
    top = ranked[:2]
    weights = (0.92, 0.08)[: len(top)]
    return rng.choices(top, weights=weights, k=1)[0]


def build_path(dist: dict, goal: tuple, blocked: set, rng: random.Random) -> list:
    """A random shortest path to ``goal``; the randomness gives the snake its zig-zag."""
    path = [goal]
    current = goal
    while dist[current] > 0:
        options = [n for n in neighbours(current, blocked)
                   if dist.get(n, -1) == dist[current] - 1]
        current = rng.choice(options)
        path.append(current)
    path.reverse()
    return path[1:]


def build_exit_path(current: tuple, text_cells: set, rng: random.Random) -> list:
    """Leave the board: reach the last column, then glide straight off-screen."""
    dist = bfs_distances(current, text_cells)
    exits = [(COLS - 1, row) for row in range(ROWS) if (COLS - 1, row) in dist]
    goal = min(exits, key=lambda cell: (dist[cell], abs(cell[1] - current[1])))
    path = build_path(dist, goal, text_cells, rng)
    path.extend((COLS + offset, goal[1]) for offset in range(OFFSCREEN + 2))
    return path


def plan_core_route(text_cells: set, pellets: dict, start: tuple,
                    rng: random.Random) -> tuple:
    """Hunt every pellet, then walk off the right edge.

    Returns the cell sequence and ``{route_index: cell}`` for every cell eaten.
    """
    route = [start]
    eaten = {}
    remaining = set(pellets)
    if start in remaining:
        remaining.discard(start)
        eaten[0] = start

    current = start
    while remaining:
        dist = bfs_distances(current, text_cells)
        reachable = [cell for cell in remaining if cell in dist]
        if not reachable:
            break
        target = choose_next_target(dist, reachable, rng)
        for cell in build_path(dist, target, text_cells, rng):
            route.append(cell)
            if cell in remaining:
                remaining.discard(cell)
                eaten[len(route) - 1] = cell
        current = target

    route.extend(build_exit_path(current, text_cells, rng))
    return route, eaten


def assemble_route(core: list, idle_steps: int, hold_steps: int,
                   spawn_row: int) -> tuple:
    """Wrap the hunting route with an off-screen idle, an entry ramp and a hold."""
    park = (-OFFSCREEN, spawn_row)
    ramp = [(col, spawn_row) for col in range(-OFFSCREEN, 0)]
    route = [park] * idle_steps + ramp + core + [core[-1]] * hold_steps
    return route, idle_steps + len(ramp)


def calculate_eat_times(eaten: dict, offset: int, step: float) -> dict:
    """Map ``route_index -> (cell, seconds)``.

    The head runs ``SNAKE_LEN - 1`` steps ahead of the shared route track, so a
    cell at route index ``k`` is reached at ``(k - lag) * step`` seconds.
    """
    lag = SNAKE_LEN - 1
    return {index + offset: (cell, (index + offset - lag) * step)
            for index, cell in eaten.items()}


# --------------------------------------------------------------------------
# CSS / SVG generation
# --------------------------------------------------------------------------

def generate_route_keyframes(route: list) -> str:
    """One shared keyframe track; every snake segment replays it with a delay.

    Only the first and last frame of a repeated position is emitted, which keeps
    the idle and hold phases perfectly still instead of slowly drifting.
    """
    points = [cell_center(col, row) for col, row in route]
    last = len(points) - 1
    frames = []
    for index, point in enumerate(points):
        is_edge = index == 0 or index == last
        starts_run = is_edge or point != points[index - 1]
        ends_run = is_edge or point != points[index + 1]
        if starts_run or ends_run:
            pct = index * 100.0 / last
            frames.append("%s%%{transform:translate(%spx,%spx)}"
                          % (fmt(pct), fmt(point[0]), fmt(point[1])))
    return "@keyframes route{" + "".join(frames) + "}"


def generate_cell_animations(eat_times: dict, cycle: float) -> tuple:
    """Per-cell vanish keyframes: pop bright, shrink away, regrow at loop end.

    Every cell shares the same cycle length, so the final regrow ramp lands on
    the loop boundary and the restart reads as the graph filling back in.
    """
    pop = 0.12 / cycle * 100
    gone = 0.34 / cycle * 100
    regrow = (cycle - REGROW_SEC) / cycle * 100

    names = {}
    rules = []
    for index in sorted(eat_times):
        cell, seconds = eat_times[index]
        at = seconds / cycle * 100
        end = min(at + gone, regrow - 0.05)
        name = "e%d" % index
        names[cell] = name
        rules.append(
            "@keyframes %s{0%%,%s%%{opacity:1;transform:scale(1)}"
            "%s%%{opacity:1;transform:scale(1.22)}"
            "%s%%{opacity:0;transform:scale(.15)}"
            "%s%%{opacity:0;transform:scale(.15)}"
            "100%%{opacity:1;transform:scale(1)}}"
            % (name, fmt(at), fmt(at + pop), fmt(end), fmt(regrow))
        )
        rules.append(".%s{animation-name:%s}" % (name, name))
    return names, rules


def generate_reveal_animations(reveal_at: float, cycle: float) -> list:
    """TERRYH stays camouflaged, then rises to full GitHub green with one pulse."""
    def pct(offset: float) -> str:
        return fmt((reveal_at + offset) / cycle * 100)

    regrow = fmt((cycle - REGROW_SEC) / cycle * 100)
    v0, v1, v2, v3 = pct(0.0), pct(0.55), pct(0.95), pct(1.45)
    return [
        "@keyframes revA{0%%,%s%%{fill:#0e4429;transform:scale(1)}"
        "%s%%{fill:#26a641;transform:scale(1.05)}"
        "%s%%{fill:#39d353;transform:scale(1.13)}"
        "%s%%{fill:#26a641;transform:scale(1)}"
        "%s%%{fill:#26a641;transform:scale(1)}"
        "100%%{fill:#0e4429;transform:scale(1)}}" % (v0, v1, v2, v3, regrow),
        "@keyframes revB{0%%,%s%%{fill:#006d32;transform:scale(1)}"
        "%s%%{fill:#26a641;transform:scale(1.05)}"
        "%s%%{fill:#39d353;transform:scale(1.13)}"
        "%s%%{fill:#39d353;transform:scale(1)}"
        "%s%%{fill:#39d353;transform:scale(1)}"
        "100%%{fill:#006d32;transform:scale(1)}}" % (v0, v1, v2, v3, regrow),
        "@keyframes glow{0%%,%s%%{opacity:0}%s%%{opacity:.5}%s%%{opacity:.26}"
        "%s%%{opacity:.26}100%%{opacity:0}}" % (v0, v2, v3, regrow),
    ]


def generate_snake(step: float) -> tuple:
    """Body segments share the route track, each lagging one cell behind the last."""
    markup = []
    rules = []
    for index in reversed(range(len(SNAKE_SEGMENTS))):
        size, radius, fill = SNAKE_SEGMENTS[index]
        half = size / 2
        parts = ['<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s"/>'
                 % (fmt(-half), fmt(-half), fmt(size), fmt(size), fmt(radius), fill)]
        if index == 0:
            parts.append('<circle cx="-2.3" cy="-1.5" r="1.15" fill="#0b3d2c"/>')
            parts.append('<circle cx="2.3" cy="-1.5" r="1.15" fill="#0b3d2c"/>')
        markup.append('<g class="seg s%d">%s</g>' % (index, "".join(parts)))
        rules.append(".s%d{animation-delay:-%ss}"
                     % (index, fmt((SNAKE_LEN - 1 - index) * step)))
    return markup, rules


def build_stylesheet(cycle: float, route_css: str, eat_rules: list,
                     reveal_rules: list, snake_rules: list) -> str:
    """Static rules plus every generated keyframe track, in one <style> block."""
    duration = fmt(cycle)
    base = [
        ".c,.t{transform-box:fill-box;transform-origin:center;"
        "animation-duration:%ss;animation-timing-function:linear;"
        "animation-iteration-count:infinite}" % duration,
        "".join(".g%d{fill:%s}" % (i, color) for i, color in enumerate(GREENS)),
        ".ta{animation-name:revA}.tb{animation-name:revB}",
        ".glow{opacity:0;animation:glow %ss linear infinite}" % duration,
        ".seg{animation:route %ss linear infinite}" % duration,
        ".seg rect{stroke:#0d1117;stroke-width:1.3}",
        "@media (prefers-reduced-motion:reduce){"
        ".snake,.food{display:none}"
        ".t{animation:none}.ta{fill:#26a641}.tb{fill:#39d353}"
        ".glow{animation:none;opacity:.26}}",
    ]
    return "".join(base + snake_rules + [route_css] + reveal_rules + eat_rules)


def generate_svg(pellets: dict, text_cells: set, eat_names: dict, css: str,
                 snake_markup: list, rng: random.Random) -> str:
    """Assemble the final document.

    Empty cells come from a single tiled pattern, so only green cells need real
    elements. Layer order is board, food, halo, TERRYH, snake.

    The snake is clipped to the card so its off-screen parking spots stay hidden
    even if a renderer gives the SVG a box wider than its viewBox.
    """
    board_w = COLS * PITCH - GAP
    board_h = ROWS * PITCH - GAP
    width = MARGIN_X * 2 + board_w
    height = MARGIN_Y * 2 + board_h

    food = []
    for cell in sorted(pellets):
        x, y = cell_center(*cell)
        food.append('<g transform="translate(%s,%s)"><use href="#q" class="c g%d %s"/></g>'
                    % (fmt(x), fmt(y), pellets[cell], eat_names[cell]))

    letters = []
    halo = []
    for cell in sorted(text_cells):
        x, y = cell_center(*cell)
        variant = "ta" if rng.random() < 0.25 else "tb"
        letters.append('<g transform="translate(%s,%s)"><use href="#q" class="t %s"/></g>'
                       % (fmt(x), fmt(y), variant))
        halo.append('<g transform="translate(%s,%s)"><use href="#q"/></g>' % (fmt(x), fmt(y)))

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %s %s" width="%s" height="%s"'
        ' role="img" aria-label="A snake eats a GitHub contribution grid and leaves the'
        ' word TERRYH"><title>TERRYH Contribution Snake</title>'
        '<defs><rect id="q" x="%s" y="%s" width="%s" height="%s" rx="%s"/>'
        '<pattern id="board" x="%s" y="%s" width="%s" height="%s" patternUnits="userSpaceOnUse">'
        '<rect width="%s" height="%s" rx="%s" fill="%s"/></pattern>'
        '<filter id="soft" x="-25%%" y="-25%%" width="150%%" height="150%%">'
        '<feGaussianBlur stdDeviation="3.2"/></filter>'
        '<clipPath id="frame"><rect width="%s" height="%s" rx="8"/></clipPath></defs>'
        '<style><![CDATA[%s]]></style>'
        '<rect width="%s" height="%s" rx="8" fill="%s"/>'
        '<rect x="%s" y="%s" width="%s" height="%s" fill="url(#board)"/>'
        '<g class="food">%s</g>'
        '<g class="glow" filter="url(#soft)" fill="#39d353">%s</g>'
        '<g class="word">%s</g>'
        '<g class="snake" clip-path="url(#frame)">%s</g></svg>'
        % (fmt(width), fmt(height), fmt(width), fmt(height),
           fmt(-CELL / 2), fmt(-CELL / 2), fmt(CELL), fmt(CELL), fmt(RADIUS),
           fmt(MARGIN_X), fmt(MARGIN_Y), fmt(PITCH), fmt(PITCH),
           fmt(CELL), fmt(CELL), fmt(RADIUS), EMPTY,
           fmt(width), fmt(height),
           css,
           fmt(width), fmt(height), BG,
           fmt(MARGIN_X), fmt(MARGIN_Y), fmt(board_w), fmt(board_h),
           "".join(food), "".join(halo), "".join(letters), "".join(snake_markup))
    )


# --------------------------------------------------------------------------
# Preview / entry point
# --------------------------------------------------------------------------

def ascii_preview(text_cells: set) -> str:
    """Render the TERRYH mask as text so the letter shapes can be eyeballed."""
    return "\n".join(
        "".join("##" if (col, row) in text_cells else ". " for col in range(COLS))
        for row in range(ROWS)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the TERRYH contribution snake SVG.")
    parser.add_argument("--preview", action="store_true",
                        help="print an ASCII view of the TERRYH mask")
    args = parser.parse_args()

    rng = random.Random(SEED)
    text_cells = build_text_cells()
    spawn_row = ROWS - 1
    start = (0, spawn_row)
    if start in text_cells:
        raise ValueError("spawn cell collides with TERRYH")

    reachable = build_initial_grid(text_cells, start)
    pellets = find_food_cells(reachable, rng)
    core, eaten = plan_core_route(text_cells, pellets, start, rng)
    if len(eaten) != len(pellets):
        raise ValueError("some pellets were never eaten")

    step = min(max(TARGET_EAT_SEC / len(core), MIN_STEP), MAX_STEP)
    idle_steps = max(SNAKE_LEN + 2, round(IDLE_SEC / step))
    hold_steps = max(SNAKE_LEN + 2, round(HOLD_SEC / step))

    # Reserve enough quiet time between the last bite and the loop restart so
    # the reveal, the snake exit and the TERRYH hold all fit.
    for _ in range(8):
        route, offset = assemble_route(core, idle_steps, hold_steps, spawn_row)
        cycle = (len(route) - 1) * step
        eat_times = calculate_eat_times(eaten, offset, step)
        last_eat = max(seconds for _, seconds in eat_times.values())
        if cycle - last_eat >= MIN_TAIL_SEC:
            break
        hold_steps += math.ceil((MIN_TAIL_SEC - (cycle - last_eat)) / step)
    else:
        raise ValueError("could not reserve the TERRYH hold window")

    reveal_at = last_eat + 0.15
    eat_names, eat_rules = generate_cell_animations(eat_times, cycle)
    reveal_rules = generate_reveal_animations(reveal_at, cycle)
    snake_markup, snake_rules = generate_snake(step)
    css = build_stylesheet(cycle, generate_route_keyframes(route),
                           eat_rules, reveal_rules, snake_rules)
    svg = generate_svg(pellets, text_cells, eat_names, css, snake_markup, rng)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(svg, encoding="utf-8")

    print("grid         : %d x %d (%d cells)" % (COLS, ROWS, COLS * ROWS))
    print("TERRYH cells : %d" % len(text_cells))
    print("pellets      : %d (all eaten)" % len(pellets))
    print("route steps  : %d (core %d)" % (len(route), len(core)))
    print("step         : %.0f ms/cell" % (step * 1000))
    print("cycle        : %.2f s (last bite %.2fs, reveal %.2fs)"
          % (cycle, last_eat, reveal_at))
    print("output       : %s (%.1f KB)" % (OUT_PATH, OUT_PATH.stat().st_size / 1024))
    if args.preview:
        print()
        print(ascii_preview(text_cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
