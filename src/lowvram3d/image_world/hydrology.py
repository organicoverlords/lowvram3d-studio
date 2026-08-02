"""Small auditable hydrology core for generated heightfields."""

from __future__ import annotations

import heapq

import numpy as np

from .contracts import ContractError


_D8_OFFSETS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
_D8_DISTANCE = np.asarray((1.0, 2**0.5, 1.0, 2**0.5, 1.0, 2**0.5, 1.0, 2**0.5))


def priority_flood_fill(height: np.ndarray) -> np.ndarray:
    """Fill closed depressions to the lowest spill elevation."""

    source = np.asarray(height, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 2 or not np.isfinite(source).all():
        raise ContractError("height must be a finite 2D array of at least 2x2")

    rows, cols = source.shape
    filled = source.copy()
    visited = np.zeros(source.shape, dtype=bool)
    queue: list[tuple[float, int, int]] = []

    def push(row: int, col: int) -> None:
        if not visited[row, col]:
            visited[row, col] = True
            heapq.heappush(queue, (float(filled[row, col]), row, col))

    for col in range(cols):
        push(0, col)
        push(rows - 1, col)
    for row in range(1, rows - 1):
        push(row, 0)
        push(row, cols - 1)

    while queue:
        elevation, row, col = heapq.heappop(queue)
        for dr, dc in _D8_OFFSETS:
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or visited[nr, nc]:
                continue
            visited[nr, nc] = True
            filled[nr, nc] = max(filled[nr, nc], elevation)
            heapq.heappush(queue, (float(filled[nr, nc]), nr, nc))

    return filled.astype(np.float32)


def d8_flow_direction(height: np.ndarray, *, cell_size: float = 1.0) -> np.ndarray:
    """Return the steepest-downhill D8 direction index, or -1 for an outlet/sink."""

    values = np.asarray(height, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ContractError("height must be a finite 2D array")
    if cell_size <= 0:
        raise ContractError("cell_size must be positive")

    rows, cols = values.shape
    direction = np.full(values.shape, -1, dtype=np.int8)
    best_slope = np.zeros(values.shape, dtype=np.float64)
    for index, ((dr, dc), distance) in enumerate(zip(_D8_OFFSETS, _D8_DISTANCE)):
        for row in range(rows):
            nr = row + dr
            if nr < 0 or nr >= rows:
                continue
            for col in range(cols):
                nc = col + dc
                if nc < 0 or nc >= cols:
                    continue
                slope = (values[row, col] - values[nr, nc]) / (distance * cell_size)
                if slope > best_slope[row, col] + 1e-12:
                    best_slope[row, col] = slope
                    direction[row, col] = index
    return direction


def flow_accumulation(direction: np.ndarray) -> np.ndarray:
    """Count upstream cells, including each cell itself."""

    flow = np.asarray(direction, dtype=np.int16)
    if flow.ndim != 2 or ((flow < -1) | (flow > 7)).any():
        raise ContractError("direction must be a 2D array containing -1 or D8 indices 0..7")

    rows, cols = flow.shape
    indegree = np.zeros(flow.shape, dtype=np.int32)
    downstream_row = np.full(flow.shape, -1, dtype=np.int32)
    downstream_col = np.full(flow.shape, -1, dtype=np.int32)
    for row in range(rows):
        for col in range(cols):
            index = int(flow[row, col])
            if index < 0:
                continue
            dr, dc = _D8_OFFSETS[index]
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            downstream_row[row, col] = nr
            downstream_col[row, col] = nc
            indegree[nr, nc] += 1

    accumulation = np.ones(flow.shape, dtype=np.float64)
    queue = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if indegree[row, col] == 0
    ]
    head = 0
    processed = 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        processed += 1
        nr, nc = downstream_row[row, col], downstream_col[row, col]
        if nr < 0:
            continue
        accumulation[nr, nc] += accumulation[row, col]
        indegree[nr, nc] -= 1
        if indegree[nr, nc] == 0:
            queue.append((int(nr), int(nc)))

    if processed != rows * cols:
        raise ContractError("flow graph contains a cycle")
    return accumulation.astype(np.float32)


def stream_mask(accumulation: np.ndarray, *, minimum_cells: float) -> np.ndarray:
    values = np.asarray(accumulation, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 1.0).any():
        raise ContractError("accumulation must be a finite 2D array with values at least one")
    if minimum_cells < 1:
        raise ContractError("minimum_cells must be at least one")
    return values >= minimum_cells
