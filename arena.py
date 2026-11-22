"""Spatial queries and bounded route searches for the Mech Royale arena."""

from array import array
from collections import defaultdict
from functools import lru_cache
from heapq import heappop, heappush
import json
import math
from pathlib import Path
import sys
from time import perf_counter


TILE = 24
CELL = 5
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
ANCHOR_ANGLES = tuple(math.radians(degrees) for degrees in (0, 30, -30, 60, -60, 90, -90, 125, -125, 180))


def contains(edges, x, y):
    inside = False
    for x1, y1, dx, dy, _ in edges:
        if (y1 > y) != (y1 + dy > y) and x < x1 + (y - y1) * dx / dy:
            inside = not inside
    return inside


def close_to_edge(edges, x, y, radius_squared):
    for x1, y1, dx, dy, length_squared in edges:
        offset_x, offset_y = x - x1, y - y1
        along = max(0.0, min(1.0, (offset_x * dx + offset_y * dy) / length_squared))
        if (offset_x - along * dx) ** 2 + (offset_y - along * dy) ** 2 < radius_squared:
            return True
    return False


def polygon_edges(vertices):
    points = list(zip(vertices[::2], vertices[1::2]))
    edges = []
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        dx, dy = x2 - x1, y2 - y1
        if dx or dy:
            edges.append((x1, y1, dx, dy, dx * dx + dy * dy))
    return edges


class Arena:
    def __init__(self, root):
        map_dir = Path(root) / "v2"
        world = json.loads((map_dir / "world.json").read_text(encoding="utf-8"))
        terrain = world["terrain"]
        self.nx, self.ny = terrain["nx"], terrain["ny"]
        self.min_x, self.min_y = terrain["minX"], terrain["minY"]
        self.step, self.scale = terrain["step"], terrain["scale"]
        bounds = world["bounds"]
        self.bounds = bounds["minX"], bounds["minY"], bounds["maxX"], bounds["maxY"]
        self.heights = array("h", (map_dir / terrain["file"]).read_bytes())
        if sys.byteorder != "little":
            self.heights.byteswap()
        self.tiles = defaultdict(list)
        for prism in world["prisms"]:
            ring = prism["ring"]
            xs, ys = ring[::2], ring[1::2]
            left, bottom, right, top = min(xs), min(ys), max(xs), max(ys)
            shape = (
                left, bottom, right, top, prism["base"], prism["top"],
                polygon_edges(ring), [polygon_edges(hole) for hole in prism["holes"]],
            )
            for i in range(math.floor(left / TILE), math.floor(right / TILE) + 1):
                for j in range(math.floor(bottom / TILE), math.floor(top / TILE) + 1):
                    self.tiles[i, j].append(shape)

    def ground(self, x, y):
        gx, gy = (x - self.min_x) / self.step, (y - self.min_y) / self.step
        gx, gy = min(max(gx, 0), self.nx - 1), min(max(gy, 0), self.ny - 1)
        i, j = min(int(gx), self.nx - 2), min(int(gy), self.ny - 2)
        dx, dy = gx - i, gy - j
        start = j * self.nx + i
        south = self.heights[start] * (1 - dx) + self.heights[start + 1] * dx
        north = self.heights[start + self.nx] * (1 - dx) + self.heights[start + self.nx + 1] * dx
        return (south * (1 - dy) + north * dy) * self.scale

    def blocked(self, x, y, feet, radius=0.0, height=0.0):
        left, bottom, right, top = self.bounds
        if x - radius < left or y - radius < bottom or x + radius > right or y + radius > top:
            return True
        for i in range(math.floor((x - radius) / TILE), math.floor((x + radius) / TILE) + 1):
            for j in range(math.floor((y - radius) / TILE), math.floor((y + radius) / TILE) + 1):
                for x1, y1, x2, y2, base, roof, ring, holes in self.tiles.get((i, j), ()):
                    if feet >= roof - 0.08 or feet + height < base:
                        continue
                    if x + radius < x1 or x - radius > x2 or y + radius < y1 or y - radius > y2:
                        continue
                    if contains(ring, x, y) and not any(contains(hole, x, y) for hole in holes):
                        return True
                    if radius and (close_to_edge(ring, x, y, radius * radius) or any(
                        close_to_edge(hole, x, y, radius * radius) for hole in holes
                    )):
                        return True
        return False

    def clear_shot(self, start, end, gravity=0.0, flight=0.0):
        dx, dy, dz = (b - a for a, b in zip(start, end))
        samples = max(2, math.ceil(math.hypot(dx, dy) / 3.0))
        arc = 0.5 * gravity * flight * flight
        for index in range(1, samples):
            fraction = index / samples
            x = start[0] + dx * fraction
            y = start[1] + dy * fraction
            z = start[2] + dz * fraction + arc * fraction * (1 - fraction)
            if z < self.ground(x, y) + 0.15 or self.blocked(x, y, z):
                return False
        return True

    def walk_clearance(self, pos, direction, width, height, step_up, distance):
        """Return the unobstructed distance along a short movement segment."""
        base_ground = self.ground(*pos[:2])
        previous = pos[2]
        samples = max(1, math.ceil(distance / 1.5))
        for index in range(1, samples + 1):
            length = distance * index / samples
            x = pos[0] + direction[0] * length
            y = pos[1] + direction[1] * length
            ground = self.ground(x, y)
            feet = max(ground, pos[2]) if pos[2] > base_ground + step_up else ground
            if feet > previous + step_up or self.blocked(x, y, feet, width, height):
                return distance * (index - 1) / samples
            previous = feet
        return distance

    def can_walk_to(self, pos, goal, width, height, step_up):
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        length = math.hypot(dx, dy)
        return length == 0.0 or self.walk_clearance(
            pos, (dx / length, dy / length), width, height, step_up, length
        ) >= length

    def safe_anchor(self, pos, goal, center, zone_radius, radius, height, step_up, threat_origins, deadline):
        """Find a traversable zone destination, or nearby cover from known pursuers."""
        x, y = pos[:2]
        goal = tuple(goal[:2])
        center = tuple(center[:2])
        initial_goal_gap = math.hypot(x - goal[0], y - goal[1])
        threats = tuple(threat_origins)[:2]
        ahead = math.atan2(goal[1] - y, goal[0] - x)
        if threats and initial_goal_gap < 7.0:
            closest = min(threats, key=lambda origin: math.hypot(x - origin[0], y - origin[1]))
            ahead = math.atan2(y - closest[1], x - closest[0])
        candidates = [goal]
        if threats:
            candidates.extend(
                (x + length * math.cos(ahead + offset), y + length * math.sin(ahead + offset))
                for length in (9.0, 17.0, 26.0)
                for offset in ANCHOR_ANGLES
            )
        else:
            # A ring can close around a solid footprint. Check both finer and wider offsets.
            candidates.append(center)
            candidates.extend(
                (goal[0] + length * math.cos(offset), goal[1] + length * math.sin(offset))
                for length in (2.0, 5.0, 10.0, 18.0)
                for offset in ANCHOR_ANGLES
            )

        clearance_radius = max(0.0, zone_radius - min(3.0, 0.18 * zone_radius))
        threat_gaps = [math.hypot(x - origin[0], y - origin[1]) for origin in threats]

        def base_score(point):
            dx, dy = point[0] - x, point[1] - y
            remaining = math.hypot(point[0] - goal[0], point[1] - goal[1])
            outside = max(0.0, math.hypot(point[0] - center[0], point[1] - center[1]) - clearance_radius)
            score = 0.8 * (initial_goal_gap - remaining) - 0.07 * math.hypot(dx, dy) - 4.0 * outside
            for origin, previous_gap in zip(threats, threat_gaps):
                gap = math.hypot(point[0] - origin[0], point[1] - origin[1])
                score += 0.7 * (gap - previous_gap) - 2.0 * max(0.0, 14.0 - gap)
            return score

        best_score, best = -math.inf, goal
        for score, point in sorted(((base_score(point), point) for point in candidates), reverse=True):
            if perf_counter() >= deadline or score + 19.0 * len(threats) < best_score:
                break
            px, py = point
            trip = math.hypot(px - x, py - y)
            feet = self.ground(px, py)
            if self.blocked(px, py, feet, radius, height):
                continue
            if threats and (not 4.0 <= trip < 31.0 or not self.can_walk_to(pos, point, radius, height, step_up)):
                continue
            torso = px, py, feet + 0.65 * height
            for origin in threats:
                if perf_counter() >= deadline:
                    break
                if not self.clear_shot(origin, torso):
                    score += 19.0
            if score > best_score:
                best_score, best = score, point
        return best

    @lru_cache(maxsize=12_000)
    def grid_height(self, i, j, width, height):
        x, y = i * CELL, j * CELL
        z = self.ground(x, y)
        return None if self.blocked(x, y, z, width, height) else z

    def next_waypoint(self, pos, goal, radius, height, step_up, deadline):
        """A bounded A* search returns useful partial routes for distant goals."""
        rounded = round(pos[0] / CELL), round(pos[1] / CELL)
        start = rounded
        target = round(goal[0] / CELL), round(goal[1] / CELL)
        if start == target:
            return goal
        width = math.ceil(radius * 2) / 2
        height = math.ceil(height)
        initial = self.grid_height(*start, width, height)
        if initial is None or abs(initial - pos[2]) > 2.0:
            if abs(self.ground(*pos[:2]) - pos[2]) > 2.0:
                return goal
            nearby = sorted(
                ((rounded[0] + dx, rounded[1] + dy) for dx, dy in NEIGHBORS),
                key=lambda node: (node[0] * CELL - pos[0]) ** 2 + (node[1] * CELL - pos[1]) ** 2,
            )
            for node in nearby:
                if perf_counter() >= deadline:
                    return goal
                z = self.grid_height(*node, width, height)
                point = node[0] * CELL, node[1] * CELL
                if z is not None and abs(z - pos[2]) <= 2.0 and self.can_walk_to(pos, point, radius, height, step_up):
                    start = node
                    break
            else:
                return goal

        def remaining(node):
            return math.hypot(node[0] - target[0], node[1] - target[1])

        frontier = [(remaining(start), 0.0, start)]
        costs = {start: 0.0}
        parents = {}
        best = None
        best_remaining = math.inf
        for _ in range(185):
            if not frontier or perf_counter() >= deadline:
                break
            _, cost, current = heappop(frontier)
            if cost != costs[current]:
                continue
            if current == target:
                break
            z = self.grid_height(*current, width, height)
            for dx, dy in NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy)
                next_cost = cost + (math.sqrt(2) if dx and dy else 1)
                if next_cost >= costs.get(neighbor, math.inf):
                    continue
                next_z = self.grid_height(*neighbor, width, height)
                if next_z is None or next_z > z + step_up * 2.0:
                    continue
                if dx and dy and (
                    self.grid_height(neighbor[0], current[1], width, height) is None
                    or self.grid_height(current[0], neighbor[1], width, height) is None
                ):
                    continue
                midpoint = (current[0] + dx * 0.5) * CELL, (current[1] + dy * 0.5) * CELL
                if self.blocked(*midpoint, self.ground(*midpoint), width, height):
                    continue
                costs[neighbor], parents[neighbor] = next_cost, current
                estimate = remaining(neighbor)
                if estimate < best_remaining:
                    best, best_remaining = neighbor, estimate
                heappush(frontier, (next_cost + 1.16 * estimate, next_cost, neighbor))

        if best is None:
            return (start[0] * CELL, start[1] * CELL) if start != rounded else goal
        route = []
        while best != start:
            route.append((best[0] * CELL, best[1] * CELL))
            best = parents[best]
        if start != rounded:
            route.append((start[0] * CELL, start[1] * CELL))
        for waypoint in route[-6:]:
            dx, dy = waypoint[0] - pos[0], waypoint[1] - pos[1]
            if math.hypot(dx, dy) > 1.2 and self.can_walk_to(pos, waypoint, radius, height, step_up):
                return waypoint
        return route[-1]
