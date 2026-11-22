"""A placement-first entry for the Builders Mech Royale simulator."""

import math
from time import perf_counter

from arena import Arena
from combat import attack
import mechfight as mf


WEAPON_VALUE = {"gun": 1, "mortar": 2, "lightning": 3, "laser": 4}
UPGRADE_VALUE = {
    "armor": 9, "plating": 10, "nanites": 9, "servos": 8, "reactor": 7,
    "jets": 6, "sensors": 5, "capacitor": 5, "gyros": 4,
    "laser_damage": 9, "laser_charge": 8, "laser_pierce": 2,
    "lightning_range": 8, "lightning_efficiency": 7, "lightning_chain": 3,
    "gun_mag": 7, "gun_damage": 6, "gun_rate": 5, "gun_velocity": 4,
    "mortar_splash": 6, "mortar_reload": 5, "mortar_velocity": 4,
}
HEADINGS = (0, 24, -24, 48, -48, 76, -76, 105, -105, 140, -140, 180)


def clamp(number, lower, upper):
    return max(lower, min(upper, number))


def angle_delta(angle, reference):
    return math.remainder(angle - reference, math.tau)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def direction(angle):
    return math.cos(angle), math.sin(angle)


def bearing(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def zone_at(obs, time):
    zone = obs["zone"]
    remaining = zone["shrink_end"] - max(obs["t"], zone["shrink_start"])
    blend = 0.0 if time < zone["shrink_start"] else 1.0
    if remaining > 0:
        blend = clamp((time - max(obs["t"], zone["shrink_start"])) / remaining, 0.0, 1.0)
    current, future = zone["center"], zone["next_center"]
    center = tuple(current[i] + (future[i] - current[i]) * blend for i in range(2))
    return center, zone["radius"] + (zone["next_radius"] - zone["radius"]) * blend


def inside_point(pos, center, radius):
    dx, dy = pos[0] - center[0], pos[1] - center[1]
    length = math.hypot(dx, dy)
    multiplier = min(1.0, max(0.0, radius) / length) if length else 0.0
    return center[0] + dx * multiplier, center[1] + dy * multiplier


class Bot:
    name = "Fieldline"

    def __init__(self, hello: mf.Hello):
        self.arena = Arena(hello["map"]["dir"])
        self.config = hello["config"]
        self.physics_step = 1 / self.config["match"]["tick_hz"]
        self.decision_time = self.physics_step * self.config["match"]["decide_every"]
        self.side = 1 if hello["seed"] % 2 else -1
        self.last_damage = -100.0
        self.damage_bearing = None
        self.last_attack = -100.0
        self.last_pos = None
        self.last_time = None
        self.stuck = 0.0
        self.moving = False
        self.boosting = False
        self.waypoint = self.route_goal = None
        self.route_time = -100.0
        self.target_id = None
        self.scout_goal = None
        self.scout_until = 0.0
        self.cover_goal = None
        self.cover_until = 0.0
        self.zone_goal = None
        self.zone_until = 0.0
        self.recovering = False

    def weapon_pickup(self, me, pickup):
        kind = pickup["kind"]
        weapons = me["weapons"]
        if kind == "gun" or any(w and w["kind"] == kind for w in weapons):
            return None
        if None in weapons:
            return {}
        # Keep the gun as a dependable option during a specialty weapon's cooldown.
        for index, weapon in enumerate(weapons):
            if weapon["kind"] != "gun" and not weapon.get("charge", 0) and (
                WEAPON_VALUE[kind] > WEAPON_VALUE[weapon["kind"]]
            ):
                return {"drop": index}
        return None

    def update_state(self, obs):
        me, now = obs["self"], obs["t"]
        if self.last_time is not None and now > self.last_time:
            elapsed = min(0.3, now - self.last_time)
            if self.moving and distance(me["pos"], self.last_pos) < elapsed * 1.5:
                self.stuck += elapsed
            else:
                self.stuck = max(0.0, self.stuck - 2 * elapsed)
        self.last_pos, self.last_time = me["pos"], now
        for event in obs["events"]:
            if event["type"] == "damaged" and event["bearing"] is not None:
                self.last_damage, self.damage_bearing = now, event["bearing"]
            elif event["type"] == "hit":
                self.last_attack = now

    def exposed_threats(self, me, threats, deadline):
        pos, height = me["pos"], me["stats"]["height"]
        torso = pos[0], pos[1], pos[2] + 0.61 * height
        result = []
        for gap, actor in threats[:6]:
            if gap > 145 or perf_counter() >= deadline:
                break
            growth = 1 + actor.get("cubes", 0) * self.config["cubes"].get("size_per_cube", 0.05)
            actor_height = self.config["mech"]["height"] * growth
            x, y, z = actor["pos"]
            origin = (x, y, z + actor_height * 0.78)
            if gap <= 12 or self.arena.clear_shot(origin, torso):
                result.append((gap, actor, origin))
        return result

    def scout(self, obs, center, radius):
        me, now = obs["self"], obs["t"]
        pos, stats = me["pos"], me["stats"]
        if self.scout_goal is not None and now < self.scout_until and self.stuck < 0.65 and (
            distance(pos, self.scout_goal) > 8 and distance(self.scout_goal, center) < radius - 14
        ):
            return self.scout_goal
        heading = bearing(pos, center)
        if self.scout_goal is not None and distance(pos, self.scout_goal) > 4:
            heading = bearing(pos, self.scout_goal)
        options = []
        for offset in (0.38, -0.58, 1.15, -1.25, 1.85, -1.95):
            vector = direction(heading + self.side * offset)
            point = inside_point((pos[0] + 62 * vector[0], pos[1] + 62 * vector[1]), center, radius - 21)
            trip = distance(pos, point)
            if trip < 10:
                continue
            clearance = self.arena.walk_clearance(
                pos, direction(bearing(pos, point)), stats["radius"] + 0.25,
                stats["height"], stats["step_up"], min(trip, 18),
            )
            options.append((clearance + 3 * math.cos(offset) + 0.08 * trip, point))
        self.scout_goal = max(options)[1] if options else center
        self.scout_until = now + 5.5
        return self.scout_goal

    def objective(self, obs, threats, pickups, urgent):
        me, now = obs["self"], obs["t"]
        pos, speed = me["pos"], me["stats"]["walk_speed"]
        health = me["hp"] / me["max_hp"]
        center, radius = zone_at(obs, now + (17 if urgent else 10))
        margin = min(28, radius * 0.34)
        safe_spot = inside_point(pos, center, radius - margin)
        if urgent:
            return inside_point(pos, center, radius - min(45, radius * 0.53)), None

        def usable(item, limit, height=4.0):
            destination = item["pos"]
            travel = distance(pos, destination)
            if travel > limit or abs(pos[2] - destination[2]) > height:
                return False
            future_center, future_radius = zone_at(obs, now + travel / speed + 8)
            return distance(destination, future_center) + min(12, future_radius * 0.35) <= future_radius

        nearest = threats[0][0] if threats else math.inf
        first_weapon = None in me["weapons"]
        options = []
        for item, _ in pickups:
            gap = distance(pos, item["pos"])
            if (nearest > 32 or gap < 9) and usable(item, 105 if first_weapon else 80):
                priority = 41 + 33 * first_weapon + 6 * WEAPON_VALUE[item["kind"]]
                options.append((priority - gap, item))
        for item in obs["cubes"]:
            gap = distance(pos, item["pos"])
            if (nearest > 34 or gap < 9) and usable(item, 95):
                options.append((84 - gap, item))
        for drop in obs.get("drops", ()):
            gap = distance(pos, drop["pos"])
            waiting = drop["holder"] in (None, me["id"]) and drop["lands_in"] <= gap / speed + 3.0
            completing = drop["holder"] == me["id"] and drop["progress"] > drop["capture_s"] - 1.2
            safe_to_capture = nearest > 43 or (completing and nearest > 22)
            if waiting and safe_to_capture and usable(drop, 130 if health < 0.8 else 105):
                options.append((108 + 67 * (1 - health) - 0.83 * gap, drop))
        if options:
            priority, item = max(options, key=lambda entry: entry[0])
            if priority > 0:
                return item["pos"], item
        if distance(pos, safe_spot) > 6:
            return safe_spot, None
        if nearest > 88 and obs["alive"] > 3 and radius > 45 and health > 0.61 and now - self.last_damage > 1.9:
            farm = []
            for crate in obs["crates"]:
                gap = distance(pos, crate["pos"])
                if usable(crate, 76, 8):
                    farm.append((62 - 0.61 * gap - 0.05 * crate["hp"], gap, crate))
            for npc in obs["mechs"]:
                if npc["kind"] == "npc" and usable(npc, 48, 7):
                    gap = distance(pos, npc["pos"])
                    farm.append((63 - 0.9 * gap - 0.03 * npc["hp"], gap, npc))
            if farm:
                priority, gap, item = max(farm, key=lambda entry: entry[0])
                if priority > 0:
                    stand = pos
                    if gap > 37:
                        dx, dy = direction(bearing(item["pos"], pos))
                        stand = item["pos"][0] + 19 * dx, item["pos"][1] + 19 * dy
                    return stand, item
        if (nearest > 103 and radius > 75 and (now < 45 or me["cubes"] < 2 and now < 92)
                and health > 0.7 and now - self.last_damage > 1.8):
            return self.scout(obs, center, radius), None
        if distance(pos, center) > radius * 0.55 and not threats:
            return inside_point(pos, center, radius * 0.54), None
        return pos, None

    def movement(self, obs, goal, threats, retreat, urgent, deadline):
        me, now = obs["self"], obs["t"]
        pos, stats = me["pos"], me["stats"]
        width, height, step_up = stats["radius"] + 0.25, stats["height"], stats["step_up"]
        safe_center, safe_radius = zone_at(obs, now + 8.0)
        if urgent and safe_radius < 85:
            if (self.zone_goal is None or now >= self.zone_until
                or distance(self.zone_goal, safe_center) > safe_radius):
                self.zone_goal = self.arena.safe_anchor(
                    pos, goal, safe_center, safe_radius, width, height, step_up, (),
                    min(deadline - 0.009, perf_counter() + 0.003),
                )
                self.zone_until = now + 0.48
            goal = self.zone_goal
        if retreat and threats and not urgent:
            if self.cover_goal is None or now >= self.cover_until or distance(pos, self.cover_goal) < 4:
                away = direction(bearing(threats[0][1]["pos"], pos) + self.side * 0.25)
                fallback = inside_point((pos[0] + 29 * away[0], pos[1] + 29 * away[1]), safe_center, safe_radius - 9)
                self.cover_goal = self.arena.safe_anchor(
                    pos, fallback, safe_center, safe_radius, width, height, step_up,
                    [origin for _, _, origin in threats[:2]], min(deadline - 0.010, perf_counter() + 0.004),
                )
                self.cover_until = now + 0.54
            goal = self.cover_goal
        travel = distance(pos, goal)
        can_route = travel > 10 and (urgent or not retreat and (not threats or threats[0][0] > 48))
        if can_route and now - self.route_time >= 0.65 and (
            self.waypoint is None or distance(goal, self.route_goal) > 13
            or distance(pos, self.waypoint) < 3 or self.stuck > 0.5 or now - self.route_time >= 2.5
        ):
            self.route_time, self.route_goal = now, goal
            probe = min(25.0, travel)
            direct = direction(bearing(pos, goal))
            if self.arena.walk_clearance(pos, direct, width, height, step_up, probe) >= probe:
                self.waypoint = goal
            else:
                self.waypoint = self.arena.next_waypoint(
                    pos, goal, width, height, step_up, deadline=min(deadline - 0.010, perf_counter() + 0.006),
                )
        waypoint = goal
        if can_route and self.waypoint is not None and distance(goal, self.route_goal) < 13:
            if distance(pos, self.waypoint) > 2.4:
                waypoint = self.waypoint

        yaw = bearing(pos, waypoint)
        close = threats[0] if threats and threats[0][0] < 109 else None
        if close and not urgent and not retreat:
            gap, threat, _ = close
            enemy_yaw = bearing(pos, threat["pos"])
            lightning = next((w for w in me["weapons"] if w and w["kind"] == "lightning"), None)
            isolated = len(threats) == 1 or threats[1][0] > 85
            preferred = 46 if lightning else 67
            if lightning and isolated and me["energy"] >= max(18, 5 * lightning["stats"]["energy"]):
                preferred = 0.74 * lightning["stats"]["range"]
            elif isolated and me["hp"] > 0.64 * me["max_hp"] and threat.get("hp", 1400) < 430:
                preferred = 43
            if travel < 8 or gap < 58:
                correction = clamp((preferred - gap) / 55, -0.54, 0.96)
                strafe = self.side if int((now + 0.13 * me["id"]) / 1.75) % 3 != 1 else -self.side
                yaw = enemy_yaw + strafe * (math.pi / 2 + correction)
        elif retreat and not threats and not urgent and now - self.last_damage < 1.6 and self.damage_bearing is not None:
            yaw = self.damage_bearing + math.pi + self.side * 0.35

        pos_x, pos_y, pos_z = pos
        near_projectiles = []
        for projectile in obs["projectiles"]:
            px, py, pz = projectile["pos"]
            vx, vy, _ = projectile["vel"]
            dx, dy = pos_x - px, pos_y - py
            if dx * dx + dy * dy < 85**2 and dx * vx + dy * vy > 0:
                near_projectiles.append(projectile)
        near_projectiles.sort(key=lambda projectile: distance(pos, projectile["pos"]))
        near_projectiles = near_projectiles[:12]
        if travel < 2.2 and not close and not near_projectiles and (not retreat or now - self.last_damage > 1.6):
            return [0.0, 0.0], False, False

        center, radius = zone_at(obs, now + 2.0)
        ahead = stats["walk_speed"] * 0.85
        mortar_gravity = self.config["weapons"]["mortar"]["gravity"]
        imminent_lasers = [(origin, direction(actor["aim"][0])) for _, actor, origin in threats[:3]
                           if actor.get("charging") and abs(angle_delta(bearing(origin, pos), actor["aim"][0])) < 0.19]
        best = None
        best_score, longest = -math.inf, 0.0
        for offset in HEADINGS:
            vector = direction(yaw + self.side * math.radians(offset))
            clear = self.arena.walk_clearance(pos, vector, width, height, step_up, ahead)
            longest = max(longest, clear)
            end_x, end_y = pos_x + vector[0] * clear, pos_y + vector[1] * clear
            zone_distance = math.hypot(end_x - center[0], end_y - center[1])
            zone_cost = max(0.0, zone_distance - radius + min(3, radius * 0.25))
            score = math.cos(math.radians(offset)) * 4.0 + 0.67 * clear - 1.4 * zone_cost
            score += 0.06 * (vector[0] * me["vel"][0] + vector[1] * me["vel"][1])
            if urgent:
                score -= 0.19 * zone_distance
            if retreat and close and not urgent:
                score -= 0.21 * distance((end_x, end_y), goal)
            for origin, beam in imminent_lasers:
                dx, dy = pos_x + 0.38 * clear * vector[0] - origin[0], pos_y + 0.38 * clear * vector[1] - origin[1]
                score -= 2.8 * max(0.0, width + 2.0 - abs(dx * beam[1] - dy * beam[0]))
            for projectile in near_projectiles:
                px, py, pz = projectile["pos"]
                vx, vy, vz = projectile["vel"]
                speed = stats["walk_speed"] if clear > 3 else 0.0
                dx, dy = px - pos_x, py - pos_y
                vx, vy = vx - vector[0] * speed, vy - vector[1] * speed
                approach = -(dx * vx + dy * vy) / max(vx * vx + vy * vy, 1)
                if not 0 < approach < 0.55:
                    continue
                projectile_z = pz + vz * approach
                if projectile["kind"] == "mortar":
                    projectile_z -= 0.5 * mortar_gravity * approach * approach
                if pos_z - 3 < projectile_z < pos_z + height + 3:
                    miss = math.hypot(dx + vx * approach, dy + vy * approach)
                    danger_radius = width + (6.5 if projectile["kind"] == "mortar" else 1.7)
                    score -= 2.2 * max(0.0, danger_radius - miss)
            if score > best_score:
                best_score, best = score, (vector, clear)

        vector, clear = best
        detour = self.stuck > 0.75 and longest < 3.2
        jump = detour and me["grounded"] and me["energy"] > 35
        if detour:
            vector, clear = direction(yaw), ahead
        if not me["grounded"]:
            boost = me["energy"] > 18 and (detour or clear < 2.5) and travel > 10
        else:
            reserve = 10 if urgent else 31
            enough_energy = me["energy"] > reserve + (0 if self.boosting else 20)
            boost = enough_energy and clear > 4.5 and (urgent or retreat or bool(near_projectiles) or bool(imminent_lasers))
        speed = 1.0 if close or retreat or urgent or travel >= 6 else clamp(travel / 6, 0.35, 1.0)
        return [speed * vector[0], speed * vector[1]], boost, jump

    def firing_time(self, weapon):
        # A cooling weapon can start firing after the torso has turned one more tick.
        wait = max(weapon["cooldown"], weapon.get("reload", 0.0))
        if weapon["kind"] != "laser" and self.physics_step + 1e-6 < wait < self.decision_time:
            return self.physics_step * math.ceil((wait - 1e-6) / self.physics_step)
        return self.physics_step

    def act(self, obs: mf.Observation) -> mf.Action:
        deadline = perf_counter() + self.config["bot"]["tick_budget_ms"] * 0.00058
        self.update_state(obs)
        me, now = obs["self"], obs["t"]
        pos, health = me["pos"], me["hp"] / me["max_hp"]
        threats = sorted(
            ((distance(pos, actor["pos"]), actor) for actor in obs["mechs"] if actor["kind"] == "mech"),
            key=lambda entry: entry[0],
        )
        exposed = self.exposed_threats(me, threats, min(deadline - 0.013, perf_counter() + 0.004))
        pickups = [(item, self.weapon_pickup(me, item)) for item in obs["pickups"]]
        pickups = [(item, swap) for item, swap in pickups if swap is not None]
        center, radius = zone_at(obs, now + 8)
        urgent = not me["in_zone"] or distance(pos, center) > radius - min(7, radius * 0.28)
        under_fire = now - self.last_damage < 3.0
        close_count = sum(gap < 83 for gap, _, _ in exposed)
        retreat = (health < 0.46 and (under_fire or len(exposed) > 0)) or close_count >= 3
        goal, objective = self.objective(obs, exposed, pickups, urgent)
        capture = objective and "capture_s" in objective and objective["lands_in"] == 0
        capturing = capture and distance(pos, goal) < objective["radius"] - 1.5 and not urgent
        quiet = not urgent and not exposed and now - self.last_damage > 1.05 and not obs["projectiles"]
        self.recovering = quiet and health < (0.79 if self.recovering else 0.52)
        heading_to_drop = objective is not None and "capture_s" in objective
        rest = self.recovering and not heading_to_drop and distance(pos, goal) < 23
        if capturing or rest:
            move, boost, jump = [0.0, 0.0], False, False
        else:
            move, boost, jump = self.movement(obs, goal, exposed, retreat, urgent, deadline)
        if rest:
            self.target_id = None
            aim = [bearing(pos, threats[0][1]["pos"]), 0.0] if threats else None
            fire = [False, False]
        else:
            aim, fire = attack(self, obs, threats, objective, retreat, deadline)
        if aim is None:
            heading = bearing(pos, goal) if distance(pos, goal) > 3 else 0.37 * now + 0.4 * me["id"]
            aim = [heading + 0.65 * math.sin(now * 1.2), 0.0]
        self.moving = math.hypot(*move) >= 0.7
        self.boosting = boost
        action: mf.Action = {"tick": obs["tick"], "move": move, "boost": boost, "jump": jump, "aim": aim, "fire": fire}
        for item, swap in sorted(pickups, key=lambda entry: -WEAPON_VALUE[entry[0]["kind"]]):
            if distance(pos, item["pos"]) <= self.config["loot"]["pickup_radius"] and abs(pos[2] - item["pos"][2]) <= 4:
                action.update(take=item["id"], **swap)
                break
        offer = obs["offer"]
        if offer and offer["choices"]:
            action["pick"] = max(
                range(len(offer["choices"])),
                key=lambda index: UPGRADE_VALUE.get(offer["choices"][index]["id"], 0),
            )
        if (not under_fire or rest) and not any(fire):
            action["reload"] = any(
                weapon and weapon["kind"] == "gun" and weapon.get("reload", 0) == 0
                and weapon.get("ammo", math.inf) <= 4 for weapon in me["weapons"]
            )
        return action


if __name__ == "__main__":
    mf.run(Bot)
