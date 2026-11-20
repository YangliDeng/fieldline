"""A placement-first entry for the Builders Mech Royale simulator."""

import math
from time import perf_counter

from arena import Arena
import mechfight as mf


WEAPON_VALUE = {"gun": 1, "mortar": 2, "lightning": 3, "laser": 4}
SHOT_VALUE = {"gun": 4.0, "laser": 6.0, "lightning": 6.5, "mortar": 6.2}
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


def target_size(actor, config):
    if actor.get("kind") == "mech":
        factor = 1 + actor.get("cubes", 0) * config["cubes"].get("size_per_cube", 0.05)
        return config["mech"]["radius"] * factor, config["mech"]["height"] * factor
    dimensions = config["npc"] if actor.get("kind") == "npc" else config["crates"]
    return dimensions["radius"], dimensions["height"]


def solve_shot(origin, target, velocity, speed, gravity, spawn_offset):
    """Find the earliest intersection with a constant-velocity target."""
    delta = tuple(b - a for a, b in zip(origin, target))
    if not speed:
        return target, 0.0
    if not gravity:
        # The spawn offset advances the projectile along the resulting aim vector.
        a = sum(v * v for v in velocity) - speed * speed
        b = 2 * (sum(d * v for d, v in zip(delta, velocity)) - speed * spawn_offset)
        c = sum(d * d for d in delta) - spawn_offset * spawn_offset
        discriminant = b * b - 4 * a * c
        if discriminant < 0 or a >= 0:
            return None
        flight = max(0.0, (-b - math.sqrt(discriminant)) / (2 * a))
        return tuple(target[i] + velocity[i] * flight for i in range(3)), flight

    def residual(t):
        dx = delta[0] + velocity[0] * t
        dy = delta[1] + velocity[1] * t
        dz = delta[2] + velocity[2] * t + 0.5 * gravity * t * t
        return math.sqrt(dx * dx + dy * dy + dz * dz) - (speed * t + spawn_offset)

    for step in range(1, 25):
        upper = step * 0.2
        if residual(upper) <= 0:
            lower = upper - 0.2
            for _ in range(12):
                middle = (upper + lower) * 0.5
                if residual(middle) > 0:
                    lower = middle
                else:
                    upper = middle
            return tuple(target[i] + velocity[i] * upper for i in range(3)), upper
    return None


class Bot:
    name = "Fieldline"

    def __init__(self, hello: mf.Hello):
        self.arena = Arena(hello["map"]["dir"])
        self.config = hello["config"]
        self.physics_step = 1 / self.config["match"]["tick_hz"]
        self.side = 1 if hello["seed"] % 2 else -1
        self.last_damage = -100.0
        self.damage_bearing = None
        self.damage_load = 0.0
        self.last_attack = -100.0
        self.last_pos = None
        self.last_time = None
        self.stuck = 0.0
        self.moving = False
        self.boosting = False
        self.waypoint = self.route_goal = None
        self.route_time = -100.0
        self.target_id = None
        self.recovering = False
        self.scout_heading = None
        self.scout_goal = None
        self.scout_until = 0.0

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
            elapsed = now - self.last_time
            self.damage_load *= math.exp(-elapsed / 2.0)
            elapsed = min(0.3, elapsed)
            if self.moving and distance(me["pos"], self.last_pos) < elapsed * 1.5:
                self.stuck += elapsed
            else:
                self.stuck = max(0.0, self.stuck - 2 * elapsed)
        self.last_pos, self.last_time = me["pos"], now
        for event in obs["events"]:
            if event["type"] == "damaged" and event["bearing"] is not None:
                self.last_damage, self.damage_bearing = now, event["bearing"]
                self.damage_load += event["amount"]
            elif event["type"] == "hit":
                self.last_attack = now

    def exposed_threats(self, obs, threats, deadline):
        me = obs["self"]
        x, y, z = me["pos"]
        torso = x, y, z + me["stats"]["height"] * 0.62
        exposed = []
        for gap, actor in threats[:6]:
            if gap > 145 or perf_counter() >= deadline:
                break
            if gap < 13 or self.arena.clear_shot(self.enemy_muzzle(actor), torso):
                exposed.append((gap, actor))
        return exposed

    def enemy_muzzle(self, actor):
        x, y, z = actor["pos"]
        _, height = target_size(actor, self.config)
        return x, y, z + height * 0.78

    def exploration_goal(self, obs, center, radius):
        me, now = obs["self"], obs["t"]
        pos, stats = me["pos"], me["stats"]
        if self.scout_goal is not None and now < self.scout_until and self.stuck < 0.65 and (
            distance(pos, self.scout_goal) > 12 and distance(self.scout_goal, center) < radius - 14
        ):
            return self.scout_goal
        if self.scout_heading is None:
            self.scout_heading = me["aim"][0]
        bias = self.scout_heading + self.side * 0.78
        best_score = -math.inf
        self.scout_goal = pos
        for offset in (0, -0.85, 0.85, -1.65, 1.65, math.pi):
            heading = bias + self.side * offset
            dx, dy = direction(heading)
            destination = inside_point((pos[0] + 105 * dx, pos[1] + 105 * dy), center, radius - 25)
            travel = distance(pos, destination)
            if travel < 15:
                continue
            direct = direction(bearing(pos, destination))
            clear = self.arena.walk_clearance(
                pos, direct, stats["radius"] + 0.3, stats["height"], stats["step_up"], min(19, travel)
            )
            score = clear + 0.075 * travel + 5 * math.cos(offset)
            if score > best_score:
                best_score, self.scout_goal = score, destination
        self.scout_heading = bearing(pos, self.scout_goal)
        self.scout_until = now + 7.5
        return self.scout_goal

    def objective(self, obs, threats, pickups, retreat, urgent):
        me, now = obs["self"], obs["t"]
        pos, speed = me["pos"], me["stats"]["walk_speed"]
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

        options = []
        nearest = threats[0][0] if threats else math.inf
        for drop in obs.get("drops", ()):
            gap = distance(pos, drop["pos"])
            missing_hp = 1 - me["hp"] / me["max_hp"]
            capturing = drop["holder"] == me["id"] and gap < drop["radius"]
            room = nearest > 31 or (capturing and drop["progress"] >= drop["capture_s"] - 1)
            if room and drop["holder"] in (None, me["id"]) and drop["lands_in"] <= gap / speed + 2 and (
                usable(drop, 130 if missing_hp > 0.30 else 95)
            ):
                options.append((104 + 68 * missing_hp - gap, drop))
        if nearest < 30 or retreat:
            if options:
                priority, drop = max(options, key=lambda entry: entry[0])
                if priority > 0 and (nearest == math.inf or distance(pos, drop["pos"]) < drop["radius"]):
                    return drop["pos"], drop
            return safe_spot, None
        for item, _ in pickups:
            if usable(item, 95):
                bonus = 23 if None in me["weapons"] else 0
                options.append((43 + bonus + 5 * WEAPON_VALUE[item["kind"]] - distance(pos, item["pos"]), item))
        for item in obs["cubes"]:
            if usable(item, 88):
                options.append((74 - distance(pos, item["pos"]), item))
        if options:
            priority, item = max(options, key=lambda entry: entry[0])
            if priority > 0:
                return item["pos"], item
        if distance(pos, safe_spot) > 6:
            return safe_spot, None
        if not threats and obs["alive"] > 3 and radius > 45:
            crates = [item for item in obs["crates"] if usable(item, 82, 5)]
            if crates:
                crate = min(crates, key=lambda item: distance(pos, item["pos"]))
                direction_to_crate = direction(bearing(crate["pos"], pos))
                stand = tuple(crate["pos"][i] + 15 * direction_to_crate[i] for i in range(2))
                return stand, crate
        if not threats and radius > 85 and (now < 54 or me["cubes"] < 3 and now < 155):
            return self.exploration_goal(obs, center, radius), None
        if distance(pos, center) > radius * 0.55 and not threats:
            return inside_point(pos, center, radius * 0.54), None
        return pos, None

    def movement(self, obs, goal, threats, retreat, urgent, deadline):
        me, now = obs["self"], obs["t"]
        pos, stats = me["pos"], me["stats"]
        width, height, step_up = stats["radius"] + 0.25, stats["height"], stats["step_up"]
        travel = distance(pos, goal)
        can_route = travel > 10 and (urgent or not threats or threats[0][0] > 40)
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
                    pos, goal, width, height, step_up, deadline=perf_counter() + 0.006,
                )
        waypoint = goal
        if can_route and self.waypoint is not None and distance(goal, self.route_goal) < 13:
            if distance(pos, self.waypoint) > 2.4:
                waypoint = self.waypoint

        yaw = bearing(pos, waypoint)
        close = threats[0] if threats and threats[0][0] < 125 else None
        running_from_sound = retreat and not close and now - self.last_damage < 1.4
        if close and not urgent:
            gap, threat = close
            enemy_yaw = bearing(pos, threat["pos"])
            lightning = next((w for w in me["weapons"] if w and w["kind"] == "lightning"), None)
            lightning_range = lightning["stats"]["range"] if lightning and me["energy"] >= 19 else 0
            use_lightning = lightning_range > 0 and len(threats) == 1
            preferred = lightning_range * 0.71 if use_lightning else 64
            if retreat:
                yaw = enemy_yaw + math.pi
            elif travel < 8 or gap < max(57, lightning_range + 12):
                correction = clamp((preferred - gap) / 44, -0.72 if use_lightning else -0.31, 1.15)
                strafe = self.side if int(now / 1.55) % 3 != 1 else -self.side
                yaw = enemy_yaw + strafe * (math.pi / 2 + correction)
        elif running_from_sound and not urgent and self.damage_bearing is not None:
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
        if travel < 2.2 and not close and not running_from_sound and not near_projectiles:
            return [0.0, 0.0], False, False

        center, radius = zone_at(obs, now + 2.0)
        ahead = stats["walk_speed"] * 0.85
        mortar_gravity = self.config["weapons"]["mortar"]["gravity"]
        candidates = []
        longest = 0.0
        for offset in HEADINGS:
            vector = direction(yaw + self.side * math.radians(offset))
            clear = self.arena.walk_clearance(pos, vector, width, height, step_up, ahead)
            longest = max(longest, clear)
            end_x, end_y = pos_x + vector[0] * clear, pos_y + vector[1] * clear
            zone_distance = math.hypot(end_x - center[0], end_y - center[1])
            zone_cost = max(0.0, zone_distance - radius + min(3, radius * 0.25))
            score = math.cos(math.radians(offset)) * 4.0 + 0.67 * clear - 1.4 * zone_cost
            if urgent:
                score -= 0.19 * zone_distance
            if retreat and close and not urgent:
                for gap, pursuer in threats[:2]:
                    if gap < 125:
                        score += 0.29 * (distance((end_x, end_y), pursuer["pos"]) - gap)
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
                    projectile_z -= 0.5 * mortar_gravity * approach**2
                if pos_z - 3 < projectile_z < pos_z + height + 3:
                    miss = math.hypot(dx + vx * approach, dy + vy * approach)
                    danger_radius = width + (6.5 if projectile["kind"] == "mortar" else 1.7)
                    score -= 2.2 * max(0.0, danger_radius - miss)
            candidates.append((score, vector, clear, end_x, end_y))

        ranked = sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
        best = ranked[0]
        if retreat and close and not urgent:
            cover_deadline = min(deadline - 0.006, perf_counter() + 0.004)
            muzzles = [self.enemy_muzzle(pursuer) for gap, pursuer in threats[:2] if gap < 125]
            above_floor = pos_z > self.arena.ground(pos_x, pos_y) + step_up
            for score, vector, clear, end_x, end_y in ranked[:6]:
                if best[0] > score + 10.0 or perf_counter() >= cover_deadline:
                    break
                if clear < 4:
                    continue
                ground = self.arena.ground(end_x, end_y)
                end_z = max(ground, pos_z) if above_floor else ground
                torso = end_x, end_y, end_z + height * 0.62
                covered = sum(not self.arena.clear_shot(muzzle, torso) for muzzle in muzzles)
                score += 5.0 * covered
                if score > best[0]:
                    best = score, vector, clear, end_x, end_y
        _, vector, clear, _, _ = best
        detour = self.stuck > 0.75 and longest < 3.2
        jump = detour and me["grounded"] and me["energy"] > 35
        if detour:
            vector, clear = direction(yaw), ahead
        if not me["grounded"]:
            boost = me["energy"] > 18 and (detour or clear < 2.5) and travel > 10
        else:
            reserve = 10 if urgent else 31
            enough_energy = me["energy"] > reserve + (0 if self.boosting else 20)
            boost = enough_energy and clear > 4.5 and (urgent or retreat or bool(near_projectiles))
        speed = 1.0 if close or running_from_sound or urgent or travel >= 6 else clamp(travel / 6, 0.35, 1.0)
        return [speed * vector[0], speed * vector[1]], boost, jump

    def shot(self, me, actor, weapon):
        stats, config = me["stats"], self.config
        kind = weapon["kind"]
        values = {**config["weapons"][kind], **weapon["stats"]}
        radius, height = target_size(actor, config)
        muzzle = stats["height"] + 0.5 if kind == "mortar" else stats["muzzle_height"]
        origin = tuple(me["pos"][i] + self.physics_step * me["vel"][i] for i in range(3))
        origin = origin[0], origin[1], origin[2] + muzzle
        velocity = actor.get("vel", (0, 0, 0))
        target = tuple(actor["pos"][i] + self.physics_step * velocity[i] for i in range(3))
        target = target[0], target[1], target[2] + 0.54 * height
        gap = math.dist(origin, target)
        if gap > values.get("range", 206) or (
            kind == "mortar" and gap < values["splash_radius"] + stats["radius"] + 4
        ):
            return None
        gravity = values.get("gravity", 0.0)
        result = solve_shot(origin, target, velocity, values.get("speed", 0), gravity, stats["radius"] + 0.5)
        if result is None:
            return None
        endpoint, flight = result
        horizontal = distance(origin, endpoint)
        pitch = math.atan2(endpoint[2] - origin[2] + 0.5 * gravity * flight * flight, horizontal)
        if not stats["min_pitch"] <= pitch <= stats["max_pitch"]:
            return None
        if not self.arena.clear_shot(origin, endpoint, gravity, flight):
            return None
        yaw = bearing(origin, endpoint)
        if kind == "lightning":
            tolerance = math.radians(values["cone_deg"] * 0.46), math.pi
        else:
            tolerance = (
                math.atan2(radius * 0.84, max(1.0, horizontal)),
                math.atan2(height * 0.38, max(1.0, horizontal)),
            )
        return yaw, pitch, tolerance, values, flight

    def attack(self, obs, threats, objective, retreat, urgent, deadline):
        me, now = obs["self"], obs["t"]
        pos, stats = me["pos"], me["stats"]
        nearest = threats[0][1] if threats else None
        if retreat:
            threats = [entry for entry in threats if entry[0] <= 42 and entry[1].get("hp", math.inf) <= 380]
        actors = [actor for _, actor in threats]
        npcs = [actor for actor in obs["mechs"] if actor["kind"] == "npc" and distance(pos, actor["pos"]) < 52]
        healing = me["hp"] < me["max_hp"] * 0.72 and now - max(self.last_damage, self.last_attack) < 10
        if not actors and not retreat and not healing and now - self.last_damage > 2:
            actors.extend(npcs)
            if objective and objective["id"] < 5000 and "hp" in objective:
                actors.append(objective)
        actors.sort(key=lambda actor: (
            distance(pos, actor["pos"]) * 0.56
            + actor.get("hp", 1200) * (0.041 if actor.get("kind") == "mech" else 0.027)
            + 8 * abs(angle_delta(bearing(pos, actor["pos"]), me["aim"][0]))
            - (5 if actor["id"] == self.target_id else 0)
        ))

        weapons = me["weapons"]
        window = self.physics_step * self.config["match"]["decide_every"]
        aiming = None
        for actor in actors[:6]:
            if perf_counter() >= deadline:
                break
            plans = [self.shot(me, actor, weapon) if weapon else None for weapon in weapons]
            priorities = []
            target_speed = math.hypot(*actor.get("vel", (0.0, 0.0, 0.0))[:2])
            target_hp = actor.get("hp", math.inf)
            ready_to_fire = [False, False]
            for index, plan in enumerate(plans):
                if plan is None:
                    continue
                shot_yaw, shot_pitch, _, values, flight = plan
                weapon = weapons[index]
                kind = weapon["kind"]
                charge = weapon.get("charge", 0.0)
                energy_cost = values["energy"] if kind == "lightning" else 0
                loaded = kind != "gun" or weapon.get("ammo", 0) > 0
                ready = (weapon["cooldown"] <= window and loaded and weapon.get("reload", 0) <= window
                         and me["energy"] >= energy_cost + (12 if retreat and energy_cost else 0))
                if kind == "laser" and (urgent or retreat) and not charge:
                    ready = False
                damage = values["damage"]
                if kind == "laser":
                    fraction = min(1.0, charge / values["charge"])
                    damage *= values["min_frac"] + fraction * (1 - values["min_frac"])
                cubes = self.config["cubes"]
                damage *= 1 + cubes["damage_per_cube"] * min(me["cubes"], cubes["stat_cap"])
                if retreat and target_hp > damage * 0.77:
                    ready = False
                if not ready:
                    if aiming is None and not retreat:
                        aiming = [shot_yaw, shot_pitch]
                    continue
                ready_to_fire[index] = True
                utility = SHOT_VALUE[kind]
                if kind == "mortar":
                    utility -= min(4.3, 0.23 * flight * target_speed + 0.95 * flight)
                utility += 6.0 if charge else 0.0
                if target_hp <= damage * 0.77:
                    utility += 2.2
                priorities.append((utility, index))
            if not priorities:
                continue
            _, primary = max(priorities)
            yaw, pitch = plans[primary][:2]
            yaw_step = self.physics_step * stats["turn_rate"]
            pitch_step = self.physics_step * stats["pitch_rate"]
            actual_yaw = me["aim"][0] + clamp(angle_delta(yaw, me["aim"][0]), -yaw_step, yaw_step)
            actual_pitch = me["aim"][1] + clamp(pitch - me["aim"][1], -pitch_step, pitch_step)
            fire = [False, False]
            for index, plan in enumerate(plans):
                if not ready_to_fire[index]:
                    continue
                shot_yaw, shot_pitch, tolerance, values, _ = plan
                weapon = weapons[index]
                aligned = abs(angle_delta(shot_yaw, actual_yaw)) <= tolerance[0] and abs(shot_pitch - actual_pitch) <= tolerance[1]
                if weapon["kind"] == "laser":
                    charge = weapon.get("charge", 0.0)
                    target_charge = values["min_charge"] if retreat or urgent else values["charge"]
                    if actor.get("hp", math.inf) < values["damage"] * values["min_frac"] * 0.85:
                        target_charge = values["min_charge"]
                    fire[index] = not retreat and index == primary and (
                        charge > 0 or abs(angle_delta(yaw, me["aim"][0])) < 0.48
                    ) and not (aligned and charge >= target_charge - 0.025)
                else:
                    fire[index] = aligned
            self.target_id = actor["id"]
            return [yaw, pitch], fire
        self.target_id = None
        if now - self.last_damage < 2.0 and self.damage_bearing is not None:
            return [self.damage_bearing, 0.0], [False, False]
        if aiming is not None:
            return aiming, [False, False]
        if nearest is not None:
            return [bearing(pos, nearest["pos"]), 0.0], [False, False]
        nearby = [sound for sound in obs["sounds"] if sound["dist"] == "near" and sound["age"] <= 0.25
                  and sound["kind"] in ("gun", "laser_charge", "laser", "lightning", "mortar", "boost", "jet")]
        if nearby:
            return [min(nearby, key=lambda sound: sound["age"])["bearing"], 0.0], [False, False]
        return None, [False, False]

    def act(self, obs: mf.Observation) -> mf.Action:
        deadline = perf_counter() + self.config["bot"]["tick_budget_ms"] * 0.00058
        self.update_state(obs)
        me, now = obs["self"], obs["t"]
        pos, health = me["pos"], me["hp"] / me["max_hp"]
        threats = sorted(
            ((distance(pos, actor["pos"]), actor) for actor in obs["mechs"] if actor["kind"] == "mech"),
            key=lambda entry: entry[0],
        )
        exposed = self.exposed_threats(obs, threats, min(deadline - 0.012, perf_counter() + 0.005))
        pickups = [(item, self.weapon_pickup(me, item)) for item in obs["pickups"]]
        pickups = [(item, swap) for item, swap in pickups if swap is not None]
        center, radius = zone_at(obs, now + 8)
        urgent = not me["in_zone"] or distance(pos, center) > radius - min(7, radius * 0.28)
        if obs["zone"]["next_radius"] == 0 and obs["zone"]["shrink_start"] < now + 9:
            urgent = urgent or distance(pos, obs["zone"]["next_center"]) > max(2, radius * 0.43)
        under_fire = now - self.last_damage < 3.0
        close_count = sum(gap < 85 for gap, _ in exposed)
        if self.recovering:
            self.recovering = health < 0.80
        else:
            taking_burst = self.damage_load > 0.22 * me["max_hp"] and health < 0.65
            self.recovering = health < 0.42 or taking_burst or (health < 0.56 and (under_fire or close_count > 0))
        hopeless_retreat = obs["alive"] <= 2 and radius < 30 and close_count > 0
        retreat = (self.recovering and not hopeless_retreat) or close_count >= 3
        goal, objective = self.objective(obs, exposed, pickups, retreat, urgent)
        capturing = (objective is not None and "capture_s" in objective and objective["lands_in"] == 0
                     and distance(pos, objective["pos"]) < objective["radius"] - 1.2 and not urgent)
        if capturing:
            move, boost, jump = [0.0, 0.0], False, False
        else:
            move, boost, jump = self.movement(obs, goal, exposed, retreat, urgent, deadline)
        aim, fire = self.attack(obs, threats, objective, retreat, urgent, deadline)
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
        if (self.target_id is None or not under_fire) and not any(fire):
            action["reload"] = any(
                weapon and weapon["kind"] == "gun" and weapon.get("reload", 0) == 0
                and weapon.get("ammo", math.inf) <= 4 for weapon in me["weapons"]
            )
        return action


if __name__ == "__main__":
    mf.run(Bot)
