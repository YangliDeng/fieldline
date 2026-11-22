"""Choose reachable firing solutions from the observation and arena geometry."""

import math
from time import perf_counter


WEAPON_PRIORITY = {"gun": 2.0, "mortar": 3.2, "lightning": 4.0, "laser": 5.0}


def _angle_delta(angle, reference):
    return math.remainder(angle - reference, math.tau)


def _bearing(origin, target):
    return math.atan2(target[1] - origin[1], target[0] - origin[0])


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dimensions(actor, config):
    if actor.get("kind") == "mech":
        factor = 1 + actor.get("cubes", 0) * config["cubes"].get("size_per_cube", 0.05)
        return config["mech"]["radius"] * factor, config["mech"]["height"] * factor
    dimensions = config["npc"] if actor.get("kind") == "npc" else config["crates"]
    return dimensions["radius"], dimensions["height"]


def _intercept(origin, target, velocity, speed, gravity, muzzle_offset):
    if not speed:
        return target, 0.0
    dx, dy, dz = (coordinate - start for coordinate, start in zip(target, origin))
    vx, vy, vz = velocity
    if not gravity:
        a = vx * vx + vy * vy + vz * vz - speed * speed
        b = 2 * (dx * vx + dy * vy + dz * vz - speed * muzzle_offset)
        c = dx * dx + dy * dy + dz * dz - muzzle_offset * muzzle_offset
        discriminant = b * b - 4 * a * c
        if a >= 0 or discriminant < 0:
            return None
        flight = max(0.0, (-b - math.sqrt(discriminant)) / (2 * a))
    else:
        def residual(elapsed):
            x = dx + vx * elapsed
            y = dy + vy * elapsed
            z = dz + vz * elapsed + 0.5 * gravity * elapsed * elapsed
            return math.sqrt(x * x + y * y + z * z) - (speed * elapsed + muzzle_offset)

        for step in range(1, 25):
            upper = step * 0.2
            if residual(upper) <= 0:
                break
        else:
            return None
        lower = upper - 0.2
        for _ in range(10):
            middle = (upper + lower) * 0.5
            if residual(middle) > 0:
                lower = middle
            else:
                upper = middle
        flight = upper
    return tuple(coordinate + component * flight for coordinate, component in zip(target, velocity)), flight


def _shot(bot, me, actor, weapon, deadline, try_edges=True):
    stats, config = me["stats"], bot.config
    kind = weapon["kind"]
    values = {**config["weapons"][kind], **weapon["stats"]}
    radius, height = _dimensions(actor, config)
    muzzle_height = stats["height"] + 0.5 if kind == "mortar" else stats["muzzle_height"]
    firing_time = bot.firing_time(weapon)
    origin = (me["pos"][0] + firing_time * me["vel"][0],
              me["pos"][1] + firing_time * me["vel"][1],
              me["pos"][2] + firing_time * me["vel"][2] + muzzle_height)
    velocity = actor.get("vel", (0.0, 0.0, 0.0))
    x, y, feet = (coordinate + firing_time * component for coordinate, component in zip(actor["pos"], velocity))
    gap = math.dist(origin, (x, y, feet + 0.54 * height))
    if gap > values.get("range", 206) or (kind == "mortar" and gap < 21):
        return None

    fractions = (0.54, 0.83, 0.25) if try_edges and kind in ("gun", "laser") else (0.54,)
    gravity = values.get("gravity", 0.0)
    speed = values.get("speed", 0.0)
    muzzle_offset = stats["radius"] + 0.5
    for fraction in fractions:
        if perf_counter() >= deadline:
            return None
        solution = _intercept(origin, (x, y, feet + fraction * height), velocity, speed, gravity, muzzle_offset)
        if solution is None:
            continue
        endpoint, flight = solution
        horizontal = _distance(origin, endpoint)
        pitch = math.atan2(endpoint[2] - origin[2] + 0.5 * gravity * flight * flight, horizontal)
        if not stats["min_pitch"] <= pitch <= stats["max_pitch"]:
            continue
        if not bot.arena.clear_shot(origin, endpoint, gravity, flight):
            continue
        if kind == "lightning":
            tolerance = math.radians(values["cone_deg"] * 0.46), math.pi
        else:
            vertical_margin = min(0.38, fraction - 0.06, 0.94 - fraction) * height
            tolerance = (math.atan2(radius * 0.84, max(1.0, horizontal)),
                         math.atan2(vertical_margin, max(1.0, horizontal)))
        return _bearing(origin, endpoint), pitch, tolerance, values, flight
    return None


def _aligned(bot, me, aim, plan, elapsed, charging):
    stats = me["stats"]
    charge_turn = bot.config["weapons"]["laser"].get("charge_turn_mult", 1.0) if charging else 1.0
    yaw_step = elapsed * stats["turn_rate"] * charge_turn
    pitch_step = elapsed * stats["pitch_rate"] * charge_turn
    actual_yaw = me["aim"][0] + max(-yaw_step, min(yaw_step, _angle_delta(aim[0], me["aim"][0])))
    actual_pitch = me["aim"][1] + max(-pitch_step, min(pitch_step, aim[1] - me["aim"][1]))
    return abs(_angle_delta(plan[0], actual_yaw)) <= plan[2][0] and abs(plan[1] - actual_pitch) <= plan[2][1]


def _ready(bot, me, weapon, values, retreat):
    reserve = 17 if retreat and values.get("energy", 0) else 0
    if me["energy"] < values.get("energy", 0) + reserve or weapon["cooldown"] >= bot.decision_time:
        return False
    if weapon["kind"] == "gun":
        reload_time = weapon.get("reload", 0.0)
        return reload_time < bot.decision_time and (reload_time > 0 or weapon.get("ammo", 0) > 0)
    return True


def _damage_scale(bot, me, actor, kind, values):
    cubes = bot.config["cubes"]
    scale = 1 + cubes["damage_per_cube"] * min(me["cubes"], cubes["stat_cap"])
    if kind == "laser" and "mortar" in actor.get("weapons", ()):
        scale *= 1 + values.get("cookoff", 0.0)
    return scale


def _laser_charge_goal(bot, me, actor, values, retreat):
    if retreat:
        return values["min_charge"]
    damage = values["damage"] * _damage_scale(bot, me, actor, "laser", values)
    if actor.get("kind") == "mech":
        # Leave margin for armor or uncertain target health at the release tick.
        damage *= 0.83
    fraction = actor.get("hp", math.inf) / max(1.0, damage)
    charge = values["charge"] * (fraction - values["min_frac"]) / max(0.01, 1 - values["min_frac"])
    return max(values["min_charge"], min(values["charge"], charge))


def _priority(bot, me, actor, weapon, plan, retreat):
    kind, values, flight = weapon["kind"], plan[3], plan[4]
    priority = WEAPON_PRIORITY[kind]
    target_weapons = actor.get("weapons", ())
    if kind == "mortar":
        target_speed = math.hypot(*actor.get("vel", (0.0, 0.0, 0.0))[:2])
        priority -= min(2.0, max(0.0, target_speed * flight - 6.5) * 0.12)
    if kind == "lightning" and "laser" in target_weapons:
        priority += 1.9 if actor.get("charging") else 0.7
    damage = values["damage"] * _damage_scale(bot, me, actor, kind, values)
    if kind != "laser" and actor.get("hp", math.inf) <= damage * 0.86:
        if _aligned(bot, me, plan[:2], plan, bot.firing_time(weapon), charging=False):
            priority += 4.0
    return weapon.get("charge", 0.0) > 0, _ready(bot, me, weapon, values, retreat), priority


def _attack_actor(bot, me, actor, plans, retreat):
    weapons = me["weapons"]
    priorities = [_priority(bot, me, actor, weapons[index], plan, retreat) if plan else None
                  for index, plan in enumerate(plans)]
    primary = max((index for index, plan in enumerate(plans) if plan), key=priorities.__getitem__)
    laser_slot = next((index for index, weapon in enumerate(weapons)
                       if weapon and weapon.get("charge", 0.0) > 0), None)
    charge_goal = None
    if laser_slot is not None and plans[laser_slot] is not None:
        laser_plan = plans[laser_slot]
        charge_goal = _laser_charge_goal(bot, me, actor, laser_plan[3], retreat)
        gun_slot = 1 - laser_slot
        gun, gun_plan = weapons[gun_slot], plans[gun_slot]
        if (charge_goal - weapons[laser_slot]["charge"] > bot.decision_time + 1e-6
                and gun_plan is not None and gun["kind"] == "gun" and priorities[gun_slot][1]
                and abs(_angle_delta(gun_plan[0], laser_plan[0])) < bot.physics_step * me["stats"]["turn_rate"]
                and abs(gun_plan[1] - laser_plan[1]) < bot.physics_step * me["stats"]["pitch_rate"]
                and _aligned(bot, me, gun_plan[:2], gun_plan, bot.firing_time(gun), charging=True)):
            primary = gun_slot

    aim = plans[primary][:2]
    fire = [False, False]
    attacks = False
    for index, plan in enumerate(plans):
        if plan is None:
            continue
        weapon = weapons[index]
        kind, values = weapon["kind"], plan[3]
        elapsed = bot.firing_time(weapon)
        aligned = _aligned(bot, me, aim, plan, elapsed, charging=laser_slot is not None)
        if kind == "laser":
            charge = weapon.get("charge", 0.0)
            if charge > 0 and charge_goal is not None:
                fire[index] = not (aligned and charge >= charge_goal - 0.025)
                attacks = True
            elif index == primary and priorities[index][1]:
                fire[index] = abs(_angle_delta(aim[0], me["aim"][0])) < 0.48
                attacks |= fire[index]
        elif priorities[index][1] and aligned:
            fire[index] = True
            attacks = True
    return list(aim), fire, attacks


def _recent_attacker(bot, obs):
    latest = getattr(bot, "combat_attacker", None)
    for event in obs["events"]:
        if event["type"] == "damaged" and event.get("by") is not None:
            latest = event["by"], obs["t"]
    bot.combat_attacker = latest
    return latest[0] if latest is not None and obs["t"] - latest[1] <= 2.5 else None


def attack(bot, obs, threats, objective, retreat, deadline):
    """Return aim and triggers, checking a second target when the first is not actionable."""
    me, now = obs["self"], obs["t"]
    pos = me["pos"]
    recent_attacker = _recent_attacker(bot, obs)
    if retreat:
        threats = [(gap, actor) for gap, actor in threats if gap <= 45 or actor.get("hp", 1400) <= 155]
    actors = [actor for _, actor in threats]
    healing = me["hp"] < me["max_hp"] * 0.72 and now - max(bot.last_damage, bot.last_attack) < 10
    if not retreat and not healing and now - bot.last_damage > 1.6:
        farmers = [actor for actor in obs["mechs"]
                   if actor["kind"] == "npc" and _distance(pos, actor["pos"]) < 52]
        if objective and "hp" in objective and objective["id"] not in {actor["id"] for actor in farmers}:
            farmers.append(objective)
        actors.extend(farmers)

    threat_distance = threats[0][0] if threats else math.inf

    def order(actor):
        gap = _distance(pos, actor["pos"])
        turret_turn = abs(_angle_delta(_bearing(pos, actor["pos"]), me["aim"][0]))
        active_laser = actor.get("charging", False) and gap < 105
        recent_pressure = actor["id"] == recent_attacker and gap < 120
        farming = actor.get("kind") != "mech"
        farm_cost = 85 if farming and threat_distance < 83 else 0
        preferred_objective = farming and objective and actor["id"] == objective["id"]
        return (0.64 * gap + 0.045 * actor.get("hp", 1400) + 9.0 * turret_turn + farm_cost
                - 13 * active_laser - 24 * recent_pressure
                - (8 if preferred_objective else 0) - (6 if actor["id"] == bot.target_id else 0))

    actors.sort(key=order)
    pending = None
    for index, actor in enumerate(actors[:6]):
        if perf_counter() >= deadline:
            break
        if pending is not None and index >= 3:
            break
        plans = [_shot(bot, me, actor, weapon, deadline, try_edges=index < 3) if weapon else None
                 for weapon in me["weapons"]]
        if not any(plans):
            continue
        aim, fire, attacks = _attack_actor(bot, me, actor, plans, retreat)
        if attacks:
            bot.target_id = actor["id"]
            return aim, fire
        if pending is None:
            pending = actor["id"], aim

    if pending is not None:
        bot.target_id = pending[0]
        return pending[1], [False, False]
    bot.target_id = None
    if now - bot.last_damage < 2.0 and bot.damage_bearing is not None:
        return [bot.damage_bearing, 0.0], [False, False]
    if threats:
        return [_bearing(pos, threats[0][1]["pos"]), 0.0], [False, False]
    hostile_sounds = ("gun", "laser_charge", "laser", "lightning", "mortar", "boost", "jet")
    nearby = [sound for sound in obs["sounds"] if sound["dist"] == "near" and sound["age"] <= 0.25
              and sound["kind"] in hostile_sounds]
    if nearby:
        return [min(nearby, key=lambda sound: sound["age"])["bearing"], 0.0], [False, False]
    return None, [False, False]
