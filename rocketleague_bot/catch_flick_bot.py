from __future__ import annotations

from dataclasses import dataclass
import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState


GRAVITY_Z = -650.0
GROUND_BALL_Z = 92.75
DRIBBLE_RANGE = 140.0
CENTER_TOLERANCE = 95.0
FLICK_COOLDOWN = 1.75
WALL_COOLDOWN = 3.0
AERIAL_COOLDOWN = 1.0
RESET_COOLDOWN = 2.2


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def flat_dist(self, other: "Vec3") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class CatchFlickBot(BaseAgent):
    def initialize_agent(self):
        self.controls = SimpleControllerState()
        self.flick_active = False
        self.flick_start = -999.0
        self.last_flick = -999.0

        self.wall_active = False
        self.wall_start = -999.0
        self.last_wall_play = -999.0

        self.aerial_active = False
        self.aerial_start = -999.0
        self.last_aerial = -999.0

        self.reset_active = False
        self.reset_start = -999.0
        self.last_reset = -999.0

    def get_output(self, packet):
        car = packet.game_cars[self.index]
        ball = packet.game_ball
        t = packet.game_info.seconds_elapsed

        car_pos = Vec3(car.physics.location.x, car.physics.location.y, car.physics.location.z)
        car_vel = Vec3(car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z)
        car_yaw = car.physics.rotation.yaw
        car_boost = car.boost
        has_wheel_contact = car.has_wheel_contact
        ball_pos = Vec3(ball.physics.location.x, ball.physics.location.y, ball.physics.location.z)
        ball_vel = Vec3(ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z)

        if self.reset_active:
            return self.step_flip_reset(t, car_pos, car_yaw, ball_pos)

        if self.aerial_active:
            return self.step_fast_aerial(t, car_pos, car_yaw, ball_pos)

        if self.wall_active:
            return self.step_wall_airdribble(t, car_pos, car_vel, car_yaw, ball_pos)

        if self.flick_active:
            return self.step_flick(t)

        if self.should_do_emergency_defense(car.team, car_pos, ball_pos, ball_vel):
            return self.step_defense(car.team, car_pos, car_vel, car_yaw, ball_pos, ball_vel)

        if self.should_start_flip_reset(t, car.team, car_boost, car_pos, ball_pos, ball_vel):
            self.reset_active = True
            self.reset_start = t
            return self.step_flip_reset(t, car_pos, car_yaw, ball_pos)

        if self.should_start_fast_aerial(t, car.team, car_boost, car_pos, ball_pos, ball_vel):
            self.aerial_active = True
            self.aerial_start = t
            return self.step_fast_aerial(t, car_pos, car_yaw, ball_pos)

        if self.should_start_wall_play(t, car.team, car_boost, car_pos, ball_pos, ball_vel):
            self.wall_active = True
            self.wall_start = t
            return self.step_wall_airdribble(t, car_pos, car_vel, car_yaw, ball_pos)

        if has_wheel_contact and ball_pos.z > 220.0 and car_boost > 45:
            # Frequent aggressive aerial takes with air-roll for mechanical pressure.
            self.aerial_active = True
            self.aerial_start = t
            return self.step_fast_aerial(t, car_pos, car_yaw, ball_pos)

        target = self.choose_target(ball_pos, ball_vel)
        on_dribble = self.is_dribble_controlled(car_pos, car_vel, car_yaw, ball_pos, ball_vel)

        if on_dribble and self.can_flick(t, car_vel, ball_pos, ball_vel):
            self.flick_active = True
            self.flick_start = t
            return self.step_flick(t)

        return self.step_catch(car_pos, car_vel, car_yaw, ball_pos, target)

    def should_start_flip_reset(
        self,
        now: float,
        team: int,
        boost: int,
        car_pos: Vec3,
        ball_pos: Vec3,
        ball_vel: Vec3,
    ) -> bool:
        if now - self.last_reset < RESET_COOLDOWN:
            return False
        if boost < 55:
            return False
        if not (380.0 < ball_pos.z < 1200.0):
            return False
        if car_pos.flat_dist(ball_pos) > 2300.0:
            return False

        offensive_space = self.distance_to_opponent_goal(team, ball_pos.y) < 3200.0
        ball_lifted = ball_vel.z > -100.0
        return offensive_space and ball_lifted

    def step_flip_reset(self, now: float, car_pos: Vec3, car_yaw: float, ball_pos: Vec3) -> SimpleControllerState:
        c = SimpleControllerState()
        dt = now - self.reset_start

        # target underside of ball (rough approximation for reset setup)
        reset_target = Vec3(ball_pos.x, ball_pos.y, ball_pos.z - 120.0)
        local_x, local_y = self.to_local_xy(car_pos, car_yaw, reset_target)
        angle = math.atan2(local_x, local_y)

        c.steer = max(-1.0, min(1.0, 3.0 * angle))
        c.yaw = c.steer
        c.throttle = 1.0

        # Heavy air-roll usage throughout reset attempt.
        c.roll = 1.0 if local_x > 0 else -1.0

        if dt < 0.10:
            c.jump = True
            c.pitch = -1.0
        elif dt < 0.18:
            c.jump = False
            c.pitch = -1.0
        elif dt < 0.30:
            c.jump = True
            c.pitch = -1.0
            c.boost = True
        elif dt < 1.15:
            c.pitch = 0.35  # nose slightly up so wheels approach underside
            c.boost = True
        elif dt < 1.40:
            # post-contact stabilization / immediate reset usage window
            c.pitch = -0.6
            c.roll = -c.roll
            c.boost = True
        else:
            self.reset_active = False
            self.last_reset = now

        return c

    def should_do_emergency_defense(self, team: int, car_pos: Vec3, ball_pos: Vec3, ball_vel: Vec3) -> bool:
        own_goal_y = -5120.0 if team == 0 else 5120.0
        danger_zone = abs(ball_pos.y - own_goal_y) < 1700.0
        moving_toward_own_goal = (ball_vel.y < -250.0) if team == 0 else (ball_vel.y > 250.0)
        wrong_side = self.distance_to_own_goal(team, car_pos.y) > self.distance_to_own_goal(team, ball_pos.y)
        return danger_zone and (moving_toward_own_goal or wrong_side)

    def step_defense(
        self,
        team: int,
        car_pos: Vec3,
        car_vel: Vec3,
        car_yaw: float,
        ball_pos: Vec3,
        ball_vel: Vec3,
    ) -> SimpleControllerState:
        c = SimpleControllerState()
        own_goal_y = -5120.0 if team == 0 else 5120.0
        block_y = own_goal_y + (700.0 if team == 0 else -700.0)
        block_x = max(-900.0, min(900.0, ball_pos.x))

        if ball_pos.z > 500.0 and self.distance_to_own_goal(team, ball_pos.y) < 1500.0:
            self.aerial_active = True
            self.aerial_start = self.get_game_tick_packet().game_info.seconds_elapsed
            self.last_aerial = self.aerial_start
            return self.step_fast_aerial(self.aerial_start, car_pos, car_yaw, ball_pos)

        if self.distance_to_own_goal(team, car_pos.y) > 850.0:
            defensive_target = Vec3(block_x, block_y, GROUND_BALL_Z)
            return self.drive_to_point(c, car_pos, car_vel, car_yaw, defensive_target, 1800.0)

        challenge_target = Vec3(ball_pos.x - ball_vel.x * 0.15, ball_pos.y - ball_vel.y * 0.15, GROUND_BALL_Z)
        controls = self.drive_to_point(c, car_pos, car_vel, car_yaw, challenge_target, 1900.0)
        controls.boost = controls.boost or self.distance_to_own_goal(team, ball_pos.y) < 1200.0
        return controls

    def should_start_fast_aerial(
        self,
        now: float,
        team: int,
        boost: int,
        car_pos: Vec3,
        ball_pos: Vec3,
        ball_vel: Vec3,
    ) -> bool:
        if now - self.last_aerial < AERIAL_COOLDOWN:
            return False
        if boost < 28:
            return False
        if ball_pos.z < 420.0:
            return False
        if car_pos.z > 140.0:
            return False

        close_enough = car_pos.flat_dist(ball_pos) < 2600.0
        attackable = self.distance_to_opponent_goal(team, ball_pos.y) < 2800.0
        fast_rising_or_dropping = abs(ball_vel.z) > 200.0
        return close_enough and (attackable or fast_rising_or_dropping)

    def step_fast_aerial(self, now: float, car_pos: Vec3, car_yaw: float, ball_pos: Vec3) -> SimpleControllerState:
        c = SimpleControllerState()
        dt = now - self.aerial_start

        local_x, local_y = self.to_local_xy(car_pos, car_yaw, ball_pos)
        angle = math.atan2(local_x, local_y)
        c.steer = max(-1.0, min(1.0, 3.0 * angle))
        c.yaw = c.steer

        # air-roll-heavy fast aerial execution
        if dt < 0.08:
            c.jump = True
            c.pitch = -1.0
            c.roll = 0.8 if local_x > 0 else -0.8
            c.throttle = 1.0
        elif dt < 0.14:
            c.jump = False
            c.pitch = -1.0
            c.roll = 0.8 if local_x > 0 else -0.8
            c.throttle = 1.0
        elif dt < 0.24:
            c.jump = True
            c.pitch = -1.0
            c.roll = 1.0 if local_x > 0 else -1.0
            c.boost = True
            c.throttle = 1.0
        elif dt < 1.05:
            c.pitch = -0.85
            c.roll = 1.0 if local_x > 0 else -1.0
            c.boost = True
            c.throttle = 1.0
        else:
            self.aerial_active = False
            self.last_aerial = now
            c.throttle = 1.0

        return c

    def should_start_wall_play(self, now: float, team: int, car_boost: int, car_pos: Vec3, ball_pos: Vec3, ball_vel: Vec3) -> bool:
        if now - self.last_wall_play < WALL_COOLDOWN:
            return False
        if car_boost < 100:
            return False
        if ball_pos.z > 170.0:
            return False

        side_close = abs(ball_pos.x) > 1900.0
        in_front_of_car = self.distance_to_opponent_goal(team, ball_pos.y) < self.distance_to_opponent_goal(team, car_pos.y) + 1200.0
        stable_ball = abs(ball_vel.z) < 250.0

        return side_close and in_front_of_car and stable_ball

    def step_wall_airdribble(self, now: float, car_pos: Vec3, car_vel: Vec3, car_yaw: float, ball_pos: Vec3) -> SimpleControllerState:
        c = SimpleControllerState()
        dt = now - self.wall_start
        wall_x = 4096.0 if ball_pos.x >= 0.0 else -4096.0
        target_goal_y = 5120.0 if self.team == 0 else -5120.0

        if dt < 1.20:
            setup_target = Vec3(wall_x * 0.9, ball_pos.y + (300 if self.team == 0 else -300), GROUND_BALL_Z)
            return self.drive_to_point(c, car_pos, car_vel, car_yaw, setup_target, 1650.0)

        if dt < 2.15:
            wall_carry_target = Vec3(wall_x * 0.98, ball_pos.y + (500 if self.team == 0 else -500), 300.0)
            controls = self.drive_to_point(c, car_pos, car_vel, car_yaw, wall_carry_target, 1500.0)
            controls.boost = controls.boost or (ball_pos.z < 320.0 and abs(ball_pos.x) > 3300.0)
            return controls

        if dt < 3.10:
            local_x, local_y = self.to_local_xy(car_pos, car_yaw, Vec3(ball_pos.x, target_goal_y, ball_pos.z + 90.0))
            angle = math.atan2(local_x, local_y)
            c.steer = max(-1.0, min(1.0, 3.0 * angle))
            c.pitch = -0.45
            c.yaw = c.steer
            c.roll = -0.25 if wall_x > 0 else 0.25
            c.jump = dt < 2.28
            c.boost = True
            c.throttle = 1.0
            return c

        self.wall_active = False
        self.last_wall_play = now
        return self.step_catch(car_pos, car_vel, car_yaw, ball_pos, self.choose_target(ball_pos, Vec3(0.0, 0.0, 0.0)))

    def choose_target(self, ball_pos: Vec3, ball_vel: Vec3) -> Vec3:
        target_time = 0.45
        predicted = ball_pos + ball_vel.scale(target_time)

        predicted.x = max(-3800.0, min(3800.0, predicted.x))
        predicted.y = max(-5000.0, min(5000.0, predicted.y))
        predicted.z = max(GROUND_BALL_Z, predicted.z + 0.5 * GRAVITY_Z * target_time * target_time)
        return predicted

    def is_dribble_controlled(self, car_pos: Vec3, car_vel: Vec3, car_yaw: float, ball_pos: Vec3, ball_vel: Vec3) -> bool:
        local_x, local_y = self.to_local_xy(car_pos, car_yaw, ball_pos)
        close_enough = abs(local_x) < CENTER_TOLERANCE and 30 < local_y < DRIBBLE_RANGE
        low_ball = 105.0 < ball_pos.z < 185.0
        speed_match = abs(self.flat_speed(car_vel) - self.flat_speed(ball_vel)) < 450.0
        return close_enough and low_ball and speed_match

    def can_flick(self, now: float, car_vel: Vec3, ball_pos: Vec3, ball_vel: Vec3) -> bool:
        if now - self.last_flick < FLICK_COOLDOWN:
            return False
        if ball_pos.z < 110.0:
            return False
        if self.flat_speed(car_vel) < 700.0:
            return False
        if abs(ball_vel.z) > 120.0:
            return False
        return True

    def step_flick(self, now: float) -> SimpleControllerState:
        dt = now - self.flick_start
        c = SimpleControllerState()

        if dt < 0.08:
            c.throttle = 1.0
        elif dt < 0.16:
            c.jump = True
            c.throttle = 1.0
        elif dt < 0.23:
            c.jump = False
            c.throttle = 1.0
        elif dt < 0.35:
            c.jump = True
            c.pitch = -1.0
            c.throttle = 1.0
        else:
            self.flick_active = False
            self.last_flick = now
            c.throttle = 1.0

        self.controls = c
        return c

    def step_catch(self, car_pos: Vec3, car_vel: Vec3, car_yaw: float, ball_pos: Vec3, target: Vec3) -> SimpleControllerState:
        c = SimpleControllerState()
        tx, ty = self.to_local_xy(car_pos, car_yaw, target)
        angle = math.atan2(tx, ty)

        c.steer = max(-1.0, min(1.0, 3.2 * angle))

        distance = car_pos.flat_dist(target)
        speed = self.flat_speed(car_vel)

        desired_speed = 1400.0
        if distance < 700.0:
            desired_speed = 1100.0
        if distance < 350.0 or ball_pos.z < 130.0:
            desired_speed = 800.0

        speed_error = desired_speed - speed
        c.throttle = max(-1.0, min(1.0, speed_error / 500.0))
        c.boost = speed_error > 350.0 and abs(c.steer) < 0.25 and c.throttle > 0.75
        c.handbrake = abs(angle) > 1.7 and speed > 800

        self.controls = c
        return c

    def drive_to_point(
        self,
        c: SimpleControllerState,
        car_pos: Vec3,
        car_vel: Vec3,
        car_yaw: float,
        target: Vec3,
        desired_speed: float,
    ) -> SimpleControllerState:
        tx, ty = self.to_local_xy(car_pos, car_yaw, target)
        angle = math.atan2(tx, ty)
        c.steer = max(-1.0, min(1.0, 3.0 * angle))

        speed_error = desired_speed - self.flat_speed(car_vel)
        c.throttle = max(-1.0, min(1.0, speed_error / 550.0))
        c.boost = speed_error > 300.0 and abs(c.steer) < 0.35 and c.throttle > 0.65
        c.handbrake = abs(angle) > 1.8 and self.flat_speed(car_vel) > 900.0
        return c

    @staticmethod
    def distance_to_opponent_goal(team: int, y_pos: float) -> float:
        goal_y = 5120.0 if team == 0 else -5120.0
        return abs(goal_y - y_pos)

    @staticmethod
    def distance_to_own_goal(team: int, y_pos: float) -> float:
        goal_y = -5120.0 if team == 0 else 5120.0
        return abs(goal_y - y_pos)

    @staticmethod
    def to_local_xy(origin: Vec3, yaw: float, target: Vec3) -> tuple[float, float]:
        dx = target.x - origin.x
        dy = target.y - origin.y
        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        local_x = dx * cos_y - dy * sin_y
        local_y = dx * sin_y + dy * cos_y
        return local_x, local_y

    @staticmethod
    def flat_speed(v: Vec3) -> float:
        return math.hypot(v.x, v.y)
