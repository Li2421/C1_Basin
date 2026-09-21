"""Legacy paired expert, ported without importing or executing VMAS/PID.

The waypoint logic, gains and the A/B asymmetry are deliberately preserved.
Only access to positions, velocities and goals is adapted to the SI environment.
"""
import numpy as np


def np_pos(agent):
    return agent.position.copy()


def np_vel(agent):
    return agent.velocity.copy()


def np_goal(agent):
    return agent.goal.copy()

KP = 1.5

V_MAX = 0.4

ENTRY_SPEED = 0.18

BAY_SPEED = 0.15

PASS_SPEED = 0.35

GOAL_APPROACH_SPEED = 0.2

GOAL_FAR_SPEED = 0.35

GOAL_SLOWDOWN_DIST = 0.45

GOAL_X_GAIN = 2.5

GOAL_Y_GAIN = 8.0

CHANNEL_X_GAIN = 3.0

CHANNEL_Y_GAIN = 1.5

CORRIDOR_X_GAIN = 1.5

CORRIDOR_Y_GAIN = 3.0

CORRIDOR_RECENTER_TOL = 0.015

CORRIDOR_RECENTER_X_GAIN = 0.3

CORRIDOR_RECENTER_Y_GAIN = 4.5

WAIT_TOL = 0.04

ENTRY_TOL = 0.01

BAY_TOL = 0.02

BAY_SETTLE_VEL_TOL = 0.01

CLEAR_TOL = 0.05

DATASET_GOAL_TOL = 0.02

SETTLE_VEL_TOL = 0.025

def goto(current_pos, target, max_speed=V_MAX):
    """
    Smooth proportional waypoint controller.

    No artificial minimum speed.
    Near waypoint, action naturally approaches zero.
    """
    target = np.asarray(target, dtype=np.float32)
    error = target - current_pos
    v = KP * error
    norm = np.linalg.norm(v)
    if norm > max_speed:
        v = v / norm * max_speed
    return v.astype(np.float32)

def goto_channel(current_pos, target_x, target_y, max_speed, x_gain=CHANNEL_X_GAIN, y_gain=CHANNEL_Y_GAIN):
    """
    Keep the agent centered while moving through the narrow passage.
    """
    target = np.array([target_x, target_y], dtype=np.float32)
    error = target - current_pos
    v = np.array([x_gain * error[0], y_gain * error[1]], dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > max_speed:
        v = v / norm * max_speed
    return v.astype(np.float32)

def goto_corridor(current_pos, target_x, max_speed, target_y=0.0, x_gain=CORRIDOR_X_GAIN, y_gain=CORRIDOR_Y_GAIN):
    """
    Keep the agent centered in the main corridor while moving horizontally.
    """
    y_error = target_y - current_pos[1]
    if abs(y_error) > CORRIDOR_RECENTER_TOL:
        x_gain = CORRIDOR_RECENTER_X_GAIN
        y_gain = CORRIDOR_RECENTER_Y_GAIN
    return goto_channel(current_pos, target_x=target_x, target_y=target_y, max_speed=max_speed, x_gain=x_gain, y_gain=y_gain)

def goal_corridor_speed(current_pos, target_x):
    if abs(target_x - current_pos[0]) > GOAL_SLOWDOWN_DIST:
        return GOAL_FAR_SPEED
    return GOAL_APPROACH_SPEED

def goto_goal(current_pos, goal_pos):
    return goto_channel(current_pos, target_x=goal_pos[0], target_y=goal_pos[1], max_speed=goal_corridor_speed(current_pos, goal_pos[0]), x_gain=GOAL_X_GAIN, y_gain=GOAL_Y_GAIN)

def reached(pos, target, tol):
    return np.linalg.norm(pos - np.asarray(target, dtype=np.float32)) <= tol

def settled(pos, vel, target, pos_tol, vel_tol=None):
    """
    用于 passage entrance / bay：
    不能只看位置，还必须让残余速度降下来。
    """
    if vel_tol is None:
        vel_tol = SETTLE_VEL_TOL
    return reached(pos, target, pos_tol) and np.linalg.norm(vel) <= vel_tol


class Expert:
    def __init__(self, env, mode):
        if mode not in (0, 1):
            raise ValueError("mode must be 0 or 1")
        self.env, self.mode, self.stage = env, mode, 0

    def action(self):
        action, self.stage = self.expert_actions(self.mode, self.stage)
        return action

    def expert_actions(self, mode, stage):
        """
        Agent 0:
            left -> right
    
        Agent 1:
            right -> left
    
        mode = 0:
            A_FIRST
            Agent 0 priority
            Agent 1 yields
    
        mode = 1:
            B_FIRST
            Agent 1 priority
            Agent 0 yields
        """
        p0 = np_pos(self.env.agents[0])
        p1 = np_pos(self.env.agents[1])
        v0 = np_vel(self.env.agents[0])
        v1 = np_vel(self.env.agents[1])
        LEFT_WAIT = np.array([-0.85, 0.0], dtype=np.float32)
        RIGHT_WAIT = np.array([0.85, 0.0], dtype=np.float32)
        ENTRY = np.array([0.0, 0.0], dtype=np.float32)
        BAY = np.array([0.0, 0.4], dtype=np.float32)
        RIGHT_PRE_GOAL = np.array([self.env.goals[0,0]-.17, 0.0], dtype=np.float32)
        LEFT_PRE_GOAL = np.array([self.env.goals[1,0]+.17, 0.0], dtype=np.float32)
        LEFT_CLEAR = np.array([-1.0, 0.0], dtype=np.float32)
        RIGHT_CLEAR = np.array([1.0, 0.0], dtype=np.float32)
        goal0 = np_goal(self.env.agents[0])
        goal1 = np_goal(self.env.agents[1])
        zero = np.zeros(2, dtype=np.float32)
        if mode == 0:
            if stage == 0:
                a0 = goto_corridor(p0, target_x=LEFT_WAIT[0], max_speed=V_MAX)
                a1 = goto_corridor(p1, target_x=RIGHT_WAIT[0], max_speed=V_MAX)
                if settled(p0, v0, LEFT_WAIT, WAIT_TOL) and settled(p1, v1, RIGHT_WAIT, WAIT_TOL):
                    stage = 1
            elif stage == 1:
                a0 = zero
                a1 = goto_corridor(p1, target_x=ENTRY[0], max_speed=ENTRY_SPEED)
                if settled(p1, v1, ENTRY, ENTRY_TOL):
                    stage = 2
            elif stage == 2:
                a0 = zero
                a1 = goto_channel(p1, target_x=0.0, target_y=BAY[1], max_speed=BAY_SPEED, x_gain=6.0)
                if settled(p1, v1, BAY, BAY_TOL):
                    stage = 3
            elif stage == 3:
                a0 = goto_corridor(p0, target_x=RIGHT_CLEAR[0], max_speed=PASS_SPEED)
                a1 = goto(p1, BAY, max_speed=BAY_SPEED)
                if reached(p0, RIGHT_CLEAR, CLEAR_TOL):
                    stage = 4
            elif stage == 4:
                a0 = goto_corridor(p0, target_x=RIGHT_PRE_GOAL[0], max_speed=PASS_SPEED)
                if reached(p1, ENTRY, ENTRY_TOL):
                    a1 = zero
                else:
                    a1 = goto_channel(p1, target_x=0.0, target_y=ENTRY[1], max_speed=BAY_SPEED, x_gain=6.0)
                if reached(p1, ENTRY, ENTRY_TOL) and np.linalg.norm(v1) <= BAY_SETTLE_VEL_TOL:
                    stage = 5
            else:
                a0 = zero if reached(p0, goal0, DATASET_GOAL_TOL) else goto_goal(p0, goal0)
                a1 = zero if reached(p1, goal1, DATASET_GOAL_TOL) else goto_goal(p1, goal1)
        elif stage == 0:
            a0 = goto_corridor(p0, target_x=LEFT_WAIT[0], max_speed=V_MAX)
            a1 = goto_corridor(p1, target_x=RIGHT_WAIT[0], max_speed=V_MAX)
            if settled(p0, v0, LEFT_WAIT, WAIT_TOL) and settled(p1, v1, RIGHT_WAIT, WAIT_TOL):
                stage = 1
        elif stage == 1:
            a0 = goto_corridor(p0, target_x=ENTRY[0], max_speed=ENTRY_SPEED)
            a1 = zero
            if settled(p0, v0, ENTRY, ENTRY_TOL):
                stage = 2
        elif stage == 2:
            a0 = goto_channel(p0, target_x=0.0, target_y=BAY[1], max_speed=BAY_SPEED, x_gain=6.0)
            a1 = zero
            if settled(p0, v0, BAY, BAY_TOL):
                stage = 3
        elif stage == 3:
            a0 = goto(p0, BAY, max_speed=BAY_SPEED)
            a1 = goto_corridor(p1, target_x=LEFT_CLEAR[0], max_speed=PASS_SPEED)
            if reached(p1, LEFT_CLEAR, CLEAR_TOL):
                stage = 4
        elif stage == 4:
            if reached(p0, ENTRY, ENTRY_TOL):
                a0 = zero
            else:
                a0 = goto_channel(p0, target_x=0.0, target_y=ENTRY[1], max_speed=BAY_SPEED, x_gain=6.0)
            a1 = goto(p1, LEFT_PRE_GOAL, max_speed=PASS_SPEED)
            if reached(p0, ENTRY, ENTRY_TOL) and np.linalg.norm(v0) <= BAY_SETTLE_VEL_TOL:
                stage = 5
        else:
            a0 = zero if reached(p0, goal0, DATASET_GOAL_TOL) else goto_goal(p0, goal0)
            a1 = zero if reached(p1, goal1, DATASET_GOAL_TOL) else goto_goal(p1, goal1)
        actions = np.stack([a0, a1]).astype(np.float32)
        return (actions, stage)
