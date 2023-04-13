import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
import random
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, configs):
        
        # Initialize robots and humans
        self.robot = None
        self.humans = []
        self.observable_dist = 2
        self.num_humans = 0
        self.actions = []

        # Initialize times
        self.global_time = 0    # Time of entire env
        self.time_step = 0.25   # Timestep for each step/action
        self.time_limit = 25    # Max time before termination
        
        # Rewards
        self.success = 1
        self.collision = -0.25
        self.close_call_reward_var = 0.05
        self.close_call_dist = 0.3

        # env_settings
        self.start_pos_radius = 5   # for setting up of start pos of hum/rob
        self.square_setting = 5     # for setting up of start pos of hum/rob
        self.random_hum_info = True   # True if hum speed/radius different from robot
        self.train = True           # True if in training mode, else test mode
        self.scenario = "circle"    # Scenario to choose between circle or square crossing
        self.config = configs

        # Visualization
        self.states = None
    
    def set_robot(self, robot):
        # robot_type = list 
        self.robot = robot
    
    def set_human_num(self, num):
        self.num_humans = num

    def set_scenario(self, scenario):
        self.scenario = scenario
    
    def generate_humans(self, num_human):
        
        # Check scenario
        if self.scenario == 'circle':
            for i in range(num_human):
                self.humans.append(self.generate_circle_crossing_humans)
        
        elif self.scenario == 'square':
            for i in range(num_human):
                self.humans.append(self.generate_square_crossing_humans)

        else:
            for i in range(num_human):
                prob = random.random.uniform()
                if prob < 0.5:
                    self.humans.append(self.generate_circle_crossing_humans)

                else:
                    self.humans.append(self.generate_square_crossing_humans)
    
    def generate_circle_crossing_humans(self):
        human = Human(self.config, 'humans')
        set = False

        if self.random_hum_info:
            human.sample_random_attributes()

        while not set:
            angle = np.random.random()  * (2 * np.pi)
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.start_pos_radius * np.cos(angle) + px_noise
            py = self.start_pos_radius * np.sin(angle) + py_noise

            for i,agent in enumerate(self.robot + self.humans):
                comfort_dist = human.radius + agent.radius + self.close_call_dist
                if norm((px - agent.px, py - agent.py)) < comfort_dist:
                    break
                
                if i == len(self.robot + self.humans) - 1:
                    set = True
        
        human.set(px, py, -px, -py, 0, 0, 0)
        return human
    
    def generate_square_crossing_humans(self):
        human = Human(self.config, 'humans')
        pos_set = False
        goal_set = False
        sign = 1

        if self.random_hum_info:
            human.sample_random_attributes()

        if np.random.random() > 0.5:
            sign = -1
        
        while not pos_set:
            px = np.random.random() * self.square_setting * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_setting

            for i,agent in enumerate(self.robot + self.humans):
                comfort_dist = human.radius + agent.radius + self.close_call_dist
                if norm((px - agent.px, py - agent.py)) < comfort_dist:
                    break
                
                if i == len(self.robot + self.humans) - 1:
                    pos_set = True
            
        while not goal_set:
            gx = np.random.random() * self.square_setting * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_setting

            for i,agent in enumerate(self.robot + self.humans):
                comfort_dist = human.radius + agent.radius + self.close_call_dist
                if norm((gx - agent.gx, gy - agent.gy)) < comfort_dist:
                    break
                
                if i == len(self.robot + self.humans) - 1:
                    goal_set = True

        human.set(px,py,gx,gy,0,0,0)
        return human
    
    def set_robot_position(self):
        self.robot_ready = []
        
        for robot in self.robot:
            if self.scenario == 'circle':
                self.set_robot_circle(robot)
            elif self.scenario == 'square':
                self.set_robot_square(robot)
            else:
                prob = random.random.uniform()
                if prob < 0.5:
                    self.set_robot_circle(robot)
                else:
                    self.set_robot_square(robot)
        
            self.robot_ready.append(robot)
    
    def set_robot_circle(self, robot):
        set = False

        while not set:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * robot.v_pref
            py_noise = (np.random.random() - 0.5) * robot.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise

            if self.robot_ready == []:
                set = True
            else:
                for i,agent in enumerate(self.robot_ready):
                    comfort_dist = robot.radius + agent.radius + self.close_call_dist
                    if norm((px - agent.px, py - agent.py)) < comfort_dist:
                        break
                
                    if i == len(self.robot + self.humans) - 1:
                        set = True
        
        robot.set(px,py,-px,-py,0,0,0)

    def set_robot_square(self, robot):
        pos_set = False
        goal_set = False
        sign = 1

        if np.random.random() > 0.5:
            sign = -1
        
        while not pos_set:
            px = np.random.random() * self.square_setting * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_setting
            
            if self.robot_ready == []:
                pos_set = True
    
            else:
                for i,agent in enumerate(self.robot_ready):
                    comfort_dist = robot.radius + agent.radius + self.close_call_dist
                    if norm((px - agent.px, py - agent.py)) < comfort_dist:
                        break
                    
                    if i == len(self.robot + self.humans) - 1:
                        pos_set = True

        while not goal_set:
            gx = np.random.random() * self.square_setting * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_setting

            if self.robot_ready == []:
                pos_set = True
            
            else:
                for i,agent in enumerate(self.robot_ready):
                    comfort_dist = robot.radius + agent.radius + self.close_call_dist
                    if norm((gx - agent.gx, gy - agent.gy)) < comfort_dist:
                        break
                
                    if i == len(self.robot + self.humans) - 1:
                        goal_set = True

        robot.set(px,py,gx,gy,0,0,0)

    def reset(self):
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        
        self.global_time = 0
        self.states = []

        self.set_robot_position()
        self.generate_humans()

        for agent in self.robot + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        # get robots current observations
        all_rob_ob = []
        for robot in self.robot:
            curr = robot.get_observable_state()
            ob = []
            for agent in self.robot + self.humans:
                if agent != robot:
                    agent_state = agent.get_observable_state()
                    dist = norm((agent_state.px - curr.px, agent_state.py - curr.py))
                    if self.observable_dist <= dist:
                        agent_state.dist = dist
                        agent_state.rr = curr.radius + agent_state.radius
                        ob.append(agent_state)
            
            all_rob_ob.append(ob)

        return all_rob_ob
    
    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update = True):
        
        huamn_actions = []
        for human in self.humans:
            human_state = human.get_observable_state
            ob = []
            
            for agent in self.robot + self.humans:
                if agent != human:
                    agent_state = agent.get_observable_state()
                    dist = norm((agent_state.px - human_state.px, agent_state.py - human_state.py))
                    if self.observable_dist <= dist:
                        agent_state.dist = dist
                        agent_state.rr = human_state.radius + agent_state.radius
                        ob.append(agent_state)
            
            huamn_actions.append(ob)
        
        dmin = [float('inf')] * len(self.robot)
        collision = [False] * len(self.robot)

        for i,robot in enumerate(self.robot):
            for j, agent in enumerate(self.robot + self.humans):
                px = agent.px - robot.px
                py = agent.py - robot.py
                if self.robot.kinematics == 'holonomic':
                    vx = agent.vx - action[i].vx
                    vy = agent.vy - action[i].vy
                else:
                    vx = agent.vx - action[i].v * np.cos(action[i].r + robot.theta)
                    vy = agent.vy - action[i].v * np.sin(action[i].r + robot.theta)
                ex = px + vx * self.time_step
                ey = py + vy * self.time_step
                # closest distance between boundaries of two agents
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - agent.radius - robot.radius
                if closest_dist < 0:
                    collision[i] = True
                    # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                    break
                elif closest_dist < dmin[i]:
                    dmin[i] = closest_dist

        
        end_pos = []
        reaching_goal = []
        reward = [0] * len(self.robot)
        done = [False] * len(self.robot)
        info = [object] * len(self.robot)
        
        for i, robot in enumerate(self.robot):
            end_pos.append(np.array(robot.compute_position(action, self.time_step)))
            reaching_goal.append(norm(end_pos[i] - np.array(robot.get_goal_position())) < robot.radius)

            if done[i] != True:
                if self.global_time >= self.time_limit - 1:
                    reward[i] = 0
                    done[i] = True
                    info[i] = Timeout()
                elif collision[i]:
                    reward[i] = self.collision_penalty
                    done[i] = True
                    info[i] = Collision()
                elif reaching_goal:
                    reward[i] = self.success_reward
                    done[i] = True
                    info[i] = ReachGoal()
                elif dmin[i] < self.discomfort_dist:
                    # only penalize agent for getting too close if it's visible
                    # adjust the reward based on FPS
                    reward[i] = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
                    done[i] = False
                    info[i] = Danger(dmin)
                else:
                    reward[i] = 0
                    done[i] = False
                    info[i] = Nothing()

        if update:
            self.states.append([[robot.get_full_state() for robot in self.robot], [human.get_full_state() for human in self.humans]])

            for i, rob_act in enumerate(action):
                self.robot[i].step(rob_act)
            
            for i, huamn_action in enumerate(huamn_actions):
                self.humans[i].step(huamn_action)
            
            robot_ob = []
            for robot in self.robot:
                curr = robot.get_observable_state()
                ob = []
                for agent in self.robot + self.humans:
                    if agent != robot:
                        agent.get_observable_state()
                        dist = norm((agent_state.px - curr.px, agent_state.py - curr.py))
                        if self.observable_dist <= dist:
                            agent_state.dist = dist
                            agent_state.rr = curr.radius + agent_state.radius
                            ob.append(agent_state)
            
                robot_ob.append(ob)
        
        else:
            robot_ob = []
            for robot in self.robot:
                curr = robot.get_observable_state()
                ob = []
                for agent in self.robot + self.humans:
                    if agent != robot:
                        agent.get_observable_state()
                        dist = norm((agent_state.px - curr.px, agent_state.py - curr.py))
                        if self.observable_dist <= dist:
                            agent_state.dist = dist
                            agent_state.rr = curr.radius + agent_state.radius
                            ob.append(agent_state)
            
                robot_ob.append(ob)

        return robot_ob, reward, done, info