import logging
import gym
import matplotlib.lines as mlines
import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.robot_num = None
        self.seeable_distance = None
        self.visible_robot = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.seeable_distance = config.getint('sim','seeable_distance')
        self.visible_robot = config.getboolean('sim','visible_robot')
        if self.config.get('humans', 'policy') == 'orca':
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        """
        Robot initialized as a list instead of class used by previous env
        Allows for multiple agent to be initilized at once
        """

        self.robots = robot
        self.robots_num = len(robot)

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        self.humans = []
        if human_num != 0:
            if rule == 'square_crossing':
                for i in range(human_num):
                    self.humans.append(self.generate_square_crossing_human())
            elif rule == 'circle_crossing':
                for i in range(human_num):
                    self.humans.append(self.generate_circle_crossing_human())

    
    def generate_circle_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in self.robots + self.humans:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, -px, -py, 0, 0, 0)
        return human

    def generate_square_crossing_human(self):
        human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = np.random.random() * self.square_width * 0.5 * sign
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in self.robots + self.humans:
                if norm((px - agent.px, py - agent.py)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = np.random.random() * self.square_width * 0.5 * -sign
            gy = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in self.robots + self.humans:
                if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        human.set(px, py, gx, gy, 0, 0, 0)
        return human
    
    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robots.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robots.get_position(), *params, self.robots.radius, self.robots.v_pref,
                     self.robots.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate(self.robots + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            for j, robot in enumerate(self.robots):
                robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.states.append(self.robots.get_full_state(), [human.get_full_state() for human in self.humans])

        del sim
        return self.human_times
    
    def generate_random_robot_position(self, rule):
        if rule == 'square_crossing':
            self.generate_square_crossing_robot()
        elif rule == 'circle_crossing':
            self.generate_circle_crossing_robot()
    
    def generate_circle_crossing_robot(self):
        set_robots = []

        for robot in self.robots:
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * robot.v_pref
                py_noise = (np.random.random() - 0.5) * robot.v_pref
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
                collide = False
                if set_robots == []:
                    pass                    
                else:
                    for agent in set_robots:
                        min_dist = robot.radius + agent.radius + self.discomfort_dist
                        if norm((px - agent.px, py - agent.py)) < min_dist or \
                                norm((px - agent.gx, py - agent.gy)) < min_dist:
                            collide = True
                            break
                if not collide:
                    break
            robot.set(px,py,-px,-py,0,0,0)
            set_robots.append(robot)

    def generate_square_crossing_robot(self):
        set_robots = []
        if np.random.random() > 0.5:
            sign = -1
        else:
            sign = 1

        for robot in self.robots:
            while True:
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                if set_robots != []:
                    for agent in set_robots:
                        min_dist = robot.radius + agent.radius + self.discomfort_dist
                        if norm((px - agent.px, py - agent.py)) < min_dist:
                            collide = True
                            break

                if not collide:
                    break

            while True:
                gx = np.random.random() * self.square_width * 0.5 * -sign
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                if set_robots != []:
                    for agent in set_robots:
                        min_dist = robot.radius + agent.radius + self.discomfort_dist
                        if norm((gx - agent.gx, gy - agent.gy)) < human.radius + agent.radius + self.discomfort_dist:
                            collide = True
                            break
                if not collide:
                    break

            robot.set(px,py,gx,gy,0,0,0)
            set_robots.append()
        
    def reset(self, mode = "test"):
        '''
        Resets the entire environment for testing or training
        Training would use randomize environment
        Testing would use general environment for consistency

        Set px, py, gx, gy, vx, vy and theta for robot and humans
        '''
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        
        self.global_time = 0  # Initialize global time
        self.human_times = [0] * self.human_num  # Initialize the timer of each human

        if mode == "test":
            self.generate_random_robot_position(self.test_sim)
            self.generate_random_human_position(self.human_num, self.test_sim)
        else:
            self.generate_random_robot_position(self.train_val_sim)
            self.generate_random_human_position(self.human_num, self.train_val_sim)
        
        for agent in self.robots + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        # Get current observation
        all_robot_ob = []
        for robot in self.robots:
            robot_ob = []
            robot_state = robot.get_observable_state()
            for agent in self.robots + self.humans:
                if agent != robot:
                    agent_state = agent.get_observable_state()
                    dist = ((agent_state.px - robot_state.px)**2 + (agent_state.py - robot_state.py)**2) ** (1/2)
                    if self.seeable_distance >= dist:
                        robot_ob.append(agent_state)

            all_robot_ob.append(robot_ob)

        return all_robot_ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)
    
    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        robot_info = []
        human_actions = []
        for human in self.humans:
            human_state = human.get_observable_state()
            agents = self.humans
            if self.visible_robot:
                agents += self.robots
            
            ob = []
            for agent in agents:
                if agent != human:
                    agent_state = agent.get_observable_state()
                    dist = ((agent_state.px - human_state.px)**2 + (agent_state.py - human_state.py)**2) ** (1/2)
                    if self.seeable_distance >= dist:
                        ob.append(agent_state)
            
            human_actions.append(ob)

        dmin = []
        collision = []
        for i,robot in enumerate(self.robots):
            rob_dmin = float('inf')
            rob_col = False

            for agent in self.humans + self.robots:
                if agent != robot:
                    px = agent.px - robot.px
                    py = agent.py - robot.py
                    if robot.kinematics == 'holonomic':
                        vx = agent.vx - action[i].vx
                        vy = agent.vy - action[i].vy
                    else:
                        vx = agent.vx - action[i].v * np.cos(action[i].r + robot.theta)
                        vy = agent.vy - action[i].v * np.sin(action[i].r + robot.theta)
                    
                    ex = px + vx * self.time_step
                    ey = py + vy * self.time_step

                    closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - agent.radius - robot.radius
                    if closest_dist < 0:
                        rob_col = True
                        break
                    elif closest_dist < rob_dmin:
                        rob_dmin = closest_dist
            
            collision.append(rob_col)
            dmin.append(rob_dmin)
        
        human_num = len(self.humans)
        for j in range(human_num):
            for k in range(j+1, human_num):
                dx = self.humans[j].px - self.humans[k].px
                dy = self.humans[j].py - self.humans[k].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[j].radius - self.humans[k].radius
                if dist<0:
                    logging.debug('Collision happens between humans in step()')
        
        for i, robot in enumerate(self.robots):
            end_pos = np.array(robot.compute_position(action[i], self.time_step))
            reaching_goal = norm(end_pos - np.array(robot.get_goal_position())) < robot.radius

            if self.global_time >= self.time_limit - 1:
                reward = 0
                done = True
                info = Timeout()

            elif collision[i]:
                reward = self.collision_penalty
                done = True
                info = Collision()
            
            elif reaching_goal:
                reward = self.success_reward
                done = True
                info = ReachGoal()
            
            elif dmin < self.discomfort_dist:
                reward = (dmin[i] - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
                done = False
                info = Danger(dmin[i])
            
            else:
                reward = 0
                done = False
                info = Nothing()
            
            robot_info.append([ob,reward,done,info])
        
        if update:
            self.states.append(1) #line 391 of crowd_sim.py