import numpy as np
import random
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, vis_range, sort_obs=False):
        world = World()
        self.np_rnd = np.random.RandomState(0)
        self.sort_obs = sort_obs
        self.vis_range = vis_range
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.world_radius = 1
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.np_rnd.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.np_rnd.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):

        return self._full_observation(agent, world)




    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew


    def _full_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            # cannot see the landmark if the distance between the landmark and the agent is larger than 0.4
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)



    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            # cannot see the landmark if the distance between the landmark and the agent is larger than 0.4
            if np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))) < self.vis_range:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(agent.state.p_pos - agent.state.p_pos)  # zero

        other_pos = []
        for other in world.agents:
            if other is agent: continue
            if np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos))) < self.vis_range:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
            else:
                other_pos.append(agent.state.p_pos - agent.state.p_pos)  # zero
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
