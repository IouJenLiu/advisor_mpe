from make_env import make_env
import numpy as np


#np.random.seed(12)
#print('123')
print('111111111111111')
env1 = make_env(scenario_name='simple_spread_v4')
env1.seed(12)
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)


print('22222222222222')
env2 = make_env(scenario_name='simple_spread_v4')
env2.seed(12)
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)


print('111111111111111')
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)

print('reset env 1 seed')
env1.seed(12)
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)
env1.reset()
print(env1.agents[0].state.p_pos, env1.agents[1].state.p_pos, env1.agents[2].state.p_pos)

print('22222222222222')
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)
env2.reset()
print(env2.agents[0].state.p_pos, env2.agents[1].state.p_pos, env2.agents[2].state.p_pos)
print('------------')