import theano
import gym
import pickle
import numpy as np
e=47

env2 = gym.make('BipedalWalker-v2')
with open('dumped/CDQN-v1-'+str(e*1000)+'epoc.dump','rb') as f:
    ContinuousDQN = pickle.load(f)
env2.monitor.start("record/", force=True, seed=0)
for _ in range(1):
    origin=env2.reset()
    ob = np.zeros((ContinuousDQN.num_frames, ContinuousDQN.input_dim), theano.config.floatX)
    ob[-1]= origin
    r_sum = 0
    steps = 0
    for _t in range(1000):

        action = ContinuousDQN.get_action(ob)
        print(action)
        for i in range(ContinuousDQN.num_frames):
            ob[i], reward, done, _ = env2.step(action.reshape(action.shape[1]))
            r_sum += reward
            if done:
                break
        if done:
            steps += (_t+1)
            break
env2.monitor.close()