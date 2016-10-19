import sys
sys.setrecursionlimit(50000)
import gym
import HumanEnv
from model_modified import *
import theano
import pickle

env = gym.make('HumanEnv-v0')
# env2 = gym.make('BipedalWalker-v2')
epochs = range(10000)
times = range(200)

def get_transaction(observation, env, action, num_frames, action_dim):
    next_state = np.zeros(observation.shape, dtype=theano.config.floatX)
    rewards = np.zeros(1)
    for i in range(num_frames):
        next_state[i], reward, done, _ = env.step(action.reshape(action_dim))

        rewards = rewards+reward
        if done:
            break
    return (observation, action, next_state, rewards, done)

ContinuousDQN=ContinuousDeepQleaner(batch_size=128, input_dim=306, num_frames=2, action_dim=16, discount=0.99, lr_policy=1e-3, lr_Q_func=1e-4, memory_capability=30000, defrozen_number=1000, param_latency=0.1, input_scale=0.5, froze_policy=1000)
count=0
print(env.action_space)

for e in epochs:
    origin = env.reset()
    observation = np.zeros((ContinuousDQN.num_frames, ContinuousDQN.input_dim), dtype=theano.config.floatX)
    observation[-1]= origin
    for t in times:
        #env.render() 
        ###########
        # exploring
        ###########
        action = ContinuousDQN.get_ou_noised_action(observation)
        # if(CinuousDQN.train_flag):
        #     print(action)ont
        # print(action.shape)
        ContinuousDQN.last_action=np.array(action).reshape((1, ContinuousDQN.action_dim))
        
        transaction = get_transaction(observation, env, action, ContinuousDQN.num_frames, ContinuousDQN.action_dim)
        ContinuousDQN.store(transaction)
        observation=transaction[2]
        count+=1
        ###########
        # trainning
        ###########
        if ContinuousDQN.train_flag:
            s, a, ns, r, tt = ContinuousDQN.get_batch()
            if count<ContinuousDQN.froze_policy:
                ContinuousDQN.train(s, a, ns, r, tt, just_Q=True)
            else:
                if count== ContinuousDQN.froze_policy:
                     print('\nstart training policy                  < < <\n')
                ContinuousDQN.train(s, a, ns, r, tt) 
            ContinuousDQN.update_target_networks()

        if transaction[4]:
            break
       
    
    if (e%100) == 0 and e!=0:
        r_sum = 0
        steps = 0
        for _ in range(10):
            origin=env2.reset()
            ob = np.zeros((ContinuousDQN.num_frames, ContinuousDQN.input_dim), theano.config.floatX)
            ob[-1]= origin       
            for _t in range(200):
                action = ContinuousDQN.get_action(ob)
                for i in range(ContinuousDQN.num_frames):
                    ob[i], reward, done, _ = env2.step(action.reshape(action.shape[1]))
                    r_sum += reward
                    steps+=1
                    if done:
                        break
                if done:
                    break
        if (e%10000==0):
            with open('dumped/CDQN-v1-'+str(e)+'epoc.dump', 'wb+') as f:
                pickle.dump(ContinuousDQN, f)

        print('> > > After ', e, ' epochs, \naverage reward ---> ', r_sum/1, '\n average timesteps ---> ', steps/1)
        print('t ---> ',count)

