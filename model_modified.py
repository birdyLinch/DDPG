import theano
import theano.tensor as T
import lasagne
import numpy as np

theano.config.floatX='float32'
theano.config.exception_verbosity='high'

class ContinuousDeepQleaner(object):
    """docstring for DeepQFunction"""
    def __init__(self, batch_size, 
            input_dim, num_frames, 
            action_dim, discount, 
            lr_policy, lr_Q_func, 
            memory_capability, defrozen_number,
            froze_policy, 
            cliff_delta=40, param_latency=0.01, 
            input_scale=0.05, max_explore_p=1.0,
            min_explore_p=0.1,
            decay_proid=150000):
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount = discount
        self.lr_policy = lr_policy
        self.lr_Q_func = lr_Q_func
        self.state_mem= np.zeros((memory_capability, num_frames, input_dim), dtype=theano.config.floatX)
        self.action_mem= np.zeros((memory_capability, action_dim), dtype=theano.config.floatX)
        self.reward_mem= np.zeros((memory_capability, 1), dtype='int32')
        self.next_states_mem=np.zeros((memory_capability, num_frames, input_dim), dtype=theano.config.floatX)
        self.terminal_mem= np.zeros((memory_capability, 1), dtype='int32')
        self.curr_idx=0
        self.train_flag = False
        self.mem_full=False
        self.defrozen_number = defrozen_number
        self.cliff_delta= cliff_delta
        #self.noise = Normalnoise(action_dim)
        self.memory_capability = memory_capability
        self.param_latency=param_latency
        self.froze_policy = froze_policy
        
        # simbolic vars
        self.states = T.tensor3('states')
        self.rewards = T.imatrix('rewards')
        self.actions = T.fmatrix('action')
        self.terminals = T.imatrix('terminals')

        #param for ou process
        self.last_action = np.zeros(())
        self.min_explore_p = min_explore_p
        self.decay_proid = decay_proid
        self.max_explore_p = max_explore_p
        self.t=0

        # shared vars
        self.states_shared=theano.shared(
            np.zeros((batch_size ,num_frames, input_dim),
            dtype=theano.config.floatX)
        )
        self.next_state_shared=theano.shared(
            np.zeros((batch_size, num_frames, input_dim),
            dtype=theano.config.floatX)
        )
        self.rewards_shared=theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            #broadcastable=True
        )
        self.action_shared=theano.shared(
            np.zeros((batch_size, action_dim), dtype=theano.config.floatX),
            #broadcastable=True
        )
        self.next_action_shared=theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX)
            #broadcastable=True
        )
        self.terminals_shared=theano.shared(
            np.zeros((batch_size, 1), dtype='int32')
            #broadcastable=True
        )
        self.sigle_state_shared=theano.shared(
            np.zeros((1, num_frames, input_dim), dtype=theano.config.floatX)
        )
        # self.single_batch_state_shared=theano.shared(
        #     np.zeros((1, num_frames, input_dim), dtype=theano.config.floatX)
        # )
        
        # build networks
        self.target_policy, tar_poli_in = self.build_policy()
        self.train_policy, train_poli_in = self.build_policy()
        self.target_Q_func, tar_Q_in1, tar_Q_in2 =  self.build_Q_function()
        self.train_Q_func, train_Q_in1, train_Q_in2 = self.build_Q_function()

        # interior symbolic var for loss function of Q
        next_target_action_out = lasagne.layers.get_output(self.target_policy, {tar_poli_in: self.states*input_scale})
        next_Q_val_out = lasagne.layers.get_output(self.target_Q_func, {tar_Q_in1: self.states*input_scale, tar_Q_in2: next_target_action_out})
        Q_val_out = lasagne.layers.get_output(self.train_Q_func, {train_Q_in1: self.states*input_scale, train_Q_in2: self.actions})
        
        # function for training Params of Q
        terminalsX = self.terminals.astype(theano.config.floatX)
        y = (self.rewards +
                  (T.ones_like(terminalsX) - terminalsX) *
                  self.discount * next_Q_val_out)

        diff = y - Q_val_out
        
        loss = T.mean(T.sqr(diff)) #+ 0.01*lasagne.regularization.regularize_network_params(self.train_Q_func, lasagne.regularization.l2)


        train_Q_params = lasagne.layers.get_all_params(self.train_Q_func, trainable=True)
        train_Q_givens={
            self.states: self.states_shared,
            self.rewards: self.rewards_shared,
            self.actions: self.action_shared,
            self.terminals: self.terminals_shared,
        }
        grad = T.grad(loss, train_Q_params)

        clipped_grad = [lasagne.updates.norm_constraint(grad[i], 100, norm_axes=tuple(range(grad[i].ndim))) for i in range(len(train_Q_params))]
        Q_updates = lasagne.updates.adam(clipped_grad, train_Q_params, self.lr_Q_func)
        
        self._train_Q = theano.function([], [loss], 
                        updates=Q_updates, 
                        #mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs),  #debug

                        givens=train_Q_givens, allow_input_downcast=True)

        # interior symbolic var for train policy
        train_action_out = lasagne.layers.get_output(self.train_policy, {train_poli_in: self.states*input_scale})
        train_Q_val_out = lasagne.layers.get_output(self.train_Q_func, {train_Q_in1: self.states*input_scale,train_Q_in2: train_action_out})

        # function for train policy
        Loss = - T.mean(train_Q_val_out) #+ 0.01*lasagne.regularization.regularize_network_params(self.train_policy, lasagne.regularization.l2)
        train_policy_params = lasagne.layers.get_all_params(self.train_policy, trainable=True)
        train_policy_givens={
            self.states: self.states_shared
        }
        policy_updates = lasagne.updates.adam(Loss, train_policy_params, self.lr_policy)

        self._train_policy = theano.function([], [Loss], updates=policy_updates, givens=train_policy_givens, allow_input_downcast=True 
                        #mode=theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs) #debug
                        )

        # forward
        action_target_policy_out = lasagne.layers.get_output(self.target_policy, {tar_poli_in: self.states*input_scale})

        self.choose_action = theano.function([], action_target_policy_out, givens={self.states: self.sigle_state_shared}, 
                        allow_input_downcast=True
                        #mode=theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs) #debug
                        )

        ###for debug
        #self.see_yi_and_Q_val = theano.function([], [Q_val_out, y], givens=train_Q_givens)

        #åct_train_policy_out = lasagne.layers.get_output(self.train_policy, {train_poli_in: self.states})
        
        #self.choose_action = theano.function([self.states], act_train_policy_out)

    def build_policy(self):
        # from lasagne.layers import cuda_convnet
        l_in = lasagne.layers.InputLayer(
            shape=(None, self.num_frames, self.input_dim)
        )
        l_0 = lasagne.layers.DenseLayer(l_in, 128, W=lasagne.init.Orthogonal(gain='relu'))
        l_1 = lasagne.layers.DenseLayer(l_0, 64, W=lasagne.init.Orthogonal(gain='relu'), nonlinearity=None)
        bn_ = lasagne.layers.BatchNormLayer(l_1, axes=(0,1), gamma=None, beta=None)
        bn_1 = lasagne.layers.NonlinearityLayer(bn_, nonlinearity=lasagne.nonlinearities.rectify)
        l_o = lasagne.layers.DenseLayer(bn_1, self.action_dim, W=lasagne.init.Orthogonal(gain='relu'), nonlinearity=None)   
        bn_2 = lasagne.layers.BatchNormLayer(l_o, axes=(0,1), gamma=None, beta=None)
        l_out = lasagne.layers.NonlinearityLayer(bn_2, nonlinearity=lasagne.nonlinearities.tanh)
        return l_out, l_in

    def build_Q_function(self):
        l_in1 = lasagne.layers.InputLayer(
            shape=(None, self.num_frames, self.input_dim)
        )
        l_dense1 = lasagne.layers.DenseLayer(
            l_in1, 128, W=lasagne.init.Orthogonal(gain='relu')
        )
        #l_bn1 = lasagne.layers.BatchNormLayer(l_dense1)
        l_dense2 = lasagne.layers.DenseLayer(
            l_dense1, 32, W=lasagne.init.Orthogonal(gain='relu')
        )
        l_in2 = lasagne.layers.InputLayer(
            shape=(None, self.action_dim),
        )
        l_decode = lasagne.layers.DenseLayer(
            l_in2, 32, W=lasagne.init.Orthogonal(gain='relu')
        )

        l_merge = lasagne.layers.ConcatLayer([l_dense2, l_decode], axis=1)
        l_dense3 = lasagne.layers.DenseLayer(
            l_merge, 48, W=lasagne.init.Orthogonal(gain='relu')
        )
        #l_bn2 = lasagne.layers.BatchNormLayer(l_dense3)
        l_out = lasagne.layers.DenseLayer(
            l_dense3, 1, W=lasagne.init.Orthogonal(gain='relu')
        )

        return l_out, l_in1, l_in2

    def store(self, transaction):
        #
        #trainsaction: tuple of (state, action, next_state, reward)
        #
        self.state_mem[self.curr_idx] = transaction[0]
        self.action_mem[self.curr_idx] = transaction[1]
        self.next_states_mem[self.curr_idx] = transaction[2]
        self.reward_mem[self.curr_idx]=transaction[3]
        self.terminal_mem[self.curr_idx]=transaction[4]
        # renew curr_idx
        self.curr_idx = (self.curr_idx+1)%self.memory_capability
        if self.curr_idx== self.defrozen_number:
            self.train_flag = True
        if self.curr_idx==(self.memory_capability-1):
            self.mem_full = True

    def get_batch(self):
        if self.mem_full:
            up_bound= self.memory_capability
        else:
            up_bound= self.curr_idx
        mask = np.random.choice(np.array(range(up_bound)), self.batch_size)
        batch_state = self.state_mem[mask]
        batch_action = self.action_mem[mask]
        batch_next_state = self.next_states_mem[mask]
        batch_reward = self.reward_mem[mask]
        batch_terminal = self.terminal_mem[mask]
        #print(self.mem_full)
        #print(batch_action)

        return batch_state, batch_action, batch_next_state, batch_reward, batch_terminal

    def train(self, batch_state, batch_action, batch_next_state, batch_reward, batch_terminal, verbose=False, just_Q=False):
        if self.train_flag:
            self.states_shared.set_value(batch_state)
            self.action_shared.set_value(batch_action)
            self.rewards_shared.set_value(batch_reward)
            self.terminals_shared.set_value(batch_terminal)
            self.next_state_shared.set_value(batch_next_state)

            Q_difference = self._train_Q()
            if verbose == True:
                print('\n\nQ_difference: ', Q_difference)
            if just_Q:
                return
            averge_Q_val = self._train_policy()
            if verbose == True:   
                print('policy average Q value: ', averge_Q_val)
            
            

    # def get_noised_action(self, sigle_state):
    #     p = self.noise.acc()
    #     if np.random.rand()<p:
    #         return np.random.rand(1,4)
    #     else:
    #         self.sigle_state_shared.set_value(sigle_state.reshape(1, self.num_frames, self.input_dim))       
    #         act = self.choose_action()
    #         noise = self.noise.get_noise()
    #         act = np.clip(act+noise, -1.0, 1.0)
    #         return act

    def get_action(self, sigle_state):
            self.sigle_state_shared.set_value(sigle_state.reshape(1, self.num_frames, self.input_dim))       
            act = self.choose_action()
            act = np.clip(act, -1.0, 1.0)
            return act

    def get_ou_noised_action(self, sigle_state):
        if self.t < self.decay_proid:
            p = self.max_explore_p - (self.max_explore_p-self.min_explore_p)/self.decay_proid * self.t
        else:
            p = self.min_explore_p
        if(np.random.rand()<p):
            return np.clip(OU_process(self.last_action), -1.0, 1.0)
        else:
            return get_action(sigle_state)

    def update_target_networks(self):
        target_poli_params = lasagne.layers.get_all_param_values(self.target_policy)
        train_poli_params = lasagne.layers.get_all_param_values(self.train_policy)
        param_to_set = []

        for i in range(len(target_poli_params)):
            param_to_set.append((self.param_latency*train_poli_params[i] 
                    +(1-self.param_latency)*target_poli_params[i]).astype(np.float32))
        lasagne.layers.set_all_param_values(
            self.target_policy, param_to_set
        )


        target_Q_params = lasagne.layers.get_all_param_values(self.target_Q_func)
        train_Q_params = lasagne.layers.get_all_param_values(self.train_Q_func)
        lasagne.layers.set_all_param_values(
            self.target_Q_func, 
            [
                (self.param_latency*train_Q_params[i] 
                    + (1-self.param_latency)*target_Q_params[i]).astype(np.float32) 
                for i in range(len(target_Q_params))
            ]
        )

# class Normalnoise(object):
#     def __init__(self, dim, sigma=0.5, random_posibility=1):
#         self.dim = dim
#         self.t = 0
#         self.sigma= sigma
#         self.p = random_posibility
#         self.last_action = np.zeros(())

#     def get_noise(self):
#         if(self.t<1000000):
#             std_scale = self.sigma - (self.sigma-0.025)/1000000 * self.t
#         else:
#             std_scale = 0.07
#         return np.random.normal(0.0, std_scale, self.dim)

#     def acc(self):
#         self.t+=1
#         if(self.t<100000):
#             return self.p-(self.p-0.1)/100000*self.t
#             if(self.t==99999):
#                 print('\n>>>>>>>>>>>>>>>>>       random pocibility down to 0.1       <<<<<<<<<<<<<<<<\n')
#         else:
#             return 0.1

def OU_process(xi, dtype=('float32'), dt=1, kappa=0.15, sigma=0.3, mu=np.zeros((1,16), dtype='float32')):
    next_xi = xi + kappa * (mu - xi) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(1,16))
    return next_xi.astype(theano.config.floatX)



