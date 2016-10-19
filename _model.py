import theano
import theano.tensor as T
import lasagne
import numpy as np

class ContinuousDeepQleaner(object):
    """docstring for DeepQFunction"""
    def __init__(self, batch_size, input_dim, num_frames, action_dim, discount, lr_policy, lr_Q_val_f, memory_capability, defrozen_number, cliff_delta=0):
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.discount = discount
        self.lr = learning_rate
        self.policy_out, _ = self.build_policy()
        self.Q_val_f_out, l_in = self.build_Q_function()
        self.state_mem= np.zeros((memory_capability, num_frames, input_dim), dtype=theano.config.floatX)
        self.action_mem= np.zeros((memory_capability, action_dim), dtype=theano.config.floatX)
        self.reward_mem= np.zeros((memory_capability, 1), dtype='int32')
        self.next_states_mem=np.zeros((memory_capability, num_frames, input_dim), dtype=theano.config.floatX)
        self.curr_idx=0
        self.train_flag = False
        self.mem_full=False
        self.defrozen_number = defrozen_number
        self.cliff_delta= cliff_delta
        self.target_q_val_f = build_policy()
        self.target_policy =  build_Q_function()

        states = T.tensor3('states')
        next_states = T.tensor3('states')
        rewards = T.col('rewards')
        action = T.fmatrix('action')
        next_action = T.fmatrix('next_action')
        terminals = T.icol('terminals')

        lasagne.random.set_rng(self.rng)

        self.input_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_dim), 
            dtype=theano.config.floatX)
        )
        self.rewards_shared=theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True)
        )
        self.action_shared=theano.shared(
            np.zeros((batch_size, action_dim), dtype=theano.config.floatX),
            broadcastable=(False, True)
        )
        self.terminals_shared=theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True)
        )
        self.states_shared=theano.shared(
            np.zeros((batch_size ,num_frames, input_dim),
            dtype=theano.config.floatX)
        )
        self.next_state_shared=theano.shared(
            np.zeros((batch_size, num_frames, input_dim),
            dtype=theano.config.floatX)
        )
        self.next_action_shared=theano.shared(
            np.zeros(batch_size, 1), 
            theano.config.floatX
        )

        policy_action = lasagne.layers.get_output(self.policy_out, states)

        target_policy_action = lasagne.layers.get_output(self.target_policy, states)
        
        q_vals = lasagne.layers.get_output(self.Q_val_f_out, 
            {
                l_in[0]: states, 
                l_in[2]: action
            })
        
        target_q_val = lasagne.layers.get_output( 
            self.target_q_val_f,
            { 
                l_in[0]: next_states,
                l_in[2]: next_action
            })

        
        terminalsX=terminals.astype(theano.config.floaX)
        yi = (rewards +
                  (T.ones_like(terminalsX) - terminalsX) *
                  self.discount * next_q_vals)

        diff = q_vals - yi
        if self.cliff_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.cliff_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.cliff_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        loss = T.mean(loss)

        train_Q_params = lasagne.layer.get_all_params(self.Q_val_f_out)
        train_Q_givens={
            states: self.states_shared,
            rewards: self.rewards_shared,
            action: self.action_shared,
            terminals: self.terminals_shared,
        }
        Q_updates = lasagne.updates.adam(loss, train_Q_params, self.lr_Q_val_f)
        self._train_Q = thenao.function([], [loss], updates, givens=train_Q_givens)
        
        train_policy_params = lasagne.layers.get_all_params(self.policy_out)
        d_train_policy_params = theano.gradient.grad()
        policy_updates = lasagne.updates.adam()
        self._q_vals = theano.function([], q_vals)


    
    def build_policy(self):
        # from lasagne.layers import cuda_convnet

        l_in = lasagne.layers.InputLayer(
            shape=(self.batch_size, self.num_frames, self.input_dim)
        )

        l_gru = lasagne.layer.GRULayer(l_in, 16)
        l_shp = lasagne.layer.ReshapeLayer(l_gru, (self.batch_size, -1))
        l_out = lasagne.DenseLayer(l_shp, self.action_dim)

        return l_out, l_in

    def build_Q_function(self):
        l_in1 = lasagne.layer.InputLayer(
            shape=(self.batch_size, self.num_frames, self.input_dim)
        )
        l_in2 = lasagne.layer.InputLayer(
            shape=(self.batch_size, self.num_frames, self.input_dim)
        )
        l_merge = lasagne.layer.MergeLayer(
            (l_in1, l_in2)
        )
        l_dense = lasagne.layer.DenseLayer(
            l_merge, 64
        )
        l_out = lasagne.layer.DenseLayer(
            l_dense, 1
        )

        return l_out,(l_in1, l_in2)

    def store(self, transaction):
        #
        #trainsaction: tuple of (state, action, next_state, reward)
        #
        self.state_mem[self.curr_idx] = transaction[0]
        self.action[self.curr_idx] = transaction[1]
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
            up_bound= self.curr_idx
        else:
            up_bound= self.memory_capability
        mask = np.random.choice(np.array(range(up_bound)), self.batch_size)
        batch_state = self.state_mem[mask]
        batch_action = self.action_mem[mask]
        batch_next_state = self.next_state[mask]
        batch_reward = self.reward_mem[mask]
        batch_terminal = self.terminal_mem[mask]

        return batch_state, batch_action, batch_next_state, batch_reward, batch_terminal

    def train(batch_state, batch_action, batch_next_state, batch_reward, batch_terminal):
        self.states_shared.set_value(batch_state)
        self.action_shared.set_value(batch_action)
        self.rewards_shared.set_value(batch_reward)
        self.terminals_shared.set_value(batch_terminal)
        self.next_state_shared.set_value(batch_next_state)

    def update_target_networks():
        

class Normalnoise():
    def __init__(self, dim):
        self.dim = dim
        self.t = 0

    def get_noise(self):
        if(self.t<10000000):
            std_scale = (self.sigma-0.025)/10000000 * self.t
        else:
            std_scale = 0.025
        self.t+=1
        return np.random.normal(0.0, std_scale, self.dim)








        