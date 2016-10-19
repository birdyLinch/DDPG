import theano
import theano.tensor as T
import lasagne
import numpy as np

theano.config.floatX='float32'
theano.config.exception_verbosity='high'

class Normalnoise(object):
    def __init__(self, dim, sigma=0.5):
        self.dim = dim
        self.t = 0
        self.sigma= sigma

    def get_noise(self):
        if(self.t<10000):
            std_scale = self.sigma - (self.sigma-0.025)/10000 * self.t
        else:
            std_scale = 0.05
        self.t+=1
        return np.random.normal(0.0, std_scale, self.dim)

class DDPG(object):
    """class ddpg used to construct a ddpg object
    according to Continuous Deep Q Learnig"""

    def __init(
            self,               batch_size, 
            input_dim,          num_frames, 
            action_dim,         lr_Q_func,
            lr_policy,          memory_capability, 
            defrozen_number,    discount=0.99,
            cliff_delta=40,     param_latency=0.01
        ):
        # model struct
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.batch_size = batch_size
        
        # model config
        self.lr_policy = lr_policy
        self.lr_Q_func = lr_Q_func
        self.discount = discount
        self.defrozen_number = defrozen_number
        self.cliff_delta= cliff_delta
        self.noise = Normalnoise(action_dim)
        self.param_latency=param_latency

        # state flag
        self.curr_idx=0
        self.train_flag = False
        self.mem_full=False        

        # memory pool
        self.memory_capability = memory_capability
        self.state_mem= np.zeros((memory_capability, num_frames, input_dim), dtype=theano.config.floatX)
        self.action_mem= np.zeros((memory_capability, action_dim), dtype=theano.config.floatX)
        self.reward_mem= np.zeros((memory_capability, 1), dtype='int32')
        self.next_states_mem=np.zeros((memory_capability, num_frames, input_dim), dtype=theano.config.floatX)
        self.terminal_mem= np.zeros((memory_capability, 1), dtype='int32')

        # simbolic vars
        self.states = T.tensor3('states')
        self.rewards = T.imatrix('rewards')
        self.actions = T.fmatrix('action')
        self.terminals = T.imatrix('terminals')

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
            np.zeros((batch_size, action_dim), dtype=theano.config.floatX)
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
        self.single_state_shared=theano.shared(
            np.zeros((1, num_frames, input_dim), dtype=theano.config.floatX)
        )

        # build networks
        self.target_policy, tar_poli_in = self.build_policy()
        self.train_policy, train_poli_in = self.build_policy()
        self.target_Q_func, tar_Q_in1, tar_Q_in2 =  self.build_Q_function()
        self.train_Q_func, train_Q_in1, train_Q_in2 = self.build_Q_function()

        # interior symbolic var for loss function of Q
        next_target_action_out = lasagne.layers.get_output(self.target_policy, {tar_poli_in: self.states})
        next_Q_val_out = lasagne.layers.get_output(self.target_Q_func, {tar_Q_in1: self.states, tar_Q_in2: next_target_action_out})
        Q_val_out = lasagne.layers.get_output(self.train_Q_func, {train_Q_in1: self.states, train_Q_in2: self.actions})





