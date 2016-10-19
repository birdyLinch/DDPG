from lasagne.layers import InputLayer, DenseLayer
import lasagne
from lasagne.updates import sgd, total_norm_constraint
import theano.tensor as T

x = T.matrix()
y = T.ivector()
l_in = InputLayer((5, 10))
l1 = DenseLayer(l_in, num_units=7, nonlinearity=T.nnet.softmax)
output = lasagne.layers.get_output(l1, x)
cost = T.mean(T.nnet.categorical_crossentropy(output, y))
all_params = lasagne.layers.get_all_params(l1)
all_grads = T.grad(cost, all_params)
scaled_grads = total_norm_constraint(all_grads[i], 5)
updates = sgd(scaled_grads, all_params, learning_rate=0.1)