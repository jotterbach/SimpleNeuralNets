import itertools
import numpy as np
import numpy.random as rd
import SimpleNeuralNets.Layers.NetworkLayer as nl
import theano
import theano.tensor as tensor


class FFNN():

    def __init__(self, input_vector, target_vector, n_in, n_hidden, n_out, hid_activations, out_activation):

        if not isinstance(input_vector.__class__, theano.tensor.TensorVariable.__class__):
            raise AssertionError("input_vector needs to be of type 'theano.tensor.TensorVariable'")

        self.input_vector = input_vector
        self.target_vector = target_vector
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.layers = self._wire_layers(hid_activations, input_vector, n_hidden, n_in, n_out, out_activation)
        self.predict = self.layers.get("layer_"+str(len(n_hidden))).output
        self.params = list(itertools.chain(*[layer.params for layer in self.layers.itervalues()]))

    def _wire_layers(self, hid_activations, input_vector, n_hidden, n_in, n_out, out_activation):
        layers = dict()
        layers["layer_0"] = nl.NetworkLayer(input_vector,
                                            n_in,
                                            n_hidden[0],
                                            activation=hid_activations[0],
                                            layer_idx=0)
        for idx in range(1, len(n_hidden)):
            layers["layer_" + str(idx)] = nl.NetworkLayer(layers.get("layer_" + str(idx - 1)).output,
                                                          n_hidden[idx - 1],
                                                          n_hidden[idx],
                                                          activation=hid_activations[idx],
                                                          layer_idx=idx)
        layers["layer_" + str(len(n_hidden))] = nl.NetworkLayer(layers.get("layer_" + str(len(n_hidden) - 1)).output,
                                                                n_hidden[-1],
                                                                n_out,
                                                                activation=out_activation,
                                                                layer_idx=len(n_hidden))
        return layers

    def compute_layer(self, input_values, layer_idx):
        compute = theano.function([self.input_vector], self.layers.get("layer_" + str(layer_idx)).output)
        return compute(input_values)

    def predict(self, input_values):
        return self.compute_layer(input_values, len(self.n_hidden))

    def print_network_graph(self):
        theano.printing.pydotprint(self.predict,
                                   var_with_name_simple=True,
                                   compact=True,
                                   outfile='nn-theano-forward_prop.png',
                                   format='png')

    def loss_function(self, n_samples, regularization_strength):
        if not isinstance(regularization_strength, float): # theano fails silently if it is integer :(
            raise AssertionError('regluarization_strength needs to be float.')

        loss = tensor.mean(tensor.pow(self.predict.reshape((n_samples, )) - self.target_vector, 2))

        reg_loss = tensor.sum(tensor.sqr(self.layers.values()[0].W))
        for layer in self.layers.values()[1:]:
            reg_loss += tensor.sum(tensor.sqr(layer.W))

        regularization = 1/n_samples * regularization_strength/2 * reg_loss

        return loss + regularization

    def compute_param_updates(self, n_samples, regularization_strength, lr):
        gparams = []
        for param in self.params:
            gparam = tensor.grad(self.loss_function(n_samples, regularization_strength), param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * lr))

        return updates

    def get_gradient(self, n_samples, regularization_strength, lr):
        return theano.function([self.input_vector, self.target_vector],
                               self.loss_function(n_samples, regularization_strength),
                               updates=tuple(self.compute_param_updates(n_samples, regularization_strength, lr)))

    def train(self, input_values, target_values, regularization_strength, learning_rate, n_iteartions=10000, print_loss=False):
        gradient_step = self.get_gradient(input_values.shape[0], regularization_strength, learning_rate)
        calculate_loss = theano.function([self.input_vector, self.target_vector], self.loss_function(input_values.shape[0], regularization_strength))

        # reinitialize weights
        for layer in self.layers.values():
            layer.W.set_value(rd.randn(layer.n_in, layer.n_out) / (layer.n_in + layer.n_out))
            layer.b.set_value(np.zeros(layer.n_out) / (layer.n_in + layer.n_out))

        for i in xrange(0, n_iteartions):
            # This will update our parameters W2, b2, W1 and b1!
            gradient_step(input_values, target_values)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" %(i, calculate_loss(input_values, target_values))
