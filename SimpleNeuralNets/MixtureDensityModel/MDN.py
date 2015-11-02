import itertools
import numpy as np
import numpy.random as rd
import SimpleNeuralNets.Layers.NetworkLayer as nl
import SimpleNeuralNets.Layers.MixtureDensityOutputLayer as mdl
import theano
import theano.tensor as tensor


class MDN(object):

    def __init__(self,
                 input_vector,
                 target_vector,
                 n_in,
                 n_hidden,
                 dimension_target_variable,
                 number_of_components,
                 hid_activations):
        if not isinstance(input_vector.__class__, theano.tensor.TensorVariable.__class__):
            raise AssertionError("input_vector needs to be of type 'theano.tensor.TensorVariable'")

        self.input_vector = input_vector
        self.target_vector = target_vector
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.number_of_components = number_of_components
        self.dimension_target_variable = dimension_target_variable
        self.n_out = (dimension_target_variable + 2) * number_of_components
        self.layers = self._wire_layers(hid_activations, input_vector, n_hidden, n_in)
        self.mu = self.layers.get("layer_"+str(len(n_hidden))).mu
        self.sigma = self.layers.get("layer_"+str(len(n_hidden))).sigma
        self.mix = self.layers.get("layer_"+str(len(n_hidden))).mix
        self.params = list(itertools.chain(*[layer.params for layer in self.layers.itervalues()]))

    def _wire_layers(self, hid_activations, input_vector, n_hidden, n_in):
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
        layers["layer_" + str(len(n_hidden))] = \
            mdl.MixtureDensityOutputLayer(layers.get("layer_" + str(len(n_hidden) - 1)).output,
                                                     n_hidden[-1],
                                                     dimension_target_variable=self.dimension_target_variable,
                                                     n_components=self.number_of_components,
                                                     layer_idx=len(n_hidden))
        return layers

    def compute_layer(self, input_values, layer_idx):
        compute = theano.function([self.input_vector], self.layers.get("layer_" + str(layer_idx)).output)
        return compute(input_values)

    def predict_mu(self, input_values):
        compute_mu = theano.function([self.input_vector], self.mu)
        return compute_mu(input_values)

    def predict_sigma(self, input_values):
        compute_sigma = theano.function([self.input_vector], self.sigma)
        return compute_sigma(input_values)

    def predict_mix(self, input_values):
        compute_mix = theano.function([self.input_vector], self.mix)
        return compute_mix(input_values)

    def predict_params(self, input_values):
        return self.predict_mu(input_values), self.predict_sigma(input_values), self.predict_mix(input_values)

    def print_network_graph(self):
        theano.printing.pydotprint(self.predict,
                                   var_with_name_simple=True,
                                   compact=True,
                                   outfile='nn-theano-forward_prop.png',
                                   format='png')

    def logsum_loss(self, n_samples, regularization_strength):
        log_sum_loss = -tensor.sum(tensor.log(
                            tensor.sum(self.mix * tensor.inv(np.sqrt(2 * np.pi) * self.sigma) *
                                       tensor.exp(tensor.neg(tensor.sqr(self.mu - self.target_vector)) *
                                                  tensor.inv(2 * tensor.sqr(self.sigma))), axis=0)
        ))

        reg_loss = tensor.sum(tensor.sqr(self.layers.values()[0].W))
        for layer in self.layers.values()[1:]:
            reg_loss += tensor.sum(tensor.sqr(layer.W))

        regularization = 1/n_samples * regularization_strength/2 * reg_loss

        return log_sum_loss + regularization

    def compute_param_updates(self, n_samples, regularization_strength, lr):
        gparams = []
        for param in self.params:
            gparam = tensor.grad(self.logsum_loss(n_samples, regularization_strength), param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * lr))

        return updates

    def get_gradient(self, n_samples, regularization_strength, lr):
        return theano.function([self.input_vector, self.target_vector],
                               self.logsum_loss(n_samples, regularization_strength),
                               updates=tuple(self.compute_param_updates(n_samples, regularization_strength, lr)))

    def train(self, input_values, target_values, regularization_strength, learning_rate, n_iteartions=10000, print_loss=False):
        gradient_step = self.get_gradient(input_values.shape[0], regularization_strength, learning_rate)
        calculate_loss = theano.function([self.input_vector, self.target_vector], self.logsum_loss(input_values.shape[0], regularization_strength))

        calculate_sigma = theano.function([self.input_vector], self.sigma)
        calculate_mu = theano.function([self.input_vector], self.mu)

        # reinitialize weights
        for layer in self.layers.values():
            layer.W.set_value(rd.randn(layer.n_in, layer.n_out) / (layer.n_in + layer.n_out))
            layer.b.set_value(np.zeros(layer.n_out) / (layer.n_in + layer.n_out))

        for i in xrange(0, n_iteartions):
            # This will update our parameters W2, b2, W1 and b1!
            grad_step = gradient_step(input_values, target_values)

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print "Loss after iteration %i: %f" % (i, calculate_loss(input_values, target_values))
                print "Sigma after iteration " + str(i) + ": ", calculate_sigma(input_values).min(), calculate_sigma(input_values).max()
                print "Mu after iteration " + str(i) + ": ", calculate_mu(input_values).min(), calculate_mu(input_values).max()
                print "gradient after iteration " + str(i) + ": ", learning_rate * grad_step
                print "\n\n"

    def _gaussian(self, x, mu, sigma):
        return np.exp(-np.power(x - mu, 2)/(2 * np.power(sigma, 2))) / np.sqrt(2 * np.pi * np.power(sigma, 2))

    def gaussian_array(self, x, y, mu, sigma, mix):
        n_dim = mu.shape[0]
        lst = []
        for idx in range(len(x)):
            val = 0
            for dim in range(n_dim):
                val += mix[dim, idx] * self._gaussian(y, mu[dim, idx], sigma[dim, idx])
            lst.append(val)
        return np.meshgrid(x, y), np.array(lst).T
