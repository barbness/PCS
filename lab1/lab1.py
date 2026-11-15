import numpy as np
class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process input data and produce ouput data.
    This is like applying a function which is called forward:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the forward function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """
    def __init__(self):
        self.output = None
        self.grad_input = None
        self.training = True

    def forward(self, data):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.update_output(data)

    def backward(self, data, grad_output):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. input (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.update_grad_input(data, grad_output)
        self.acc_grad_parameters(data, grad_output)
        return self.grad_input


    def update_output(self, data):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the output field.

        Make sure to both store the data in output field and return it.
        """

        # The easiest case:

        self.output = data
        return self.output

        pass

    def update_grad_input(self, data, grad_output):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in gradInput. Also, the gradInput state variable is updated accordingly.

        The shape of gradInput is always the same as the shape of input.

        Make sure to both store the gradients in gradInput field and return it.
        """

        # The easiest case:

        self.grad_input = grad_output
        return self.grad_input

        pass

    def acc_grad_parameters(self, data, grad_output):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zero_grad_parameters(self):
        """
        Zeroes gradParams variable if the module has params.
        """
        pass

    def get_parameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def get_grad_parameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def repr(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"

#Поверх него вы легко можете написать аналог ещё одного класса:

class Sequential(Module):
    """
         This class implements a container, which processes input data sequentially.

         input is processed by each module (layer) in self.modules consecutively.
         The resulting array is called output.
    """
    def __init__(self):
        super(Sequential, self).__init__()
        self.mods = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.mods.append(module)

    def update_output(self, data):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})
        """
        y = data
        for m in self.mods:
          y = m.forward(y)
        self.output = y
        return self.output

    def backward(self, data, grad_output):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)

        To each module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for i-th layer the output of module[i] (just the same input as in forward pass)
        and NOT input to this Sequential module.

        """
        t = [] #len(t)  eq len(self.modules)+1
        y = input
        t.append(y)
        for m in self.mods:
          y = m.forward(y)
          t.append(y)

        y = grad_output
        for i in range(len(self.mods)):
          k = len(self.mods)-i-1
          y = self.mods[k].backward(t[k],y)
        self.grad_input = y
        return self.grad_input


    def zero_grad_parameters(self):
        for module in self.mods:
            module.zero_grad_parameters()

    def get_parameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.get_parameters() for x in self.mods]

    def get_grad_parameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.get_grad_parameters() for x in self.mods]

    def repr(self):
        string = "".join([str(x) + '\n' for x in self.mods])
        return string

    def getitem(self,x):
        return self.mods.getitem(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.mods:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.mods:
            module.evaluate()

# !! Ваша задача дописать код базовых слоёв НС: (там где стоит #...) !!

class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization BUT with a problem was detected by Xavier
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def update_output(self, data):
        # y = W^T*x + b
        self.output = np.dot(data, self.W.T) + self.b

    def update_grad_input(self, data, grad_output):
        #dL/dX = dL/dY * W
        self.grad_input = np.dot(grad_output, self.W)
        return self.grad_input

    def acc_grad_parameters(self, data, grad_output):
        #dL/dW = (dL/dY)^T * X
        self.gradW += np.dot(grad_output.T, data)
        #dL/db = сумма dL/dY по всем объектам батча
        self.gradb += np.sum(grad_output, axis=0)

    def zero_grad_parameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def get_parameters(self):
        return [self.W, self.b]

    def get_grad_parameters(self):
        return [self.gradW, self.gradb]

    def repr(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q

class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()

    def update_output(self, data):
        # start with normalization for numerical stability
        self.output = np.subtract(data, data.max(axis=1, keepdims=True))
        #Экспонента и деление на сумму экспонент по строке
        exp_input = np.exp(self.output)
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output

    def update_grad_input(self, data, grad_output):
        #Локальный градиент: dL/dx_i = p_i * (dL/dy_i - \sum_j dL/dy_j * p_j)
        #где p_i = self.output
        sum_grad_out_p = np.sum(grad_output * self.output, axis=1, keepdims=True)
        self.grad_input = self.output * (grad_output - sum_grad_out_p)
        return self.grad_input

    def repr(self):
        return "SoftMax"

"""## 3. LogSoftMax
- input:   **batch_size x n_feats**
- output: **batch_size x n_feats**

$\text{logsoftmax}(x)_i = \log\text{softmax}(x)_i = x_i - \log {\sum_j \exp x_j}$

The main goal of this layer is to be used in computation of log-likelihood loss.
"""

class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()

    def update_output(self, data):
        # start with normalization for numerical stability
        shifted_input = np.subtract(data, data.max(axis=1, keepdims=True))
        #log_sum_exp = log(sum(exp(x)))
        log_sum_exp = np.log(np.sum(np.exp(shifted_input), axis=1, keepdims=True))
        self.output = shifted_input - log_sum_exp
        return self.output

    def update_grad_input(self, data, grad_output):
        #dL/dx = dL/dy - sum(dL/dy, axis=1) * softmax(x)
        softmax = np.exp(self.output)
        self.grad_input = grad_output - np.sum(grad_output, axis=1, keepdims=True) * softmax
        return self.grad_input

    def repr(self):
        return "LogSoftMax"

"""## 4. Batch normalization
One of the most significant recent ideas that impacted NNs a lot is [Batch normalization](http://arxiv.org/abs/1502.03167). The idea is simple, yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the way through NN. This improves the convergence for deep models letting it train them for days but not weeks. You are to implement the first part of the layer: features normalization. The second part (ChannelwiseScaling layer) is implemented below.

- input:   **batch_size x n_feats**
- output: **batch_size x n_feats**

The layer should work as follows. While training (self.training == True) it transforms input as $$y = \frac{x - \mu}  {\sqrt{\sigma + \epsilon}}$$
where $\mu$ and $\sigma$ - mean and variance of feature values in batch and $\epsilon$ is just a small number for numericall stability. Also during training, layer should maintain exponential moving average values for mean and variance:
    self.moving_mean = self.moving_mean * alpha + batch_mean * (1 - alpha)
    self.moving_variance = self.moving_variance * alpha + batch_variance * (1 - alpha)
During testing (self.training == False) the layer normalizes input using moving_mean and moving_variance.

Note that decomposition of batch normalization on normalization itself and channelwise scaling here is just a common implementation choice. In general "batch normalization" always assumes normalization + scaling.
"""
# 4. BatchNormalization
class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def update_output(self, data):
        if self.training:
            batch_mean = np.mean(data, axis=0)
            batch_variance = np.var(data, axis=0)

            if self.moving_mean is None:
                self.moving_mean = batch_mean.copy()
                self.moving_variance = batch_variance.copy()
            else:
                self.moving_mean = (self.moving_mean * self.alpha + batch_mean * (1. - self.alpha))
                self.moving_variance = (self.moving_variance * self.alpha + batch_variance * (1. - self.alpha))

            self.output = (data - batch_mean) / np.sqrt(batch_variance + self.EPS)
        else:
            self.output = (data - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return self.output

    def update_grad_input(self, data, grad_output):
        if self.training:
            batch_size = data.shape[0]
            batch_mean = np.mean(data, axis=0)
            batch_variance = np.var(data, axis=0)

            x_centered = input - batch_mean
            inv_std = 1. / np.sqrt(batch_variance + self.EPS)

            self.grad_input = (inv_std / batch_size) * (batch_size * grad_output - np.sum(grad_output, axis=0) - x_centered * (inv_std ** 2) * np.sum(grad_output * x_centered, axis=0))
        else:
            self.grad_input = grad_output / np.sqrt(self.moving_variance + self.EPS)

        return self.grad_input

    def __repr__(self):
        return "BatchNormalization"

## 5. Dropout
#Implement [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). The idea and implementation is really simple:
# just multimply the input by $Bernoulli(p)$ mask. Here $p$ is probability of an element to be zeroed.

#This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.

#While training (self.training == True) it should sample a mask on each iteration (for every batch),
# zero out elements and multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values
# which will be in test mode. When testing this module should implement identity transform i.e. self.output = input.

#- input:   **batch_size x n_feats**
#- output: **batch_size x n_feats**


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.mask = None

    def update_output(self, data):
        if self.training:
            self.mask = (
                np.random.binomial(1, 1. - self.p, size=data.shape) / (1. - self.p)
            )
            self.output = data * self.mask
        else:
            self.output = data
        return self.output

    def update_grad_input(self, data, grad_output):
        if self.training:
            self.grad_input = grad_output * self.mask
        else:
            self.grad_input = grad_output
        return self.grad_input

    def __repr__(self):
        return "Dropout"

"""# Activation functions

Here's the complete example for the Rectified Linear Unit non-linearity (aka ReLU):
"""
class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()

    def update_output(self, data):
        self.output = np.maximum(input, 0)
        return self.output

    def update_grad_input(self, data, grad_output):
        self.grad_input = np.multiply(grad_output , data > 0)
        return self.grad_input

    def repr(self):
        return "ReLU"

"""## 6. Leaky ReLU
Implement [Leaky Rectified Linear Unit](http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs). Expriment with slope.
"""

class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()
        self.slope = slope

    def update_output(self, data):
        self.output = np.where(data > 0, data, data * self.slope)
        return self.output

    def update_grad_input(self, data, grad_output):
        self.grad_input = np.where(data > 0, grad_output, grad_output * self.slope)
        return self.grad_input

    def __repr__(self):
        return "LeakyReLU"