import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    x[x<0] = 0.0
    return x

def leaky_relu(x):
    c = 1e-2
    x[x<0] = c * x[x<0]

class MLP():
    """
    A multilayer perceptron, currently with no gradients
    """
    def __init__(self, input_dim, output_dim, hid_dim=[32], act=np.tanh):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.act = act
        self.final_act = True
        
        self.num_parameters = self.input_dim * self.hid_dim[0]
        self.num_parameters += self.hid_dim[-1] * self.output_dim
        for hid_idx in range(1, len(hid_dim)):
            self.num_parameters += self.hid_dim[hid_idx-1] \
                    * self.hid_dim[hid_idx]

        self.mean = np.zeros((self.num_parameters))
        self.covar =  np.eye(self.num_parameters)

        self.init_params()

    def forward(self, x):

        
        x = np.matmul(x, self.w_x2h) + self.b_x2h
        x = self.act(x)

        for hid_idx in range(1,len(self.hid_dim)):
            x = np.matmul(x, self.w_hid[hid_idx]) + self.b_hid[hid_idx]
            x = self.act(x)
            
        x = np.matmul(x, self.w_h2y) + self.b_h2y

        if self.final_act:
            x = self.act(x)


        return x

    def init_params(self):
        
        self.parameters = np.random.multivariate_normal(self.mean, self.covar)

        start = 0
        end = self.input_dim * self.hid_dim[0]
        self.w_x2h = self.parameters[start:end]\
                .reshape(self.input_dim, self.hid_dim[0])/ self.input_dim
        self.b_x2h = np.zeros((1,self.hid_dim[0]))

        start = end

        self.w_hid = {}
        self.b_hid = {}
        for hid_idx in range(1,len(self.hid_dim)):
            end = start + int(self.hid_dim[hid_idx-1] * self.hid_dim[hid_idx])
        

            self.w_hid[hid_idx] = self.parameters[start:end]\
                    .reshape(self.hid_dim[hid_idx-1], self.hid_dim[hid_idx])\
                    / self.hid_dim[hid_idx-1]

            start = end
            
            self.b_hid[hid_idx] = np.zeros((1,self.hid_dim[hid_idx]))

        end = start + self.hid_dim[-1] * self.output_dim

        self.w_h2y = self.parameters[start:end]\
                .reshape(self.hid_dim[-1], self.output_dim)\
                /self.hid_dim[-1]

        self.b_h2y = np.zeros((1, self.output_dim))


if __name__ == "__main__":

    policy = MLP(4,1,[32,32])

    x = np.random.randn(100,4)
    y = policy.forward(x)
