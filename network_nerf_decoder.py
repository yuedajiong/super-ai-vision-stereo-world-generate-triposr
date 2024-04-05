import torch
import einops

class NeRFMLP(torch.nn.Module):
    def __init__(self, n_hidden_layers):
        super().__init__()
        self.in_channels = 3 * 40
        self.n_neurons = 64
        self.n_hidden_layers = n_hidden_layers
        self.activation = ["silu","relu"][0]
        self.bias: bool = True
        self.weight_init = "kaiming_uniform"
        self.bias_init = None

        def make_linear(dim_in, dim_out, bias=True, weight_init=None, bias_init=None):
            layer = torch.nn.Linear(dim_in, dim_out, bias=bias)
            if weight_init is None:
                pass
            elif weight_init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:
                raise NotImplementedError
            if bias:
                if bias_init is None:
                    pass
                elif bias_init == "zero":
                    torch.nn.init.zeros_(layer.bias)
                else:
                    raise NotImplementedError
            return layer

        def make_activation(activation):
            if activation == "relu":
                return torch.nn.ReLU(inplace=True)
            elif activation == "silu":
                return torch.nn.SiLU(inplace=True)
            else:
                raise NotImplementedError

        layers = [make_linear(self.in_channels, self.n_neurons, bias=self.bias, weight_init=self.weight_init, bias_init=self.bias_init), make_activation(self.activation)]
        for i in range(self.n_hidden_layers - 1):
            layers += [make_linear(self.n_neurons, self.n_neurons, bias=self.bias, weight_init=self.weight_init, bias_init=self.bias_init), make_activation(self.activation)]
        layers += [make_linear(self.n_neurons, 4, bias=self.bias, weight_init=self.weight_init, bias_init=self.bias_init)]  #4=density:1+features:3
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])
        features = self.layers(x).reshape(*x.shape[:-1], -1)
        return {"density": features[..., 0:1], "features": features[..., 1:4]}

