import torch.nn as nn

def init_layer(layer):
    
    """
    Initialize the weights and bias of a Linear or Convolutional layer.

    This function applies Xavier (Glorot) uniform initialization to the
    layer's weight tensor and sets the bias to zero if a bias term exists.
    Xavier initialization helps maintain stable gradient magnitudes
    during training, especially for deep networks.

    Parameters
    ----------
    layer : torch.nn.Module
        A PyTorch layer with a `weight` attribute, such as `nn.Linear`,
        `nn.Conv1d`, `nn.Conv2d`, or `nn.Conv3d`. If the layer has a bias,
        it will be initialized to zero.
    """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """
    Initialize a Batch Normalization layer.

    This function sets the BatchNorm scale parameter (weight, γ) to 1
    and the shift parameter (bias, β) to 0. This ensures that the
    BatchNorm layer initially performs an identity transformation.

    Parameters
    ----------
    bn : torch.nn.modules.batchnorm._BatchNorm
        A PyTorch BatchNorm layer (e.g., `nn.BatchNorm1d`,
        `nn.BatchNorm2d`, or `nn.BatchNorm3d`).
    """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def _same_padding(kernel_size):
    """ 
    Determines the size of 'padding' in CovBlocks to obtain same padding given:
    - kernel_size: int or tuple
    """
    if isinstance(kernel_size, int):
        return kernel_size // 2
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    raise TypeError(f"Unsupported kernel_size type: {type(kernel_size)}")


class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        num_convs:int=2, 
        kernel_size=3, # tuple or int
        stride:int=1, 
        padding=None, 
        activation:str='relu', 
        norm:bool=True, # If true, applies nn.BatchNorm2d between convolutions
        dropout=None, # If true, applies nn.Dropout2d between convolutions
        pool:bool=True,
        pool_size=(2,2), # tuple or int
        pool_type:str='avg',
        bias=False,
        init_weights=False, # If true, initializes weights with Xavier on every conv in the block
    ):
        super().__init__()

        if padding is None or padding == "same":
            padding = _same_padding(kernel_size)
        
        # Build Block, with num_convolutions
        self.layers = []
        for layer in range(num_convs):

            # Define channel structure outside 
            # (Allows for iteratable recudtion inside of the block, in the future)
            in_ch = in_channels if layer == 0 else out_channels
            out_ch = out_channels

            self.layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride if layer == 0 else 1, # Changing stride != 1: Resnet downsampling style
                    padding=padding,
                    bias=bias,
                )
            )
            if norm:
                self.layers.append(nn.BatchNorm2d(out_channels))
            # Apply activation after CONV (+ BN)
            self.layers.append(self._get_activation(activation))
            if dropout is not None:
                # (Apply dropout after CONV (+ BN + Activation) + DO)
                self.layers.append(nn.Dropout2d(dropout))

        # Sequence layered convolutions into the block
        self.conv = nn.Sequential(*self.layers)

        # Apply Pooling
        self.pool_type = pool_type
        self.pool_size = pool_size
        if pool:
            if pool_type == 'max':
                self.pool = nn.MaxPool2d(self.pool_size)
            elif pool_type == "avg":
                self.pool = nn.AvgPool2d(self.pool_size)
            elif pool_type == "avg+max": 
                # Will be applied in: forward()
                self.pool = None
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}")
        else:
            self.pool = None
        
        # Init starting weights
        if init_weights:
            self.init_weight()
    
    def init_weight(self):
        def init_fn(module):
            if isinstance(module, nn.Conv2d):
                init_layer(module)
            elif isinstance(module, nn.BatchNorm2d):
                init_bn(module)

        self.apply(init_fn)
    
    def forward(self, x):

        # Apply the input into the sequential block
        x = self.conv(x) 

        if self.pool:
            if self.pool_type == "avg+max":
                # Apply flexible pooling
                x = nn.functional.avg_pool2d(x, self.pool_size) + nn.functional.max_pool2d(x, self.pool_size)
            else:
                # Apply single pooling set in __init__
                x = self.pool(x) 
        return x

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        if activation == "leaky_relu":
            return nn.LeakyReLU(0.1, inplace=True)
        if activation == "elu":
            return nn.ELU(inplace=True)
        if activation == "gelu":
            return nn.GELU()
        if activation == "identity" or activation is None:
            return nn.Identity()
        raise ValueError(f"Unsupported activation: {activation}")