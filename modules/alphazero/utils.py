"""
Taken from https://github.com/opendilab/LightZero. Thanks to them!
"""
from collections import namedtuple

import torch
from torch import nn
from typing import Optional, Tuple, TypeVar, List, Union, Callable

SequenceType = TypeVar('SequenceType', List, Tuple, namedtuple)


def build_normalization(norm_type: str, dim: Optional[int] = None) -> nn.Module:
    r"""
    Overview:
        Build the corresponding normalization module
    Arguments:
        - norm_type (:obj:`str`): type of the normaliztion, now support ['BN', 'LN', 'IN', 'SyncBN']
        - dim (:obj:`int`): dimension of the normalization, when norm_type is in [BN, IN]
    Returns:
        - norm_func (:obj:`nn.Module`): the corresponding batch normalization function

    .. note::
        For beginers, you can refer to <https://zhuanlan.zhihu.com/p/34879333> to learn more about batch normalization.
    """
    if dim is None:
        key = norm_type
    else:
        if norm_type in ['BN', 'IN']:
            key = norm_type + str(dim)
        elif norm_type in ['LN', 'SyncBN']:
            key = norm_type
        else:
            raise NotImplementedError("not support indicated dim when creates {}".format(norm_type))
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN1': nn.InstanceNorm1d,
        'IN2': nn.InstanceNorm2d,
        'SyncBN': nn.SyncBatchNorm,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))


def sequential_pack(layers: list) -> nn.Sequential:
    r"""
    Overview:
        Pack the layers in the input list to a `nn.Sequential` module.
        If there is a convolutional layer in module, an extra attribute `out_channels` will be added
        to the module and set to the out_channel of the conv layer.
    Arguments:
        - layers (:obj:`list`): the input list
    Returns:
        - seq (:obj:`nn.Sequential`): packed sequential container
    """
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in reversed(layers):
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def conv2d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        pad_type: str = 'zero',
        activation: nn.Module = None,
        norm_type: str = None,
        num_groups_for_gn: int = 1,
        bias: bool = True
) -> nn.Sequential:
    r"""
    Overview:
        Create a 2-dim convolution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - kernel_size (:obj:`int`): Size of the convolving kernel.
        - stride (:obj:`int`): Stride of the convolution.
        - padding (:obj:`int`): Zero-padding added to both sides of the input.
        - dilation (:obj:`int`): Spacing between kernel elements.
        - groups (:obj:`int`): Number of blocked connections from input channels to output channels.
        - pad_type (:obj:`str`): the way to add padding, include ['zero', 'reflect', 'replicate'], default: None.
        - activation (:obj:`nn.Module`): the optional activation function.
        - norm_type (:obj:`str`): The type of the normalization, now support ['BN', 'LN', 'IN', 'GN', 'SyncBN'],
            default set to None, which means no normalization.
        - num_groups_for_gn (:obj:`int`): Number of groups for GroupNorm.
        - bias (:obj:`bool`): whether adds a learnable bias to the nn.Conv2d. Default set to True.
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the 2 dim convlution layer

    .. note::

        Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
    """
    block = []
    assert pad_type in ['zero', 'reflect', 'replication'], "invalid padding type: {}".format(pad_type)
    if pad_type == 'zero':
        pass
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
        padding = 0
    elif pad_type == 'replication':
        block.append(nn.ReplicationPad2d(padding))
        padding = 0
    block.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    )
    if norm_type is not None:
        if norm_type == 'LN':
            # LN is implemented as GroupNorm with 1 group.
            block.append(nn.GroupNorm(1, out_channels))
        elif norm_type == 'GN':
            block.append(nn.GroupNorm(num_groups_for_gn, out_channels))
        elif norm_type in ['BN', 'IN', 'SyncBN']:
            block.append(build_normalization(norm_type, dim=2)(out_channels))
        else:
            raise KeyError(
                "Invalid value in norm_type: {}. The valid norm_type are "
                "BN, LN, IN, GN and SyncBN.".format(norm_type)
            )

    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


class ResBlock(nn.Module):
    r"""
    Overview:
        Residual Block with 2D convolution layers, including 3 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+
            downsample block: used in EfficientZero
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________ 3*3*C ____________________/+
    Interfaces:
        forward
    """

    def __init__(
        self,
        in_channels: int,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = 'BN',
        res_type: str = 'basic',
        bias: bool = True,
        out_channels: Union[int, None] = None,
    ) -> None:
        """
        Overview:
            Init the 2D convolution residual block.
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor.
            - activation (:obj:`nn.Module`): the optional activation function.
            - norm_type (:obj:`str`): type of the normalization, default set to 'BN'(Batch Normalization), \
                supports ['BN', 'LN', 'IN', 'GN', 'SyncBN', None].
            - res_type (:obj:`str`): type of residual block, supports ['basic', 'bottleneck', 'downsample']
            - bias (:obj:`bool`): whether adds a learnable bias to the conv2d_block. default set to True.
            - out_channels (:obj:`int`): Number of channels in the output tensor, default set to None,
                which means out_channels = in_channels.
        """
        super(ResBlock, self).__init__()
        self.act = activation
        assert res_type in ['basic', 'bottleneck',
                            'downsample'], 'residual type only support basic and bottleneck, not:{}'.format(res_type)
        self.res_type = res_type
        if out_channels is None:
            out_channels = in_channels
        if self.res_type == 'basic':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
            )
        elif self.res_type == 'bottleneck':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv3 = conv2d_block(
                out_channels, out_channels, 1, 1, 0, activation=None, norm_type=norm_type, bias=bias
            )
        elif self.res_type == 'downsample':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 3, 2, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
            )
            self.conv3 = conv2d_block(in_channels, out_channels, 3, 2, 1, activation=None, norm_type=None, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.res_type == 'bottleneck':
            x = self.conv3(x)
        elif self.res_type == 'downsample':
            identity = self.conv3(identity)
        x = self.act(x + identity)
        return x


def MLP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    layer_num: int,
    layer_fn: Callable = None,
    activation: nn.Module = None,
    norm_type: str = None,
    use_dropout: bool = False,
    dropout_probability: float = 0.5,
    output_activation: bool = True,
    output_norm: bool = True,
    last_linear_layer_init_zero: bool = False
):
    r"""
    Overview:
        create a multi-layer perceptron using fully-connected blocks with activation, normalization and dropout,
        optional normalization can be done to the dim 1 (across the channels).
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - layer_num (:obj:`int`): Number of layers.
        - layer_fn (:obj:`Callable`): Layer function.
        - activation (:obj:`nn.Module`): The optional activation function.
        - norm_type (:obj:`str`): The type of the normalization.
        - use_dropout (:obj:`bool`): Whether to use dropout in the fully-connected block.
        - dropout_probability (:obj:`float`): The probability of an element to be zeroed in the dropout. Default: 0.5.
        - output_activation (:obj:`bool`): Whether to use activation in the output layer. If True,
            we use the same activation as front layers. Default: True.
        - output_norm (:obj:`bool`): Whether to use normalization in the output layer. If True,
            we use the same normalization as front layers. Default: True.
        - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last linear layer
            (including w and b), which can provide stable zero outputs in the beginning,
            usually used in the policy network in RL settings.
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block.

    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html).
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    # The last layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))
    """
    In the final layer of a neural network, whether to use normalization and activation are typically determined
    based on user specifications. These specifications depend on the problem at hand and the desired properties of
    the model's output.
    """
    if output_norm is True:
        # The last layer uses the same norm as front layers.
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
    if output_activation is True:
        # The last layer uses the same activation as front layers.
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    if last_linear_layer_init_zero:
        # Locate the last linear layer and initialize its weights and biases to 0.
        for _, layer in enumerate(reversed(block)):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return sequential_pack(block)


class DownSample(nn.Module):

    def __init__(self, observation_shape: SequenceType, out_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 norm_type: Optional[str] = 'BN',
                 ) -> None:
        """
        Overview:
            Define downSample convolution network. Encode the observation into hidden state.
            This network is often used in video games like Atari. In board games like go and chess,
            we don't need this module.
        Arguments:
            - observation_shape (:obj:`SequenceType`): The shape of observation space, e.g. [C, W, H]=[12, 96, 96]
                for video games like atari, RGB 3 channel times stack 4 frames.
            - out_channels (:obj:`int`): The output channels of output hidden state.
            - activation (:obj:`nn.Module`): The activation function used in network, defaults to nn.ReLU(). \
                Use the inplace operation to speed up.
            - norm_type (:obj:`Optional[str]`): The normalization type used in network, defaults to 'BN'.
        """
        super().__init__()
        assert norm_type in ['BN', 'LN'], "norm_type must in ['BN', 'LN']"

        self.conv1 = nn.Conv2d(
            observation_shape[0],
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,  # disable bias for better convergence
        )
        if norm_type == 'BN':
            self.norm1 = nn.BatchNorm2d(out_channels // 2)
        elif norm_type == 'LN':
            self.norm1 = nn.LayerNorm([out_channels // 2, observation_shape[-2] // 2, observation_shape[-1] // 2])

        self.resblocks1 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels // 2,
                    activation=activation,
                    norm_type='BN',
                    res_type='basic',
                    bias=False
                ) for _ in range(1)
            ]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResBlock(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            activation=activation,
            norm_type='BN',
            res_type='downsample',
            bias=False
        )
        self.resblocks2 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [
                ResBlock(
                    in_channels=out_channels, activation=activation, norm_type='BN', res_type='basic', bias=False
                ) for _ in range(1)
            ]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C_in, W, H)`, where B is batch size, C_in is channel, W is width, \
                H is height.
            - output (:obj:`torch.Tensor`): :math:`(B, C_out, W_, H_)`, where B is batch size, C_out is channel, W_ is \
                output width, H_ is output height.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        output = self.pooling2(x)
        return output
