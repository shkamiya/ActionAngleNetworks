import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, List, Optional, Sequence
import optax

class GSBlock(nn.Module):
    dim_config: int # dimenion of configuration space
    dim_hidden: int # hidden dimension of MLP
    activation: Callable = nn.relu 

    def setup(self):

        self.W_affine = self.param("W_affine", nn.initializers.lecun_normal(),
                                    (self.dim_hidden, self.dim_config))
        self.B_bias = self.param("B_bias", nn.initializers.zeros, (self.dim_hidden,))
        self.A_diag = self.param("A_diag", nn.initializers.ones, (self.dim_hidden,))
        C_tmp = self.param("C_tmp", nn.initializers.zeros, (self.dim_config, self.dim_config))
        self.C_sym = (C_tmp + C_tmp.T) / 2.0  # 対称化

    @nn.compact
    def __call__(self, x):
        # x: (B, dim_config)
        y = self.activation(x @ self.W_affine.T + self.B_bias) # (B, dim_hidden)
        z = (y * self.A_diag) @ self.W_affine
        
        return z + x @ self.C_sym # (B, dim_config)

class GSympNet(nn.Module):
    dim_config: int # dimenion of configuration space
    dim_hidden: int # hidden dimension of MLP
    num_blocks: int
    block_fcn: Callable = GSBlock # fucntion to generate block
    activation: Callable = nn.relu    

    def setup(self):
        assert self.num_blocks % 2 == 0, "num_blocks must be even number"
        self.blocks =[
            self.block_fcn(
                dim_config=self.dim_config,
                dim_hidden=self.dim_hidden,
                activation=self.activation,
                name=f'gsblock_{i}'
                ) for i in range(self.num_blocks)
        ]

    def __call__(self, q, p):
        for idx, blk in enumerate(self.blocks):
            if idx % 2 == 0:
                p = p + blk(q)
            else:
                q = q + blk(p)
        return q, p # I^x, I^y
    
    # 逆写像：層を逆順に、加算の符号を反転（可逆三角写像の基本）
    def inverse(self, q, p):
        for idx in reversed(range(self.num_blocks)):
            blk = self.blocks[idx]
            if idx % 2 == 0:           # p' = p + f(q)  =>  p = p' - f(q)
                p = p - blk(q)
            else:                    # q' = q + f(p)  =>  q = q' - f(p)
                q = q - blk(p)
        return q, p

class PolarCoordinates(nn.Module):
    dim_config: int # dimenion of configuration space

    def setup(self):
        pass

    def __call__(self, Ix, Iy):
        I = jnp.sqrt(Ix**2 + Iy**2)
        theta = jnp.arctan2(Iy, Ix)
        return I, theta

class GotosCanonicalPolarCoordinates(nn.Module):
    dim_config: int # dimenion of configuration space

    def setup(self):
        pass

    def __call__(self, Ix, Iy):
        I = 0.5 * (Ix**2 + Iy**2)
        theta = jnp.arctan2(-Iy, Ix)
        return I, theta

class InversePolarCoordinates(nn.Module):
    dim_config: int # dimenion of configuration space

    def setup(self):
        pass

    def __call__(self, I, theta):
        Ix = jnp.sqrt(I) * jnp.cos(theta)
        Iy = jnp.sqrt(I) * jnp.sin(theta)
        return Ix, Iy

class InverseGotosCanonicalPolarCoordinates(nn.Module):
    dim_config: int # dimenion of configuration space

    def setup(self):
        pass

    def __call__(self, I, theta):
        Ix = jnp.sqrt(2*I) * jnp.cos(theta)
        Iy = -jnp.sqrt(2*I) * jnp.sin(theta)
        return Ix, Iy

class MLP(nn.Module):
    dim_input: int
    dim_output: int
    dim_hidden: int
    activation: Callable = nn.relu
    res_connection: bool = False
    
    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.dim_hidden)(x)
        y = self.activation(y)
        y = nn.Dense(self.dim_output)(y)
        return y if self.res_connection == False else y + x

class MLPFlexible(nn.Module):
    dim_input: int
    dim_hidden_list: Sequence[int] # list of hidden dimensions
    dim_output: Optional[int] = None
    activation: Callable = nn.relu
    res_connection: bool = False

    @nn.compact
    def __call__(self, x):
        y = x
        for dim_h in self.dim_hidden_list:
            y = nn.Dense(dim_h)(y)
            y = self.activation(y)
        y = nn.Dense(self.dim_output)(y)
        return y if self.res_connection == False else y + x