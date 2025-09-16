import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, List, Optional
from action_angle_networks.sk_layers import (
    MLP,
    GSympNet,
    MLPFlexible,
    PolarCoordinates,
    InversePolarCoordinates,
    GotosCanonicalPolarCoordinates,
    InverseGotosCanonicalPolarCoordinates
)


class MyActionAngleNetwork(nn.Module):
    dim_config: int # dimenion of configuration space
    dim_hidden: int # hidden dimension of MLP
    num_gsblocks: int
    type_polar: str = "canonical" # "canonical" or "normal"
    activation: Callable = nn.relu
    mlp_res_connection: bool = False
    theta_predictor: str = "mlp" # "mlp" 
    dim_hidden_list: Optional[List[int]] = None # if theta_predictor is "gradient", this is used for H_of_I

    def setup(self):
        self.gsymp_net = GSympNet(
            dim_config=self.dim_config,
            dim_hidden=self.dim_hidden,
            num_blocks=self.num_gsblocks,
            activation=self.activation
        )
        if self.type_polar == "normal":
            self.to_polar = PolarCoordinates(dim_config=self.dim_config)
            self.inv_polar = InversePolarCoordinates(dim_config=self.dim_config)
        elif self.type_polar == "canonical":
            self.to_polar = GotosCanonicalPolarCoordinates(dim_config=self.dim_config)
            self.inv_polar = InverseGotosCanonicalPolarCoordinates(dim_config=self.dim_config)

        if self.theta_predictor == "mlp":
            self.theta_generator = MLP(
                dim_input=self.dim_config,
                dim_output=self.dim_config,
                dim_hidden=self.dim_hidden,
                activation=self.activation,
                res_connection=self.mlp_res_connection,
            )
        elif self.theta_predictor == "gradient":
            dim_hidden_list = [self.dim_hidden, self.dim_hidden] if self.dim_hidden_list is None else self.dim_hidden_list
            H_of_I = MLPFlexible(
                dim_input=self.dim_config,
                dim_output=1, # output scalar Hamiltonian
                dim_hidden_list=dim_hidden_list,
                activation=self.activation,
                res_connection=self.mlp_res_connection,
            )
            self.theta_generator = lambda I: jax.vmap(jax.grad(H_of_I))(I)

    def __call__(self, q, p, delta_t, train: bool = True):
        # q, p: (B, n)
        # delta_t: (B,) or scalar

        # (q, p) -> (Ix, Iy)
        Ix, Iy = self.gsymp_net(q, p)
        # Ix, Iy: (B, n)

        # (Ix, Iy) -> (I, theta)
        I, theta = self.to_polar(Ix, Iy)
        # I, theta: (B, n)

        # I_ = I as I should be constant, only renew theta
        theta_ = theta + delta_t * self.theta_generator(I)
        # theta_: (B, n)

        # wrap theta to [-pi, pi]
        theta_ = (theta_ + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # inverse from future: (Ix_, Iy_) -> (q_, p_)
        Ix_, Iy_ = self.inv_polar(I, theta_)
        q_, p_ = self.gsymp_net.inverse(Ix_, Iy_)

        return q_, p_, I, theta
