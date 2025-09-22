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
    dim_hidden: int = 64 # hidden dimension of MLP
    num_gsblocks: int = 20
    type_polar: str = "canonical" # "canonical" or "normal"
    activation: Callable = nn.relu
    mlp_res_connection: bool = False
    theta_predictor: str = "mlp" # "mlp" 
    dim_hidden_list: Optional[List[int]] = None # if theta_predictor is "gradient", this is used for H_of_I
    #normalize_qp: bool = False # whether to normalize (q,p) before feeding into GSympNet
    learn_scale: bool = False # whether to learn scale parameters for (q,p)

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
            self.H_of_I = MLPFlexible(
                dim_input=self.dim_config,
                dim_output=1, # output scalar Hamiltonian
                dim_hidden_list=dim_hidden_list,
                activation=self.activation,
                res_connection=self.mlp_res_connection,
            ) # (B,2n) -> (B,1)

            #self.theta_generator = lambda I: jax.vmap(jax.grad(H_of_I))(I)
            # <- ここで呼ぶとCallCompactUnknownErrorになる
        
        if self.learn_scale:
            self.scale_q = self.param('scale_q', nn.initializers.ones, (self.dim_config,))
            self.scale_p = self.param('scale_p', nn.initializers.ones, (self.dim_config,))
        else:
            self.scale_q = 1.0
            self.scale_p = 1.0

    def __call__(self, q, p, delta_t, train: bool = True):
        # q, p: (B, n)
        # delta_t: (B,) or scalar
        # if self.normalize_qp:
        #     mean_q = jnp.mean(q, axis=0, keepdims=True)
        #     mean_p = jnp.mean(p, axis=0, keepdims=True)
        #     std_q = jnp.std(q, axis=0, keepdims=True)
        #     std_p = jnp.std(p, axis=0, keepdims=True)
        #     q = (q - mean_q) / std_q
        #     p = (p - mean_p) / std_p

        # scale q, p
        q_scale= self.scale_q / jnp.sqrt( self.scale_q * self.scale_p )
        p_scale = self.scale_p / jnp.sqrt( self.scale_q * self.scale_p )
        q = q * q_scale[None,...]
        p = p * p_scale[None,...]

        # (q, p) -> (Ix, Iy)
        Ix, Iy = self.gsymp_net(q, p)
        # Ix, Iy: (B, n)

        # (Ix, Iy) -> (I, theta)
        I, theta = self.to_polar(Ix, Iy)
        # I, theta: (B, n)

        # I_ = I as I should be constant, only renew theta
        if self.theta_predictor == "mlp":
            theta_ = theta + delta_t * self.theta_generator(I)
        elif self.theta_predictor == "gradient":
            def hamil_sum_and_hamil(II):
                HH = self.H_of_I(II)   # (B,)
                return HH.sum(), HH
             # Here, you can do ".sum()" as long as batches do not mix.
            
            grad_hamil, hamil = jax.grad(hamil_sum_and_hamil, has_aux=True)(I)

            # def H_scalar(II):
            #     return self.H_of_I(II[None, :]).squeeze() # (n,)→scalar
            # omega = jax.vmap(jax.grad(H_scalar))(I)
            theta_ = theta + delta_t * grad_hamil

        #theta_ = theta + delta_t * self.theta_generator(I)
        # theta_: (B, n)

        # wrap theta to [-pi, pi]
        theta_ = (theta_ + jnp.pi) % (2 * jnp.pi) - jnp.pi

        # inverse from future: (Ix_, Iy_) -> (q_, p_)
        Ix_, Iy_ = self.inv_polar(I, theta_)
        q_, p_ = self.gsymp_net.inverse(Ix_, Iy_)

        q_ = q_ * 1./q_scale[None,...]
        p_ = p_ * 1./p_scale[None,...]

        # if self.normalize_qp:
        #     q_ = q_ * std_q + mean_q
        #     p_ = p_ * std_p + mean_p

        return q_, p_, I, theta

    def _scale_qp(self, q, p):
        if self.learn_scale:
            q_scale = self.scale_q / jnp.sqrt(self.scale_q * self.scale_p)
            p_scale = self.scale_p / jnp.sqrt(self.scale_q * self.scale_p)
        else:
            q_scale = 1.0
            p_scale = 1.0
        q_s = q * q_scale
        p_s = p * p_scale
        return q_s, p_s, q_scale, p_scale

    # --- (q,p) → I を取り出すだけのヘルパ ---
    def get_actions(self, q, p):
        """q,p:(B,n) → I:(B,n) を返す（GSymp→極座標まで辿る）。"""
        q_s, p_s, *_ = self._scale_qp(q, p)
        Ix, Iy = self.gsymp_net(q_s, p_s)
        I, _theta = self.to_polar(Ix, Iy)
        return I

    # --- H(I) を直接計算（batched I を受け取り (B,) を返す）---
    def h_of_I(self, I):
        if not hasattr(self, "H_of_I"):
            raise ValueError("theta_predictor='gradient' のときだけ H_of_I が定義されます。")
        H = self.H_of_I(I)            # (B,1)
        return H.squeeze(-1)          # (B,)

    # --- (q,p) から H を推定（あなたの learned_hamiltonian 相当）---
    def hamiltonian(self, q, p):
        """(B,n),(B,n) → (B,)"""
        I = self.get_actions(q, p)
        return self.h_of_I(I)

    # --- 角速度 ω = ∂H/∂I（= dH/dI）を返す（必要なら）---
    def omega(self, q, p):
        """(B,n),(B,n) → (B,n), with ω_i = ∂H/∂I_i"""
        I = self.get_actions(q, p)
        # batched 勾配を安全に取る：合計にしてから grad
        def h_sum(I_):
            return self.h_of_I(I_).sum()
        dH_dI = jax.grad(h_sum)(I)    # (B,n)
        return dH_dI
