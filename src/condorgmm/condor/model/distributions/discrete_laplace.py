import jax.numpy as jnp
from genjax import Const
import genjax
from ...types import FloatFromDiscreteSet
from ...utils import unwrap
from typing import cast
from genjax.typing import IntArray

### Discretized Laplace ###


def sum_exponential_over_range(minx, maxx, log_base):
    val = (jnp.exp(log_base * minx) - jnp.exp(log_base * (maxx + 1))) / (
        1 - jnp.exp(log_base)
    )
    return jnp.where(minx <= maxx, val, 0)


def normalizing_const_for_discretized_laplace_for_center_in_range(
    minx: int, maxx: int, center: IntArray, scale: float
):
    ## Derivation of the first computed expression:
    #   sum_{x=minx}^{center} exp(-|x - center| / scale)
    # = sum_{x=minx}^{center} exp(-(center - x) / scale)
    # = sum_{y = 0}^{center - minx} exp(-y / scale) [set y = center - x]
    p_minx_to_center = sum_exponential_over_range(0, center - minx, -1.0 / scale)

    ## Derivation of the second computed expression:
    #   sum_{x=center+1}^{maxx} exp(-|x - center| / scale)
    # = sum_{x=center+1}^{maxx} exp(-(x - center) / scale)
    # = sum_{y = 1}^{maxx - center} exp(-y / scale) [set y = x - center]
    p_center_plus_1_to_maxx = sum_exponential_over_range(1, maxx - center, -1.0 / scale)

    ## Final return is the sum of these two expressions.
    return p_minx_to_center + p_center_plus_1_to_maxx


def normalizing_const_for_discretized_laplace(
    minx: int, maxx: int, center: IntArray, scale: float
):
    in_range = normalizing_const_for_discretized_laplace_for_center_in_range(
        minx, maxx, center, scale
    )

    ## Derivation of the first computed expression:
    #   sum_{x=minx}^{maxx} exp(-(x - center) / scale)
    # = sum_{y=minx - center}^{maxx - center} exp(-y / scale) [set y = x - center]
    p_if_center_below_minx = sum_exponential_over_range(
        minx - center, maxx - center, -1.0 / scale
    )

    ## Derivation of the second computed expression:
    #   sum_{x=minx}^{maxx} exp(-(center - x) / scale)
    # = sum_{y=center - maxx}^{center - minx} exp(-y / scale) [set y = center - x]
    p_if_center_above_maxx = sum_exponential_over_range(
        center - maxx, center - minx, -1.0 / scale
    )

    ## Final return is a 3-way among these two expressions,
    # and the expression for when center is in range.
    return jnp.where(
        minx <= center,
        jnp.where(center <= maxx, in_range, p_if_center_above_maxx),
        p_if_center_below_minx,
    )


def discretized_laplace_logpdf(x, center, scale, minx, maxx):
    assert isinstance(center, int) or jnp.issubdtype(
        center.dtype, jnp.integer
    ), "center must be an integer"
    assert isinstance(x, int) or jnp.issubdtype(
        x.dtype, jnp.integer
    ), f"x must be an integer but x = {x}"

    # convert to int32, in case these are given as uint8.
    # (if we keep these as uint8, arithmetic will be done mod 256,
    # which is not what we want)
    center = jnp.array(center, dtype=jnp.int32)
    x = jnp.array(x, dtype=jnp.int32)

    # PDF is exp(-|x - center| / scale) / Z
    Z = normalizing_const_for_discretized_laplace(minx, maxx, center, scale)
    return (-jnp.abs(x - center) / scale) - jnp.log(Z)


@genjax.Pytree.dataclass
class DiscretizedLaplace(genjax.ExactDensity):
    def sample(self, key, center, scale, minx: int | Const, maxx: int | Const):
        minx_val = cast(int, unwrap(minx))
        maxx_val = cast(int, unwrap(maxx))
        return (
            genjax.categorical(
                discretized_laplace_logpdf(
                    jnp.arange(minx_val, maxx_val + 1),
                    center,
                    scale,
                    minx_val,
                    maxx_val,
                )
            )(key)
            + minx_val
        )

    def logpdf(self, x, center, scale, minx: int | Const, maxx: int | Const):
        minx, maxx = unwrap(minx), unwrap(maxx)
        return discretized_laplace_logpdf(x, center, scale, minx, maxx)

    @property
    def __doc__(self):
        return DiscretizedLaplace.__doc__


discretized_laplace = DiscretizedLaplace()


@genjax.Pytree.dataclass
class IndexSpaceDiscretizedLaplace(genjax.ExactDensity):
    def sample(self, key, mean: FloatFromDiscreteSet, scale):
        idx = discretized_laplace.sample(
            key,
            mean.idx,
            scale,
            Const(0),
            Const(mean.domain.values.size - 1),
        )
        return FloatFromDiscreteSet(idx=idx, domain=mean.domain)

    def logpdf(self, x, mean: FloatFromDiscreteSet, scale):
        return discretized_laplace.logpdf(
            x.idx,
            mean.idx,
            scale,
            Const(0),
            Const(mean.domain.values.size - 1),
        )

    @property
    def __doc__(self):
        return IndexSpaceDiscretizedLaplace.__doc__


index_space_discretized_laplace = IndexSpaceDiscretizedLaplace()
