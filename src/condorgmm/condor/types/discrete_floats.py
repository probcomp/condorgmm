from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import jax
from ..utils import MyPytree, find_first_above
from genjax import Pytree
from genjax.typing import IntArray


@dataclass
class _Domain:
    # JAX array of the values in this domain.
    # At runtime, this will live on the GPU, and
    # accessing `FloatFromDiscreteSet.value` will
    # index into this array.
    values: jnp.ndarray

    # A copy of the values on the CPU, for use at compile time.
    _numpy_values: np.ndarray

    def __init__(self, values):
        self.values = values
        self._numpy_values = np.array(values)

    def __eq__(self, other):
        return bool(np.all(self._numpy_values == other._numpy_values))

    def __hash__(self):
        return hash(tuple(self._numpy_values))


@Pytree.dataclass
class Domain(MyPytree):
    _dom: _Domain = Pytree.static()

    def __init__(self, values):
        self._dom = _Domain(values)  # type: ignore

    def __len__(self):
        return len(self._dom.values)

    @property
    def values(self):
        return self._dom.values

    @property
    def discrete_float_values(self):
        return jax.vmap(lambda idx: FloatFromDiscreteSet(idx=idx, domain=self))(
            jnp.arange(self.values.shape[0])
        )

    def first_value_above(self, val) -> "FloatFromDiscreteSet":
        idx = find_first_above(self.values, val)
        return FloatFromDiscreteSet(idx=idx, domain=self)


@Pytree.dataclass
class FloatFromDiscreteSet(MyPytree):
    idx: IntArray
    domain: Domain = Pytree.static()

    @property
    def value(self):
        return self.domain.values[self.idx]

    @property
    def shape(self):
        return self.idx.shape

    def tile(self, *tile_args, **tile_kwargs):
        return FloatFromDiscreteSet(
            idx=jnp.tile(self.idx, *tile_args, **tile_kwargs), domain=self.domain
        )

    def __eq__(self, other):
        return self.domain == other.domain and jnp.all(
            jnp.array(self.idx) == jnp.array(other.idx)
        )
