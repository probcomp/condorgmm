from typing import Self
import jax
from genjax import PythonicPytree, Const
from dataclasses import replace
import jax.numpy as jnp


class MyPytree(PythonicPytree):
    def replace(self: Self, *args, do_replace_none=False, **kwargs) -> Self:
        if len(kwargs) > 0:
            assert (
                len(args) == 0
            ), "Cannot mix positional and keyword arguments in replace."
            return self.replace(kwargs)

        assert len(args) == 1 and isinstance(
            args[0], dict
        ), "Expected a single dictionary argument."
        new_fields = args[0]

        def recurse(k, v):
            current_val = self.__getattribute__(k)
            if isinstance(v, dict) and isinstance(current_val, MyPytree):
                return current_val.replace(v)
            else:
                return v

        new_fields = {
            k: recurse(k, v)
            for k, v in new_fields.items()
            if do_replace_none or v is not None
        }
        return replace(self, **new_fields)  # type: ignore

    @staticmethod
    def eq(x, y):
        # See https://github.com/probcomp/genjax/issues/1441 for why
        # I didn't just override __eq__.
        # (Could get the __eq__ override to work with a bit more effort, however.)
        if jax.tree_util.tree_structure(x) != jax.tree_util.tree_structure(y):
            return False
        leaves1 = jax.tree_util.tree_leaves(x)
        leaves2 = jax.tree_util.tree_leaves(y)
        bools = [jnp.all(l1 == l2) for l1, l2 in zip(leaves1, leaves2)]
        return jnp.all(jnp.array(bools))

    # The __getitem__ override is needed for GenJAX versions
    # prior to https://github.com/probcomp/genjax/pull/1440.
    def __getitem__(self, idx) -> Self:
        return jax.tree_util.tree_map(lambda v: v[idx], self)


def unwrap(x):
    if isinstance(x, Const):
        return x.val
    else:
        return x


def normalize(arr):
    return arr / jnp.sum(arr)


def mywhere(b, x, y):
    assert len(x.shape) == len(y.shape)
    if len(b.shape) == len(x.shape):
        return jnp.where(b, x, y)
    else:
        return jnp.where(
            b[:, *(None for _ in range(len(x.shape) - len(b.shape)))], x, y
        )


def replace_slots_in_seq(
    seq: MyPytree,  # Batched, (T, N, ...)
    replacements: MyPytree,  # Batched, (N, ...)
    do_replace: jnp.ndarray,  # (T, N)
):
    return jax.tree.map(
        lambda s, r: jax.vmap(
            lambda x, rep, do_rep: mywhere(do_rep, rep, x), in_axes=(0, None, 0)
        )(s, r, do_replace),
        seq,
        replacements,
    )


def uniformly_replace_slots_in_seq(
    seq: MyPytree,  # Batched, (T, N, ...)
    replacements: MyPytree,  # Batched, (N, ...)
    do_replace: jnp.ndarray,  # (N,)
):
    T = len(seq)
    return replace_slots_in_seq(seq, replacements, jnp.tile(do_replace, (T, 1)))


def find_first_above(values, threshold):
    first = jnp.argmax(values >= threshold)
    return jnp.where(jnp.logical_and(first == 0, values[0] < threshold), -1, first)
