import jax.numpy as jnp
from genjax import Pytree
from condorgmm.condor.utils import (
    MyPytree,
    mywhere,
    replace_slots_in_seq,
    uniformly_replace_slots_in_seq,
)
from genjax.typing import Array


def test_mypytree():
    @Pytree.dataclass
    class MySubclass(MyPytree):
        m: Array | int
        n: Array | int

    @Pytree.dataclass
    class MyClass(MyPytree):
        a: Array | int
        b: Array | int
        c: list | int = Pytree.static()
        d: MySubclass = Pytree.field()

    s = MySubclass(1, 2)
    x = MyClass(a=jnp.array(1), b=2, c=3, d=s)
    y = MyClass(a=jnp.array(1), b=2, c=3, d=s)
    assert x == y
    assert x.replace(a=2) == MyClass(a=2, b=2, c=3, d=s)

    # print(jax.tree_util.tree_structure(MyClass(jnp.ones(3), jnp.ones(3), jnp.array([0, 0, 0]), s)) == jax.tree_util.tree_structure(MyClass(jnp.ones(3), jnp.ones(3), jnp.array([0, 0, 0]), s)))

    assert MyPytree.eq(x, y)
    assert MyPytree.eq(
        MyClass(jnp.ones(3), jnp.ones(3), [0, 0, 0], s),
        MyClass(jnp.ones(3), jnp.ones(3), [0, 0, 0], s),
    )
    assert not MyPytree.eq(
        MyClass(jnp.ones(3), jnp.ones(3), [0, 0, 0], s),
        MyClass(jnp.ones(3), jnp.ones(3), [0, 0, 1], s),
    )
    assert not MyPytree.eq(
        MyClass(jnp.ones(3), jnp.ones(3), [0, 0, 0], s),
        MyClass(jnp.ones(3), jnp.ones(3), [0, 0], s),
    )
    assert not MyPytree.eq(
        MyClass(jnp.ones(3), jnp.array([0, 1, 0]), [0, 0, 0], s),
        MyClass(jnp.ones(3), jnp.ones(3), [0, 0, 0], s),
    )

    assert x.replace({"d": {"m": 5}}) == x.replace(d=x.d.replace(m=5))
    assert x.replace({"d": {"m": 5, "n": 6}}) == x.replace(d=x.d.replace(m=5, n=6))
    assert x.replace({"d": {"m": 5}, "a": 2}) == x.replace(d=x.d.replace(m=5), a=2)
    assert x.replace({"d": {"m": 5, "n": 6}, "a": 2}) == MyClass(
        a=2, b=2, c=3, d=MySubclass(m=5, n=6)
    )


def test_mywhere():
    b = jnp.array([0, 1, 0, 1], dtype=bool)
    x = jnp.array([1, 2, 3, 4])
    y = jnp.array([5, 6, 7, 8])
    assert jnp.all(mywhere(b, x, y) == jnp.array([5, 2, 7, 4]))
    x = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = jnp.array([[9, 10], [11, 12], [13, 14], [15, 16]])
    assert jnp.all(mywhere(b, x, y) == jnp.array([[9, 10], [3, 4], [13, 14], [7, 8]]))


def test_replace_slots_in_seq():
    T, N = 3, 4
    seq = jnp.arange(T * N).reshape(T, N)
    replacements = -jnp.arange(N)

    do_replace = jnp.array([[0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
    assert jnp.all(
        replace_slots_in_seq(seq, replacements, do_replace)  # type: ignore
        == jnp.array([[0, -1, 2, -3], [0, 5, -2, 7], [0, -1, -2, -3]])
    )

    do_replace = jnp.array([0, 1, 0, 1], dtype=bool)
    assert jnp.all(
        uniformly_replace_slots_in_seq(seq, replacements, do_replace)  # type: ignore
        == jnp.array([[0, -1, 2, -3], [4, -1, 6, -3], [8, -1, 10, -3]])
    )

    @Pytree.dataclass
    class MyClass(MyPytree):
        a: Array | int
        b: Array | int
        c: int = Pytree.static()

    seqc = MyClass(a=seq[:, :, None] * jnp.ones(2), b=seq, c=False)
    replc = MyClass(a=replacements[:, None] * jnp.ones(2), b=replacements, c=False)
    assert MyPytree.eq(
        uniformly_replace_slots_in_seq(seqc, replc, do_replace),
        MyClass(
            a=jnp.array(
                [
                    [[0, 0], [-1, -1], [2, 2], [-3, -3]],
                    [[4, 4], [-1, -1], [6, 6], [-3, -3]],
                    [[8, 8], [-1, -1], [10, 10], [-3, -3]],
                ]
            ),
            b=jnp.array([[0, -1, 2, -3], [4, -1, 6, -3], [8, -1, 10, -3]]),
            c=False,
        ),
    )
