from typing import Tuple, Dict, TypeVar
import jax
import jax.numpy as jnp
from genjax import Pytree
from ..utils import MyPytree
from ..types import CondorGMMState

### Utilities for instrumenting inference to get intermediate states and logging ###

T = TypeVar("T")


@Pytree.dataclass
class InferenceStateSequence[T](MyPytree):
    states: T  # batched (n,)
    label_idxs: jnp.ndarray  # batched (n,) int array
    labels: list[str] = Pytree.static()

    def get_label(self, idx: int) -> str:
        return self.labels[self.label_idxs[idx]]

    @property
    def all_labels(self):
        return [self.get_label(i) for i in range(len(self))]

    def __len__(self):
        return len(self.label_idxs)


@Pytree.dataclass
class InferenceMetadata[T](Pytree):
    visited_states: InferenceStateSequence[T] | None
    misc: Dict

    def __post_init__(self):
        assert (
            isinstance(self.visited_states, InferenceStateSequence)
            or self.visited_states is None
        )
        assert isinstance(self.misc, dict)


@Pytree.dataclass
class InferenceLoggingConfig(Pytree):
    log: bool = Pytree.static(default=False)
    log_global_param_move_details: bool = Pytree.static(default=False)


Metadata = InferenceMetadata[CondorGMMState]
LogConfig = InferenceLoggingConfig
Seq = InferenceStateSequence
no_metadata = InferenceMetadata(None, {})
default_config = InferenceLoggingConfig(log=False)


def simplemeta(
    st: CondorGMMState, c: LogConfig, str_label: str, misc={}
) -> InferenceMetadata:
    if c.log:
        return InferenceMetadata(
            InferenceStateSequence(
                states=jax.tree.map(lambda x: jnp.array(x)[None, ...], st),
                label_idxs=jnp.array([0]),
                labels=[str_label],
            ),
            misc,
        )
    else:
        return no_metadata


def wrap(
    st: CondorGMMState, c: LogConfig, str_label: str, misc={}
) -> Tuple[CondorGMMState, Metadata]:
    return st, simplemeta(st, c, str_label, misc)


def metadata_from_state_list(
    states: list[T], labels: list[str], miscs: list[Dict] | None = None
):
    if miscs is None:
        miscs = [{} for _ in states]

    return InferenceMetadata(
        InferenceStateSequence(
            states=jax.tree.map(lambda *sts: jnp.stack([*sts]), *states),
            label_idxs=jnp.arange(len(states)),
            labels=labels,
        ),
        {
            f"for_{label}": misc
            for label, misc in zip(labels, miscs)
            if len(misc.keys()) != 0
        },
    )


def _sequence_states(
    s1: InferenceStateSequence, s2: InferenceStateSequence
) -> InferenceStateSequence:
    return InferenceStateSequence(
        states=jax.tree.map(lambda x, y: _concat([x, y]), s1.states, s2.states),
        label_idxs=_concat([s1.label_idxs, s2.label_idxs + len(s1.labels)]),
        labels=s1.labels + s2.labels,
    )


def _sequence_metadata(m1: Metadata, m2: Metadata) -> Metadata:
    if m1.visited_states is None:
        return m2
    if m2.visited_states is None:
        return m1
    return InferenceMetadata(
        _sequence_states(m1.visited_states, m2.visited_states), {**m1.misc, **m2.misc}
    )


R = TypeVar("R", InferenceStateSequence, InferenceMetadata)


def sequence_one(a: R, b: R) -> R:
    if isinstance(a, InferenceStateSequence):
        return _sequence_states(a, b)  # type: ignore
    if isinstance(a, InferenceMetadata):
        return _sequence_metadata(a, b)  # type: ignore


def sequence(*args: R) -> R:
    if len(args) < 1:
        raise ValueError("sequence requires at least one argument")

    if len(args) < 2:
        return args[0]

    a = sequence_one(args[0], args[1])
    for i, x in enumerate(args[2:], start=2):
        a = sequence_one(a, x)
    return a


def prepend_misc(meta: Metadata, prefix):
    return InferenceMetadata(
        meta.visited_states, {f"{prefix}::{k}": v for k, v in meta.misc.items()}
    )


def flatten_state_sequence(seq: InferenceStateSequence) -> InferenceStateSequence:
    return InferenceStateSequence(
        states=jax.tree.map(
            lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), seq.states
        ),
        label_idxs=seq.label_idxs.reshape(-1),
        labels=seq.labels,
    )


def flatten_metadata(meta: Metadata) -> Metadata:
    if meta.visited_states is not None:
        return InferenceMetadata(flatten_state_sequence(meta.visited_states), meta.misc)
    return meta


def prepend_label(meta: InferenceMetadata, label: str):
    if meta.visited_states is None:
        states = None
    else:
        states = meta.visited_states.replace(
            labels=[f"{label}::{lab}" for lab in meta.visited_states.labels]
        )
    final_meta = prepend_misc(InferenceMetadata(states, meta.misc), label)
    assert (
        isinstance(final_meta.visited_states, InferenceStateSequence)
        or final_meta.visited_states is None
    )
    assert isinstance(final_meta.misc, dict)
    return final_meta


### Misc. utilities ###


def _concat(vals, axis=0):
    if vals[0].shape == ():
        for v in vals:
            assert v.dtype == vals[0].dtype
            assert v.shape == ()
            return vals[0]
    return jnp.concatenate(vals, axis=axis)
