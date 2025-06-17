import jax.numpy as jnp
from ..utils import MyPytree
from genjax import Pytree
import warnings
from genjax.typing import FloatArray

GAVE_WARNING = False


@Pytree.dataclass
class Intrinsics(MyPytree):
    fx: FloatArray
    fy: FloatArray
    cx: FloatArray
    cy: FloatArray
    near: FloatArray
    far: FloatArray
    image_height: int = Pytree.static()
    image_width: int = Pytree.static()

    def upscale(self, factor):
        global GAVE_WARNING
        if not GAVE_WARNING:
            warnings.warn(
                "Called Intrinsics.upscale, but the cx and cy computations are slightly incorrect."
            )
            GAVE_WARNING = True

        return Intrinsics(
            self.fx * factor,
            self.fy * factor,
            self.cx * factor,
            self.cy * factor,
            self.near,
            self.far,
            self.image_height * factor,
            self.image_width * factor,
        )

    def downscale(self, factor):
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        new_cx = cx / factor - 0.5 / factor + 0.5
        new_cy = cy / factor - 0.5 / factor + 0.5
        new_fx = fx / factor
        new_fy = fy / factor

        return Intrinsics(
            new_fx,
            new_fy,
            new_cx,
            new_cy,
            self.near,
            self.far,
            self.image_height // factor,
            self.image_width // factor,
        )

    def crop(self, miny, maxy, minx, maxx):
        warnings.warn(
            "Called Intrinsics.crop.  TODO: double check the cx and cy computation in Intrinsics.crop."
        )

        return Intrinsics(
            self.fx,
            self.fy,
            self.cx - minx,
            self.cy - miny,
            self.near,
            self.far,
            maxy - miny,
            maxx - minx,
        )


@Pytree.dataclass
class ImageWithIntrinsics(MyPytree):
    image: jnp.ndarray  # (H, W, ...)
    intrinsics: Intrinsics

    def downscale(self, factor):
        return ImageWithIntrinsics(
            image=self.image[::factor, ::factor],
            intrinsics=self.intrinsics.downscale(factor),
        )

    def crop(self, miny, maxy, minx, maxx):
        return ImageWithIntrinsics(
            image=self.image[miny:maxy, minx:maxx],
            intrinsics=self.intrinsics.crop(miny, maxy, minx, maxx),
        )
