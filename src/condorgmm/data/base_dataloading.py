from abc import abstractmethod
import trimesh
from .frame import Frame
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class Video:
    def __init__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Frame:
        raise NotImplementedError

    @abstractmethod
    def get_object_mesh_from_id(self, id: int) -> trimesh.Trimesh:
        raise NotImplementedError

    @abstractmethod
    def get_object_name_from_id(self, id: int) -> str:
        raise NotImplementedError

    def downscale(self, factor: int) -> "DownscaledVideo":
        return DownscaledVideo(self, factor)
    
    def upscale(self, factor: int) -> "UpscaledVideo":
        return UpscaledVideo(self, factor)

    def crop(self, miny, maxy, minx, maxx) -> "CroppedVideo":
        return CroppedVideo(self, miny, maxy, minx, maxx)

    def load_frames(self, indices=None):
        if indices is None:
            indices = range(len(self))
        with ThreadPoolExecutor() as executor:
            frames = list(
                tqdm(
                    executor.map(lambda i: self[i], indices),
                    total=len(indices),
                )
            )
        return frames

    def load_all_frames(self):
        return self.load_frames()


class DownscaledVideo(Video):
    def __init__(self, video: Video, factor: int):
        self.video = video
        self.factor = factor

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx: int):
        return self.video[idx].downscale(self.factor)

    def get_object_mesh_from_id(self, id: int) -> trimesh.Trimesh:
        return self.video.get_object_mesh_from_id(id)

    def get_object_name_from_id(self, id: int) -> str:
        return self.video.get_object_name_from_id(id)
    
class UpscaledVideo(Video):
    def __init__(self, video: Video, factor: int):
        self.video = video
        self.factor = factor

    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx: int):
        return self.video[idx].upscale(self.factor)
    
    def get_object_mesh_from_id(self, id: int) -> trimesh.Trimesh:
        return self.video.get_object_mesh_from_id(id)

    def get_object_name_from_id(self, id: int) -> str:
        return self.video.get_object_name_from_id(id)


class CroppedVideo(Video):
    def __init__(self, video: Video, miny, maxy, minx, maxx):
        self.video = video
        self.miny = miny
        self.maxy = maxy
        self.minx = minx
        self.maxx = maxx

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx: int):
        return self.video[idx].crop(self.miny, self.maxy, self.minx, self.maxx)

    def get_object_mesh_from_id(self, id: int) -> trimesh.Trimesh:
        return self.video.get_object_mesh_from_id(id)

    def get_object_name_from_id(self, id: int) -> str:
        return self.video.get_object_name_from_id(id)
