import cv2
import glob
import os
from pathlib import Path


class ImageInput:
    def read(self):
        raise NotImplementedError

    def is_open(self):
        raise NotImplementedError

    def close(self):
        pass

    def __len__(self):
        return self.get_frame_count()

    def __call__(self):
        return self.read()

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_open():
            return self.read()
        else:
            raise StopIteration

    @staticmethod
    def create(src):
        src = Path(src)
        if src.suffix in (".mp4", ".avi", ".mov"):
            return CapInput(src)
        else:
            return ImageFolderInput(src)


class CapInput(ImageInput):
    def __init__(self, src):
        super().__init__()

        src = str(src)
        try:
            src = int(src)
        except ValueError:
            pass

        self.src = src

        # self.size = None
        self.cap = None
        self.frame_count = None

        self._open()

    def _open(self):
        cap = cv2.VideoCapture(self.src)
        self.cap = cap
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert self.cap.isOpened(), f"CapInput failed to open '{self.src}'"

    def get_frame_count(self):
        return self.frame_count

    def read(self):
        ret = False

        while not ret:
            if not self.cap.isOpened():
                self.close()
                frame = None
                break

            ret, frame = self.cap.read()

        return frame

    def is_open(self):
        return self.cap.isOpened()

    def close(self):
        self.cap.release()


class ImageFolderInput(ImageInput):
    def __init__(self, path,):
        path = Path(path)
        self.files = sorted(path.glob("*")) if path.is_dir() else [str(path)]
        self.frame_idx = 0

        assert len(self.files) > 0

    def read(self):
        img = cv2.imread(str(self.files[self.frame_idx]))
        self.frame_idx += 1

        return img

    def is_open(self):
        return self.frame_idx < len(self.files)

    def get_frame_count(self):
        return len(self.files)

    def close(self):
        pass