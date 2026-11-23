"""Selective HDF5 observations and lossless frame-indexed deterministic inputs."""

import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

LAYOUT = "frame-indexed-v1"


class ObservationFiles:
    """At most four open files with 1 MiB chunk caches; no image corpus cache.

    Closing releases handles. A subsequent read reopens them, including after a
    process fork. Views own this pool, so a record can safely outlive its dataset.
    """

    def __init__(self, *, temporary=None):
        self.temporary = temporary
        self.files = OrderedDict()
        self.columns = {}
        self.pid = os.getpid()

    def get(self, path):
        if self.pid != os.getpid():
            self.close()
            self.pid = os.getpid()
        path = str(path)
        if path not in self.files:
            if len(self.files) == 4:
                oldest, file = self.files.popitem(last=False)
                self.columns.pop(oldest, None)
                file.close()
            self.files[path] = h5py.File(path, "r", rdcc_nbytes=1024**2)
        self.files.move_to_end(path)
        return self.files[path]

    def observations(self, path):
        path = str(path)
        file = self.get(path)
        if path not in self.columns:
            source = file["observations"]
            self.columns[path] = (
                source if isinstance(source, h5py.Dataset) else {key: source[key] for key in source}
            )
        return self.columns[path]

    def close(self):
        self.columns.clear()
        for file in self.files.values():
            file.close()
        self.files.clear()

    def __getstate__(self):
        return dict(self.__dict__, files=OrderedDict(), columns={})

    def __del__(self):
        self.close()
        if self.temporary is not None:
            self.temporary.cleanup()


def read_rows(values, rows, columns=slice(None)):
    """Read contiguous runs, preserving duplicates and caller order.

    HDF5 point selections of wide image rows are much slower than a few ordinary
    slices. Sorting small row indexes does not alter sample order or eligibility.
    """
    rows = np.asarray(rows, dtype=np.int64)
    width = len(range(values.shape[1])[columns])
    if not len(rows):
        return np.empty((0, width), dtype=values.dtype)
    unique, inverse = np.unique(rows, return_inverse=True)
    result = np.empty((len(unique), width), dtype=values.dtype)
    boundaries = np.r_[0, np.flatnonzero(np.diff(unique) != 1) + 1, len(unique)]
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        result[begin:end] = values[int(unique[begin]) : int(unique[end - 1]) + 1, columns]
    return result[inverse]


class ObservationArray:
    """A row/column view of dense, compressed, VDS or frame-indexed HDF5 data.

    Only indexing/explicit NumPy conversion reads values. Creating a view never
    reads images. NumPy conversion is intentionally available for small legacy
    callers; dataset sampling and normalization always select rows first.
    """

    ndim = 2

    def __init__(self, path, files, shape, dtype, *, rows=None, columns=None):
        self.path, self.files = Path(path).resolve(), files
        self.source_shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.rows = range(shape[0]) if rows is None else rows
        self.columns = range(shape[1]) if columns is None else columns
        self.shape = (len(self.rows), len(self.columns))

    @classmethod
    def open(cls, path, files, source):
        if isinstance(source, h5py.Group):
            if source.attrs.get("layout") != LAYOUT:
                raise ValueError("Unsupported indexed observation layout")
            frames, indices, proprio = (source[k] for k in ("frames", "indices", "proprio"))
            if (
                frames.ndim != 2
                or indices.ndim != 2
                or proprio.shape != (len(indices), 18)
                or indices.dtype.kind not in "iu"
            ):
                raise ValueError("Invalid indexed observation shapes")
            width = frames.shape[1] + (source["coverage"].shape[1] if "coverage" in source else 0)
            shape = (len(indices), indices.shape[1] * width + 18)
            dtype = np.result_type(frames.dtype, proprio.dtype)
        else:
            shape, dtype = source.shape, source.dtype
            if len(shape) != 2:
                raise ValueError("Invalid observation matrix shape")
        return cls(path, files, shape, dtype)

    def __len__(self):
        return self.shape[0]

    def view(self, *, rows=None, columns=None):
        selected_rows = self.rows
        if rows is not None:
            selected_rows = (
                selected_rows[rows] if isinstance(rows, slice) else np.asarray(selected_rows)[rows]
            )
        selected_columns = self.columns if columns is None else self.columns[columns]
        return type(self)(
            self.path,
            self.files,
            self.source_shape,
            self.dtype,
            rows=selected_rows,
            columns=selected_columns,
        )

    def same_view(self, other):
        return (
            isinstance(other, ObservationArray)
            and self.files is other.files
            and self.path == other.path
            and np.array_equal(self.rows, other.rows)
            and self.columns == other.columns
        )

    def __getitem__(self, key):
        row, column = key if isinstance(key, tuple) else (key, slice(None))
        scalar = np.isscalar(row)
        rows = np.atleast_1d(np.asarray(self.rows)[row])
        source = self.files.observations(self.path)
        columns = self.columns[column]
        scalar_column = isinstance(columns, int)
        if scalar_column:
            columns = range(columns, columns + 1)
        selection = slice(columns.start, columns.stop, columns.step)
        if isinstance(source, h5py.Dataset):
            # Select columns before reading, especially for proprioception-only access.
            result = read_rows(source, rows, selection)
        else:
            width = source["frames"].shape[1] + (
                source["coverage"].shape[1] if "coverage" in source else 0
            )
            visual_width = source["indices"].shape[1] * width
            if columns.start >= visual_width:
                result = read_rows(source["proprio"], rows)[
                    :,
                    slice(columns.start - visual_width, columns.stop - visual_width, columns.step),
                ]
            else:
                indices = read_rows(source["indices"], rows)
                if (indices < 0).any() or (indices >= len(source["frames"])).any():
                    raise ValueError("Invalid observation frame index")
                # Deduplicate the active batch/history, never all possible windows.
                unique, inverse = np.unique(indices, return_inverse=True)
                frames = read_rows(source["frames"], unique)
                if "coverage" in source:
                    frames = np.concatenate((frames, read_rows(source["coverage"], unique)), axis=1)
                visual = frames[inverse].reshape(len(rows), visual_width)
                result = np.concatenate((visual, read_rows(source["proprio"], rows)), axis=1)
                result = result[:, selection]
        if scalar:
            result = result[0]
        if scalar_column:
            result = result[..., 0]
        return result

    def __array__(self, dtype=None, copy=None):
        result = self[:]
        return (
            np.asarray(result, dtype=dtype) if copy is not True else np.array(result, dtype=dtype)
        )


def finite_observations(values):
    """Scan with at most 8 MiB of visual rows plus the temporary finite mask."""
    if (
        isinstance(values, ObservationArray)
        and isinstance(values.rows, range)
        and values.rows == range(values.source_shape[0])
        and values.columns == range(values.source_shape[1])
    ):
        source = values.files.observations(values.path)
        if isinstance(source, dict):
            indices = source["indices"]
            for start in range(0, len(indices), 4096):
                block = indices[start : start + 4096]
                if (block < 0).any() or (block >= len(source["frames"])).any():
                    return False
            return all(
                finite_observations(source[key])
                for key in ("frames", "proprio", "coverage")
                if key in source
            )
    block = max(1, 8 * 1024**2 // (values.shape[1] * np.dtype(values.dtype).itemsize))
    return all(
        np.isfinite(values[start : start + block]).all() for start in range(0, len(values), block)
    )


class FrameObservationWriter:
    """Stream observations to disk, indexing repeated whole image blocks when possible.

    A bounded previous-frame cache matches exact values, including dtype. No
    encoder weights or learned features are cached. The caller's binding remains
    the sole preprocessing recipe. The temporary file lives as long as its view.
    """

    def __init__(self, n_rows, binding, *, dtype="f8"):
        self.temporary = tempfile.TemporaryDirectory(prefix="alexdoor-observations-")
        self.path = Path(self.temporary.name) / "episode.hdf5"
        self.file = h5py.File(self.path, "x")
        self.count = 0
        self.n_rows = n_rows
        width, recent = binding.config["visual_dims"]
        self.dense = None
        if recent % width:
            self.dense = self.file.create_dataset(
                "observations", shape=(n_rows, binding.obs_dim), dtype=dtype
            )
            return
        group = self.file.create_group("observations")
        group.attrs["layout"] = LAYOUT
        self.width, self.blocks = width, 1 + recent // width
        learned = binding.config.get("encoder") == "rgbd-conv-v1"
        image_width = width * 4 // 5 if learned else width
        self.coverage = (
            group.create_dataset(
                "coverage",
                shape=(0, width - image_width),
                maxshape=(None, width - image_width),
                dtype="f4",
                chunks=(1, width - image_width),
                compression="lzf",
            )
            if learned
            else None
        )
        self.frames = group.create_dataset(
            "frames",
            shape=(0, image_width),
            maxshape=(None, image_width),
            dtype="f4" if learned else dtype,
            chunks=(1, image_width),
            compression=None if learned else "lzf",
        )
        self.indices = group.create_dataset("indices", shape=(n_rows, self.blocks), dtype="i8")
        self.proprio = group.create_dataset("proprio", shape=(n_rows, 18), dtype=dtype)
        self.cache = []
        self.previous = self.previous_indices = None

    def append(self, features):
        if self.dense is not None:
            self.dense[self.count] = features
            self.count += 1
            return
        visual = features[:-18]
        if self.previous is not None and np.array_equal(
            visual.view("u1"), self.previous.view("u1")
        ):
            indices = self.previous_indices
        else:
            indices, cache = [], []
            for block in visual.reshape(self.blocks, self.width):
                index = next(
                    (
                        i
                        for i, old in self.cache + cache
                        if np.array_equal(block.view("u1"), old.view("u1"))
                    ),
                    None,
                )
                if index is None:
                    # RGBDEncoder produces float32 images. Use their exact native
                    # values; widen losslessly if another producer needs float64.
                    if self.frames.dtype.itemsize < block.dtype.itemsize and not np.array_equal(
                        block.astype(self.frames.dtype).astype(block.dtype), block
                    ):
                        group = self.file["observations"]
                        for key in ("frames", "coverage"):
                            if key not in group:
                                continue
                            old = group[key]
                            wide = group.create_dataset(
                                "wide_frames",
                                shape=old.shape,
                                maxshape=old.maxshape,
                                dtype=block.dtype,
                                chunks=old.chunks,
                                compression=old.compression,
                            )
                            for start in range(0, len(old), 32):
                                wide[start : start + 32] = old[start : start + 32]
                            del group[key]
                            group.move("wide_frames", key)
                        self.frames = group["frames"]
                        self.coverage = group.get("coverage")
                    index = len(self.frames)
                    self.frames.resize(index + 1, axis=0)
                    width = self.frames.shape[1]
                    self.frames[index] = block[:width]
                    if self.coverage is not None:
                        self.coverage.resize(index + 1, axis=0)
                        self.coverage[index] = block[width:]
                indices.append(index)
                cache.append((index, block.copy()))
            self.cache = cache
            self.previous, self.previous_indices = visual.copy(), indices
        self.indices[self.count] = indices
        self.proprio[self.count] = features[-18:]
        self.count += 1

    def finish(self):
        if self.count != self.n_rows:
            raise ValueError("Missing prepared observation rows")
        self.file.close()
        files = ObservationFiles(temporary=self.temporary)
        with h5py.File(self.path, "r") as source:
            return ObservationArray.open(self.path, files, source["observations"])

    def close(self):
        self.file.close()
