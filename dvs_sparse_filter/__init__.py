from __future__ import annotations
import cmaes
import os
import copy
import dataclasses
import event_stream
import dvs_sparse_filter_extension
import h5py
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.pyplot as plt
import pathlib
import numpy
from numpy.lib.stride_tricks import as_strided
import json
import cv2
from astropy.wcs import WCS
import astropy.units as u
from skimage.color import rgb2gray
import io
import matplotlib.cm as cm
from matplotlib.patches import RegularPolygon, Ellipse, Circle, Polygon
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageFilter
import scipy.optimize
import astrometry
import logging
from skimage.measure import label, regionprops
from scipy.ndimage import convolve
import typing
import torch
import yaml
import bisect
from scipy.signal import find_peaks
from skimage import measure
import random
from matplotlib.colors import to_rgba
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
import hdbscan
from sklearn.cluster import DBSCAN
from typing import Tuple, List
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
import plotly.graph_objects as go
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont, ImageOps
import colorsys
import time
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, average_precision_score, mean_squared_error
from math import exp, isfinite
from scipy.linalg import expm, lstsq
from collections import deque

# import cache
# import stars
import astropy
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.visualization.wcsaxes import WCSAxes

from astroquery.gaia import Gaia
astrometry.SolutionParameters

petroff_colors_6 = [
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
]
petroff_colors_8 = [
    "#1845fb",
    "#ff5e02",
    "#c91f16",
    "#c849a9",
    "#adad7d",
    "#86c8dd",
    "#578dff",
    "#656364",
]
petroff_colors_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]


def print_message(message, color='default', style='normal'):
    styles = {
        'default': '\033[0m',  # Reset to default
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    
    colors = {
        'default': '',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    
    print(f"{styles[style]}{colors[color]}{message}{styles['default']}")


def read_es_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    with event_stream.Decoder(path) as decoder:
        return (
            decoder.width,
            decoder.height,
            numpy.concatenate([packet for packet in decoder]),
        )


def write_es_file(
    path: typing.Union[pathlib.Path, str],
    event_type: str,
    width: int,
    height: int,
    events: numpy.ndarray
) -> None:
    with event_stream.Encoder(path, event_type, width, height) as encoder:
        encoder.write(events)

def read_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    data = numpy.asarray(h5py.File(path, "r")["/FalconNeuro"], dtype=numpy.uint32)
    events = numpy.zeros(data.shape[1], dtype=event_stream.dvs_dtype)
    events["t"] = data[3]
    events["x"] = data[0]
    events["y"] = data[1]
    events["on"] = data[2] == 1
    return numpy.max(events["x"].max()) + 1, numpy.max(events["y"]) + 1, events  # type: ignore


def read_es_or_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    if pathlib.Path(path).with_suffix(".es").is_file():
        return read_es_file(path=pathlib.Path(path).with_suffix(".es"))
    elif pathlib.Path(path).with_suffix(".h5").is_file():
        return read_h5_file(path=pathlib.Path(path).with_suffix(".h5"))
    raise Exception(
        f"neither \"{pathlib.Path(path).with_suffix('.es')}\" nor \"{pathlib.Path(path).with_suffix('.h5')}\" exist"
    )


@dataclasses.dataclass
class RotatedEvents:
    eventsrot: numpy.ndarray

@dataclasses.dataclass
class CumulativeMap:
    pixels: numpy.ndarray


def remove_hot_pixels_percentile(events: numpy.ndarray, ratio: float):
    assert 0.0 <= ratio <= 1.0
    count = numpy.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    numpy.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    threshold = numpy.percentile(count, 100.0 * (1.0 - ratio))
    mask = count[events["x"], events["y"]] > threshold
    labels = numpy.zeros(len(events["x"]), dtype=int)
    labels[mask] = 1
    filtered_events = events[~mask]
    return labels


def without_most_active_pixels(events: numpy.ndarray, ratio: float):
    assert ratio >= 0.0 and ratio <= 1.0
    count = numpy.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    numpy.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    return events[count[events["x"], events["y"]]<= numpy.percentile(count, 100.0 * (1.0 - ratio))]

def with_most_active_pixels(events: numpy.ndarray):
    return events[events["x"], events["y"]]

# velocity in px/us
def warp(events: numpy.ndarray, velocity: tuple[float, float]):
    warped_events = numpy.array(
        events, dtype=[("t", "<u8"), ("x", "<f8"), ("y", "<f8"), ("on", "?")]
    )
    warped_events["x"] -= velocity[0] * warped_events["t"]
    warped_events["y"] -= velocity[1] * warped_events["t"]
    warped_events["x"] = numpy.round(warped_events["x"])
    warped_events["y"] = numpy.round(warped_events["y"])
    return warped_events

def unwarp(warped_events: numpy.ndarray, velocity: tuple[float, float]):
    events = numpy.zeros(
        len(warped_events),
        dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")],
    )
    events["t"] = warped_events["t"]
    events["x"] = numpy.round(
        warped_events["x"] + velocity[0] * warped_events["t"]
    ).astype("<u2")
    events["y"] = numpy.round(
        warped_events["y"] + velocity[1] * warped_events["t"]
    ).astype("<u2")
    events["on"] = warped_events["on"]
    return events



def smooth_histogram(warped_events: numpy.ndarray):
    return dvs_sparse_filter_extension.smooth_histogram(warped_events)

def accumulate(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return CumulativeMap(
        pixels=dvs_sparse_filter_extension.accumulate(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
        ),
        offset=0
    )

def accumulate_timesurface(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
    tau: int,
):
    return CumulativeMap(
        pixels=dvs_sparse_filter_extension.accumulate_timesurface(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
            tau,
        ),
        offset=0
    )

def accumulate_pixel_map(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    accumulated_pixels, event_indices_list = dvs_sparse_filter_extension.accumulate_pixel_map(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )
    
    # Convert event_indices_list to a numpy array if needed
    event_indices_np = numpy.array(event_indices_list, dtype=object)

    return {
        'cumulative_map': CumulativeMap(
            pixels=accumulated_pixels,
            offset=0
        ),
        'event_indices': event_indices_np
    }


def accumulate_cnt(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return CumulativeMap(
        pixels=dvs_sparse_filter_extension.accumulate_cnt(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
        ),
        offset=0
    )

def accumulate_cnt_rgb(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    label: numpy.ndarray,
    velocity: tuple[float, float],
):
    accumulated_pixels, label_image = dvs_sparse_filter_extension.accumulate_cnt_rgb(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        label,  # Assuming 'l' labels are 32-bit integers
        velocity[0],
        velocity[1],
    )
    return CumulativeMap(
        pixels=accumulated_pixels,
        offset=0
    ), label_image

class CumulativeMap:
    def __init__(self, pixels, offset=0):
        self.pixels = pixels
        self.offset = offset


#####################################################################
DEFAULT_TIMESTAMP = -1
class STDFFilter:
    def __init__(self, chip_size, num_must_be_correlated=3, dt=1e3, subsample_by=0, let_first_event_through=True, filter_hot_pixels=True):
        self.sx, self.sy = chip_size
        self.num_must_be_correlated = num_must_be_correlated
        self.dt = dt
        self.subsample_by = subsample_by
        self.let_first_event_through = let_first_event_through
        self.filter_hot_pixels = filter_hot_pixels
        self.timestamp_image = numpy.full((self.sx, self.sy), DEFAULT_TIMESTAMP)
        self.density_matrix = numpy.zeros((self.sx, self.sy, 8), dtype=int)
        self.total_event_count = 0

    def filter_packet(self, events):
        boolean_mask = numpy.zeros(len(events), dtype=bool)
        if self.timestamp_image is None:
            self.allocate_maps()

        filtered_events = []
        for i, e in enumerate(events):
            self.total_event_count += 1
            ts = e['t']
            x, y = e['x'] >> self.subsample_by, e['y'] >> self.subsample_by

            if x < 0 or x >= self.sx or y < 0 or y >= self.sy:
                continue

            if self.timestamp_image[x, y] == DEFAULT_TIMESTAMP:
                self.timestamp_image[x, y] = ts
                if self.let_first_event_through:
                    filtered_events.append(e)
                continue

            ncorrelated = 0
            for eid in range(7, 0, -1):
                self.density_matrix[x, y, eid] = self.density_matrix[x, y, eid - 1]

            self.density_matrix[x, y, 0] = ts
            for eid in range(1, 8):
                eid_ts = self.density_matrix[x, y, eid]
                delta_t = ts - eid_ts
                if delta_t > self.dt:
                    self.density_matrix[x, y, eid] = 0

            for xx in range(x - 2, x + 3):
                for yy in range(y - 2, y + 3):
                    if xx < 0 or xx >= self.sx or yy < 0 or yy >= self.sy:
                        continue
                    if self.filter_hot_pixels and xx == x and yy == y:
                        continue

                    last_t = self.timestamp_image[xx, yy]
                    delta_t = ts - last_t
                    if delta_t < self.dt and last_t != DEFAULT_TIMESTAMP:
                        ncorrelated += 1
                        for k in range(1, 8):
                            last_t2 = self.density_matrix[xx, yy, k]
                            if last_t2 == 0:
                                break
                            delta_t2 = ts - last_t2
                            if delta_t2 < self.dt:
                                ncorrelated += 1
                                if ncorrelated >= self.num_must_be_correlated:
                                    break

            if ncorrelated >= self.num_must_be_correlated:
                filtered_events.append(e)
            else:
                boolean_mask[i] = True
            self.timestamp_image[x, y] = ts

        filtered_events = numpy.array(filtered_events, dtype=events.dtype)
        return boolean_mask, filtered_events

    def reset_filter(self):
        self.timestamp_image.fill(DEFAULT_TIMESTAMP)
        self.density_matrix.fill(0)

    def init_filter(self, chip_size):
        self.sx, self.sy = chip_size
        self.reset_filter()

    def allocate_maps(self):
        self.timestamp_image = numpy.full((self.sx, self.sy), DEFAULT_TIMESTAMP)
        self.density_matrix = numpy.zeros((self.sx, self.sy, 8), dtype=int)

    def initialize_last_times_map_for_noise_rate(self, noise_rate_hz, last_timestamp_us):
        random.seed()
        for array_row in range(self.sx):
            for i in range(self.sy):
                p = random.random()
                t = -noise_rate_hz * numpy.log(1 - p)
                t_us = int(1000000 * t)
                self.timestamp_image[array_row, i] = last_timestamp_us - t_us

    def set_num_must_be_correlated(self, num_must_be_correlated):
        self.num_must_be_correlated = max(1, min(num_must_be_correlated, 9))



class InceptiveFilter:
    def __init__(self, multiTriggerWindow):
        self.multiTriggerWindow = multiTriggerWindow

    def find_ie(self, idx, ts, p):
        idx = numpy.sort(idx)
        idxp = numpy.where(p[idx] > 0)[0]
        idxn = numpy.where(p[idx] <= 0)[0]

        ie_idxp, ie_idxn = [], []

        if idxp.size > 0:
            multiEventp = numpy.concatenate(([False], numpy.diff(ts[idx[idxp]]) <= self.multiTriggerWindow))
            ieIdxp = numpy.diff(numpy.concatenate(([0], multiEventp))) == 1
            idxp = idxp[ieIdxp]

        if idxn.size > 0:
            multiEventn = numpy.concatenate(([False], numpy.diff(ts[idx[idxn]]) <= self.multiTriggerWindow))
            ieIdxn = numpy.diff(numpy.concatenate(([0], multiEventn))) == 1
            idxn = idxn[ieIdxn]

        return numpy.concatenate((idx[idxp], idx[idxn]))

    def find_te(self, idx, ts, p):
        idx = numpy.sort(idx)
        idxp = numpy.where(p[idx] > 0)[0]
        idxn = numpy.where(p[idx] <= 0)[0]

        te_idxp, te_idxn = [], []

        if idxp.size > 0:
            multiEventp = numpy.concatenate(([False], numpy.diff(ts[idx[idxp]]) <= self.multiTriggerWindow))
            teIdxp = numpy.abs(numpy.diff(numpy.concatenate(([0], multiEventp)))) == 1
            idxp = idxp[teIdxp]

        if idxn.size > 0:
            multiEventn = numpy.concatenate(([False], numpy.diff(ts[idx[idxn]]) <= self.multiTriggerWindow))
            teIdxn = numpy.abs(numpy.diff(numpy.concatenate(([0], multiEventn)))) == 1
            idxn = idxn[teIdxn]

        return numpy.concatenate((idx[idxp], idx[idxn]))

    def ie(self, x, y, ts, p, sensorDim):
        numEvents = len(x)
        hasDataIdx = numpy.ravel_multi_index((y, x), sensorDim)
        N = numpy.arange(numEvents)

        isIE = numpy.zeros(numEvents, dtype=bool)
        unique_indices, inverse_indices = numpy.unique(hasDataIdx, return_inverse=True)
        for i in range(len(unique_indices)):
            idx = N[inverse_indices == i]
            if len(idx) > 0:
                ie_result = self.find_ie(idx, ts, p)
                if len(ie_result) > 0:
                    isIE[ie_result.astype(int)] = True

        return isIE

    def te(self, x, y, ts, p, sensorDim):
        numEvents = len(x)
        hasDataIdx = numpy.ravel_multi_index((y, x), sensorDim)
        N = numpy.arange(numEvents)

        isTE = numpy.zeros(numEvents, dtype=bool)
        unique_indices, inverse_indices = numpy.unique(hasDataIdx, return_inverse=True)
        for i in range(len(unique_indices)):
            idx = N[inverse_indices == i]
            if len(idx) > 0:
                te_result = self.find_te(idx, ts, p)
                if len(te_result) > 0:
                    isTE[te_result.astype(int)] = True

        return isTE

    def filter(self, td):
        x = td['x'].flatten()
        y = td['y'].flatten()
        p = td['on'].flatten()
        p[p == 0] = 1
        t = td['t'].flatten()
        t = t - t[0]

        sensorDim = [numpy.max(y) + 1, numpy.max(x) + 1]

        isIE = self.ie(x, y, t, p, sensorDim)
        isTE = self.te(x, y, t, p, sensorDim)

        isNoise = numpy.logical_not(isIE) & numpy.logical_not(isTE)
        return isNoise


class EventFlowFilter:
    def __init__(self, resolution, duration=1e5, search_radius=3, float_threshold=20.0):
        self.mWidth = resolution[0]
        self.mHeight = resolution[1]
        self.mDuration = duration
        self.mSearchRadius = search_radius
        self.mFloatThreshold = float_threshold

        self.mDeque = deque()

    def fit_event_flow(self, event):
        # Default flow value will be infinity
        flow = numpy.inf
        
        # Search spatio-temporal related events
        candidate_events = []
        for deque_event in self.mDeque:
            if (abs(int(event['x']) - int(deque_event['x'])) <= self.mSearchRadius) and \
               (abs(int(event['y']) - int(deque_event['y'])) <= self.mSearchRadius):
                candidate_events.append(deque_event)

        # Calculate flow
        if len(candidate_events) > 3:
            A = numpy.zeros((len(candidate_events), 3))
            b = numpy.zeros(len(candidate_events))
            for i, candidate_event in enumerate(candidate_events):
                A[i, 0] = candidate_event['x']
                A[i, 1] = candidate_event['y']
                A[i, 2] = 1.0
                b[i] = (int(candidate_event['t']) - int(event['t'])) * 1E-3

            # Solve
            X, residuals, rank, s = lstsq(A, b)
            if X[0] != 0 and X[1] != 0:  # Avoid division by zero
                flow = numpy.sqrt((1 / X[0]) ** 2 + (1 / X[1]) ** 2)

        return flow

    def evaluate(self, event):
        # Calculate density in spatio-temporal neighborhood
        flow = self.fit_event_flow(event)

        # Evaluate
        is_stars = (flow <= self.mFloatThreshold)

        # Update deque
        while self.mDeque and (int(event['t']) - int(self.mDeque[0]['t']) >= self.mDuration):
            self.mDeque.popleft()
        self.mDeque.append(event)

        return is_stars

    def retain(self, event):
        return self.evaluate(event)

    def accept(self, events):
        boolean_mask = numpy.zeros(len(events), dtype=bool)
        filtered_events = []

        for i, event in enumerate(events):
            if self.retain(event):
                filtered_events.append(event)
            else:
                boolean_mask[i] = True

        filtered_events = numpy.array(filtered_events, dtype=events.dtype)
        return boolean_mask, filtered_events

    def __lshift__(self, events):
        return self.accept(events)


class KhodamoradiNoiseFilter:
    def __init__(self, resolution, duration=1e5, int_threshold=1):
        self.mWidth = resolution[0]
        self.mHeight = resolution[1]
        self.mDuration = duration
        self.mIntThreshold = int_threshold
        self.intenrenr = 0

        self.xCols = numpy.zeros(self.mWidth, dtype=[('t', 'int64'), ('x', 'int16'), ('y', 'int16'), ('p', 'bool')])
        self.yRows = numpy.zeros(self.mHeight, dtype=[('t', 'int64'), ('x', 'int16'), ('y', 'int16'), ('p', 'bool')])
        
    def search_correlation(self, event):
        support = 0
        x, y, t, p = event['x'], event['y'], event['t'], event['on']
        x_minus_one = (x > 0)
        x_plus_one = (x < (self.mWidth - 1))
        y_minus_one = (y > 0)
        y_plus_one = (y < (self.mHeight - 1))

        if x_minus_one:
            x_prev = self.xCols[x - 1]
            if (t - x_prev['t']) <= self.mDuration and p == x_prev['p']:
                if ((y_minus_one and x_prev['y'] == y - 1) or x_prev['y'] == y or (y_plus_one and x_prev['y'] == y + 1)):
                    support += 1

        x_cell = self.xCols[x]
        if (t - x_cell['t']) <= self.mDuration and p == x_cell['p']:
            if ((y_minus_one and x_cell['y'] == y - 1) or (y_plus_one and x_cell['y'] == y + 1)):
                support += 1

        if x_plus_one:
            x_next = self.xCols[x + 1]
            if (t - x_next['t']) <= self.mDuration and p == x_next['p']:
                if ((y_minus_one and x_next['y'] == y - 1) or x_next['y'] == y or (y_plus_one and x_next['y'] == y + 1)):
                    support += 1

        if y_minus_one:
            y_prev = self.yRows[y - 1]
            if (t - y_prev['t']) <= self.mDuration and p == y_prev['p']:
                if ((x_minus_one and y_prev['x'] == x - 1) or y_prev['x'] == x or (x_plus_one and y_prev['x'] == x + 1)):
                    support += 1

        y_cell = self.yRows[y]
        if (t - y_cell['t']) <= self.mDuration and p == y_cell['p']:
            if ((x_minus_one and y_cell['x'] == x - 1) or (x_plus_one and y_cell['x'] == x + 1)):
                support += 1

        if y_plus_one:
            y_next = self.yRows[y + 1]
            if (t - y_next['t']) <= self.mDuration and p == y_next['p']:
                if ((x_minus_one and y_next['x'] == x - 1) or y_next['x'] == x or (x_plus_one and y_next['x'] == x + 1)):
                    support += 1

        return support
    
    def evaluate(self, event):
        support = self.search_correlation(event)
        
        # if support > 2:
        #     self.intenrenr += 1
        #     print(support)
        # print(self.intenrenr)

        is_stars = (support >= self.mIntThreshold)
        target_dtype = numpy.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')])
        new_event = numpy.zeros(1, dtype=target_dtype)
        for field in target_dtype.names:
            new_event[field] = event[field]

        self.xCols[event['x']] = new_event[0]
        self.yRows[event['y']] = new_event[0]

        return is_stars

    def retain(self, event):
        return self.evaluate(event)

    def accept(self, events):
        boolean_mask = numpy.zeros(len(events), dtype=bool)
        filtered_events = []

        for i, event in enumerate(events):
            if self.retain(event):
                filtered_events.append(event)
            else:
                boolean_mask[i] = True

        filtered_events = numpy.array(filtered_events, dtype=events.dtype)
        return boolean_mask, filtered_events

    def __lshift__(self, events):
        return self.accept(events)


class YangNoiseFilter:
    def __init__(self, resolution, duration=2000, search_radius=1, int_threshold=1):
        self.mWidth = resolution[0]
        self.mHeight = resolution[1]
        self.mDuration = duration
        self.mSearchRadius = search_radius
        self.mIntThreshold = int_threshold

        self.mOffsets = self._initialize_offsets(search_radius)
        self.mTimestampMat = numpy.zeros((self.mWidth, self.mHeight), dtype=numpy.int64)
        self.mPolarityMat = numpy.zeros((self.mWidth, self.mHeight), dtype=numpy.uint8)

    def _initialize_offsets(self, radius):
        offsets = []
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x != 0 or y != 0:
                    offsets.append((x, y))
        return offsets

    def calculate_density(self, event):
        density = 0

        for delta in self.mOffsets:
            x = event['x'] + delta[0]
            y = event['y'] + delta[1]

            if x < 0 or y < 0 or x >= self.mWidth or y >= self.mHeight:
                continue

            if event['t'] - self.mTimestampMat[x, y] > self.mDuration:
                continue

            if event['on'] != self.mPolarityMat[x, y]:
                continue

            density += 1

        return density

    def evaluate(self, event):
        density = self.calculate_density(event)
        is_stars = density >= self.mIntThreshold

        self.mTimestampMat[event['x'], event['y']] = event['t']
        self.mPolarityMat[event['x'], event['y']] = event['on']

        return is_stars

    def retain(self, event):
        return self.evaluate(event)

    def accept(self, events):
        boolean_mask = numpy.zeros(len(events), dtype=bool)
        filtered_events = []

        for i, event in enumerate(events):
            if self.retain(event):
                filtered_events.append(event)
            else:
                boolean_mask[i] = True

        filtered_events = numpy.array(filtered_events, dtype=events.dtype)
        return boolean_mask, filtered_events

    def __lshift__(self, events):
        return self.accept(events)


class TimeSurfaceFilter:
    def __init__(self, resolution, decay, search_radius, float_threshold):
        self.mWidth = resolution[0]
        self.mHeight = resolution[1]
        self.mDecay = decay
        self.mSearchRadius = search_radius
        self.mFloatThreshold = float_threshold
        self.mOffsets = self._initialize_offsets(search_radius)
        
        event_dtype = numpy.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')])
        self.mPos = numpy.zeros((self.mWidth, self.mHeight), dtype=event_dtype)
        self.mNeg = numpy.zeros((self.mWidth, self.mHeight), dtype=event_dtype)

    def _initialize_offsets(self, radius):
        offsets = []
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x != 0 or y != 0:
                    offsets.append((x, y))
        return offsets

    def fit_time_surface(self, event):
        support = 0
        diff_time = 0
        
        cell = self.mPos if event['on'] else self.mNeg
        for delta in self.mOffsets:
            x = event['x'] + delta[0]
            y = event['y'] + delta[1]

            if x < 0 or y < 0 or x >= self.mWidth or y >= self.mHeight:
                continue

            if cell[x, y]['t'] == 0:
                continue

            time_diff = (numpy.int64(cell[x, y]['t']) - numpy.int64(event['t'])) / self.mDecay
            if isfinite(time_diff) and -700 < time_diff < 700:  # Ensure time_diff is within a reasonable range for exp()
                diff_time += exp(time_diff)
                support += 1

        surface = 0 if support == 0 else diff_time / support
        return surface

    def evaluate(self, event):
        surface = self.fit_time_surface(event)
        is_stars = surface >= self.mFloatThreshold
        target_dtype = numpy.dtype([('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')])
        new_event = numpy.zeros(1, dtype=target_dtype)
        for field in target_dtype.names:
            new_event[field] = event[field]

        if event['on']:
            self.mPos[event['x'], event['y']] = new_event[0]
        else:
            self.mNeg[event['x'], event['y']] = new_event[0]

        return is_stars

    def retain(self, event):
        return self.evaluate(event)

    def accept(self, events):
        filtered_events = []
        boolean_mask = numpy.zeros(len(events), dtype=bool)
        for i, event in enumerate(events):
            if self.retain(event):
                filtered_events.append(event)
            else:
                boolean_mask[i] = True
        return boolean_mask, numpy.array(filtered_events, dtype=events.dtype)

    def __lshift__(self, events):
        return self.accept(events)


class DoubleFixedWindowFilter:
    def __init__(self, sx, sy, wlen, disThr, useDoubleMode, numMustBeCorrelated):
        self.sx = sx
        self.sy = sy
        self.wlen = wlen
        self.disThr = disThr
        self.useDoubleMode = useDoubleMode
        self.numMustBeCorrelated = numMustBeCorrelated
        self.subsampleBy = 0
        self.lastREvents = numpy.full((wlen, 2), -1, dtype=numpy.int64)
        self.lastNEvents = numpy.full((wlen, 2), -1, dtype=numpy.int64) if useDoubleMode else None
        self.fillindex = 0
        self.ts = 0

    def filter_packet(self, events):
        boolean_mask = numpy.zeros(len(events), dtype=bool)
        filtered_events = []
        for idx, e in enumerate(events):
            self.ts = e['t']
            x = e['x'] >> self.subsampleBy
            y = e['y'] >> self.subsampleBy

            if x < 0 or x > self.sx or y < 0 or y > self.sy:
                continue

            if self.fillindex < self.wlen and self.lastREvents[self.wlen - 1][0] == -1:
                self.lastREvents[self.fillindex][0] = e['x']
                self.lastREvents[self.fillindex][1] = e['y']
                self.fillindex += 1
                continue

            ncorrelated = 0

            if self.useDoubleMode:
                dwlen = self.wlen // 2 if self.wlen > 1 else 1
                noiseflag = True

                for i in range(dwlen):
                    dis = abs(numpy.int64(e['x']) - self.lastREvents[i][0]) + abs(numpy.int64(e['y']) - self.lastREvents[i][1])
                    if dis < self.disThr:
                        ncorrelated += 1
                        if ncorrelated == self.numMustBeCorrelated:
                            noiseflag = False
                            break

                if ncorrelated < self.numMustBeCorrelated:
                    for i in range(dwlen):
                        dis = abs(numpy.int64(e['x']) - self.lastNEvents[i][0]) + abs(numpy.int64(e['y']) - self.lastNEvents[i][1])
                        if dis < self.disThr:
                            ncorrelated += 1
                            if ncorrelated == self.numMustBeCorrelated:
                                noiseflag = False
                                break

                if not noiseflag:
                    filtered_events.append(e)
                    for i in range(dwlen - 1):
                        self.lastREvents[i][0] = self.lastREvents[i + 1][0]
                        self.lastREvents[i][1] = self.lastREvents[i + 1][1]
                    self.lastREvents[dwlen - 1][0] = e['x']
                    self.lastREvents[dwlen - 1][1] = e['y']
                else:
                    boolean_mask[idx] = True
                    for i in range(dwlen - 1):
                        self.lastNEvents[i][0] = self.lastNEvents[i + 1][0]
                        self.lastNEvents[i][1] = self.lastNEvents[i + 1][1]
                    self.lastNEvents[dwlen - 1][0] = e['x']
                    self.lastNEvents[dwlen - 1][1] = e['y']
            else:
                noiseflag = True
                for i in range(self.wlen):
                    dis = abs(numpy.int64(e['x']) - self.lastREvents[i][0]) + abs(numpy.int64(e['y']) - self.lastREvents[i][1])
                    if dis < self.disThr:
                        ncorrelated += 1
                        if ncorrelated == self.numMustBeCorrelated:
                            noiseflag = False
                            break

                if not noiseflag:
                    filtered_events.append(e)
                else:
                    boolean_mask[idx] = True

                for i in range(self.wlen - 1):
                    self.lastREvents[i][0] = self.lastREvents[i + 1][0]
                    self.lastREvents[i][1] = self.lastREvents[i + 1][1]
                self.lastREvents[self.wlen - 1][0] = e['x']
                self.lastREvents[self.wlen - 1][1] = e['y']

        filtered_events_array = numpy.array(filtered_events, dtype=events.dtype)
        return boolean_mask, filtered_events_array



DEFAULT_TIMESTAMP = -1
class SpatioTemporalCorrelationFilter:
    def __init__(self, size_x, size_y, subsample_by=1, num_must_be_correlated=2, shot_noise_correlation_time_s=0.1):
        self.num_must_be_correlated = num_must_be_correlated  # k (constant)
        self.shot_noise_correlation_time_s = shot_noise_correlation_time_s  # tau (variable for ROC)
        self.filter_alternative_polarity_shot_noise_enabled = False
        self.subsample_by = subsample_by
        self.size_x = size_x
        self.size_y = size_y
        self.sxm1 = size_x - 1
        self.sym1 = size_y - 1
        self.ssx = self.sxm1 >> subsample_by
        self.ssy = self.sym1 >> subsample_by
        self.timestamp_image = numpy.full((self.ssx + 1, self.ssy + 1), DEFAULT_TIMESTAMP, dtype=numpy.int32)
        self.pol_image = numpy.zeros((self.ssx + 1, self.ssy + 1), dtype=numpy.int8)
        self.reset_shot_noise_test_stats()

    def reset_shot_noise_test_stats(self):
        self.num_shot_noise_tests = 0
        self.num_alternating_polarity_shot_noise_events_filtered_out = 0

    def reset_filter(self):
        self.timestamp_image.fill(DEFAULT_TIMESTAMP)
        self.reset_shot_noise_test_stats()

    def filter_packet(self, events):
        boolean_mask = numpy.ones(len(events), dtype=bool)
        dt = int(round(self.shot_noise_correlation_time_s * 1e6))
        filtered_events = []
        for i, event in enumerate(events):
            ts, x, y, on = event["t"], event["x"], event["y"], event["on"]
            x >>= self.subsample_by
            y >>= self.subsample_by
            if not (0 <= x <= self.ssx and 0 <= y <= self.ssy):
                boolean_mask[i] = False #True
                continue
            if self.timestamp_image[x, y] == DEFAULT_TIMESTAMP:
                self.store_timestamp_polarity(x, y, ts, on)
                boolean_mask[i] = False #True
                continue
            ncorrelated = self.count_correlated_events(x, y, ts, dt)
            if ncorrelated < self.num_must_be_correlated:
                boolean_mask[i] = False #True
                continue
            if self.filter_alternative_polarity_shot_noise_enabled and self.test_filter_out_shot_noise_opposite_polarity(x, y, ts, on):
                boolean_mask[i] = False #True
                continue
            filtered_events.append(event)
            self.store_timestamp_polarity(x, y, ts, on)

        filtered_events_array = numpy.array(filtered_events, dtype=events.dtype)
        return boolean_mask, filtered_events_array

    def count_correlated_events(self, x, y, ts, dt):
        ncorrelated = 0
        for xx in range(max(0, x - 1), min(self.ssx, x + 1) + 1):
            for yy in range(max(0, y - 1), min(self.ssy, y + 1) + 1):
                if xx == x and yy == y:
                    continue
                last_ts = self.timestamp_image[xx, yy]
                if 0 <= ts - last_ts < dt:
                    ncorrelated += 1
        return ncorrelated

    def store_timestamp_polarity(self, x, y, ts, on):
        self.timestamp_image[x, y] = ts
        self.pol_image[x, y] = 1 if on else -1

    def test_filter_out_shot_noise_opposite_polarity(self, x, y, ts, on):
        prev_ts = self.timestamp_image[x, y]
        prev_pol = self.pol_image[x, y]
        if on == (prev_pol == 1):
            return False
        dt = ts - prev_ts
        if dt > self.shot_noise_correlation_time_s * 1e6:
            return False
        self.num_alternating_polarity_shot_noise_events_filtered_out += 1
        return True


def plot_roc_curve(fpr, tpr, filter_method, recording_name):
    plt.figure()
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, color='b', linewidth=1, marker='o', markersize=5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xticks([i * 0.1 for i in range(11)])
    plt.yticks([i * 0.1 for i in range(11)])
    plt.savefig(f"/media/samiarja/VERBATIM HD/Denoising/{filter_method}/{recording_name}/{filter_method}_roc_curve.png")
    # plt.show()


def roc_val(input_events, detected_noise, ground_truth):
    """
    Calculate the true positive (TP), true negative (TN), false positive (FP), and false negative (FN) rates.

    Parameters:
    detected_noise (numpy.array or list): Binary array or list where:
        - 0 indicates noise
        - 1 indicates signal
    ground_truth (numpy.array or list): Binary array or list where:
        - 0 indicates noise
        - 1 indicates signal

    Returns:
    tuple: Normalized values of TP, TN, FP, and FN in the form (TP, TN, FP, FN)
    """
    detected_noise  = numpy.array(detected_noise)
    ground_truth    = numpy.array(ground_truth)

    # TP: Signal correctly labeled as signal
    TP = numpy.sum((detected_noise == 0) & (ground_truth == 1))
    # TN: Noise correctly labeled as noise
    TN = numpy.sum((detected_noise == 1) & (ground_truth == 0))
    # FP: Noise incorrectly labeled as signal
    FP = numpy.sum((detected_noise == 0) & (ground_truth == 0))
    # FN: Signal incorrectly labeled as noise
    FN = numpy.sum((detected_noise == 1) & (ground_truth == 1))
    
    total = len(ground_truth)
    normalized_TP = TP / total
    normalized_TN = TN / total
    normalized_FP = FP / total
    normalized_FN = FN / total
    
    
    HP_idx    = numpy.where(numpy.logical_or(ground_truth==2,ground_truth==1)) #only hot pixels
    HP_no_idx = numpy.where(numpy.logical_or(ground_truth==0,ground_truth==1))[0] #only noise
    
    # Compute precision, recall, and F1-score
    groundtruth_noise = ground_truth[HP_no_idx]
    actually_detected_noise = detected_noise[HP_no_idx]
    precision_noise   = 100*(precision_score(groundtruth_noise, actually_detected_noise, pos_label=1))
    recall_noise      = 100*(recall_score(groundtruth_noise, actually_detected_noise, pos_label=1))
    f1_noise          = 100*(f1_score(groundtruth_noise, actually_detected_noise, pos_label=1))
    
    hot_pixel_gt = ground_truth[HP_idx]
    hot_pixel_gt[hot_pixel_gt==2] = 0
    detected_hot_pixels = detected_noise[HP_idx]
    precision_hp   = 100*(precision_score(hot_pixel_gt, detected_hot_pixels, pos_label=1))
    recall_hp      = 100*(recall_score(hot_pixel_gt, detected_hot_pixels, pos_label=1))
    f1_hp          = 100*(f1_score(hot_pixel_gt, detected_hot_pixels, pos_label=1))
    
    signal_intersection     = numpy.where(numpy.logical_and(detected_noise == 1,ground_truth == 1))
    noise_intersection      = numpy.where(numpy.logical_and(detected_noise == 0,ground_truth == 0))
    hot_pixel_intersection  = numpy.where(numpy.logical_and(detected_noise == 0,ground_truth == 2))

    SR = 100*(len(signal_intersection[0])/len(numpy.where(ground_truth == 1)[0]))
    NR = 100*(len(noise_intersection[0])/len(numpy.where(ground_truth == 0)[0]))
    HPR = 100*(len(hot_pixel_intersection[0])/len(numpy.where(ground_truth == 2)[0]))
    DA = (SR+NR)/2
    HDA = (SR+NR+HPR)/3
    
    print(f"SR: {SR:.2f}% NR: {NR:.2f}% HPR: {HPR:.3f}% DA: {DA:.2f}% HDA: {HDA:.3f}%")

    return precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA

def CrossConv_HotPixelFilter(input_events,ratio_par):
    print("Start removing hot pixels...")

    x = input_events['x']
    y = input_events['y']
    x_max, y_max = x.max() + 1, y.max() + 1

    # Create a 2D histogram of event counts
    event_count = numpy.zeros((x_max, y_max), dtype=int)
    numpy.add.at(event_count, (x, y), 1)

    kernels = [
        numpy.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    ]

    # Apply convolution with each kernel and calculate ratios
    shifted = [convolve(event_count, k, mode="constant", cval=0.0) for k in kernels]
    max_shifted = numpy.maximum.reduce(shifted)
    ratios = event_count / (max_shifted + 1.0)
    smart_mask = ratios < ratio_par #this should be 2 ideally

    yhot, xhot      = numpy.where(~smart_mask)
    label_hotpix    = numpy.zeros(len(input_events), dtype=bool)
    for xi, yi in zip(xhot, yhot):
        label_hotpix |= (y == xi) & (x == yi)
    label_hotpix_binary = label_hotpix.astype(int)

    jj              = numpy.where(label_hotpix_binary==0)[0]
    output_events   = input_events[jj]
    
    print(f'Number of input event: {len(input_events["x"])}')
    print(f'Number of hot pixel events: {numpy.sum(label_hotpix)}')
    
    detected_noise  = label_hotpix_binary
    detected_noise  = 1 - detected_noise
    return output_events, detected_noise

def CrossConv(input_events, ground_truth, time_window):
    print("Start processing CrossConv filter...")
    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]

    x = input_events['x']
    y = input_events['y']
    x_max, y_max = x.max() + 1, y.max() + 1

    # Create a 2D histogram of event counts
    event_count = numpy.zeros((x_max, y_max), dtype=int)
    numpy.add.at(event_count, (x, y), 1)

    kernels = [
        numpy.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    ]

    # Apply convolution with each kernel and calculate ratios
    shifted = [convolve(event_count, k, mode="constant", cval=0.0) for k in kernels]
    max_shifted = numpy.maximum.reduce(shifted)
    ratios = event_count / (max_shifted + 1.0)
    smart_mask = ratios < 2.0 #this should be 2 ideally

    yhot, xhot      = numpy.where(~smart_mask)
    label_hotpix    = numpy.zeros(len(input_events), dtype=bool)
    for xi, yi in zip(xhot, yhot):
        label_hotpix |= (y == xi) & (x == yi)
    label_hotpix_binary = label_hotpix.astype(int)

    print(f'Number of input event: {len(input_events["x"])}')

    jj              = numpy.where(label_hotpix_binary==0)[0]
    output_events   = input_events[jj]

    print(f'Number of detected noise event: {numpy.sum(label_hotpix)}')

    detected_noise  = label_hotpix_binary
    detected_noise  = 1 - detected_noise
    ground_truth    = ground_truth[ii]
    
    ###################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # jj = numpy.where(detected_noise==0)[0]
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[jj],
    #                                                       detected_noise[jj].astype(numpy.int32),
    #                                                       (vx_velocity_raw[jj],vy_velocity_raw[jj]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # zz = numpy.where(detected_noise==1)[0]
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[zz],
    #                                                         detected_noise[zz].astype(numpy.int32),
    #                                                         (vx_velocity_raw[zz],vy_velocity_raw[zz]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    ##################################################################################

    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA  = roc_val(input_events, detected_noise, ground_truth)
    performance     = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise

def STCF(input_events, ground_truth, time_window):
    print("Start processing STCF filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    print(f'Number of input event: {len(input_events["x"])}')

    filter_instance             = SpatioTemporalCorrelationFilter(size_x=x_max, size_y=y_max, num_must_be_correlated=2, shot_noise_correlation_time_s=0.1)
    boolean_mask, output_events = filter_instance.filter_packet(input_events)

    print(f'Number of detected noise event: {len(input_events) - len(output_events)}')
    detected_noise = boolean_mask.astype(int)
    
    ###################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    ###################################################################################


    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise


def DFWF(input_events, ground_truth, time_window):
    print("Start processing DWF/FWF filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1
    
    print(f'Number of input event: {len(input_events["x"])}')

    filter_instance             = DoubleFixedWindowFilter(sx=x_max, 
                                                          sy=y_max,
                                                          wlen=175, 
                                                          disThr=100, 
                                                          useDoubleMode=False, 
                                                          numMustBeCorrelated=8)
    boolean_mask, output_events = filter_instance.filter_packet(input_events)

    print(f'Number of detected noise event: {len(input_events) - len(output_events)}')
    detected_noise = boolean_mask.astype(int)
    
    detected_noise = 1 - detected_noise
    ###################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    ##################################################################################


    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise

def TS(input_events, ground_truth, time_window):
    print("Start processing TS filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    print(f'Number of input event: {len(input_events["x"])}')

    time_surface_filter = TimeSurfaceFilter(resolution=(x_max, y_max))
    boolean_mask, output_events = time_surface_filter << input_events

    print(f'Number of detected noise event: {len(input_events) - len(output_events)}')

    detected_noise = boolean_mask.astype(int)
    
    detected_noise = 1 - detected_noise

    ###################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    ##################################################################################

    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise


def YNoise(input_events, ground_truth, time_window):
    print("Start processing YNoise filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    print(f'Number of input event: {len(input_events["x"])}')

    yang_noise_filter             = YangNoiseFilter(resolution=(x_max, y_max))
    boolean_mask, output_events   = yang_noise_filter << input_events

    print(f'Number of detected noise event: {len(input_events) - len(output_events)}')

    detected_noise = boolean_mask.astype(int)
    
    detected_noise = 1 - detected_noise

    ##################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    #################################################################################

    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise



def KNoise(input_events, ground_truth, time_window):
    print("Start processing KNoise filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    print(f'Number of input event: {len(input_events["x"])}')

    khodamoradi_noise             = KhodamoradiNoiseFilter(resolution=(x_max, y_max))
    boolean_mask, output_events = khodamoradi_noise << input_events

    print(f'Number of detected noise event: {len(input_events) - len(output_events)}')

    detected_noise = boolean_mask.astype(int)
    
    detected_noise = 1 - detected_noise

    ##################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    #################################################################################

    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise


def EvFlow(input_events, ground_truth, time_window):
    print("Start processing EvFlow filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    print(f'Number of input event: {len(input_events["x"])}')

    event_flow                  = EventFlowFilter(resolution=(x_max, y_max))
    boolean_mask, output_events = event_flow << input_events

    print(f'Number of detected noise event: {len(input_events) - len(output_events)}')

    detected_noise = boolean_mask.astype(int)
    
    detected_noise = 1 - detected_noise

    ##################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    ################################################################################

    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise


def IETS(input_events, ground_truth, time_window):
    print("Start processing IETS filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    # ii = numpy.where(numpy.logical_and(input_events["t"] > 0, input_events["t"] < time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    print(f'Number of input event: {len(input_events["x"])}')

    multiTriggerWindow = 1e6
    inceptive_filter = InceptiveFilter(multiTriggerWindow)
    detected_noise = inceptive_filter.filter(input_events)

    output_events = input_events[~detected_noise]
    detected_noise = detected_noise.astype(int)
    detected_noise = 1 - detected_noise

    ##################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    ################################################################################

    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise


def STDF(input_events, ground_truth, time_window):
    print("Start processing STDF filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] >= time_window[0], input_events["t"] <= time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    print(f'Number of input event: {len(input_events["x"])}')

    stdf = STDFFilter((x_max, y_max), num_must_be_correlated=3, dt=50000)
    boolean_mask, output_events = stdf.filter_packet(input_events)

    print(f'Number of detected noise event: {len(input_events) - len(output_events)}')

    detected_noise = boolean_mask.astype(int)
    detected_noise = 1 - detected_noise

    #################################################################################
    # vx_velocity_raw = numpy.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    # vy_velocity_raw = numpy.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       input_events["label"].astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events,
    #                                                       detected_noise.astype(numpy.int32),
    #                                                       (vx_velocity_raw,vy_velocity_raw))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==0],
    #                                                       detected_noise[detected_noise==0].astype(numpy.int32),
    #                                                       (vx_velocity_raw[detected_noise==0],vy_velocity_raw[detected_noise==0]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    
    # cumulative_map_object, seg_label = accumulate_cnt_rgb((1280,720),input_events[detected_noise==1],
    #                                                         detected_noise[detected_noise==1].astype(numpy.int32),
    #                                                         (vx_velocity_raw[detected_noise==1],vy_velocity_raw[detected_noise==1]))
    # warped_image_segmentation_raw    = rgb_render_white(cumulative_map_object, seg_label)
    # warped_image_segmentation_raw.show()
    ###############################################################################

    precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = roc_val(input_events, detected_noise, ground_truth)
    performance = [precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA]

    return output_events, performance, detected_noise



def filter_events(input_events, ground_truth, time_window, method="STCF", save_performance=True):
    # Get the method function based on the method name
    method_function = globals().get(method)
    
    if method_function is None:
        raise ValueError(f"Method {method} not found")
    
    # Execute the method function
    start_time = time.time()
    output_events, performance, detected_noise = method_function(input_events, ground_truth, time_window)
    end_time = time.time()
    
    # Save performance metrics if requested
    if save_performance:
        performance += f" | Execution time: {end_time - start_time} seconds"
    
    return output_events, performance, detected_noise if save_performance else output_events

###########################################################################

def linearClassifier(Xtrain, Ytrain, Xtest,YtestGroundTruth):
    YtestOutput = weightMapping(Xtrain, Ytrain, Xtest)
    accuracy, rmse, misclassified_count, YtestOutputMaxed = CalculateClassifierPerformance(Xtrain, Ytrain, Xtest, YtestGroundTruth, YtestOutput)
    return accuracy, YtestOutputMaxed, YtestOutput

def ELMClassifier(Xtrain, Ytrain, Xtest,YtestGroundTruth,NUM_ELM_SIMULATIONS=10,ELM_hiddenLayerSizes=[50]):
    ELM_regularizationFactors       = [1e-6]
    rescaleInputsToNonlinearRegion  = 4

    ELM_accuracyArray = numpy.zeros((len(ELM_hiddenLayerSizes), len(ELM_regularizationFactors), NUM_ELM_SIMULATIONS))
    ELM_RMSEArray = numpy.zeros_like(ELM_accuracyArray)
    ELM_resultArray = numpy.empty_like(ELM_accuracyArray, dtype=object)

    for simELMIndex in tqdm(range(NUM_ELM_SIMULATIONS)):
        for regFacIndex, regFac in enumerate(ELM_regularizationFactors):
            for hiddenLayerSizeIndex, hiddenLayerSize in enumerate(ELM_hiddenLayerSizes):
                YtestOutput, hiddenLayerWeights, outputWeights = ELMWeight(Xtrain, Ytrain, Xtest, hiddenLayerSize, rescaleInputsToNonlinearRegion, regFac)
                accuracy, rmse, misclassified_count, YtestOutputMaxed = CalculateClassifierPerformance(Xtrain, Ytrain, Xtest, YtestGroundTruth, YtestOutput)

                ELM_resultArray[hiddenLayerSizeIndex, regFacIndex, simELMIndex] = {
                                'ELM_accuracy': accuracy,
                                'ELM_RMSE': rmse,
                                'YtestOutputELM': YtestOutput,
                                'YtestOutputMaxed': YtestOutputMaxed
                            }

                ELM_accuracyArray[hiddenLayerSizeIndex, regFacIndex, simELMIndex] = accuracy
                ELM_RMSEArray[hiddenLayerSizeIndex, regFacIndex, simELMIndex] = rmse
        
    mean_ELM_accuracies = numpy.mean(ELM_accuracyArray, axis=2)
    std_ELM_accuracies = numpy.std(ELM_accuracyArray, axis=2)
    bestHLNsIndex, bestELMRegFactIndex = numpy.unravel_index(numpy.argmax(mean_ELM_accuracies, axis=None), mean_ELM_accuracies.shape)
    highestMean_ELM_accuracy = mean_ELM_accuracies[bestHLNsIndex, bestELMRegFactIndex]
    stdOfbestMean_ELM_accuracy = std_ELM_accuracies[bestHLNsIndex, bestELMRegFactIndex]
    bestELMRegFact = ELM_regularizationFactors[bestELMRegFactIndex]
    bestHLNs = ELM_hiddenLayerSizes[bestHLNsIndex]

    best_YtestOutputMaxed = ELM_resultArray[bestHLNsIndex, bestELMRegFactIndex, 0]['YtestOutputMaxed']
    best_YtestOutputProb  = ELM_resultArray[bestHLNsIndex, bestELMRegFactIndex, 0]['YtestOutputELM']
    return highestMean_ELM_accuracy, stdOfbestMean_ELM_accuracy, best_YtestOutputMaxed, best_YtestOutputProb

def weightMapping(Xtrain, Ytrain, Xtest):
    regularizationFactor                 = 1e-6
    testingPresentations, inputChannels  = Xtest.shape
    XtX = numpy.dot(Xtrain.T, Xtrain) + regularizationFactor * numpy.eye(inputChannels)
    XtY = numpy.dot(Xtrain.T, Ytrain)
    linearInputToOutputMapping = numpy.linalg.solve(XtX, XtY).T
    YtestOutput = numpy.dot(Xtest, linearInputToOutputMapping.T)
    return YtestOutput

def CalculateClassifierPerformance(Xtrain, Ytrain, Xtest, YtestGroundTruth, YtestOutput):
    if YtestGroundTruth.shape != YtestOutput.shape:
        raise ValueError("YtestGroundTruth and YtestOutput must have the same shape")

    testingPresentations, numLabels = YtestGroundTruth.shape
    misclassified_indices = []
    misclassified_correct_labels = []
    misclassified_incorrect_labels = []

    YtestOutputMaxed = numpy.zeros_like(YtestGroundTruth)
    for pres in range(testingPresentations):
        maxYtestOutputValue = numpy.max(YtestOutput[pres, :])
        if numpy.all(YtestOutput[pres, :] == 0):
            YtestOutputMaxed[pres, :] = numpy.zeros_like(YtestOutput[pres, :])
        elif numpy.all(YtestOutput[pres, :] == maxYtestOutputValue):
            YtestOutputMaxed[pres, :] = numpy.ones_like(YtestOutput[pres, :])
        else:
            YtestOutputMaxed[pres, :] = (YtestOutput[pres, :] == maxYtestOutputValue).astype(float)

        if not numpy.array_equal(YtestOutputMaxed[pres, :], YtestGroundTruth[pres, :]):
            misclassified_indices.append(pres)
            misclassified_correct_labels.append(YtestGroundTruth[pres, :])
            misclassified_incorrect_labels.append(YtestOutputMaxed[pres, :])

    misclassified_indices = numpy.array(misclassified_indices, dtype=int)
    misclassified_correct_labels = numpy.array(misclassified_correct_labels, dtype=float)
    misclassified_incorrect_labels = numpy.array(misclassified_incorrect_labels, dtype=float)

    correctly_labeled_positives_TP = numpy.sum(numpy.logical_and(YtestGroundTruth, YtestOutputMaxed), axis=0)  # TP
    correctly_labeled_negatives_TN = numpy.sum(numpy.logical_and(~YtestGroundTruth.astype(bool), ~YtestOutputMaxed.astype(bool)), axis=0)  # TN

    actual_positives = numpy.sum(YtestGroundTruth, axis=0)  # TP + FN
    actual_negatives = numpy.sum(~YtestGroundTruth.astype(bool), axis=0)  # TN + FP

    sensitivity = correctly_labeled_positives_TP / actual_positives  # TP / (TP + FN)
    specificity = correctly_labeled_negatives_TN / actual_negatives  # TN / (TN + FP)
    informedness = sensitivity + specificity - 1
    
    rmse = numpy.sqrt(mean_squared_error(YtestGroundTruth, YtestOutput))
    accuracy = (1 - len(misclassified_indices) / testingPresentations) * 100

    return accuracy, rmse, len(misclassified_indices), YtestOutputMaxed


def ELMWeight(Xtrain, Ytrain, Xtest, hiddenLayerSize, rescaleInputsToNonlinearRegion, regularizationFactor):
    inputChannels = Xtrain.shape[1]

    # Initialize hidden layer weights
    hiddenLayerWeights = numpy.random.rand(inputChannels, hiddenLayerSize) - 0.5

    # Calculate hidden layer outputs for training data
    hiddenLayerWeightedInputs = Xtrain @ hiddenLayerWeights
    hiddenLayerOutputs = 1 / (1 + numpy.exp(-rescaleInputsToNonlinearRegion * hiddenLayerWeightedInputs)) - 0.5

    # Calculate output weights
    hiddenLayerOutputs_T = hiddenLayerOutputs.T
    outputWeights = numpy.linalg.solve(
        hiddenLayerOutputs_T @ hiddenLayerOutputs + regularizationFactor * numpy.eye(hiddenLayerSize),
        hiddenLayerOutputs_T @ Ytrain
    ).T

    # Calculate hidden layer outputs for test data
    hiddenLayerWeightedInputsForXtest = Xtest @ hiddenLayerWeights
    hiddenLayerOutputsForTest = 1 / (1 + numpy.exp(-rescaleInputsToNonlinearRegion * hiddenLayerWeightedInputsForXtest)) - 0.5

    # Calculate YtestOutput
    YtestOutput = hiddenLayerOutputsForTest @ outputWeights.T

    return YtestOutput, hiddenLayerWeights, outputWeights


def viz_LiC_ELM(dataname, xCoordArraytest, yCoordArraytest, tsCoordArraytest, YtestOutputMaxedLiC,
                YtestOutputMaxedELM, Ytest, YtestOutputProbLiC, YtestOutputProbELM):
    if not os.path.exists(f"./output/benchmark/{dataname}"):
        os.mkdir(f"./output/benchmark/{dataname}")
    
    sortIdx = numpy.argsort(tsCoordArraytest.flatten())
    events = {
        'x': xCoordArraytest[sortIdx].flatten(),
        'y': yCoordArraytest[sortIdx].flatten(),
        't': tsCoordArraytest[sortIdx].flatten(),
        'yPredLiC': YtestOutputMaxedLiC[[sortIdx],0].flatten().astype(numpy.int32),
        'yPredELM': YtestOutputMaxedELM[[sortIdx],0].flatten().astype(numpy.int32),
        'yTest': Ytest[[sortIdx],0].flatten().astype(numpy.int32),
        'vx_zero': numpy.zeros_like(xCoordArraytest[[sortIdx]]).flatten(),
        'vy_zero': numpy.zeros_like(yCoordArraytest[[sortIdx]]).flatten(),
        'yPredLiC_onehot': YtestOutputMaxedLiC,
        'yPredELM_onehot': YtestOutputMaxedELM,
        'yPredLiC_prob': YtestOutputProbLiC,
        'yPredELM_prob': YtestOutputProbELM
    }

    noise_LiC = numpy.where(events["yPredLiC"] == 1)[0]
    events_indexed_stars_LiC = {key: value[noise_LiC] for key, value in events.items()}

    signal_ELM = numpy.where(events["yPredELM"] == 1)[0]
    events_indexed_stars_ELM = {key: value[signal_ELM] for key, value in events.items()}

    numpy.savez(f"./output/benchmark/{dataname}/processed_events", **events)

    cumulative_map_object  = accumulate((1280,720),events,(0,0))
    warped_image_segmentation   = render(cumulative_map_object, colormap_name="magma", gamma=lambda image: image ** (1 / 5))
    warped_image_segmentation.save(f"./output/benchmark/{dataname}/{dataname}_all_events_image.png")

    cumulative_map_LiC  = accumulate((1280,720),events_indexed_stars_LiC,(0,0))
    warped_image_segmentation_LiC   = render(cumulative_map_LiC, colormap_name="magma", gamma=lambda image: image ** (1 / 5))
    warped_image_segmentation_LiC.save(f"./output/benchmark/{dataname}/{dataname}_filtered_events_LiC_stars.png")

    cumulative_map_object_ELM  = accumulate((1280,720),events_indexed_stars_ELM,(0,0))
    warped_image_segmentation_ELM   = render(cumulative_map_object_ELM, colormap_name="magma", gamma=lambda image: image ** (1 / 5))
    warped_image_segmentation_ELM.save(f"./output/benchmark/{dataname}/{dataname}_filtered_events_ELM_stars.png")

    
    cumulative_map_LiC, seg_label_LiC   = accumulate_cnt_rgb((1280, 720), events, events["yPredLiC"], (events["vx_zero"], events["vy_zero"]))
    warped_image_segmentation_LiC       = rgb_render_advanced(cumulative_map_LiC, seg_label_LiC)
    warped_image_segmentation_LiC.save(f"./output/benchmark/{dataname}/{dataname}_yPredLiC_labelled_image.png")

    cumulative_map_ELM, seg_label_ELM   = accumulate_cnt_rgb((1280, 720), events, events["yPredELM"], (events["vx_zero"], events["vy_zero"]))
    warped_image_segmentation_ELM       = rgb_render_advanced(cumulative_map_ELM, seg_label_ELM)
    warped_image_segmentation_ELM.save(f"./output/benchmark/{dataname}/{dataname}_yPredELM_labelled_image.png")

    fprLiC, tprLiC, _ = roc_curve(Ytest[:,0], YtestOutputProbLiC[:,0], pos_label=1)
    fprELM, tprELM, _ = roc_curve(Ytest[:,0], YtestOutputProbELM[:,0], pos_label=1)
    aucLiC            = roc_auc_score(Ytest[:,0], YtestOutputProbLiC[:,0])
    aucELM            = roc_auc_score(Ytest[:,0], YtestOutputProbELM[:,0])

    plt.figure(figsize=(12, 6))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'roc LiC curve: {aucLiC:.3f}, roc ELM curve: {aucELM:.3f}')
    plt.plot(fprLiC, tprLiC, color='b', linewidth=1)
    plt.plot(fprELM, tprELM, color='g', linewidth=1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(["LiC","ELM"])
    plt.savefig(f"./output/benchmark/{dataname}/roc_curve_LiC_ELM.png")


def FEAST_training(parent_folder):
    epoch               = 10
    nNeuron_noise       = 200
    nNeuron_stars       = int(numpy.round((nNeuron_noise/2)))
    nNeuron_satellites  = int(numpy.round((nNeuron_noise/2)))
    thresholdRise       = 0.001
    thresholdFall       = 0.002
    tau                 = 1e6
    eta_noise           = 0.0005
    eta_stars           = 0.0001
    eta_satellites      = 0.01
    R                   = 5
    D                   = 2 * R + 1

    w_satellites = numpy.random.rand(D * D, nNeuron_satellites)
    for iNeuron in range(nNeuron_satellites):
        w_satellites[:, iNeuron] /= numpy.linalg.norm(w_satellites[:, iNeuron])
        
    w_stars = numpy.random.rand(D * D, nNeuron_stars)
    for iNeuron in range(nNeuron_stars):
        w_stars[:, iNeuron] /= numpy.linalg.norm(w_stars[:, iNeuron])
        
    w_noise = numpy.random.rand(D * D, nNeuron_noise)
    for iNeuron in range(nNeuron_noise):
        w_noise[:, iNeuron] /= numpy.linalg.norm(w_noise[:, iNeuron])
        
    recording_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

    training_folders  = recording_folders[:60]

    # # Combine training and testing folders, marking their type
    # combined_folders = [(f, "TrainingStars") for f in training_folders]
    
    for iteration in range(epoch):
        print_message(f'Epoch: {iteration}', color='yellow', style='bold')

        # Shuffle the order of the first 60 directories
        random.shuffle(training_folders)
        
        # Combine training folders with their type
        combined_folders = [(f, "TrainingStars") for f in training_folders]

        # Processing all folders
        f_idx = 0
        for recording_name, validation_type in combined_folders:
            print(f"{validation_type}  Processing file: {recording_name}")
            folder_path = os.path.join(parent_folder, recording_name)
            
            # Find the .es file in the folder
            es_file_path = None
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".es"):
                    es_file_path = os.path.join(folder_path, file_name)
                    break
            
            width, height, events = read_es_file(es_file_path)
            sensor_size = (width, height)        
            
            if os.path.exists(f"{parent_folder}/{recording_name}/labelled_events_v2.0.0.npy"):
                labels = numpy.load(f"{parent_folder}/{recording_name}/labelled_events_v2.0.0.npy")
            else:
                labels = numpy.load(f"{parent_folder}/{recording_name}/labelled_events_v2.5.0.npy")
            
            events_labels = numpy.zeros(len(events["x"]), dtype=int)

            label_dict = {}
            for row in labels:
                coord = (row['x'], row['y'])
                if row['label'] != 0:  # Exclude noise labels
                    label_dict[coord] = row['label']

            # Assign labels to events based on pixel coordinates
            for i, event in enumerate(events):
                coord = (event['x'], event['y'])
                if coord in label_dict:
                    events_labels[i] = label_dict[coord]

            # Create a structured array with the new labels column
            events_with_labels = numpy.zeros(len(events), dtype=[('t', '<u8'),
                                                            ('x', '<u2'),
                                                            ('y', '<u2'),
                                                            ('on', '?'),
                                                            ('label', '<i2')])
            events_with_labels['t']     = events['t']
            events_with_labels['x']     = events['x']
            events_with_labels['y']     = events['y']
            events_with_labels['on']    = events['on']
            events_with_labels['label'] = events_labels

            nnn = numpy.where(numpy.logical_and(events_with_labels['t'] > 10e6, events_with_labels['t'] < events_with_labels['t'][-1]))
            events = events_with_labels[nnn]
            events["label"][events["label"] > 0] = 1
            events["label"][events["label"] < 0] = 2
            events["t"] = events["t"] - events["t"][0]
            
            nTD = len(events['x'])
            xs = numpy.max(events['x']) + 1
            ys = numpy.max(events['y']) + 1
        
            # for epochIndex in range(epoch):
            threshArray_stars = numpy.zeros(nNeuron_stars) + 0.001
            threshArray_satellites = numpy.zeros(nNeuron_satellites) + 0.001
            threshArray_noise = numpy.zeros(nNeuron_noise) + 0.001
            T_stars = numpy.full((xs, ys), -numpy.inf)
            P_stars = numpy.zeros((xs, ys))
            T_satellites = numpy.full((xs, ys), -numpy.inf)
            P_satellites = numpy.zeros((xs, ys))
            T_noise = numpy.full((xs, ys), -numpy.inf)
            P_noise = numpy.zeros((xs, ys))
            
            for idx in tqdm(range(nTD)):
                x = int(events['x'][idx])
                y = int(events['y'][idx])
                p = int(events['on'][idx])
                if p == 0:
                    p = 1
                t = int(events['t'][idx])
                class_label = int(events['label'][idx])
                
                if class_label == 1: #stars
                    T_stars[x, y] = t
                    P_stars[x, y] = p

                    if (x - R > 0) and (x + R < xs) and (y - R > 0) and (y + R < ys):
                        ROI = P_stars[x - R:x + R + 1, y - R:y + R + 1] * numpy.exp((T_stars[x - R:x + R + 1, y - R:y + R + 1] - t) / tau)
                        ROI_norm = ROI / numpy.linalg.norm(ROI)
                        ROI_ARRAY = numpy.outer(ROI_norm.flatten(), numpy.ones(nNeuron_stars))
                        dotProds = numpy.sum(w_stars * ROI_ARRAY, axis=0)
                        C = dotProds * (dotProds > threshArray_stars)
                        winnerNeuron = numpy.argmax(C)

                        if numpy.all(C == 0):
                            threshArray_stars -= thresholdFall
                        else:
                            w_stars[:, winnerNeuron] += eta_stars * ROI.flatten()
                            w_stars[:, winnerNeuron] /= numpy.linalg.norm(w_stars[:, winnerNeuron])
                            threshArray_stars[winnerNeuron] += thresholdRise

                if class_label == 2: #satellites
                    T_satellites[x, y] = t
                    P_satellites[x, y] = p

                    if (x - R > 0) and (x + R < xs) and (y - R > 0) and (y + R < ys):
                        ROI = P_satellites[x - R:x + R + 1, y - R:y + R + 1] * numpy.exp((T_satellites[x - R:x + R + 1, y - R:y + R + 1] - t) / tau)
                        ROI_norm = ROI / numpy.linalg.norm(ROI)
                        ROI_ARRAY = numpy.outer(ROI_norm.flatten(), numpy.ones(nNeuron_satellites))
                        dotProds = numpy.sum(w_satellites * ROI_ARRAY, axis=0)
                        C = dotProds * (dotProds > threshArray_satellites)
                        winnerNeuron = numpy.argmax(C)

                        if numpy.all(C == 0):
                            threshArray_satellites -= thresholdFall
                        else:
                            w_satellites[:, winnerNeuron] += eta_satellites * ROI.flatten()
                            w_satellites[:, winnerNeuron] /= numpy.linalg.norm(w_satellites[:, winnerNeuron])
                            threshArray_satellites[winnerNeuron] += thresholdRise
                            
                elif class_label == 0:
                    T_noise[x, y] = t
                    P_noise[x, y] = p

                    if (x - R > 0) and (x + R < xs) and (y - R > 0) and (y + R < ys):
                        ROI = P_noise[x - R:x + R + 1, y - R:y + R + 1] * numpy.exp((T_noise[x - R:x + R + 1, y - R:y + R + 1] - t) / tau)
                        ROI_norm = ROI / numpy.linalg.norm(ROI)
                        ROI_ARRAY = numpy.outer(ROI_norm.flatten(), numpy.ones(nNeuron_noise))
                        dotProds = numpy.sum(w_noise * ROI_ARRAY, axis=0)
                        C = dotProds * (dotProds > threshArray_noise)
                        winnerNeuron = numpy.argmax(C)

                        if numpy.all(C == 0):
                            threshArray_noise -= thresholdFall
                        else:
                            w_noise[:, winnerNeuron] += eta_noise * ROI.flatten()
                            w_noise[:, winnerNeuron] /= numpy.linalg.norm(w_noise[:, winnerNeuron])
                            threshArray_noise[winnerNeuron] += thresholdRise
            
            w_signal = numpy.concatenate((w_satellites, w_stars), axis=1)
            numpy.save("./output/feast/feast_weights.npy", w_signal, w_noise)
            
        # nNeuron = nNeuron_satellites + nNeuron_stars
        # sqNeuron = int(numpy.ceil(numpy.sqrt(nNeuron)))
        # fig, axes = plt.subplots(sqNeuron, sqNeuron, figsize=(15, 15))
        # fig.suptitle('Weights signal', fontsize=16)

        # for iNeuron in range(nNeuron):
        #     ax = axes[iNeuron // sqNeuron, iNeuron % sqNeuron]
        #     wShow = combined_weights[:, iNeuron].reshape(D, D)
        #     # wShow[R, R] = numpy.nan  # Setting the center to NaN
        #     cax = ax.imshow(wShow, cmap='hot')
        #     ax.axis('off')
        #     fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        
        # fig.savefig(f'./output/feast/feast_trained_weight_signal_epoch_{f_idx}.png')
        
        # fig2, axes2 = plt.subplots(sqNeuron, sqNeuron, figsize=(15, 15))
        # fig2.suptitle('Weights noise', fontsize=16)

        # for iNeuron in range(nNeuron_noise):
        #     ax = axes2[iNeuron // sqNeuron, iNeuron % sqNeuron]
        #     wShow = w_noise[:, iNeuron].reshape(D, D)
        #     # wShow[R, R] = numpy.nan  # Setting the center to NaN
        #     cax = ax.imshow(wShow, cmap='hot')
        #     ax.axis('off')
        #     fig2.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

        # fig2.savefig(f'./output/feast/feast_trained_weight_noise_epoch_{f_idx}.png')
        # f_idx += 1
        
    return w_signal, w_noise

def FEAST_inference(input_events, ground_truth, signalweight, noiseweight):
    downSampleFactor    = 5
    tau                 = 1e6
    beta                = 0.5
    R                   = 5
    D                   = 2 * R + 1
    pooling_wind        = 3
    wFrozen             = numpy.hstack((signalweight, noiseweight))
    neuron_labels       = numpy.hstack((numpy.ones(signalweight.shape[1]), 
                                        numpy.zeros(noiseweight.shape[1]))).astype(int)
    nNeuron             = len(neuron_labels)
    nTD                 = len(input_events['x'])

    xs = numpy.max(input_events['x']) + 1
    ys = numpy.max(input_events['y']) + 1

    T = -numpy.inf * numpy.ones((xs, ys))
    P = numpy.zeros((xs, ys))

    predicted_labels = numpy.full(nTD, numpy.nan)
    T_F = -numpy.inf * numpy.ones((xs, ys, nNeuron))

    T_Fd = -numpy.inf * numpy.ones((round(xs / downSampleFactor), round(ys / downSampleFactor), nNeuron))
    P_Fd = numpy.zeros_like(T_Fd)
    T_FdSimple = T_Fd.copy()

    oneMinusBeta = 1 - beta
    countArray = numpy.full(nNeuron, 10)
    X_original = numpy.eye(nTD, pooling_wind * pooling_wind * nNeuron)

    xdMax = xs / downSampleFactor
    ydMax = ys / downSampleFactor

    Valididx = 0
    validnewTD0 = 0
    validnewTD1 = 0
    newTD0 = []
    newTD1 = []
    coordinate = numpy.zeros((nTD, 4))
    labels = numpy.full_like(input_events['x'], numpy.nan)

    for idx in tqdm(range(nTD)):
        x = int(input_events['x'][idx])
        y = int(input_events['y'][idx])
        xd = round(x / downSampleFactor)
        yd = round(y / downSampleFactor)
        p = int(input_events['on'][idx])
        if p == 0:
            p = 1
        t = int(input_events['t'][idx])

        T[x, y] = t
        P[x, y] = p

        if (x - R > 0) and (x + R < xs) and (y - R > 0) and (y + R < ys):
            ROI = P[x - R:x + R + 1, y - R:y + R + 1] * numpy.exp((T[x - R:x + R + 1, y - R:y + R + 1] - t) / tau)

            if xd > 1 and yd > 1 and xd < xdMax and yd < ydMax:
                ROI_norm = ROI / numpy.linalg.norm(ROI, 'fro')
                dotProds = numpy.sum(wFrozen * ROI_norm.ravel()[:, None], axis=0)
                winnerNeuron = numpy.argmax(dotProds)
                Valididx += 1

                countArray[winnerNeuron] += 1

                predicted_labels[Valididx] = neuron_labels[winnerNeuron]

                T_F[x, y, winnerNeuron] = t
                T_FdSimple[xd, yd, winnerNeuron] = t

                if numpy.isinf(T_FdSimple[xd, yd, winnerNeuron]):
                    T_FdSimple[xd, yd, winnerNeuron] = t
                else:
                    T_FdSimple[xd, yd, winnerNeuron] = oneMinusBeta * T_FdSimple[xd, yd, winnerNeuron] + beta * t

                x_start = max(xd - 1, 0)
                x_end = min(xd + 2, T_FdSimple.shape[0])
                y_start = max(yd - 1, 0)
                y_end = min(yd + 2, T_FdSimple.shape[1])

                T_Fd_featureContext = T_FdSimple[x_start:x_end, y_start:y_end, :]

                if T_Fd_featureContext.shape[0] < 3 or T_Fd_featureContext.shape[1] < 3:
                    T_Fd_featureContext = numpy.pad(T_Fd_featureContext, 
                                                 ((0, max(0, 3 - T_Fd_featureContext.shape[0])), 
                                                  (0, max(0, 3 - T_Fd_featureContext.shape[1])), 
                                                  (0, 0)), 'constant', constant_values=-numpy.inf)

                S_Fd_featureContext = numpy.exp((T_Fd_featureContext - t) / tau)
                X_original[Valididx, :] = S_Fd_featureContext.ravel()
                coordinate[Valididx, 0] = x
                coordinate[Valididx, 1] = y
                coordinate[Valididx, 2] = t

                if ground_truth[idx] == 0:
                    coordinate[Valididx, 3] = 0
                    validnewTD0 += 1
                    labels[Valididx] = 0

                elif ground_truth[idx] == 1:
                    coordinate[Valididx, 3] = 1
                    validnewTD1 += 1
                    labels[Valididx] = 1

    where_are_NaNs = numpy.isnan(predicted_labels)
    predicted_labels[where_are_NaNs] = 0
    return X_original, coordinate, labels, predicted_labels


def cross_validation(coordinate, X_original, labels, trainTestSplitRatio=0.5):
    # Extract coordinates and timestamps
    xCoordArray = coordinate[:, 0]
    yCoordArray = coordinate[:, 1]
    tsCoordArray = coordinate[:, 2]

    # Convert X and Y to single precision
    X = X_original.astype(numpy.float32)
    Y = labels.astype(numpy.float32)

    # Find indices of NaN values in Y
    findNaN = numpy.where(numpy.isnan(Y))[0]

    if len(findNaN) > 0:
        X = X[:findNaN[0], :]
        Y = Y[:findNaN[0]]

        xCoordArray = xCoordArray[:findNaN[0]]
        yCoordArray = yCoordArray[:findNaN[0]]
        tsCoordArray = tsCoordArray[:findNaN[0]]

    nEventAfterSkip = X.shape[0]

    # Shuffle the data
    shuffledIndex = numpy.random.permutation(nEventAfterSkip)
    X_shuffled = X[shuffledIndex, :]
    Y_shuffled = Y[shuffledIndex]

    xCoordArray_shuffle = xCoordArray[shuffledIndex]
    yCoordArray_shuffle = yCoordArray[shuffledIndex]
    tsCoordArray_shuffle = tsCoordArray[shuffledIndex]

    # Split into training and test sets
    splitIndex = int(numpy.floor(nEventAfterSkip * trainTestSplitRatio))
    
    Xtrain = X_shuffled[:splitIndex, :]
    Xtest = X_shuffled[splitIndex:, :]

    Ytrain = Y_shuffled[:splitIndex]
    Ytest = Y_shuffled[splitIndex:]

    xCoordArraytest = xCoordArray_shuffle[splitIndex:]
    yCoordArraytest = yCoordArray_shuffle[splitIndex:]
    tsCoordArraytest = tsCoordArray_shuffle[splitIndex:]

    # Replace NaN values in Ytrain and Ytest with 0
    Ytrain[numpy.isnan(Ytrain)] = 0
    Ytest[numpy.isnan(Ytest)] = 0

    # Create the second column for binary classification
    Ytrain = numpy.column_stack((Ytrain, 1 - Ytrain))
    Ytest = numpy.column_stack((Ytest, 1 - Ytest))

    return Xtrain, Ytrain, Xtest, Ytest, xCoordArraytest, yCoordArraytest, tsCoordArraytest



def accumulate4D_placeholder(sensor_size, events, linear_vel, angular_vel, zoom):
    # Placeholder function to simulate accumulate4D.
    # You'll need to replace this with the actual PyTorch-compatible implementation.
    return torch.randn(sensor_size[0], sensor_size[1])

def accumulate4D_torch(sensor_size, events, linear_vel, angular_vel, zoom):
    # Convert tensors back to numpy arrays for the C++ function
    t_np = events["t"].cpu().numpy()
    x_np = events["x"].cpu().numpy()
    y_np = events["y"].cpu().numpy()

    # Get the 2D image using the C++ function
    image_np = dvs_sparse_filter_extension.accumulate4D(
        sensor_size[0],
        sensor_size[1],
        t_np.astype("<f8"),
        x_np.astype("<f8"),
        y_np.astype("<f8"),
        linear_vel[0],
        linear_vel[1],
        angular_vel[0],
        angular_vel[1],
        angular_vel[2],
        zoom,
    )

    # Convert numpy array to PyTorch tensor
    image_tensor = torch.tensor(image_np).float().to(linear_vel.device)
    return image_tensor


def save_conf(method: str, condition: str, field_center: Tuple[float, float], speed: float, t_start: float, window_size: float, sliding_window:float, vx: float, vy: float, contrast: float, total_events: float):
    results_folder = f"./output/{condition}_speed_survey_{field_center[0]}_{field_center[1]}_{speed}/{method}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)
    config_file_name = os.path.join(results_folder, f"{method}_configuration.yaml")
    config_data = {
        "method": method,
        "condition": condition,
        "field_center": f"({field_center[0]},{field_center[1]})",  # Format as a string to maintain format
        "speed": speed,
        "t_start": t_start,
        "window_size": window_size,
        "vx": vx,
        "vy": vy,
        "contrast": contrast,
        "total_events": total_events
    }
    if sliding_window is not None:
        config_data["sliding_window"] = sliding_window
    def tuple_representer(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', f'({data[0]},{data[1]})')
    yaml.add_representer(tuple, tuple_representer)
    with open(config_file_name, "w") as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
    return config_file_name


def remove_hot_pixels_cedric(td, std_dev_threshold):
    # Convert MATLAB's 1-indexing to Python's 0-indexing by adjusting td's x and y
    sensor_size = (numpy.max(td['y'])+1, numpy.max(td['x'])+1)
    
    # Initialize the histogram
    histogram = numpy.zeros(sensor_size)
    
    # Build the histogram from event data
    for y, x in zip(td['y'], td['x']):
        histogram[y, x] += 1
    
    # Use standard deviation threshold to determine hot pixels
    valid_histogram_values = histogram[histogram > 0]
    mean_val = numpy.mean(valid_histogram_values)
    std_dev = numpy.std(valid_histogram_values)
    threshold = mean_val + std_dev_threshold * std_dev
    top_n_hot_pixels_indices = numpy.where(histogram > threshold)
    
    # Convert array indices to x and y coordinates
    yhot, xhot = top_n_hot_pixels_indices
    
    # Adjust for 0-indexing used in Python
    xhot = xhot
    yhot = yhot
    
    # Identify events to remove
    is_hot_event = numpy.zeros(len(td['x']), dtype=bool)
    for x, y in zip(xhot, yhot):
        is_hot_event |= (td['x'] == x) & (td['y'] == y)
    
    td_clean = td[~is_hot_event]
    
    print('Number of hot pixels detected:', len(xhot))
    print('Number of events removed due to hot pixels:', numpy.sum(is_hot_event))
    
    return td_clean


def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
    """
    Accumulate x and y coords to an image using bilinear interpolation
    @param pxs Numpy array of integer typecast x coords of events
    @param pys Numpy array of integer typecast y coords of events
    @param dxs Numpy array of residual difference between x coord and int(x coord)
    @param dys Numpy array of residual difference between y coord and int(y coord)
    @returns Image
    From: https://github.com/TimoStoff/event_utils/blob/master/lib/representations/image.py#L102
    """
    img.index_put_((pys,   pxs  ), weights*(1.0-dxs)*(1.0-dys), accumulate=True)
    img.index_put_((pys,   pxs+1), weights*dxs*(1.0-dys), accumulate=True)
    img.index_put_((pys+1, pxs  ), weights*(1.0-dxs)*dys, accumulate=True)
    img.index_put_((pys+1, pxs+1), weights*dxs*dys, accumulate=True)
    return img


def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(1280, 720), clip_out_of_range=True,
        interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    From: https://github.com/TimoStoff/event_utils/blob/master/lib/representations/image.py#L46
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs*mask, ys*mask
            img.index_put_((ys, xs), ps, accumulate=True)
        except Exception as e:
            print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
            raise e
    return img

def events_to_image(xs, ys, ps, sensor_size=(1280, 720), interpolation=None, padding=False, meanval=False, default=0):
    """
    Place events into an image using numpy
    @param xs x coords of events
    @param ys y coords of events
    @param ps Event polarities/weights
    @param sensor_size The size of the event camera sensor
    @param interpolation Whether to add the events to the pixels by interpolation (values: None, 'bilinear')
    @param padding If true, pad the output image to include events otherwise warped off sensor
    @param meanval If true, divide the sum of the values by the number of events at that location
    @returns Event image from the input events
    From: https://github.com/TimoStoff/event_utils/blob/master/lib/representations/image.py#L5
    """
    img_size = (sensor_size[0]+1, sensor_size[1]+1)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        xt, yt, pt = torch.from_numpy(xs.astype(numpy.int16)), torch.from_numpy(ys.astype(numpy.int16)), torch.from_numpy(ps.astype(numpy.int16))
        xt, yt, pt = xt.float(), yt.float(), pt.float()
        img = events_to_image_torch(xt, yt, pt, clip_out_of_range=False, interpolation='bilinear', padding=padding)
        img[img==0] = default
        img = img.numpy()
        if meanval:
            event_count_image = events_to_image_torch(xt, yt, torch.ones_like(xt),
                    clip_out_of_range=True, padding=padding)
            event_count_image = event_count_image.numpy()
    else:
        coords = numpy.stack((xs, ys))
        try:
            abs_coords = numpy.ravel_multi_index(coords, img_size)
        except ValueError:
            print("Issue with input arrays! minx={}, maxx={}, miny={}, maxy={}, coords.shape={}, \
                    sum(coords)={}, sensor_size={}".format(numpy.min(xs), numpy.max(xs), numpy.min(ys), numpy.max(ys),
                        coords.shape, numpy.sum(coords), img_size))
            raise ValueError
        img = numpy.bincount(abs_coords, weights=ps, minlength=img_size[0]*img_size[1])
        img = img.reshape(img_size)
        if meanval:
            event_count_image = numpy.bincount(abs_coords, weights=numpy.ones_like(xs), minlength=img_size[0]*img_size[1])
            event_count_image = event_count_image.reshape(img_size)
    if meanval:
        img = numpy.divide(img, event_count_image, out=numpy.ones_like(img)*default, where=event_count_image!=0)
    return img[0:sensor_size[0], 0:sensor_size[1]]


def hot_pixels_filter(events, PercentPixelsToRemove=0.01, minimumNumOfEventsToCheckHotNess=100 ):
    xMax = max(events["x"])+1
    yMax = max(events["y"])+1
    bigNumber=1e99
    nEvent = len(events['x'])
    tCell = [[[] for _ in range(yMax)] for _ in range(xMax)]
    eventCount = numpy.zeros((xMax, yMax), dtype=int)
    
    # Populate eventCount and tCell
    for ii in range(nEvent):
        x = events['x'][ii]
        y = events['y'][ii]
        t = events['t'][ii]
        eventCount[x, y] += 1
        tCell[x][y].append(t)
    
    duration = numpy.zeros((xMax, yMax)) - (events['t'][-1] - events['t'][0])
    errorTime = numpy.zeros((xMax, yMax)) + bigNumber
    
    # Calculate duration and errorTime
    for x in range(xMax):
        for y in range(yMax):
            if eventCount[x, y] > minimumNumOfEventsToCheckHotNess:
                duration[x, y] = tCell[x][y][-1] - tCell[x][y][0]
                linspace = numpy.linspace(tCell[x][y][0], tCell[x][y][-1], eventCount[x, y])
                errorTime[x, y] = numpy.sum(numpy.abs(numpy.array(tCell[x][y]) - linspace))
                
    hotPixelMeasure = eventCount * duration / errorTime
    
    # Determine hot pixel threshold
    hotPixelMeasureThreshold = numpy.percentile(hotPixelMeasure.ravel(), (100 - PercentPixelsToRemove))
    
    hotPixelImage = numpy.zeros((xMax, yMax), dtype=int)
    xHotPixelArray, yHotPixelArray = [], []
    
    # Identify hot pixels
    for x in range(xMax):
        for y in range(yMax):
            if hotPixelMeasure[x, y] > hotPixelMeasureThreshold:
                xHotPixelArray.append(x)
                yHotPixelArray.append(y)
                hotPixelImage[x, y] = 1
                
    return xHotPixelArray, yHotPixelArray

def remove_hot_pixels(xs, ys, ts, ps, sensor_size=(1280, 720), num_hot=10):
    """
    Given a set of events, removes the 'hot' pixel events.
    Accumulates all of the events into an event image and removes
    the 'num_hot' highest value pixels.
    @param xs Event x coords
    @param ys Event y coords
    @param ts Event timestamps
    @param ps Event polarities
    @param sensor_size The size of the event camera sensor
    @param num_hot The number of hot pixels to remove
    From: https://github.com/TimoStoff/event_utils/blob/master/lib/util/event_util.py
    """
    img = events_to_image(xs, ys, ps, interpolation=None, meanval=False, sensor_size=sensor_size)
    img_copy = img
    hot = numpy.array([])
    hot_coors = []
    for i in range(num_hot):
        maxc = numpy.unravel_index(numpy.argmax(img), sensor_size)
        # vertical_flip = (sensor_size[0] - 1 - maxc[0], maxc[1])
        # vertical_then_horizontal_flip = (sensor_size[0] - 1 - vertical_flip[0], sensor_size[1] - 1 - vertical_flip[1])

        hot_coors.append(maxc)
        #print("{} = {}".format(maxc, img[maxc]))
        img[maxc] = 0
        h = numpy.where((xs == maxc[0]) & (ys == maxc[1]))
        hot = numpy.concatenate((hot, h[0]))
        # Example assuming `hot` should be an array of indices
        hot_indices = numpy.array(hot, dtype=int)  # Ensure `hot` is an integer array
        # xs, ys, ts, ps = (
        #     numpy.delete(xs, hot_indices),
        #     numpy.delete(ys, hot_indices),
        #     numpy.delete(ts, hot_indices),
        #     numpy.delete(ps, hot_indices),
        # )

    print(f"Number of hot pixels: {num_hot}")
    print(f"Number of hot events: {len(hot)}")
    plt.figure(figsize=(12, 7))
    plt.imshow(img_copy)
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')
    for idx in range(len(hot_coors)):
        circle = Circle((hot_coors[idx][0], hot_coors[idx][1]), color='red', radius=10, fill=False,alpha=0.65)
        plt.gca().add_patch(circle)
    # plt.colorbar()
    plt.show()
    return xs, ys, ts, ps


def cmax_full_window(sensor_size: Tuple[int, int], events: numpy.ndarray, events_filtered: numpy.ndarray):
    best_velocity, highest_variance = find_best_velocity_iteratively(sensor_size, events_filtered, increment=500)
    warped_image_zero = accumulate(sensor_size, events, velocity=(0,0))
    warped_image = accumulate(sensor_size, events, velocity=best_velocity)
    return warped_image_zero, warped_image, best_velocity[0], best_velocity[1], highest_variance



def cmax_slidding_window(sensor_size: Tuple[int, int], events: numpy.ndarray, events_filtered: numpy.ndarray, sliding_window: float):
    min_t = numpy.min(events['t'])
    max_t = numpy.max(events['t'])

    start_t = min_t
    end_t = start_t + sliding_window

    best_velocity_vec = []
    first_iteration = True  # Flag to indicate the first iteration

    # Initialize best_velocity outside of the loop to be used as initial_velocity
    best_velocity = None

    while start_t <= max_t:
        events_subset_filtered = events_filtered[(events_filtered['t'] >= start_t) & (events_filtered['t'] < end_t)]

        # best_velocity, highest_variance = find_best_velocity_iteratively(sensor_size, events_subset_filtered, increment=500)

        if first_iteration:
            best_velocity, highest_variance = find_best_velocity_iteratively(sensor_size, events_subset_filtered, increment=500)
            first_iteration = False  # After the first iteration, set this to False
        else:
            best_velocity, highest_variance = find_best_velocity_with_initialisation(sensor_size, events_subset_filtered, initial_velocity=best_velocity, iterations=10)

        best_velocity_vec.append(best_velocity)
        start_t = end_t
        end_t = start_t + sliding_window
    
    vx_avg = sum(pair[0] for pair in best_velocity_vec) / len(best_velocity_vec)
    vy_avg = sum(pair[1] for pair in best_velocity_vec) / len(best_velocity_vec)

    warped_image_zero = accumulate(sensor_size, events, velocity=(0,0))
    warped_image = accumulate(sensor_size, events, velocity=(vx_avg, vy_avg))

    objective_loss = intensity_variance(sensor_size, events, (vx_avg, vy_avg))
    print(f"vx: {best_velocity[0] * 1e6} vy: {best_velocity[1] * 1e6} contrast: {objective_loss:.5f}")
    return warped_image_zero, warped_image, vx_avg, vy_avg, objective_loss


def cmax_slidding_window_overlap(sensor_size: Tuple[int, int], events: numpy.ndarray, events_filtered: numpy.ndarray, sliding_window: float):
    min_t = numpy.min(events['t'])
    max_t = numpy.max(events['t'])

    stride = sliding_window / 4  # Calculate the stride as one-fourth of the window size
    start_t = min_t
    end_t = start_t + sliding_window

    best_velocity_vec = []
    first_iteration = True  # Flag to indicate the first iteration
    
    # Initialize best_velocity outside of the loop to be used as initial_velocity
    best_velocity = None

    while start_t <= max_t:
        events_subset_filtered = events_filtered[(events_filtered['t'] >= start_t) & (events_filtered['t'] < end_t)]

        if first_iteration:
            best_velocity, highest_variance = find_best_velocity_iteratively(sensor_size, events_subset_filtered, increment=500)
            first_iteration = False  # After the first iteration, set this to False
        else:
            best_velocity, highest_variance = find_best_velocity_with_initialisation(sensor_size, events_subset_filtered, initial_velocity=best_velocity, iterations=10)

        best_velocity_vec.append(best_velocity)
        start_t += stride  # Increment start_t by the stride to create overlap
        end_t = start_t + sliding_window  # Recalculate end_t based on the new start_t
    
    vx_avg = sum(pair[0] for pair in best_velocity_vec) / len(best_velocity_vec)
    vy_avg = sum(pair[1] for pair in best_velocity_vec) / len(best_velocity_vec)

    warped_image_zero = accumulate(sensor_size, events, velocity=(0,0))
    warped_image = accumulate(sensor_size, events, velocity=(vx_avg, vy_avg))

    # Assuming 'intensity_variance' and 'accumulate' are functions you have defined elsewhere
    objective_loss = intensity_variance(sensor_size, events, (vx_avg, vy_avg))
    print(f"vx: {vx_avg * 1e6} vy: {vy_avg * 1e6} contrast: {objective_loss:.5f}")
    return warped_image_zero, warped_image, vx_avg, vy_avg, objective_loss


def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    input_img_mask  = numpy.array(A.convert('L'))

    median = cv2.medianBlur(input_img_mask, 9)

    A = numpy.pad(input_img_mask, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))
   

#detect sources in target frequency:
def detect_sources(warped_accumulated_frame, filtered_image_path):
    print('Detecting sources in image frame using masking and region props')
    # detect sources using region props
    filtered_warped_image   = warped_accumulated_frame.filter(PIL.ImageFilter.MedianFilter(9))
    input_img_mask          = numpy.array(filtered_warped_image.convert('L'))
    # filtered_image          = cv2.medianBlur(input_img_mask, 9)
    blurred_image           = cv2.GaussianBlur(input_img_mask, (3, 3), 0)
    _, thresh_image         = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    input_img_mask_labelled = measure.label(thresh_image)
    
    #detect sources as pixel regions covered by event mask
    regions                 = measure.regionprops(label_image=input_img_mask_labelled, intensity_image=thresh_image)

    filtered_regions = [region for region in regions if (region['intensity_max'] * region['area']) > 1]
    # filtered_regions = [region for region in regions if (region.intensity_max * region.area) > 1]

    print(f'Removed {len(regions)-len(filtered_regions)} of {len(regions)} as single event sources with {len(filtered_regions)} sources remaining')

    #sort sources by the area in descending order
    sources = sorted(filtered_regions, key=lambda x: x['area'], reverse=True)

    formatted_sources = {'sources_pix':[], 'sources_astro':[]}

    brightest_sources = sources[:300]
    brightest_source_positions = [[float(source['centroid_weighted'][0]), float(source['centroid_weighted'][1])] for source in brightest_sources]
    source_positions = [[float(source['centroid_weighted'][0]), float(source['centroid_weighted'][1])] for source in sources]
    
    print_message(f'Number of stars extracted: {len(filtered_regions)}', color='yellow', style='bold')

    filtered_warped_image.save(filtered_image_path)
    return sources, source_positions, input_img_mask, filtered_warped_image, input_img_mask_labelled


def display_img_sources(source_extraction_img_path, regions, intensity_img, img_mask, title='Detected Sources', scaling_factor=3, colour_source_overlay=False):
    intensity_img = numpy.array(intensity_img)

    intensity_img_pil = Image.fromarray(numpy.uint8(intensity_img))

    # Create a drawing context
    draw = ImageDraw.Draw(intensity_img_pil)

    for region in regions:
        if region['intensity_max'] > 1:
            centroid = region.centroid
            minr, minc, maxr, maxc = region.bbox
            radius = 10 #numpy.mean([(maxr - minr), (maxc - minc)]) * scaling_factor / 2

            # Calculate the bounding box for the circle
            left = centroid[1] - radius
            top = centroid[0] - radius
            right = centroid[1] + radius
            bottom = centroid[0] + radius
            bbox = [left, top, right, bottom]

            # Draw a red circle as an ellipse within the bounding box
            draw.ellipse(bbox, outline="red", width=2)

            # If you want to overlay specific regions with a color, you'd have to handle that manually
            # For example, drawing semi-transparent rectangles (more complex in PIL)

    # Save the image
    intensity_img_pil.save(source_extraction_img_path, "PNG")


def detect_sources_cca(warped_accumulated_frame):
    """
    Detects sources in an image and filters them based on their properties.
    :param warped_accumulated_frame: PIL Image of the input image.
    :param filtered_image_path: Path where the filtered image will be saved.
    :return: Tuple of sources, source_positions, focus_frame_mask, filtered_warped_image, and focus_frame_mask_labelled
    """
    image_np = numpy.array(warped_accumulated_frame)
    
    # Convert to grayscale for processing
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image_np

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0) #(13,13)
    
    # Threshold the image
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations
    kernel = numpy.ones((3, 3), numpy.uint8)
    closed = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Label connected components
    label_image = measure.label(closed)
    regions = measure.regionprops(label_image=label_image, intensity_image=closed)
    
    filtered_regions = [region for region in regions if region.area > 1]
    print(f'Removed {len(regions)-len(filtered_regions)} of {len(regions)} sources, {len(filtered_regions)} sources remaining')
    
    sources = sorted(filtered_regions, key=lambda x: x.area, reverse=True)
    brightest_sources = sources[:1000]
    brightest_source_positions = [region.centroid for region in brightest_sources]
    source_positions = [region.centroid for region in filtered_regions]
    
    # Prepare to draw circles on the original image using PIL
    output_image = warped_accumulated_frame.copy()
    draw = ImageDraw.Draw(output_image)
    
    # Draw circles on the original image using PIL
    for centroid in source_positions:
        cx, cy = int(centroid[1]), int(centroid[0])  # Swap x and y for image coordinates
        draw.ellipse([(cx-5, cy-5), (cx+5, cy+5)], outline=(0, 255, 0), width=3)
    
    # Save the modified image using PIL to maintain original format and color space
    # output_image.save(filtered_image_path)
    filtered_warped_image = output_image
    
    return sources, brightest_source_positions, closed, filtered_warped_image, label_image


def cone_search_gaia(ra, dec, search_radius):
    # Perform a cone search for the whole field with a radius of 1 degree
    radius = search_radius * u.degree
    Gaia.ROW_LIMIT = 100000
    field_centre_ra = ra
    field_centre_dec = dec
    field_centre_coord = SkyCoord(ra=field_centre_ra, dec=field_centre_dec, unit=(u.deg, u.deg), frame='icrs')
    print(f'Cone searching GAIA for {Gaia.ROW_LIMIT} sources in radius {radius} (deg) around field centre {field_centre_ra}, {field_centre_dec}')
    job = Gaia.cone_search_async(coordinate=field_centre_coord, radius=radius)
    gaia_sources = job.get_results()
    return gaia_sources

def associate_sources_new(sources_calibrated, solution, wcs_calibration, gaia_sources, results_folder_method, save_fig=True):

    #need to search for every source, not just the ones from the brightest sources sublist or the matching sources sublist
    sources_ra = [source['centroid_radec_deg'][0][0] for source in sources_calibrated['sources_astro']]
    sources_dec = [source['centroid_radec_deg'][0][1] for source in sources_calibrated['sources_astro']]

    filtered_gaia_sources = []
    for filtered_source in gaia_sources:
        # for gaia_source in gaia_sources:
        position_pix = wcs_calibration.all_world2pix(filtered_source['ra'], filtered_source['dec'], 0) 
        # print(position_pix)
        if 0 < position_pix[0] < 1280 and 0 < position_pix[1] < 720:
                filtered_gaia_sources.append(filtered_source['phot_g_mean_mag'])

    # Create a list of SkyCoord objects for detected source positions
    source_coords = SkyCoord(ra=sources_ra, dec=sources_dec, unit=(u.degree, u.degree), frame='icrs')
    gaia_source_coords = SkyCoord(ra=gaia_sources['ra'], dec=gaia_sources['dec'], unit=(u.deg, u.deg), frame='icrs')
    all_gaia_mags = [source['phot_g_mean_mag'] for source in gaia_sources]
    
    # Loop through each source and find the closest match locally
    match_idxs, d2d, _ = match_coordinates_sky(source_coords, gaia_source_coords, nthneighbor=1)
    # match_idxs, d2d, _ = match_coordinates_sky(source_coords, SkyCoord(ra=filtered_gaia_source_ra, dec=filtered_gaia_source_dec, unit=(u.deg, u.deg), frame='icrs'), nthneighbor=1)
    duplicate_indices = numpy.where(numpy.bincount(match_idxs) > 1)[0]

    match_mags = []

    #update all sources with associated astrophyisical characteristics of the matching astrometric source
    for i, match_idx in enumerate(match_idxs):
            closest_source = gaia_sources[match_idx]
            position_pix = wcs_calibration.all_world2pix(closest_source['ra'], closest_source['dec'], 0)

            sources_calibrated['sources_astro'][i]['matching_source_ra'] = closest_source['ra']
            sources_calibrated['sources_astro'][i]['matching_source_dec'] = closest_source['dec']
            sources_calibrated['sources_astro'][i]['matching_source_x'] = (position_pix[0])
            sources_calibrated['sources_astro'][i]['matching_source_y'] = (position_pix[1])
            sources_calibrated['sources_astro'][i]['match_error_arcsec'] = d2d[i].arcsecond
            sources_calibrated['sources_astro'][i]['match_error_pix'] = d2d[i].arcsecond / solution.best_match().scale_arcsec_per_pixel
            sources_calibrated['sources_astro'][i]['mag'] = closest_source['phot_g_mean_mag']
            # sources_calibrated['sources_astro'][i]['ID'] = closest_source['source_id']

            match_mags.append(closest_source['phot_g_mean_mag'])

            #check to make sure that the associated source is actually within the image, and store if so
            if 0 < position_pix[0] < 720 and 0 < position_pix[1] < 1280:
                sources_calibrated['sources_astro'][i]['match_in_fov'] = True
            else:
                sources_calibrated['sources_astro'][i]['match_in_fov'] = False

            #flag duplicates 
            if i in duplicate_indices:
                sources_calibrated['sources_astro'][i]['duplicate'] = True
            else:
                sources_calibrated['sources_astro'][i]['duplicate'] = False

    low_error_associated_sources =  [source['match_error_pix'] for source in sources_calibrated['sources_astro'] if source['match_error_pix'] <= 3]
    sources_calibrated['num_good_associated_sources_astro'] = len(low_error_associated_sources)
    print(f'min mag: {numpy.min(match_mags)}')


    filtered_mags = [source['mag'] for source in sources_calibrated['sources_astro'] if source['duplicate'] == False]

    range_of_interest = (5, 16)
    hist_data, bin_edges = numpy.histogram(filtered_mags, bins=100, range=range_of_interest)
    gaia_hist_data, _ = numpy.histogram(filtered_gaia_sources, bins=100, range=range_of_interest)

    # Find the maximum occurrence (height of the tallest bin) within the specified range
    max_occurrence = max(hist_data.max(), gaia_hist_data.max())

    # Plot the histograms
    plt.hist(filtered_mags, bins=100, alpha=0.95, edgecolor='green', histtype='step', label='Detected Sources', range=range_of_interest)
    plt.hist(filtered_gaia_sources, bins=100, alpha=0.95, edgecolor='orange', histtype='step', label='Catalogue Sources', range=range_of_interest)

    plt.ylabel('Occurrences')
    plt.xlabel('G-Band Magnitude')
    plt.xlim(range_of_interest)
    # Adjust the ylim based on the maximum occurrence, adding some padding for visual clarity
    plt.ylim([0, max_occurrence + max_occurrence * 0.1])  # Adding 10% padding

    plt.axvline(x=14.45, color='magenta', linestyle='--', label='Previously reported\nmag limit\n(Ralph et al. 2022)')
    plt.grid('on')
    plt.legend()

    if save_fig: plt.savefig(f'{results_folder_method}/15_histogram_of_star_magnitudes.png')
    # plt.show()
    return sources_calibrated


def astrometric_calibration_new(source_pix_positions, centre_ra, centre_dec):

    #parse to astrometry with priors on pixel scale and centre, then get matches
    # logging.getLogger().setLevel(logging.INFO)

    #number of sources to input for calibration, usually take the first 20 (brightest since sorted)
    number_sources_to_calibrate_from = min(100, len(source_pix_positions))
    
    if number_sources_to_calibrate_from > 0:

        solver = astrometry.Solver(
            astrometry.series_5200_heavy.index_files(
                cache_directory="./astrometry_cache",
                scales={5},
            )
        )
        solution = solver.solve(
            stars=source_pix_positions[0:number_sources_to_calibrate_from],
            # size_hint=None, 
            # position_hint=None,
            size_hint=astrometry.SizeHint(
                lower_arcsec_per_pixel=1.0,
                upper_arcsec_per_pixel=2.0,
            ),
            position_hint=astrometry.PositionHint(
                ra_deg = centre_ra,
                dec_deg = centre_dec,
                radius_deg = 3, #1 is good, 3 also works, trying 5
            ),
            solution_parameters=astrometry.SolutionParameters(
            tune_up_logodds_threshold=None, # None disables tune-up (SIP distortion)
            output_logodds_threshold=14.0, #14.0 good, andre uses 1.0
            # logodds_callback=logodds_callback
            logodds_callback=lambda logodds_list: astrometry.Action.CONTINUE #CONTINUE or STOP
            )
        )
        

        if solution.has_match():
            detected_sources = {'pos_x':[], 'pos_y':[]}
            print('Solution found')
            print(f'Centre ra, dec: {solution.best_match().center_ra_deg=}, {solution.best_match().center_dec_deg=}')
            print(f'Pixel scale: {solution.best_match().scale_arcsec_per_pixel=}')
            print(f'Number of sources found: {len(solution.best_match().stars)}')

            wcs_calibration = astropy.wcs.WCS(solution.best_match().wcs_fields)
            pixels = wcs_calibration.all_world2pix([[star.ra_deg, star.dec_deg] for star in solution.best_match().stars],0,)

            for idx, star in enumerate(solution.best_match().stars):
                detected_sources['pos_x'].append(pixels[idx][0])
                detected_sources['pos_y'].append(pixels[idx][1])

        else:
            print(f'\n *** No astrometric solution found ***')
            detected_sources = None
            wcs_calibration = None

    else:
        print(f'\n *** No astrometric solution found, too few input sources ***')
        detected_sources = None
        wcs_calibration = None
        solution = None

    return solution, detected_sources, wcs_calibration

def run_astrometry_new(sources, source_positions, centre_ra, centre_dec, gaia_sources, results_folder_method):

    print(f'Astrometrically calibrating field centred at {centre_ra}, {centre_dec}')
    solution, detected_sources, wcs_calibration = astrometric_calibration_new(source_positions, centre_ra, centre_dec)
    sources_calibrated = None
    
    if solution is not None:
        if solution.has_match():
            #main container, list of the two pixel and astro/wcs space source entries (dicts)
            sources_calibrated = {'sources_pix':[], 'sources_astro':[]}

            #convert region props object to dict
            for source in sources:
                source_info = {attr: getattr(source, attr) for attr in dir(source)}
                sources_calibrated['sources_pix'].append(source_info)

            #make an astro source entry for each source
            for idx, source in enumerate(sources_calibrated['sources_pix']):
                source_astro = {'astro_ID':None, 'centroid_radec_deg':None, 'mag':None}
                source_astro['centroid_radec_deg'] = wcs_calibration.all_pix2world([sources_calibrated['sources_pix'][idx]['centroid_weighted']], 0,)
                sources_calibrated['sources_astro'].append(source_astro)

            #find matching astrophysical sources for each detected source in the pixel space
            print('Searching GAIA for associated astrophyical sources')
            sources_calibrated = associate_sources_new(sources_calibrated, solution, wcs_calibration, gaia_sources, results_folder_method, save_fig=True)

            sources_calibrated['pixel_scale_arcsec'] = solution.best_match().scale_arcsec_per_pixel
            sources_calibrated['field_centre_radec_deg'] = [solution.best_match().center_ra_deg, solution.best_match().center_dec_deg]
            sources_calibrated['wcs'] = wcs_calibration
            sources_calibrated['detected_sources_pix'] = len(sources_calibrated['sources_pix'])
    
    return sources_calibrated, detected_sources, wcs_calibration



def display_astro_matches(input_img, detected_sources, sources_calibrated, source_positions, wcs_calibration, results_filename, save_fig=False):
    plt.figure(figsize=(12, 7))
    plt.title('Comparison of input brightest detected sources (magenta) with astometric calibrated and associated sources (cyan)',fontsize=10)
    plt.imshow(input_img, vmin=0, vmax=3)
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')

    for source in source_positions:
        circle = Circle((int(source[1]), int(source[0])), color='magenta', radius=7.5, fill=False,alpha=0.65)
        plt.gca().add_patch(circle)

    for idx in range(len(detected_sources['pos_x'])):
        circle = Circle((int(detected_sources['pos_y'][idx]), int(detected_sources['pos_x'][idx])), color='cyan', radius=7.5, fill=False,alpha=0.65)
        plt.gca().add_patch(circle)

    plt.grid(color='white', linestyle='dotted', alpha=0.7)
    plt.tight_layout()
    if save_fig: plt.savefig(f'{results_filename}/3_astro_input_sources_vs_matched_astrometry.png')
    # plt.show()


    # Plotting the image with WCS axis
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': wcs_calibration})
    ax.set_title('Comparison of all detected sources (magenta) with associated astrophysical sources (cyan)')

    # Make the x-axis (RA) tick intervals and grid dense
    ax.coords[0].set_ticks(spacing=0.1*u.deg, color='white', size=6)
    ax.coords[0].grid(color='white', linestyle='solid', alpha=0.7)
    # Customize the y-axis (Dec) tick intervals and labels
    ax.coords[1].set_ticks(spacing=0.1*u.deg, color='white', size=6)
    ax.coords[1].grid(color='white', linestyle='solid', alpha=0.7)

    # Set the number of ticks and labels for the y-axis
    ax.coords[1].set_ticks(number=10)
    ax.set_xlabel('Right Ascention (ICRS deg)')
    ax.set_ylabel('Declination (ICRS deg)')

    # Display the image
    im = ax.imshow(input_img, origin='lower', cmap='viridis', vmin=0, vmax=3)

    for source in sources_calibrated['sources_pix']:
        circle = Circle((source['centroid_weighted'][1], source['centroid_weighted'][0]), color='magenta', radius=3, fill=False,alpha=0.5)
        ax.add_patch(circle)

    for source in sources_calibrated['sources_astro']:
        circle = Circle((source['matching_source_y'], source['matching_source_x']), color='cyan', radius=7.5, fill=False,alpha=0.5)
        ax.add_patch(circle)

    plt.tight_layout()
    if save_fig: plt.savefig(f'{results_filename}/4_astro_detected_vs_matched_from_gaia.png')
    # plt.show()

    #*******************************************************************************************

    #scatter all sources detected and catalogued in image
    plt.figure(figsize=(12, 7))
    plt.title('Comparison of input brightest detected sources (magenta) with astometric calibrated and associated sources (cyan)',fontsize=10)
    # plt.imshow(input_img, vmin=0, vmax=3)
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')
    plt.axvline(x=0, color='magenta', linestyle='--')
    plt.axvline(x=1280, color='magenta', linestyle='--')
    plt.axhline(y=0, color='magenta', linestyle='--')
    plt.axhline(y=720, color='magenta', linestyle='--')

    catalog_sources_x = [source['centroid_weighted'][0] for source in sources_calibrated['sources_pix']]
    catalog_sources_y = [source['centroid_weighted'][1] for source in sources_calibrated['sources_pix']]

    detected_sources_x = [source['matching_source_x'] for source in sources_calibrated['sources_astro']]
    detected_sources_y = [source['matching_source_y'] for source in sources_calibrated['sources_astro']]

    plt.scatter(catalog_sources_y, catalog_sources_x)
    plt.scatter(detected_sources_y, detected_sources_x)
    plt.grid(color='white', linestyle='solid', alpha=0.7)
    plt.tight_layout()
    
    if save_fig: plt.savefig(f'{results_filename}/5_astro_input_sources_vs_matched_astrometry.png')
    # plt.show()

    #*******************************************************************************************

    # Plotting the image with WCS axis with source mags
    fig, ax = plt.subplots(figsize=(24, 16), subplot_kw={'projection': wcs_calibration})
    ax.set_title('Comparison of all detected sources (magenta) with associated astrophysical sources (cyan) and mags',fontsize=10)

    # Make the x-axis (RA) tick intervals and grid dense
    ax.coords[0].set_ticklabel(size=20, weight='bold', color='black')
    ax.coords[0].set_ticks(spacing=0.1*u.deg, color='white', size=6)
    ax.coords[0].grid(color='white', linestyle='solid', alpha=0.7)
    # Customize the y-axis (Dec) tick intervals and labels
    ax.coords[1].set_ticklabel(size=20, weight='bold', color='black')
    ax.coords[1].set_ticks(spacing=0.1*u.deg, color='white', size=6)
    ax.coords[1].grid(color='white', linestyle='solid', alpha=0.7)

    # Set the number of ticks and labels for the y-axis
    ax.coords[1].set_ticks(number=10)
    ax.set_xlabel('Right Ascention (ICRS deg)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Declination (ICRS deg)', fontsize=18, fontweight='bold')

    # Display the image
    im = ax.imshow(input_img, origin='lower', cmap='viridis', vmin=0, vmax=3)

    max_radius = 20  # Maximum radius for the brightest sources
    min_radius = 3   # Minimum radius for the dimmest sources

    mag_min = numpy.min([source['mag'] for source in sources_calibrated['sources_astro']])
    mag_max = numpy.max([source['mag'] for source in sources_calibrated['sources_astro']])

    # Calculate the radius inversely proportional to the magnitude
    for source in sources_calibrated['sources_astro']:
        center_x = source['matching_source_y']
        center_y = source['matching_source_x']
        
        # Normalize magnitude between 0 and 1, then invert the scale for radius calculation
        normalized_mag = (source['mag'] - mag_min) / (mag_max - mag_min)
        radius = (1 - normalized_mag) * (max_radius - min_radius) + min_radius
            
        # Generate hexagon points
        num_vertices = 6
        angle = numpy.linspace(0, 2 * numpy.pi, num_vertices, endpoint=False)
        hexagon = numpy.array([[center_x + radius * numpy.cos(a), center_y + radius * numpy.sin(a)] for a in angle])
            
        # Create and add hexagon patch
        hexagon_patch = Polygon(hexagon, closed=True, color='red', fill=False, alpha=1.0, linewidth=2)
        ax.add_patch(hexagon_patch)

        # Add magnitude text label
        ax.text(center_x + 1, center_y + 1, str(numpy.round(source['mag'], 2)), color='yellow', va='bottom')

    plt.tight_layout()
    if save_fig: plt.savefig(f'{results_filename}/6_astro_matched_from_gaia_labelled.png')
    # plt.show()

    #*******************************************************************************************
    fig, ax = plt.subplots(figsize=(24, 16), subplot_kw={'projection': wcs_calibration})
    ax.set_title('Comparison of all detected sources (magenta) with associated astrophysical sources (cyan) and mags', fontsize=24, fontweight='bold')

    # Customize RA and Dec axes
    ax.coords[0].set_ticklabel(size=20, weight='bold', color='black')
    ax.coords[0].set_ticks(spacing=0.1*u.deg, color='white', size=16)
    ax.coords[0].grid(color='white', linestyle='solid', alpha=0.7)
    ax.coords[1].set_ticklabel(size=20, weight='bold', color='black')
    ax.coords[1].set_ticks(spacing=0.1*u.deg, color='white', size=16)
    ax.coords[1].grid(color='white', linestyle='solid', alpha=0.7)
    ax.coords[1].set_ticks(number=10)
    ax.set_xlabel('Right Ascention (ICRS deg)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Declination (ICRS deg)', fontsize=18, fontweight='bold')

    im = ax.imshow(input_img, origin='lower', cmap='viridis', vmin=0, vmax=3)

    # Normalize magnitude for alpha calculation
    max_radius = 20  # Maximum radius for the brightest sources
    min_radius = 3   # Minimum radius for the dimmest sources

    mag_min = numpy.min([source['mag'] for source in sources_calibrated['sources_astro']])
    mag_max = numpy.max([source['mag'] for source in sources_calibrated['sources_astro']])

    # Calculate the radius inversely proportional to the magnitude
    for source in sources_calibrated['sources_astro']:
        if 0 <= source['mag'] <= 16:  # Only process sources within the specified magnitude range
            center_x = source['matching_source_y']
            center_y = source['matching_source_x']
            
            # Normalize magnitude between 0 and 1, then invert the scale for radius calculation
            normalized_mag = (source['mag'] - mag_min) / (mag_max - mag_min)
            radius = (1 - normalized_mag) * (max_radius - min_radius) + min_radius
            
            # Generate hexagon points
            num_vertices = 6
            angle = numpy.linspace(0, 2 * numpy.pi, num_vertices, endpoint=False)
            hexagon = numpy.array([[center_x + radius * numpy.cos(a), center_y + radius * numpy.sin(a)] for a in angle])
            
            # Create and add hexagon patch
            hexagon_patch = Polygon(hexagon, closed=True, color='red', fill=False, alpha=1.0, linewidth=2)
            ax.add_patch(hexagon_patch)

            # Add magnitude text label
            ax.text(center_x + 1, center_y + 1, str(numpy.round(source['mag'], 2)), color='yellow', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{results_filename}/7_astro_matched_from_gaia_labelled.png')
    # plt.show()

    #*******************************************************************************************

    plt.figure(figsize=(12, 7))
    plt.title('Focused comparison of all detected sources (magenta) with associated astrophysical sources (cyan)',fontsize=10)
    plt.imshow(input_img, vmin=0, vmax=1)
    plt.xlim([600, 700])
    plt.ylim([300, 400])
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')

    for source in sources_calibrated['sources_pix']:
        circle = Circle((source['centroid_weighted'][1], source['centroid_weighted'][0]), color='magenta', radius=1.5, fill=False,alpha=0.5)
        plt.gca().add_patch(circle)

    for source in sources_calibrated['sources_astro']:
        circle = Circle((source['matching_source_y'], source['matching_source_x']), color='cyan', radius=2, fill=False,alpha=0.5)
        plt.gca().add_patch(circle)

    plt.grid(color='white', linestyle='solid', alpha=0.7)
    plt.tight_layout()
    if save_fig: plt.savefig(f'{results_filename}/8_centre_crop-astro_detected_vs_matched_from_gaia.png')
    # plt.show()

    #*******************************************************************************************
    mags = [source['mag'] for source in sources_calibrated['sources_astro']]
    # Set matplotlib style and figure parameters
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = [16, 10]
    plt.rcParams["font.size"] = 20
    # Create a figure and subplot
    figure, subplot = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    # Define window size and create a Hamming window
    window_size = 10
    window = scipy.signal.windows.hamming(window_size * 2 + 1)
    window /= numpy.sum(window)
    # Initialize true positives and match distances
    true_positives = numpy.zeros(len(source_positions), dtype=numpy.float64)
    match_distances = numpy.full(len(source_positions), numpy.nan, dtype=numpy.float64)
    # Calculate distances and find matches
    for index, gaia_pixel_position in enumerate(source_positions):
        distances = numpy.hypot(numpy.array(detected_sources["pos_x"]) - gaia_pixel_position[0], 
                            numpy.array(detected_sources["pos_y"]) - gaia_pixel_position[1])
        closest = numpy.argmin(distances)
        if distances[closest] <= 1.0:  # Tolerance for match
            true_positives[index] = 1.0
            match_distances[index] = distances[closest]

    # Calculate recall using convolution
    recall = scipy.signal.convolve(numpy.concatenate((numpy.repeat(true_positives[0], window_size), true_positives, numpy.repeat(true_positives[-1], window_size))), window, mode="valid")

    # Plot recall curve against magnitudes
    subplot.plot(numpy.sort(mags), recall, c=petroff_colors_6[0], linestyle="-", linewidth=3.0)
    subplot.axhline(y=0.5, color="#000000", linestyle="--")
    subplot.set_xticks(numpy.arange(5, 19), minor=False)
    subplot.set_yticks(numpy.linspace(0.0, 1.0, 11, endpoint=True), minor=False)
    subplot.set_xlim(left=5.0, right=18.5)
    subplot.set_ylim(bottom=-0.05, top=1.05)
    subplot.grid(visible=True, which="major")
    subplot.grid(visible=True, which="minor")
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.set_xlabel("Magnitude")
    subplot.set_ylabel("Recall (ratio of detected over total stars)")
    figure.savefig(f'{results_filename}/9_recall_curve.png')
    # Close the figure
    plt.close(figure)

def format_results(sources_calibrated, source_positions, field_solved):

    if field_solved:
        # get the most important values and save to disk
        # put in float to ensure it can be dumped as json later
        matching_errors = [float(source['match_error_pix']) for source in sources_calibrated['sources_astro']]
        matching_errors_arcsec = [float(source['match_error_arcsec']) for source in sources_calibrated['sources_astro']]
        event_counts = [float(source['event_count']) for source in sources_calibrated['sources_pix']]
        event_rates = [float(source['event_rate']) for source in sources_calibrated['sources_pix']]
        mags = [float(source['mag']) for source in sources_calibrated['sources_astro']]
        areas = [float(source['equivalent_diameter_area']) for source in sources_calibrated['sources_pix']]
        areas_arcsec = [float(area * sources_calibrated['pixel_scale_arcsec']) for area in areas]
        matched_in_fov = [source['match_in_fov'] for source in sources_calibrated['sources_astro']]
        print(f'second occurrence min mag: {numpy.min(mags)}')

        #one produced for each dataset
        output_results = {
            'matching_errors': matching_errors,
            'mean_matching_errors': numpy.mean(matching_errors),
            'std_matching_errors':numpy.std(matching_errors),
            'median_matching_errors':numpy.median(matching_errors),
            'matching_errors_arcsec': matching_errors_arcsec,
            'mean_matching_errors_arcsec': numpy.mean(matching_errors_arcsec),
            'std_matching_errors_arcsec':numpy.std(matching_errors_arcsec),
            'median_matching_errors_arcsec':numpy.median(matching_errors_arcsec),
            'event_counts': event_counts,
            'event_rates': event_rates,
            'mags': mags,
            'areas': areas,
            'areas_arcsec': areas_arcsec, 
            'astrometric_solution': True, 
            'detected_sources': len(source_positions),
            'matched_sources': len(mags),
            'matched_sources_within_fov':matched_in_fov
        }
    else:
        output_results = {
                           'astrometric_solution':False, 
                           'detected_sources': len(source_positions),
                           'matched_sources':0
                           }
    return output_results

def astrometric_calibration(source_pix_positions, centre_ra, centre_dec):

    #parse to astrometry with priors on pixel scale and centre, then get matches
    # logging.getLogger().setLevel(logging.INFO)

    solver = astrometry.Solver(
        astrometry.series_5200_heavy.index_files(
            cache_directory="./astrometry_cache",
            scales={5},
        )
    )

    solution = solver.solve(
        stars=source_pix_positions[0:100],
        # size_hint=None, 
        # position_hint=None,
        size_hint=astrometry.SizeHint(
            lower_arcsec_per_pixel=1.0,
            upper_arcsec_per_pixel=2.0,
        ),
        position_hint=astrometry.PositionHint(
            ra_deg=centre_ra,
            dec_deg=centre_dec,
            radius_deg=5, #1 is good, 3 also works, trying 5
        ),
        solution_parameters=astrometry.SolutionParameters(
        tune_up_logodds_threshold=None, # None disables tune-up (SIP distortion)
        output_logodds_threshold=14.0, #14.0 good, andre uses 1.0
        # logodds_callback=logodds_callback
        logodds_callback=lambda logodds_list: astrometry.Action.STOP #CONTINUE or STOP
        )
    )
    
    detected_sources = {'pos_x':[], 'pos_y':[]}

    if solution.has_match():
        print('Solution found')
        print(f'Centre ra, dec: {solution.best_match().center_ra_deg=}, {solution.best_match().center_dec_deg=}')
        print(f'Pixel scale: {solution.best_match().scale_arcsec_per_pixel=}')
        print_message(f'Number of astrophysical sources found: {len(solution.best_match().stars)}', color='yellow', style='bold')
        # print(f'Number of sources found: {len(solution.best_match().stars)}')

        wcs_calibration = astropy.wcs.WCS(solution.best_match().wcs_fields)
        pixels = wcs_calibration.all_world2pix([[star.ra_deg, star.dec_deg] for star in solution.best_match().stars],0,)

        for idx, star in enumerate(solution.best_match().stars):
            detected_sources['pos_x'].append(pixels[idx][0])
            detected_sources['pos_y'].append(pixels[idx][1])

    else:
        print_message(f'\n *** No astrometric solution found ***', color='red', style='bold')

    return solution, detected_sources, wcs_calibration


def associate_sources(sources_calibrated, solution, wcs_calibration):
    #need to search for every source, not just the ones from the brightest sources sublist or the matching sources sublist
    sources_ra = [source['centroid_radec_deg'][0][0] for source in sources_calibrated['sources_astro']]
    sources_dec = [source['centroid_radec_deg'][0][1] for source in sources_calibrated['sources_astro']]

    # Create a list of SkyCoord objects for your source positions
    source_coords = SkyCoord(ra=sources_ra, dec=sources_dec, unit=(u.degree, u.degree), frame='icrs')

    # Perform a cone search for the whole field with a radius of 1 degree
    radius = 0.5 * u.degree
    Gaia.ROW_LIMIT = 120000
    field_centre_ra = solution.best_match().center_ra_deg
    field_centre_dec = solution.best_match().center_dec_deg
    field_centre_coord = SkyCoord(ra=field_centre_ra, dec=field_centre_dec, unit=(u.deg, u.deg), frame='icrs')
    print(f'Cone searching GAIA for {Gaia.ROW_LIMIT} sources in radius {radius} (deg) around field centre {field_centre_ra}, {field_centre_dec}')
    job = Gaia.cone_search_async(coordinate=field_centre_coord, radius=radius)
    result = job.get_results()

    # Loop through each source and find the closest match locally
    idx, d2d, _ = match_coordinates_sky(source_coords, SkyCoord(ra=result['ra'], dec=result['dec'], unit=(u.deg, u.deg), frame='icrs'))

    #update all sources with associated astrophyisical characteristics of the matching astrometric source
    for i, source_coord in enumerate(source_coords):

        closest_match_idx = idx[i]
        closest_source = result[closest_match_idx]
        position_pix = wcs_calibration.all_world2pix(closest_source['ra'], closest_source['dec'], 0)
        
        sources_calibrated['sources_astro'][i]['matching_source_ra'] = closest_source['ra']
        sources_calibrated['sources_astro'][i]['matching_source_dec'] = closest_source['dec']
        sources_calibrated['sources_astro'][i]['matching_source_x'] = int(position_pix[0])
        sources_calibrated['sources_astro'][i]['matching_source_y'] = int(position_pix[1])
        sources_calibrated['sources_astro'][i]['match_error_asec'] = d2d[i].arcsecond
        sources_calibrated['sources_astro'][i]['match_error_pix'] = d2d[i].arcsecond / solution.best_match().scale_arcsec_per_pixel
        sources_calibrated['sources_astro'][i]['mag'] = closest_source['phot_g_mean_mag']

    #we define the number of astrometrically associated sources as the number of detected sources
    # which are within 3 pixels, or 5.55 arcseconds of the closest associated source
    low_error_associated_sources =  [source['match_error_pix'] for source in sources_calibrated['sources_astro'] if source['match_error_pix'] <= 3]
    sources_calibrated['num_good_associated_sources_astro'] = len(low_error_associated_sources)

    return sources_calibrated

def run_astrometry(astrometry_output, sources, events, warped_accumulated_frame, source_positions, mount_position):

    focus_frame = numpy.array(warped_accumulated_frame.convert('L'))
    mount_ra    = mount_position["ra"]
    mount_dec   = mount_position["dec"]

    print(f'Astrometrically calibrating field centred at {mount_ra}, {mount_dec}')
    solution, detected_sources, wcs_calibration = astrometric_calibration(source_positions, mount_ra, mount_dec)

    #main container, list of the two pixel and astro/wcs space source entries (dicts)
    sources_calibrated = {'sources_pix':[], 'sources_astro':[]}

    #convert region props object to dict
    for source in sources:
        source_info = {attr: getattr(source, attr) for attr in dir(source)}
        sources_calibrated['sources_pix'].append(source_info)

    #make a pixel space source entry for each source
    duration = (events["t"][-1] - events["t"][0]) * 1e6 #seconds
    for idx, source in enumerate(sources_calibrated['sources_pix']):
        event_count = numpy.sum(focus_frame[source['coords'][:,0], source['coords'][:,1]])
        sources_calibrated['sources_pix'][idx]['event_count'] = event_count
        sources_calibrated['sources_pix'][idx]['event_rate'] = event_count/duration

    #make an astro source entry for each source
    for idx, source in enumerate(sources_calibrated['sources_pix']):
        source_astro = {'astro_ID':None, 'centroid_radec_deg':None, 'mag':None}
        source_astro['centroid_radec_deg'] = wcs_calibration.all_pix2world([sources_calibrated['sources_pix'][idx]['centroid_weighted']], 0,)
        sources_calibrated['sources_astro'].append(source_astro)

    #find matching astrophysical sources for each detected source in the pixel space
    print('Searching GAIA for associated astrophyical sources')
    sources_calibrated = associate_sources(sources_calibrated, solution, wcs_calibration)

    print_message(f"Associated gaia astrophysical sources: {len(sources_calibrated['sources_astro'])}", color='yellow', style='bold')

    sources_calibrated['pixel_scale_arcsec'] = solution.best_match().scale_arcsec_per_pixel
    sources_calibrated['field_centre_radec_deg'] = [solution.best_match().center_ra_deg, solution.best_match().center_dec_deg]
    sources_calibrated['wcs'] = wcs_calibration
    sources_calibrated['detected_sources_pix'] = len(sources_calibrated['sources_pix'])

    # #save the final calibrated source array to disk
    # with open(astrometry_output, 'w') as file:
    #     json.dump(sources_calibrated['sources_pix'], file)

    return sources_calibrated, detected_sources, wcs_calibration


def recall_curve(recall_curve, sources_calibrated, source_positions, detected_sources, petroff_colors_6):
    mags = [source['mag'] for source in sources_calibrated['sources_astro']]
    matplotlib.style.use("default")
    matplotlib.rcParams["figure.figsize"] = [16, 10]
    matplotlib.rcParams["font.size"] = 20
    figure, subplot = matplotlib.pyplot.subplots(nrows=1, ncols=1, layout="constrained")
    window_size = 20
    window = scipy.signal.windows.hamming(window_size * 2 + 1)
    window /= numpy.sum(window)

    true_positives = numpy.zeros(len(source_positions), dtype=numpy.float64)
    match_distances = numpy.full(len(source_positions), numpy.nan, dtype=numpy.float64)
    for index, gaia_pixel_position in enumerate(source_positions):
        distances = numpy.hypot(numpy.array(detected_sources["pos_x"]) - gaia_pixel_position[0], numpy.array(detected_sources["pos_y"]) - gaia_pixel_position[1])
        closest = numpy.argmin(distances)
        if distances[closest] <= 5.0:  # Tolerance for match
            # stars_pixel_positions = numpy.delete(stars_pixel_positions, closest, axis=0)
            true_positives[index] = 1.0
            match_distances[index] = distances[closest]

    recall = scipy.signal.convolve(numpy.concatenate((numpy.repeat(true_positives[0], window_size), true_positives, numpy.repeat(true_positives[-1], window_size))), window, mode="valid")
    subplot.plot(numpy.sort(mags), recall, c=petroff_colors_6[0], linestyle="-", linewidth=3.0)
    subplot.axhline(y=0.5, color="#000000", linestyle="--")
    subplot.set_xticks(numpy.arange(5, 19), minor=False)
    subplot.set_yticks(numpy.linspace(0.0, 1.0, 11, endpoint=True), minor=False)
    subplot.set_xlim(left=5.0, right=18.5)
    subplot.set_ylim(bottom=-0.05, top=1.05)
    subplot.grid(visible=True, which="major")
    subplot.grid(visible=True, which="minor")
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.set_xlabel("Magnitude")
    subplot.set_ylabel("Recall (ratio of detected over total stars)")
    figure.savefig(recall_curve)
    plt.close(figure)

def display_calibration(astrometry_display_path, warped_accumulated_frame,source_positions,detected_sources):
    focus_frame = numpy.array(warped_accumulated_frame)
    plt.figure(figsize=(12, 7))
    plt.title('Comparison of input brightest detected sources (magenta) with astometric calibrated and associated sources (cyan)',fontsize=10)
    plt.imshow(focus_frame, vmin=0, vmax=3)
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')

    for source in source_positions:
        circle = Circle((int(source[1]), int(source[0])), color='magenta', radius=7.5, fill=False, alpha=0.65)
        plt.gca().add_patch(circle)

    for idx in range(len(detected_sources['pos_x'])):
        circle = Circle((int(detected_sources['pos_y'][idx]), int(detected_sources['pos_x'][idx])), color='cyan', radius=7.5, fill=False,alpha=0.65)
        plt.gca().add_patch(circle)

    plt.grid(color='white', linestyle='solid', alpha=0.7)
    plt.savefig(astrometry_display_path, dpi=200)


def display_calibration_wcs(astrometry_display_wcs_path, warped_accumulated_frame, wcs_calibration, sources_calibrated):
    focus_frame = numpy.array(warped_accumulated_frame)
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': wcs_calibration})
    ax.set_title('Focused comparison of all detected sources (magenta) with associated astrophysical sources (cyan)',fontsize=15)

    # Make the x-axis (RA) tick intervals and grid dense
    ax.coords[0].set_ticks(spacing=0.1*u.deg, color='white', size=6)
    ax.coords[0].grid(color='white', linestyle='solid', alpha=0.7)
    # Customize the y-axis (Dec) tick intervals and labels
    ax.coords[1].set_ticks(spacing=0.1*u.deg, color='white', size=6)
    ax.coords[1].grid(color='white', linestyle='solid', alpha=0.7)

    # Set the number of ticks and labels for the y-axis
    ax.coords[1].set_ticks(number=10)
    ax.set_xlabel('Right Ascention (ICRS deg)')
    ax.set_ylabel('Declination (ICRS deg)')

    # Display the image
    im = ax.imshow(focus_frame, origin='lower', cmap='viridis', vmin=0, vmax=3)

    for source in sources_calibrated['sources_pix']:
        circle = Circle((source['centroid'][1], source['centroid'][0]), color='magenta', radius=3, fill=False,alpha=0.5)
        ax.add_patch(circle)

    for source in sources_calibrated['sources_astro']:
        circle = Circle((source['matching_source_y'], source['matching_source_x']), color='cyan', radius=7.5, fill=False,alpha=0.5)
        ax.add_patch(circle)

    plt.savefig(astrometry_display_wcs_path, dpi=200)


def astrometry_stat(sources_calibrated, error_pix_path, error_arcsec_pos, gband_mag_path, gband_rate, diam_rate):
    matching_errors = [source['match_error_pix'] for source in sources_calibrated['sources_astro']]
    matching_errors_asec = [source['match_error_asec'] for source in sources_calibrated['sources_astro']]
    event_counts = [source['event_count'] for source in sources_calibrated['sources_pix']]
    event_rates = [source['event_rate'] for source in sources_calibrated['sources_pix']]
    mags = [source['mag'] for source in sources_calibrated['sources_astro']]
    areas = [source['equivalent_diameter_area'] for source in sources_calibrated['sources_pix']]
    areas_arcsec = [area * sources_calibrated['pixel_scale_arcsec'] for area in areas]

    plt.figure(figsize=(10, 6))
    plt.hist(matching_errors, bins=50)
    plt.ylabel('Occurances')
    plt.xlabel('Astrometric association error magnitude (pix)')
    plt.axvline(x=3, color='magenta', linestyle='--', label='Matching error cut-off (3 pix)')
    plt.legend()
    plt.grid('on')
    plt.xlim(min(matching_errors)-2, 10)
    plt.savefig(error_pix_path, dpi=200)

    plt.figure(figsize=(10, 6))
    plt.hist(matching_errors_asec, bins=50)
    plt.ylabel('Occurances')
    plt.xlabel('Astrometric association error magnitude (arcsec)')
    plt.axvline(x=3*sources_calibrated['pixel_scale_arcsec'], color='magenta', linestyle='--', label='Matching error cut-off')
    plt.legend()
    plt.xlim(min(matching_errors_asec)-2, 3*sources_calibrated['pixel_scale_arcsec']+10)
    plt.grid('on')
    plt.savefig(error_arcsec_pos, dpi=200)

    plt.figure(figsize=(10, 6))
    plt.hist(mags, bins=50)
    plt.ylabel('Occurances')
    plt.xlabel('G-Band Magnitude')
    plt.axvline(x=14.45, color='magenta', linestyle='--', label='Previously reported\nmag limit\n(Ralph et al. 2022)')
    plt.legend()
    plt.xlim(min(mags)-2, 16)
    plt.grid('on')
    plt.savefig(gband_mag_path, dpi=200)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(mags, event_rates, c=areas)
    plt.ylabel('Event rate (eps)')
    plt.xlabel('G-Band Magnitude')
    plt.axvline(x=14.45, color='magenta', linestyle='--', label='Previously reported\nmag limit\n(Ralph et al. 2022)')
    plt.legend()
    plt.xlim(min(mags)-2, 16)
    plt.grid('on')
    cbar = plt.colorbar(sc)
    cbar.set_label('Source area (pix)')
    plt.savefig(gband_rate, dpi=200)

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(event_rates, areas, c=mags)
    plt.ylabel('Event rate (eps)')
    plt.xlabel('Equivalent Diameter (pix)')
    plt.grid('on')
    cbar = plt.colorbar(sc)
    cbar.set_label('Astrometric association error magnitude (arcsec)',fontsize=15)
    plt.savefig(diam_rate, dpi=200)


def filter_warped_stars(warped_image: numpy.ndarray):
    filtered_warped_image = warped_image.filter(PIL.ImageFilter.MedianFilter(5))
    return filtered_warped_image



def binarise_warped_image(warped_image_filtered: numpy.ndarray, threshold: float):
    threshold = numpy.percentile(warped_image_filtered, threshold) #highlight the brightest 1% of pixels
    image_file = warped_image_filtered.convert('L')
    image_file = image_file.point( lambda p: 255 if p > threshold else 0 )
    binarise_warped_image = image_file.convert('1')
    return binarise_warped_image



def source_finder(binary_warped_image: numpy.ndarray, warped_accumulated_frame: numpy.ndarray):
    if isinstance(binary_warped_image, Image.Image):
        # Ensure binary_mask is a boolean array for processing
        binary_mask = numpy.array(binary_warped_image.convert('L'), dtype=numpy.uint8)
        binary_mask = binary_mask > 0  # Convert to boolean where nonzero is True/foreground
    else:
        binary_mask = binary_warped_image
    
    # Label the image, considering background as 0
    labels_array, maximum_label = label(binary_mask, connectivity=1, background=0, return_num=True)
    
    # Generate a colormap for the labels, ensuring the background remains black
    colormap = cm.get_cmap('tab20b', maximum_label + 1)
    colored_labels = colormap(numpy.linspace(0, 1, maximum_label + 1))
    
    # Explicitly set the first color (background) to black (RGB)
    colored_labels[0] = numpy.array([0.0, 0.0, 0.0, 1.0])

    # Applying colormap to each label
    colored_labels_image = numpy.zeros((*labels_array.shape, 4))  # Initialize RGBA image
    for label_id in range(maximum_label + 1):
        mask = labels_array == label_id
        colored_labels_image[mask] = colored_labels[label_id]

    # Convert the float [0, 1] RGBA values to uint8 [0, 255]
    colored_labels_image_uint8 = (colored_labels_image * 255).astype(numpy.uint8)
    
    # Convert the RGBA image to a PIL Image
    colored_labels_image_pil = Image.fromarray(colored_labels_image_uint8, 'RGBA')
    draw = ImageDraw.Draw(colored_labels_image_pil)
    draw_org = ImageDraw.Draw(warped_accumulated_frame)

    # Drawing circles around each region's centroid and collecting stars_pixel_positions
    stars_pixel_positions = []
    stars_pixel_diameters = []
    for region in regionprops(labels_array):
        if region.label == 0:  # Skip the background
            continue
        centroid = region.centroid
        diameter = region.equivalent_diameter_area
        stars_pixel_positions.append((centroid[1], centroid[0]))  # Append (x, y) format
        stars_pixel_diameters.append((diameter))  # Append stars diameters

        # Define the bounding box for the circle
        left = centroid[1] - 10
        top = centroid[0] - 10
        right = centroid[1] + 10
        bottom = centroid[0] + 10
        
        # Draw a circle around the centroid
        draw.ellipse([left, top, right, bottom], outline="red",width=2)
        draw_org.ellipse([left, top, right, bottom], outline="red",width=2)

    stars_pixel_positions_array = numpy.array(stars_pixel_positions)
    stars_pixel_diameters_array = numpy.array(stars_pixel_diameters)
    number_of_classes = maximum_label

    print_message(f"Total stars detected: {number_of_classes}", color='yellow', style='bold')
    return colored_labels_image_pil, warped_accumulated_frame, stars_pixel_positions_array, stars_pixel_diameters_array


def source_finder_robust(warped_image_filtered, overlay_image_path):
    """
    Detect stars in an image, filtering out single-pixel noise and artifacts.

    Parameters:
    - warped_image_filtered: PIL.Image.Image object of a filtered warped image.

    Returns:
    - A PIL.Image.Image object with detected stars marked and circled.
    - A list of tuples containing the center points (x, y) of each detected star.
    - A list of diameters for each detected star.
    """
    # Convert PIL Image to a NumPy array (OpenCV compatible)
    image_array = numpy.array(warped_image_filtered)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Use cv2.threshold to create a binary image for contour detection
    _, thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours (potential stars)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out single-pixel contours
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 1]

    # Initialize a copy of the image to draw the filtered contours
    processed_image = image_array.copy()

    # Initialize lists to hold center points and diameters
    centers = []
    diameters = []

    def get_enclosing_circle(contour):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius

    # Draw the enclosing circle for each filtered contour and collect center points and diameters
    for contour in filtered_contours:
        center, radius = get_enclosing_circle(contour)
        diameter = radius * 2
        centers.append(center)
        diameters.append(diameter)
        cv2.circle(processed_image, center, radius, (0, 255, 0), 2)
        cv2.circle(processed_image, center, 2, (0, 0, 255), -1)
    
    star_pixel_positions = numpy.array(centers)
    plt.style.use("dark_background")
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["font.size"] = 16
    figure, subplot = plt.subplots(nrows=1, ncols=1, layout="constrained")
    accumulated_frame_array = numpy.array(warped_image_filtered)
    subplot.imshow(accumulated_frame_array, cmap='gray', origin='lower')
    subplot.scatter(star_pixel_positions[:, 0], star_pixel_positions[:, 1], s=550, marker="o", facecolors="none", edgecolors='red', linewidths=2, label='Source finder')

    plt.savefig(overlay_image_path)
    plt.close(figure)
    # Convert the processed NumPy array back to a PIL Image
    processed_image_pil = Image.fromarray(processed_image)
    # print_message(f"Total stars detected: {len(radius)}", color='yellow', style='bold')
    im_flip = ImageOps.flip(processed_image_pil)
    return im_flip, numpy.array(centers), numpy.array(diameters)


def stars_clustering(events, method="spectral_clustering", neighbors=30, opt_clusters=20, min_cluster_size=400, eps=10, min_samples=150):
    pol = events["on"]
    ts = events["t"] / 1e6
    x = events["x"]
    y = events["y"]
    ALL = len(pol)
    selected_events = numpy.array([y, x, ts * 0.0001, pol * 0]).T
    adMat_cleaned = kneighbors_graph(selected_events, n_neighbors=neighbors)
    if method == "spectral_clustering":
        clustering = SpectralClustering(n_clusters=opt_clusters, random_state=0,
                                        affinity='precomputed_nearest_neighbors',
                                        n_neighbors=neighbors, assign_labels='kmeans',
                                        n_jobs=-1).fit_predict(adMat_cleaned)
    elif method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        clusterer.fit(selected_events)
        clustering = clusterer.labels_
    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
        clusterer.fit(selected_events)
        clustering = clusterer.labels_
    else:
        raise ValueError("Invalid clustering method")
    return clustering


def astrometry_fit(stars_pixel_positions, stars_pixel_diameters, metadata, slice):
    # Set logging level to INFO to see astrometry process details
    logging.getLogger().setLevel(logging.INFO)

    if slice:
        sorted_indices = numpy.argsort(stars_pixel_diameters)[::-1]
        slice_length   = int(numpy.ceil(len(stars_pixel_diameters) / 2))

        sorted_diameters_sliced = stars_pixel_diameters[sorted_indices][:slice_length]
        sorted_positions_sliced = stars_pixel_positions[sorted_indices][:slice_length]
    else:
        sorted_indices = numpy.argsort(stars_pixel_diameters)[::-1]
        sorted_positions_sliced = stars_pixel_positions[sorted_indices]
        sorted_positions_sliced = sorted_positions_sliced
    
    # Initialize the astrometry solver with index files
    solver = astrometry.Solver(
        astrometry.series_5200_heavy.index_files(
            cache_directory="astrometry_cache",
            scales={4},
        )
    )
    # Solve the astrometry using the provided star center coordinates
    solution = solver.solve(
        stars=sorted_positions_sliced,
        size_hint=astrometry.SizeHint(
        lower_arcsec_per_pixel=1.0,
        upper_arcsec_per_pixel=3.0,
    ),
        position_hint=astrometry.PositionHint(
        ra_deg=metadata["ra"],
        dec_deg=metadata["dec"],
        radius_deg=1,
    ),
        solution_parameters=astrometry.SolutionParameters(),
    )
    
    # Ensure that a match has been found
    assert solution.has_match()

    return solution


def astrometry_fit_with_gaia(observation_time, warped_image, stars_pixel_positions, stars_pixel_diameters, metadata, slice):
    # Set logging level to INFO to see astrometry process details
    logging.getLogger().setLevel(logging.INFO)

    if slice:
        sorted_indices = numpy.argsort(stars_pixel_diameters)[::-1]
        slice_length   = int(numpy.ceil(len(stars_pixel_diameters) / 2))

        sorted_diameters_sliced = stars_pixel_diameters[sorted_indices][:slice_length]
        sorted_positions_sliced = stars_pixel_positions[sorted_indices][:slice_length]
    else:
        sorted_indices = numpy.argsort(stars_pixel_diameters)[::-1]
        sorted_positions_reordered = stars_pixel_positions[sorted_indices]
        sorted_positions_sliced = sorted_positions_reordered

    # Initialize the astrometry solver with index files
    solver = astrometry.Solver(
        astrometry.series_5200_heavy.index_files(
            cache_directory="astrometry_cache",
            scales={4,5,6},
        )
    )
    solution = cache.load(
        "astrometry",
        lambda: solver.solve(
            stars=sorted_positions_sliced,
            size_hint=astrometry.SizeHint(
            lower_arcsec_per_pixel=1.0,
            upper_arcsec_per_pixel=3.0,
            ),
            position_hint=astrometry.PositionHint(
            ra_deg=metadata["ra"],
            dec_deg=metadata["dec"],
            radius_deg=1,
            ),
            solution_parameters=astrometry.SolutionParameters(
                sip_order=0,
                tune_up_logodds_threshold=None,
                logodds_callback=lambda logodds_list: (
                    astrometry.Action.STOP
                    if logodds_list[0] > 100.0
                    else astrometry.Action.CONTINUE
                ),
            ),
        ),
    )
    match = solution.best_match()

    # download stars from Gaia
    gaia_stars = stars.gaia(
        center_ra_deg=match.center_ra_deg,
        center_dec_deg=match.center_dec_deg,
        radius=numpy.hypot(
            warped_image.pixels.shape[0], warped_image.pixels.shape[1]
        )
        * match.scale_arcsec_per_pixel
        / 3600.0,
        cache_key="gaia",
        observation_time=observation_time,
    )

    if len(sorted_positions_sliced) == 0:
        tweaked_wcs = match.astropy_wcs()
    else:
        tweaked_wcs = cache.load(
                "tweaked_wcs",
                lambda: stars.tweak_wcs(
                    accumulated_frame=warped_image,
                    initial_wcs=match.astropy_wcs(),
                    gaia_stars=gaia_stars[gaia_stars["phot_g_mean_mag"] < 15],
                    stars_pixel_positions=sorted_positions_sliced,
                ),
            )

    return tweaked_wcs, gaia_stars


def astrometry_overlay(output_path, solution, colored_labels_image_pil):
    # Convert PIL Image to a format that can be used with Matplotlib
    img_buffer = io.BytesIO()
    colored_labels_image_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_data = plt.imread(img_buffer)

    # Setup the plot with WCS projection
    match = solution.best_match()
    wcs = WCS(match.wcs_fields)
    fig, ax = plt.subplots(subplot_kw={'projection': wcs})
    ax.imshow(img_data, origin='lower')

    # Convert star positions from RA/Dec to pixels
    stars = wcs.all_world2pix([[star.ra_deg, star.dec_deg] for star in match.stars], 0)

    # Retrieve and scale star magnitudes
    magnitudes = numpy.array([star.metadata["mag"] for star in match.stars])
    scaled_magnitudes = 1.0 - (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    sizes = scaled_magnitudes * 1000  # Adjust the scaling factor as needed

    # Plot hexagons for stars
    for (x, y), size in zip(stars, sizes):
        hexagon = RegularPolygon((x, y), numVertices=6, radius=size**0.5, orientation=0, 
                                 facecolor='none', edgecolor='white', linewidth=1.5, transform=ax.get_transform('pixel'))
        ax.add_patch(hexagon)

    # Calculate and plot grid lines for RA and DEC
    ra = numpy.arange(int(match.center_ra_deg - 1), int(match.center_ra_deg + 1), 1) * u.deg
    dec = numpy.arange(int(match.center_dec_deg - 1), int(match.center_dec_deg + 1), 1) * u.deg
    ra_grid, dec_grid = numpy.meshgrid(ra, dec)
    ax.coords.grid(True, color='blue', ls='solid', alpha=1.0)

    ax.set_xlim(0, img_data.shape[1])
    ax.set_ylim(0, img_data.shape[0])

    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    fig.savefig(output_path, dpi=100)
    return stars


def gaia_stars_processing(accumulated_frame, tweaked_wcs, gaia_stars):
    # Convert Gaia star positions from celestial coordinates to pixel coordinates
    gaia_pixel_positions = tweaked_wcs.all_world2pix(numpy.array([gaia_stars["ra"], gaia_stars["dec"]]).transpose(), 0)
    rounded_gaia_pixel_positions = numpy.round(gaia_pixel_positions).astype(numpy.uint16)
    accumulated_frame_array = numpy.array(accumulated_frame)
    
    # Apply mask to filter out positions outside the image boundaries
    gaia_base_mask = numpy.logical_and.reduce((
        rounded_gaia_pixel_positions[:, 0] >= 0,
        rounded_gaia_pixel_positions[:, 1] >= 0,
        rounded_gaia_pixel_positions[:, 0] < accumulated_frame_array.shape[1],
        rounded_gaia_pixel_positions[:, 1] < accumulated_frame_array.shape[0],
    ))
    
    # Apply the base mask to pixel and world coordinates
    gaia_pixel_positions = gaia_pixel_positions[gaia_base_mask]
    gaia_world_positions = numpy.array([gaia_stars["ra"], gaia_stars["dec"]]).transpose()[gaia_base_mask]
    rounded_gaia_pixel_positions = rounded_gaia_pixel_positions[gaia_base_mask]
    
    # Assuming a simple threshold for and_mask creation to filter valid Gaia positions
    # This part might need to be adjusted based on the specific use case
    and_mask = accumulated_frame_array[:,:,0] > 0 # Adjust according to the actual condition
    gaia_mask = and_mask[rounded_gaia_pixel_positions[:, 1], rounded_gaia_pixel_positions[:, 0]]
    
    # Apply the additional gaia_mask
    gaia_pixel_positions = gaia_pixel_positions[gaia_mask]
    gaia_world_positions = gaia_world_positions[gaia_mask]
    gaia_magnitudes = gaia_stars["phot_g_mean_mag"][gaia_base_mask][gaia_mask]

    return gaia_pixel_positions, gaia_world_positions, gaia_magnitudes


def astrometry_overlay_with_gaia(accumulated_frame, stars_pixel_positions, tweaked_wcs, gaia_stars, petroff_colors_6, astrometry_gaia_path):
    """
    Visualizes the tweaked WCS on the accumulated image by plotting Gaia stars and detected stars
    over the accumulated frame.
    """
    # Apply styling for visualization
    matplotlib.style.use("dark_background")
    matplotlib.rcParams["figure.figsize"] = [20, 12]
    matplotlib.rcParams["font.size"] = 16
    
    # Create figure and subplot
    figure, subplot = plt.subplots(nrows=1, ncols=1, layout="constrained")

    # Display the accumulated frame
    # Convert the PIL image to a NumPy array for display
    accumulated_frame_array = numpy.array(accumulated_frame)
    subplot.imshow(accumulated_frame_array, cmap='gray', origin='lower')

    gaia_pixel_positions, gaia_world_positions, gaia_magnitudes = gaia_stars_processing(accumulated_frame_array, tweaked_wcs,gaia_stars)

    # Plot Gaia stars with vertically flipped positions
    if len(gaia_magnitudes) > 0:
        subplot.scatter(gaia_pixel_positions[:, 0], gaia_pixel_positions[:,1], s=(gaia_magnitudes.max() - gaia_magnitudes) * 4, c=petroff_colors_6[0], label='Gaia Stars')

    # # # Plot additional solution data
    # # if solution:
    # #     match = solution.best_match()
    # #     wcs = WCS(match.wcs_fields)
        
    # #     # Assuming match.stars contains RA and Dec for each star
    # #     stars = wcs.all_world2pix([[star.ra_deg, star.dec_deg] for star in match.stars], 0)
        
    # #     # Plot hexagons for stars from the solution
    # #     for (x, y) in stars:
    # #         hexagon = RegularPolygon((x, y), numVertices=6, radius=10, orientation=0,
    # #                                  facecolor='none', edgecolor='white', linewidth=1.5)
    # #         subplot.add_patch(hexagon)

    # # Plot detected stars
    # if len(stars_pixel_positions) > 0:
    #     subplot.scatter(stars_pixel_positions[:, 0], stars_pixel_positions[:, 1], s=550, marker="o", facecolors="none", edgecolors='red', linewidths=2, label='Source finder')
    
    # Save the figure
    figure.savefig(astrometry_gaia_path)
    plt.close(figure)



def evaluate_astrometry(accumulated_frame, astrometry_stars, tweaked_wcs, gaia_stars, stars_pixel_positions, petroff_colors_6, astrometry_gaia_mag_path, gaia_matches_path, astrometry_final, tolerance=15.0):
    matplotlib.style.use("default")
    matplotlib.rcParams["figure.figsize"] = [16, 10]
    matplotlib.rcParams["font.size"] = 20
    figure, subplot = matplotlib.pyplot.subplots(nrows=1, ncols=1, layout="constrained")
    window_size = 20
    window = scipy.signal.windows.hamming(window_size * 2 + 1)
    window /= numpy.sum(window)

    gaia_pixel_positions, gaia_world_positions, gaia_magnitudes = gaia_stars_processing(accumulated_frame, tweaked_wcs, gaia_stars)

    true_positives = numpy.zeros(len(gaia_pixel_positions), dtype=numpy.float64)
    match_distances = numpy.full(len(gaia_pixel_positions), numpy.nan, dtype=numpy.float64)
    matched_star_positions = []
    true_positive_counter = 0
    all_matching_data = []
    for index, gaia_pixel_position in enumerate(gaia_pixel_positions):
        if len(stars_pixel_positions) == 0:
            continue
        distances = numpy.hypot(stars_pixel_positions[:, 0] - gaia_pixel_position[0], stars_pixel_positions[:, 1] - gaia_pixel_position[1])
        closest = numpy.argmin(distances)
        if distances[closest] <= tolerance:  # Tolerance for match
            # stars_pixel_positions = numpy.delete(stars_pixel_positions, closest, axis=0)
            true_positives[index] = 1.0
            match_distances[index] = distances[closest]
            matched_star_positions.append(stars_pixel_positions[closest])
            all_matching_data.append([stars_pixel_positions[closest][0],stars_pixel_positions[closest][1],gaia_pixel_position[0],gaia_pixel_position[1],gaia_magnitudes[closest]])
            true_positive_counter+=1

    matched_star_positions      = numpy.array(matched_star_positions)
    all_matching_data           = numpy.array(all_matching_data)

    unique_star_pos             = numpy.array(list(set(tuple(p) for p in matched_star_positions)))
    _, unique_indices           = numpy.unique(all_matching_data[:, :2], axis=0, return_index=True)
    unique_star_pos_gaia_mag    = all_matching_data[numpy.sort(unique_indices)]

    recall = scipy.signal.convolve(numpy.concatenate((numpy.repeat(true_positives[0], window_size), true_positives, numpy.repeat(true_positives[-1], window_size))), window, mode="valid")
    subplot.plot(gaia_magnitudes, recall, c=petroff_colors_6[0], linestyle="-", linewidth=3.0)
    subplot.axhline(y=0.5, color="#000000", linestyle="--")
    subplot.set_xticks(numpy.arange(5, 19), minor=False)
    subplot.set_yticks(numpy.linspace(0.0, 1.0, 11, endpoint=True), minor=False)
    subplot.set_xlim(left=5.0, right=18.5)
    subplot.set_ylim(bottom=-0.05, top=1.05)
    subplot.grid(visible=True, which="major")
    subplot.grid(visible=True, which="minor")
    subplot.spines["top"].set_visible(False)
    subplot.spines["right"].set_visible(False)
    subplot.set_xlabel("Magnitude")
    subplot.set_ylabel("Recall (ratio of detected over total stars)")
    figure.savefig(astrometry_gaia_mag_path)
    plt.close(figure)

    # Plotting additional image with true positive matches
    plt.style.use("dark_background")
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["font.size"] = 16
    figure, subplot = plt.subplots(nrows=1, ncols=1, layout="constrained")
    accumulated_frame_array = numpy.array(accumulated_frame)
    subplot.imshow(accumulated_frame_array, cmap='gray', origin='lower')
    matched_positions = gaia_pixel_positions[true_positives == 1]
    gaia_magnitudes_true_positives = gaia_magnitudes[true_positives == 1]

    if len(matched_positions) > 0:
        subplot.scatter(matched_positions[:, 0], matched_positions[:, 1], s=(gaia_magnitudes.max() - gaia_magnitudes_true_positives) * 4, c=petroff_colors_6[0], label='Gaia Stars - True Positives')
        # for position in matched_positions:
        #     circle = Circle((position[0], position[1]), radius=5, edgecolor='red', facecolor='none', linewidth=2)
        #     subplot.add_patch(circle)
        # subplot.scatter(stars_pixel_positions[:, 0], stars_pixel_positions[:, 1], s=200, marker="o", facecolors="none", edgecolors='green', linewidths=2, label='Source finder')
        subplot.scatter(matched_star_positions[:, 0], matched_star_positions[:, 1], s=550, marker="o", facecolors="none", edgecolors='red', linewidths=2, label='Source finder')
    # subplot.legend()
    plt.savefig(gaia_matches_path)
    plt.close(figure)

    plt.style.use("dark_background")
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["font.size"] = 16

    figure, ax = plt.subplots(nrows=1, ncols=1, layout="constrained")
    ax.imshow(accumulated_frame_array, origin='lower')

    # figure, ax = plt.subplots(nrows=1, ncols=1, layout="constrained")
    # accumulated_frame_array = numpy.array(accumulated_frame)
    # ax.imshow(accumulated_frame_array, cmap='gray', origin='lower')

    x_coords = unique_star_pos_gaia_mag[:, 0]
    y_coords = unique_star_pos_gaia_mag[:, 1]
    stars = unique_star_pos_gaia_mag[:,0:2]
    magnitudes = unique_star_pos_gaia_mag[:, -1]

    # Scale magnitudes for hexagon sizes
    scaled_magnitudes = 1.0 - (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    sizes = scaled_magnitudes * 1000  # Adjust the scaling factor as needed

    # for (x, y), size in zip(stars, sizes):
    #     hexagon = RegularPolygon((x, y), numVertices=6, radius=size**0.5, orientation=0, 
    #                              facecolor='none', edgecolor='white', linewidth=1.5, transform=ax.get_transform('pixel'))
    #     ax.add_patch(hexagon)

    # Plot hexagons for each star
    for x, y, size in zip(x_coords, y_coords, sizes):
        hexagon = RegularPolygon((x, y), numVertices=6, radius=size**0.5, orientation=0, 
                                 facecolor='none', edgecolor='white', linewidth=1.5)
        ax.add_patch(hexagon)

    ax.set_xlim([x_coords.min() - 10, x_coords.max() + 10])
    ax.set_ylim([y_coords.min() - 10, y_coords.max() + 10])
    ax.set_aspect('equal') 

    plt.savefig(astrometry_final)
    plt.close(figure)
    
    print_message(f"Total extracted stars: {len(stars_pixel_positions[:, 0])}", color='yellow', style='bold')
    print_message(f"Total astrometry stars: {len(astrometry_stars[:, 0])}", color='yellow', style='bold')
    print_message(f"Total gaia stars: {len(gaia_pixel_positions[:, 0])}", color='yellow', style='bold')
    print_message(f"Total detected stars: {len(unique_star_pos)}", color='red', style='bold')
    





# def astrometry_overlay_with_gaia2(warped_accumulated_frame, solution, tweaked_wcs, gaia_stars, stars_pixel_positions, petroff_colors_6, astrometry_gaia_path, astrometry_gaia_mag_path):
#     """
#     Visualizes the tweaked WCS on the warped accumulated image by plotting Gaia stars and detected stars
#     over the image, and overlaying solution data.
#     """
#     # Apply styling for visualization
#     matplotlib.style.use("dark_background")
#     plt.rcParams["figure.figsize"] = [20, 12]
#     plt.rcParams["font.size"] = 16

#     # Prepare the figure and subplot
#     figure, subplot = plt.subplots(nrows=1, ncols=1, layout="constrained")

#     # Display the warped accumulated frame directly
#     subplot.imshow(warped_accumulated_frame)

#     # Convert Gaia star positions from celestial coordinates to pixel coordinates
#     gaia_pixel_positions = tweaked_wcs.all_world2pix(numpy.array([gaia_stars["ra"], gaia_stars["dec"]]).transpose(), 0)

#     # Apply a mask to filter out positions outside the image boundaries
#     valid_gaia_positions_mask = (gaia_pixel_positions[:, 0] >= 0) & (gaia_pixel_positions[:, 0] < warped_accumulated_frame.width) & \
#                                 (gaia_pixel_positions[:, 1] >= 0) & (gaia_pixel_positions[:, 1] < warped_accumulated_frame.height)
    
    
#     gaia_pixel_positions = gaia_pixel_positions[valid_gaia_positions_mask]
#     gaia_magnitudes = gaia_stars["phot_g_mean_mag"][valid_gaia_positions_mask]

#     # Flip the y-coordinates of gaia_pixel_positions
#     max_y = numpy.max(gaia_pixel_positions[:, 1])
#     flipped_y = max_y - gaia_pixel_positions[:, 1]

#     # Plot Gaia stars with vertically flipped positions
#     if len(gaia_magnitudes) > 0:
#         subplot.scatter(gaia_pixel_positions[:, 0], flipped_y, s=(gaia_magnitudes.max() - gaia_magnitudes) * 4, c=petroff_colors_6[0], label='Gaia Stars')

#     # Plot additional solution data
#     if solution:
#         match = solution.best_match()
#         wcs = WCS(match.wcs_fields)
        
#         # Assuming match.stars contains RA and Dec for each star
#         stars = wcs.all_world2pix([[star.ra_deg, star.dec_deg] for star in match.stars], 0)
        
#         # Plot hexagons for stars from the solution
#         for (x, y) in stars:
#             hexagon = RegularPolygon((x, y), numVertices=6, radius=10, orientation=0,
#                                      facecolor='none', edgecolor='white', linewidth=1.5)
#             subplot.add_patch(hexagon)

#     subplot.legend()

#     # Save the figure
#     figure.savefig(astrometry_gaia_path)
#     plt.close(figure)



# def save_solution_to_json(solution, output_path):
#     if solution.has_match():
#         data_to_save = []
#         for star in solution.best_match().stars:
#             star_data = {
#                 "ra_deg": star.ra_deg,
#                 "dec_deg": star.dec_deg,
#                 "metadata": star.metadata
#             }
#             data_to_save.append(star_data)

#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(data_to_save, f, ensure_ascii=False, indent=4)
#     else:
#         print("No astrometry match found. No data saved.")



def analyze_star_data(json_path):
    # Read the JSON data from the file
    with open(json_path, 'r', encoding='utf-8') as file:
        stars_data = json.load(file)
    
    # Extract magnitudes from the data
    magnitudes = [star['metadata']['mag'] for star in stars_data if 'mag' in star['metadata']]
    ra = [star['metadata']['ra'] for star in stars_data if 'ra' in star['metadata']]
    dec = [star['metadata']['dec'] for star in stars_data if 'dec' in star['metadata']]

    # Calculate the required information
    num_stars = len(magnitudes)
    min_mag = min(magnitudes) if magnitudes else None
    max_mag = max(magnitudes) if magnitudes else None
    
    # Print the results
    print_message(f"Number of stars: {num_stars}", color='yellow', style='bold')
    print_message(f"Min mag: {min_mag}", color='green', style='bold')
    print_message(f"Max mag: {max_mag}", color='red', style='bold')
    print(f"All mags: {magnitudes}")
    print(f"All ra: {ra}")
    print(f"All dec: {dec}")

    # Plotting the histogram of magnitudes
    plt.figure(figsize=(10, 6))
    plt.hist(magnitudes, bins='auto', color='skyblue', alpha=0.7, rwidth=0.85)
    plt.title('Histogram of Star Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('# detected sources')
    plt.grid(axis='y', alpha=0.75)
    
    plt.show()
    
    return num_stars, min_mag, max_mag, magnitudes, ra, dec


def find_best_velocity_iteratively(sensor_size: Tuple[int, int], events: numpy.ndarray, increment=100):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity = None
    highest_variance = float('-inf')
    
    variances = []  # Storing variances for each combination of velocities
    
    for vy in tqdm(range(-1000, 1001, increment)):
        for vx in range(-1000, 1001, increment):
            current_velocity = (vx / 1e6, vy / 1e6)
            
            optimized_velocity = optimize_local(sensor_size=sensor_size,
                                                events=events,
                                                initial_velocity=current_velocity,
                                                tau=1000,
                                                heuristic_name="variance",
                                                method="Nelder-Mead",
                                                callback=None)
            
            objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
            
            variances.append((optimized_velocity, objective_loss))
            
            if objective_loss > highest_variance:
                highest_variance = objective_loss
                best_velocity = optimized_velocity
    
    # Converting variances to a numpy array for easier handling
    variances = numpy.array(variances, dtype=[('velocity', float, 2), ('variance', float)])
    print(f"vx: {best_velocity[0] * 1e6} vy: {best_velocity[1] * 1e6} contrast: {highest_variance}")
    return best_velocity, highest_variance



def accumulate4D(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    linear_vel: tuple[float, float],
    angular_vel: tuple[float, float, float],
    zoom: float,
):
    return CumulativeMap(
        pixels=dvs_sparse_filter_extension.accumulate4D(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            linear_vel[0],
            linear_vel[1],
            angular_vel[0],
            angular_vel[1],
            angular_vel[2],
            zoom,
        ),
        offset=0
    )

def accumulate4D_cnt(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    linear_vel: numpy.ndarray,
    angular_vel: numpy.ndarray,
    zoom: numpy.ndarray,
):
    return CumulativeMap(
        pixels=dvs_sparse_filter_extension.accumulate4D_cnt(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            linear_vel[0],
            linear_vel[1],
            angular_vel[0],
            angular_vel[1],
            angular_vel[2],
            zoom,
        ),
        offset=0
    )


def geometric_transformation(
        resolution: float, 
        rotation_angle: float
):
    rotated_particles = dvs_sparse_filter_extension.geometricTransformation(
        resolution, 
        rotation_angle)
    return rotated_particles


def render(
    cumulative_map: CumulativeMap,
    colormap_name: str,
    gamma: typing.Callable[[numpy.ndarray], numpy.ndarray],
    bounds: typing.Optional[tuple[float, float]] = None,
):
    colormap = matplotlib.pyplot.get_cmap(colormap_name) # type: ignore
    if bounds is None:
        bounds = (cumulative_map.pixels.min(), cumulative_map.pixels.max())
    scaled_pixels = gamma(
        numpy.clip(
            ((cumulative_map.pixels - bounds[0]) / (bounds[1] - bounds[0])),
            0.0,
            1.0,
        )
    )
    image = PIL.Image.fromarray(
        (colormap(scaled_pixels)[:, :, :3] * 255).astype(numpy.uint8)
    )
    return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

def generate_palette(cluster_count):
    """Generates a color palette for a given number of clusters."""
    palette = []
    for i in range(cluster_count):
        hue = i / cluster_count
        lightness = 0.5  # Middle value ensures neither too dark nor too light
        saturation = 0.9  # High saturation for vibrant colors
        rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue, lightness, saturation))
        palette.append(rgb)
    return palette

def see_cluster_color(events, cluster):
    """Processes events and generates an image."""
    # Generate color palette
    palette = generate_palette(max(cluster))
    
    # Extract dimensions and event count
    xs, ys = int(events["x"].max()) + 1, int(events["y"].max()) + 1
    event_count = events.shape[0]
    
    # Initialize arrays
    wn = numpy.full((xs, ys), -numpy.inf)
    img = numpy.full((xs, ys, 3), 255, dtype=numpy.uint8)
    
    # Process each event
    for idx in tqdm(range(event_count)):
        x = events["x"][idx]
        y = events["y"][idx]
        label = cluster[idx]
        if label < 0:
            label = 1
        
        wn[x, y] = label + 1
        img[wn == 0] = [0, 0, 0]
        img[wn == label + 1] = palette[label - 1]
    return numpy.rot90(img, -1)

def get_high_intensity_bbox(image):
    """Return the bounding box of the region with the highest intensity in the image."""
    # Convert the image to grayscale
    gray = image.convert("L")
    arr = numpy.array(gray)
    threshold_value = arr.mean() + arr.std()
    high_intensity = (arr > threshold_value).astype(numpy.uint8)
    labeled, num_features = scipy.ndimage.label(high_intensity)
    slice_x, slice_y = [], []

    for i in range(num_features):
        slice_xi, slice_yi = scipy.ndimage.find_objects(labeled == i + 1)[0]
        slice_x.append(slice_xi)
        slice_y.append(slice_yi)

    if not slice_x:
        return None

    max_intensity = -numpy.inf
    max_intensity_index = -1
    for i, (slice_xi, slice_yi) in enumerate(zip(slice_x, slice_y)):
        if arr[slice_xi, slice_yi].mean() > max_intensity:
            max_intensity = arr[slice_xi, slice_yi].mean()
            max_intensity_index = i

    return (slice_y[max_intensity_index].start, slice_x[max_intensity_index].start, 
            slice_y[max_intensity_index].stop, slice_x[max_intensity_index].stop)



def generate_combined_image(sensor_size, events, labels, vx, vy):
    unique_labels = numpy.unique(labels)
    total_events = len(events)

    # Generate the first warped image to determine its width and height
    first_warped_image = accumulate_cnt(sensor_size, 
                                        events=events[labels == unique_labels[0]], 
                                        velocity=(vx[labels == unique_labels[0]], vy[labels == unique_labels[0]]))
    warped_image_rendered = render(first_warped_image, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
    
    num_cols = min(4, len(unique_labels))
    num_rows = int(numpy.ceil(len(unique_labels) / 4))
    combined_final_segmentation = Image.new('RGB', (num_cols * warped_image_rendered.width, num_rows * warped_image_rendered.height))
    
    try:
        font = ImageFont.truetype("./src/Roboto-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    variances = []
    max_variance_index = -1
    previous_coordinates = (0, 0)
    
    # Initialize variables to store the required additional information
    max_variance_events = None
    max_variance_velocity = None
    max_intensity_pixel_center = None
    
    for i, label in enumerate(unique_labels):
        sub_events = events[labels == label]
        sub_vx = vx[labels == label]
        sub_vy = vy[labels == label]
        
        warped_image = accumulate_pixel_map(sensor_size, events=sub_events, velocity=(sub_vx[0], sub_vy[0]))
        cumulative_map = warped_image['cumulative_map']
        event_indices = warped_image['event_indices']
        flipped_event_indices = event_indices[::-1]
        warped_image_rendered = render(cumulative_map, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
        variance = variance_loss_calculator(cumulative_map.pixels)

        x_coordinate = (i % 4) * warped_image_rendered.width
        y_coordinate = (i // 4) * warped_image_rendered.height
        
        combined_final_segmentation.paste(warped_image_rendered, (x_coordinate, y_coordinate))
        variances.append(variance)
        max_var = numpy.argmax(variances)
        
        if i == max_var:
            if max_variance_index != -1:
                combined_final_segmentation.paste(warped_image_rendered, previous_coordinates)
            
            draw = ImageDraw.Draw(combined_final_segmentation)
            
            # Update the additional information variables
            max_variance_events = sub_events
            max_variance_velocity = (sub_vx[0], sub_vy[0])
            
            # Draw bounding box on the new image with max variance
            draw.rectangle([(x_coordinate, y_coordinate), 
                            (x_coordinate + warped_image_rendered.width, y_coordinate + warped_image_rendered.height)],
                           outline=(255, 0, 0), width=5)
            
            # Draw circle for the pixel with maximum intensity
            max_intensity_pixel = numpy.unravel_index(numpy.argmax(cumulative_map.pixels), cumulative_map.pixels.shape)
            
            # Flip the y-coordinate vertically
            flipped_y = cumulative_map.pixels.shape[0] - max_intensity_pixel[0]
            
            # Update the center of the pixel with the highest intensity
            max_intensity_pixel_center = (x_coordinate + max_intensity_pixel[1], y_coordinate + flipped_y)
            
            circle_radius = 50  # radius of the circle
            draw.ellipse([(max_intensity_pixel_center[0] - circle_radius, max_intensity_pixel_center[1] - circle_radius),
                          (max_intensity_pixel_center[0] + circle_radius, max_intensity_pixel_center[1] + circle_radius)],
                         outline=(0, 255, 0), width=2)  # green color
            
            max_variance_index = i
            previous_coordinates = (x_coordinate, y_coordinate)

    return combined_final_segmentation, max_variance_events, max_variance_velocity, max_intensity_pixel_center


def motion_selection(sensor_size, events, labels, vx, vy):
    '''
    Apply the same speed on all the cluster, pick the cluster that has the maximum contrast
    '''
    variances       = []
    unique_labels   = numpy.unique(labels)
    for i, label in enumerate(unique_labels):
        sub_events      = events[labels == label]
        warped_image    = accumulate(sensor_size, events=sub_events, velocity=(vx[0], vy[0]))
        variance        = variance_loss_calculator(warped_image.pixels)
        variances.append(variance)
    return unique_labels[numpy.argmax(variances)]


def events_trimming(sensor_size, events, labels, vx, vy, winner, circle_radius, nearby_radius):
    """
    Trims events based on intensity and a specified circle radius and nearby pixels.

    Parameters:
    - sensor_size: Tuple indicating the dimensions of the sensor.
    - events: Array containing event data.
    - labels: Array of labels corresponding to events.
    - vx, vy: Arrays of x and y velocities for each event.
    - winner: The winning label for which events are to be trimmed.
    - circle_radius: Radius for trimming around high intensity pixel.
    - nearby_radius: Radius to select nearby pixels around each pixel.

    Returns:
    - List of selected event indices after trimming.
    - Centroid of the selected events (x, y).
    """
    # Filter events, velocities by winner label
    sub_events, sub_vx, sub_vy = events[labels == winner], vx[labels == winner], vy[labels == winner]
    # Compute warped image and retrieve cumulative map and event indices
    warped_image = accumulate_pixel_map(sensor_size, events=sub_events, velocity=(sub_vx[0], sub_vy[0]))
    cumulative_map, event_indices = warped_image['cumulative_map'], warped_image['event_indices']
    # Determine max intensity pixel and its flipped y-coordinate
    max_y, max_x = numpy.unravel_index(numpy.argmax(cumulative_map.pixels), cumulative_map.pixels.shape)
    flipped_y = cumulative_map.pixels.shape[0] - max_y
    # Create a mask centered on the max intensity pixel
    y, x = numpy.ogrid[:cumulative_map.pixels.shape[0], :cumulative_map.pixels.shape[1]]
    main_mask = (x - max_x)**2 + (y - flipped_y)**2 <= circle_radius**2
    
    # Mask for nearby pixels
    nearby_mask = numpy.zeros_like(main_mask)
    for i in range(cumulative_map.pixels.shape[0]):
        for j in range(cumulative_map.pixels.shape[1]):
            if main_mask[i, j]:
                y_nearby, x_nearby = numpy.ogrid[max(0, i-nearby_radius):min(cumulative_map.pixels.shape[0], i+nearby_radius+1), 
                                                 max(0, j-nearby_radius):min(cumulative_map.pixels.shape[1], j+nearby_radius+1)]
                mask = (x_nearby - j)**2 + (y_nearby - i)**2 <= nearby_radius**2
                nearby_mask[y_nearby, x_nearby] = mask
    
    combined_mask = main_mask | nearby_mask

    # Mask the flipped pixels to identify high intensity regions
    marked_image_np = numpy.flipud(cumulative_map.pixels) * combined_mask
    # Extract event indices for non-zero pixels
    selected_events_indices = numpy.concatenate(event_indices[::-1][numpy.where(marked_image_np != 0)]).tolist()
    return selected_events_indices


def compute_centroid(selected_events, label, winner_class, events_filter_raw, current_indices, 
                     label_after_segmentation, vx_after_segmentation, vy_after_segmentation, sub_vx, sub_vy):
    """
    Compute the centroid for selected events based on the winner class.

    Parameters:
    - selected_events: Array of selected events.
    - label: Array of labels corresponding to the events.
    - winner_class: The winning class for which centroid is computed.
    - events_filter_raw: Raw event data.
    - current_indices: Current indices of the events.
    - label_after_segmentation: Global array or passed array for labels after segmentation.
    - vx_after_segmentation: Global array or passed array for vx after segmentation.
    - vy_after_segmentation: Global array or passed array for vy after segmentation.
    - sub_vx: Array of x velocities for each event.
    - sub_vy: Array of y velocities for each event.

    Returns:
    - centroid_x: x-coordinate of the centroid.
    - centroid_y: y-coordinate of the centroid.
    """
    selected_indices = numpy.where(label == winner_class)[0][selected_events]
    label_after_segmentation[current_indices[selected_indices]] = winner_class
    vx_after_segmentation[current_indices[selected_indices]] = sub_vx[0]
    vy_after_segmentation[current_indices[selected_indices]] = sub_vy[0]

    selected_events_for_centroid = events_filter_raw[label_after_segmentation == winner_class]
    centroid_x = numpy.mean(selected_events_for_centroid['x'])
    centroid_y = numpy.mean(selected_events_for_centroid['y'])

    return centroid_x, centroid_y

def generate_combined_image_no_label(sensor_size, events, labels, vx, vy):
    """
    Generate a combined image for given events, labels, and velocities.

    Parameters:
    - events: Array of event data.
    - labels: Array of labels for each event.
    - vx: Array of x velocities.
    - vy: Array of y velocities.

    Returns:
    - A PIL Image combining the warped images for each label.
    """
    
    unique_labels = numpy.unique(labels)
    sub_vx = vx[labels == unique_labels[0]]
    sub_vy = vy[labels == unique_labels[0]]

    # Generate the first warped image to determine its width and height
    first_warped_image = accumulate(sensor_size, 
                                        events=events, 
                                        velocity=(sub_vx[0],sub_vy[0]))
    warped_image_rendered = render(first_warped_image, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
    
    # Determine the number of rows and columns for the final image
    num_cols = min(4, len(unique_labels))
    num_rows = int(numpy.ceil(len(unique_labels) / 4))
    
    # Initialize the final combined image
    combined_final_segmentation = Image.new('RGB', (num_cols * warped_image_rendered.width, num_rows * warped_image_rendered.height))
    
    # Load a font for the text
    try:
        font = ImageFont.truetype("./src/Roboto-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, label in enumerate(unique_labels):
        # Filter events, vx, and vy based on the label value
        # sub_events = events[labels == label]
        sub_vx = vx[labels == label]
        sub_vy = vy[labels == label]
        
        # Generate the warped image for this subset of events
        warped_image = accumulate(sensor_size, events=events, velocity=(sub_vx[0], sub_vy[0]))
        warped_image_rendered = render(warped_image, colormap_name="magma", gamma=lambda image: image ** (1 / 3))

        # Compute the x and y coordinates for pasting based on the index
        x_coordinate = (i % 4) * warped_image_rendered.width
        y_coordinate = (i // 4) * warped_image_rendered.height
        
        # Paste this warped image into the final combined image
        combined_final_segmentation.paste(warped_image_rendered, (x_coordinate, y_coordinate))
    return combined_final_segmentation


def generate_overlay_and_indices(blur_map, warped_image, removeFactor=1, flipped_event_indices=None):
    """
    Generate an overlay image and unique indices to remove based on the provided blur_map and warped_image.

    Parameters:
    - blur_map: Numpy array representing the blur map.
    - warped_image: PIL Image object.
    - removeFactor: Factor to determine the intensity of the overlay.
    - flipped_event_indices: Numpy array representing the indices of flipped events.

    Returns:
    - overlay_image: PIL Image object.
    - unique_indices_to_remove: Numpy array of indices.
    """
    
    # Convert blur_map to PIL Image and adjust range
    blur_map_image = Image.fromarray(numpy.uint8(blur_map * 255), 'L')
    blur_map_image = ImageOps.flip(blur_map_image)
    
    blur_image_np = (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min())
    blur_image_np = numpy.flipud(blur_image_np)
    
    image_np = numpy.array(warped_image.convert('RGBA'))
    marked_image_np = image_np.copy()
    marked_image_np[..., :3] = [0, 0, 255]
    marked_image_np[..., 3] = (blur_image_np * removeFactor * 255).astype(numpy.uint8)
    marked_image_np = numpy.where(marked_image_np > numpy.mean(marked_image_np.flatten()), marked_image_np, 0).astype(numpy.uint8)
    
    overlay_image = Image.alpha_composite(Image.fromarray(image_np), Image.fromarray(marked_image_np))
    
    if flipped_event_indices is not None:
        sharp_pixel_coords = numpy.where(marked_image_np[..., 3] != 0)
        all_indices_to_remove = numpy.concatenate(flipped_event_indices[sharp_pixel_coords]).astype(int)
        unique_indices_to_remove = numpy.unique(all_indices_to_remove)
        unique_indices_to_remove = numpy.sort(unique_indices_to_remove)
    else:
        unique_indices_to_remove = None

    return blur_map_image, overlay_image, unique_indices_to_remove


def rgb_render_advanced(cumulative_map_object, l_values):
    """Render the cumulative map using RGB values based on the frequency of each class."""
    
    def generate_intense_palette(n_colors):
        """Generate an array of intense and bright RGB colors, avoiding blue."""
        # Define a base palette with intense and bright colors
        base_palette = numpy.array([
            [255, 255, 255],  # Bright white
            [255, 0, 0],  # Intense red
            [0, 255, 0],  # Intense green
            [255, 255, 150],  # Intense yellow
            [21,  185, 200],  # blue
            [255, 175, 150],  # Coral
            [255, 150, 200],  # Magenta
            [150, 255, 150],  # Intense green
            [150, 255, 200],  # Aqua
            [255, 200, 150],  # Orange
            [200, 255, 150],  # Light green with more intensity
            [255, 225, 150],  # Gold
            [255, 150, 175],  # Raspberry
            [175, 255, 150],  # Lime
            [255, 150, 255],  # Strong pink
            # Add more colors if needed
        ], dtype=numpy.uint8)
        # Repeat the base palette to accommodate the number of labels
        palette = numpy.tile(base_palette, (int(numpy.ceil(n_colors / base_palette.shape[0])), 1))
        return palette[:n_colors]  # Select only as many colors as needed

    cumulative_map = cumulative_map_object.pixels
    height, width = cumulative_map.shape
    rgb_image = numpy.zeros((height, width, 3), dtype=numpy.uint8)  # Start with a black image

    unique, counts = numpy.unique(l_values[l_values != 0], return_counts=True)
    # Sort the indices of the unique array based on the counts in descending order
    sorted_indices = numpy.argsort(counts)[::-1]
    # Retrieve the sorted labels, excluding label 0
    sorted_unique = unique[sorted_indices]

    # Now we explicitly add the label 0 at the beginning of the sorted_unique array
    sorted_unique = numpy.concatenate(([0], sorted_unique))

    palette = generate_intense_palette(len(sorted_unique))
    color_map = dict(zip(sorted_unique, palette))
    
    for label, color in color_map.items():
        mask = l_values == label
        norm_intensity = cumulative_map[mask] / (cumulative_map[mask].max() + 1e-9)
        norm_intensity = numpy.power(norm_intensity, 0.2)  # Increase the color intensity
        blended_color = color * norm_intensity[:, numpy.newaxis]
        rgb_image[mask] = numpy.clip(blended_color, 0, 255)
    
    image = Image.fromarray(rgb_image)
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def save_events_to_es(events, filename, ordering):
    import loris
    events_array = numpy.zeros((len(events), 4))
    events_array[:, 0] = events['t']
    events_array[:, 1] = events['x']
    events_array[:, 2] = events['y']
    events_array[:, 3] = events['on'].astype(int)
    final_array = numpy.asarray(events_array)
    loris.write_events_to_file(final_array, filename, ordering)
    print("File: " + filename + " converted to .es")

def save_events_to_txt(events, txt_path, chunk_size):
    with open(txt_path, 'w') as f:
        for idx in tqdm(range(0, len(events), chunk_size)):
            for j in range(idx, min(idx + chunk_size, len(events))):
                # Format each event with a single space between elements
                event_str = f"{numpy.float64((events['t'][j] - events['t'][0])/1e6):.6f} {int(events['x'][j])} {int(events['y'][j])} {int(events['on'][j])}\n"
                f.write(event_str)
    print(f"{txt_path} file's saved")


def rgb_render_white(cumulative_map_object, l_values):
    """Render the cumulative map using RGB values based on the frequency of each class, with a white background and save as PNG."""
    
    def generate_intense_palette(n_colors):
        """Generate an array of intense and bright RGB colors, avoiding blue."""
        base_palette = numpy.array([
            [255, 255, 255],  # White (will invert to black)
            [0, 255, 255],    # Cyan (will invert to red)
            [255, 0, 255],    # Magenta (will invert to green)
            [255, 255, 0],    # Yellow (will invert to blue)
            [0, 255, 0],      # Green (will invert to magenta)
            [255, 0, 0],      # Red (will invert to cyan)
            [0, 128, 128],    # Teal (will invert to a light orange)
            [128, 0, 128],    # Purple (will invert to a light green)
            [255, 165, 0],    # Orange (will invert to a blue-green)
            [128, 128, 0],    # Olive (will invert to a light blue)
            [255, 192, 203],  # Pink (will invert to a light green-blue)
            [165, 42, 42],    # Brown (will invert to a light blue-green)
            [0, 100, 0],      # Dark Green (will invert to a light magenta)
            [173, 216, 230],  # Light Blue (will invert to a darker yellow)
            [245, 222, 179],  # Wheat (will invert to a darker blue)
            [255, 20, 147],   # Deep Pink (will invert to a light cyan-green)
            [75, 0, 130],     # Indigo (will invert to a lighter yellow)
            [240, 230, 140],  # Khaki (will invert to a light blue)
            [0, 0, 128],      # Navy (will invert to a light yellow)
        ], dtype=numpy.uint8)
        palette = numpy.tile(base_palette, (int(numpy.ceil(n_colors / base_palette.shape[0])), 1))
        return palette[:n_colors]


    cumulative_map = cumulative_map_object.pixels
    height, width = cumulative_map.shape
    rgb_image = numpy.ones((height, width, 3), dtype=numpy.uint8) * 255  # Start with a white background

    unique, counts = numpy.unique(l_values[l_values != 0], return_counts=True)
    sorted_indices = numpy.argsort(counts)[::-1]
    sorted_unique = unique[sorted_indices]
    sorted_unique = numpy.concatenate(([0], sorted_unique))

    palette = generate_intense_palette(len(sorted_unique))
    color_map = dict(zip(sorted_unique, palette))
    
    for label, color in color_map.items():
        mask = l_values == label
        if numpy.any(mask):
            norm_intensity = cumulative_map[mask] / (cumulative_map[mask].max() + 1e-9)

            norm_intensity = numpy.power(norm_intensity, 0.01)
            blended_color = color * norm_intensity[:, numpy.newaxis]
            rgb_image[mask] = numpy.clip(blended_color, 0, 255)
    
    image = Image.fromarray(rgb_image, 'RGB')
    inverted_image = ImageOps.invert(image)
    inverted_image = inverted_image.transpose(Image.FLIP_TOP_BOTTOM)
    return inverted_image



def rgb_render(cumulative_map_object, l_values):
    """Render the cumulative map using HSV values based on the frequency of each class."""
    def generate_palette_hsv(n_colors):
        """Generate an array of HSV colors and convert them to RGB."""
        hues = numpy.linspace(0, 1, n_colors, endpoint=False)
        hsv_palette = numpy.stack([hues, numpy.ones_like(hues), numpy.ones_like(hues)], axis=-1)
        rgb_palette = matplotlib.colors.hsv_to_rgb(hsv_palette)
        return (rgb_palette * 255).astype(numpy.uint8)

    cumulative_map = cumulative_map_object.pixels
    height, width = cumulative_map.shape
    rgb_image = numpy.ones((height, width, 3), dtype=numpy.uint8) * 255

    unique, counts = numpy.unique(l_values, return_counts=True)
    sorted_indices = numpy.argsort(counts)[::-1]
    sorted_unique = unique[sorted_indices]

    palette = generate_palette_hsv(len(sorted_unique))
    color_map = dict(zip(sorted_unique, palette))
    color_map[0] = numpy.array([255, 255, 255], dtype=numpy.uint8)

    for label, color in color_map.items():
        mask = l_values == label
        norm_intensity = cumulative_map[mask] / (cumulative_map[mask].max() + 1e-9)
        norm_intensity = numpy.power(norm_intensity, 0.3)  # Increase the color intensity
        blended_color = color * norm_intensity[:, numpy.newaxis]
        rgb_image[mask] = numpy.clip(blended_color, 0, 255)
    
    # Ensuring that any unprocessed region is set to white
    unprocessed_mask = numpy.all(rgb_image == [255, 255, 255], axis=-1)
    rgb_image[unprocessed_mask] = [255, 255, 255]

    image = PIL.Image.fromarray(rgb_image)
    # rotated_image = image.rotate(180)
    return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)



def render_3d(variance_loss_3d: numpy.ndarray):
    x, y, z = numpy.indices(variance_loss_3d.shape)
    values = variance_loss_3d.flatten()
    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values,
        isomin=0.2,
        isomax=numpy.max(variance_loss_3d),
        opacity=0.6,
        surface_count=1,
    ))
    fig.show()


def render_histogram(cumulative_map: CumulativeMap, path: pathlib.Path, title: str):
    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.hist(cumulative_map.pixels.flat, bins=200, log=True)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel("Event count")
    matplotlib.pyplot.ylabel("Pixel count")
    matplotlib.pyplot.savefig(path)
    matplotlib.pyplot.close()


def intensity_variance(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return dvs_sparse_filter_extension.intensity_variance(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )


def intensity_variance_ts(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
    tau: int,
):
    return dvs_sparse_filter_extension.intensity_variance_ts(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
        tau,
    )

@dataclasses.dataclass
class CumulativeMap:
    pixels: numpy.ndarray
    offset: Tuple[float, float]

def accumulate_warped_events_square(warped_x: torch.Tensor, warped_y: torch.Tensor):
    x_minimum = float(warped_x.min())
    y_minimum = float(warped_y.min())
    xs = warped_x - x_minimum + 1.0
    ys = warped_y - y_minimum + 1.0
    pixels = torch.zeros((int(torch.ceil(ys.max())) + 2, int(torch.ceil(xs.max())) + 2))
    xis = torch.floor(xs).long()
    yis = torch.floor(ys).long()
    xfs = xs - xis.float()
    yfs = ys - yis.float()
    for xi, yi, xf, yf in zip(xis, yis, xfs, yfs):
        pixels[yi, xi] += (1.0 - xf) * (1.0 - yf)
        pixels[yi, xi + 1] += xf * (1.0 - yf)
        pixels[yi + 1, xi] += (1.0 - xf) * yf
        pixels[yi + 1, xi + 1] += xf * yf
    return CumulativeMap(
        pixels=pixels,
        offset=(-x_minimum + 1.0, -y_minimum + 1.0),
    )

def center_events(eventx, eventy):
    center_x = eventx.max() / 2
    center_y = eventy.max() / 2
    eventsx_centered = eventx - center_x
    eventsy_centered = eventy - center_y
    return eventsx_centered, eventsy_centered


def warp_4D(events, linear_vel, angular_vel, zoom, deltat):
    wx, wy, wz = angular_vel
    vx, vy = linear_vel
    eventsx, eventsy = center_events(events[0,:], events[1,:])
    rot_mat = torch.tensor([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=torch.float32)
    event_stack = torch.stack((eventsx, eventsy, torch.ones(len(events[0,:])))).t().float()
    deltat = torch.from_numpy(deltat).float()
    rot_exp = (rot_mat * deltat[:, None, None]).float()
    rot_exp = torch.matrix_exp(rot_exp)
    rot_events = torch.einsum("ijk,ik->ij", rot_exp, event_stack)
    warpedx_scale = (1 - deltat * zoom) * rot_events[:, 0]
    warpedy_scale = (1 - deltat * zoom) * rot_events[:, 1]
    warpedx_trans = warpedx_scale - deltat * vx
    warpedy_trans = warpedy_scale - deltat * vy
    return warpedx_trans, warpedy_trans


def opt_loss_py(events, linear_vel, angular_vel, zoom, deltat):
    warpedx, warpedy    = warp_4D(events, linear_vel, angular_vel, zoom, deltat)
    warped_image        = accumulate_warped_events_square(warpedx, warpedy)
    objective_func      = variance_loss_calculator(warped_image)
    save_img(warped_image, "./")
    return objective_func

def opt_loss_cpp(events, sensor_size, linear_vel, angular_vel, zoom):
    # Convert events numpy array to a PyTorch tensor
    events_tensor = {}
    for key in events.dtype.names:
        if events[key].dtype == numpy.uint64:
            events_tensor[key] = torch.tensor(numpy.copy(events[key]).astype(numpy.int64)).to(linear_vel.device)
        elif events[key].dtype == numpy.uint16:
            events_tensor[key] = torch.tensor(numpy.copy(events[key]).astype(numpy.int32)).to(linear_vel.device)
        elif events[key].dtype == numpy.bool_:
            events_tensor[key] = torch.tensor(numpy.copy(events[key]).astype(numpy.int8)).to(linear_vel.device)

    warped_image = accumulate4D_torch(sensor_size=sensor_size,
                                events=events_tensor,
                                linear_vel=linear_vel,
                                angular_vel=angular_vel,
                                zoom=zoom)

    # Convert warped_image to a PyTorch tensor if it's not already one
    if not isinstance(warped_image, torch.Tensor):
        warped_image_tensor = torch.tensor(warped_image.pixels).float()
    else:
        warped_image_tensor = warped_image
        
    objective_func = variance_loss_calculator_torch(warped_image_tensor)
    objective_func = objective_func.float()
    return objective_func


def variance_loss_calculator_torch(evmap):
    flattening = evmap.view(-1)  # Flatten the tensor
    res = flattening[flattening != 0]
    return -torch.var(res)

def variance_loss_calculator(evmap):
    pixels = evmap
    flattening = pixels.flatten()
    res = flattening[flattening != 0]
    return torch.var(torch.from_numpy(res))


def save_img(warped_image, savefileto):
    image = render(
    warped_image,
    colormap_name="magma",
    gamma=lambda image: image ** (1 / 3))
    filename = "eventmap_wz.jpg"
    filepath = os.path.join(savefileto, filename)
    image.save(filepath)

def rad2degree(val):
    return val/numpy.pi*180.

def degree2rad(val):
    return val/180*numpy.pi

def generate_warped_images(events: numpy.ndarray,
                           sensor_size: Tuple[int, int],
                           linear_velocity: numpy.ndarray, 
                           angular_velocity: numpy.ndarray, 
                           scale: numpy.ndarray, 
                           tmax: float,
                           savefileto: str) -> None:

    for iVel in tqdm(range(len(linear_velocity))):
        linear          = linear_velocity[iVel]
        angular         = angular_velocity[iVel]
        zoom            = scale[iVel]
        vx              = -linear / 1e6
        vy              = -43 / 1e6
        wx              = 0.0 / 1e6
        wy              = 0.0 / 1e6
        wz              = (0.0 / tmax) / 1e6
        zooms           = (0.0 / tmax) / 1e6

        warped_image = accumulate4D(sensor_size=sensor_size,
                                    events=events,
                                    linear_vel=(vx,vy),
                                    angular_vel=(wx,wy,wz),
                                    zoom=zooms)

        image = render(warped_image,
                       colormap_name="magma",
                       gamma=lambda image: image ** (1 / 3))
        new_image = image.resize((500, 500))
        filename = f"eventmap_wz_{wz*1e6:.2f}_z_{zooms*1e6:.2f}_vx_{vx*1e6:.4f}_vy_{vy*1e6:.4f}_wx_{wx*1e6:.2f}_wy_{wy*1e6:.2f}.jpg"
        filepath = os.path.join(savefileto, filename)
        new_image.save(filepath)
    return None


def generate_3Dlandscape(events: numpy.ndarray,
                       sensor_size: Tuple[int, int],
                       linear_velocity: numpy.ndarray, 
                       angular_velocity: numpy.ndarray, 
                       scale: numpy.ndarray, 
                       tmax: float,
                       savefileto: str) -> None:
    nvel = len(angular_velocity)
    trans=0
    rot=0
    variance_loss = numpy.zeros((nvel*nvel,nvel))
    for iVelz in tqdm(range(nvel)):
        wx              = 0.0 / 1e6
        wy              = 0.0 / 1e6
        wz              = (angular_velocity[iVelz] / tmax) / 1e6
        for iVelx in range(nvel):
            vx          = linear_velocity[iVelx] / 1e6
            for iVely in range(nvel):
                vy          = linear_velocity[iVely] / 1e6
                warped_image = accumulate4D(sensor_size=sensor_size,
                                            events=events,
                                            linear_vel=(vx,vy),
                                            angular_vel=(wx,wy,wz),
                                            zoom=0)
                var = variance_loss_calculator(warped_image.pixels)
                variance_loss[trans,rot] = var
                trans+=1
        rot+=1
        trans=0
    
    reshaped_variance_loss = variance_loss.reshape(nvel, nvel, nvel)
    sio.savemat(savefileto+"reshaped_variance_loss.mat",{'reshaped_variance_loss':numpy.asarray(reshaped_variance_loss)})
    render_3d(reshaped_variance_loss)
    return None


def random_velocity(opt_range):
    return (random.uniform(-opt_range / 1e6, opt_range / 1e6), 
            random.uniform(-opt_range / 1e6, opt_range / 1e6))


def find_best_velocity_with_initialisation(sensor_size: Tuple[int, int], events: numpy.ndarray, initial_velocity:int, iterations: int):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity = None
    highest_variance = float('-inf')
    for _ in range(iterations):
        optimized_velocity = optimize_local(sensor_size=sensor_size,
                                                          events=events,
                                                          initial_velocity=initial_velocity,
                                                          tau=1000,
                                                          heuristic_name="variance",
                                                          method="Nelder-Mead",
                                                          callback=None)
        objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
        print("iter. vx: {}    iter. vy: {}    contrast: {}".format(optimized_velocity[0] * 1e6, optimized_velocity[1] * 1e6, objective_loss))
        if objective_loss > highest_variance:
            highest_variance = objective_loss
            best_velocity = optimized_velocity
        initial_velocity = optimized_velocity
    return best_velocity, highest_variance

def find_best_velocity(sensor_size: Tuple[int, int], events: numpy.ndarray, opt_range:int, iterations: int):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity = None
    highest_variance = float('-inf')
    for _ in range(iterations):
        initial_velocity = random_velocity(opt_range)
        optimized_velocity = optimize_local(sensor_size=sensor_size,
                                                          events=events,
                                                          initial_velocity=initial_velocity,
                                                          tau=1000,
                                                          heuristic_name="variance",
                                                          method="Nelder-Mead",
                                                          callback=None)
        objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
        print("iter. vx: {}    iter. vy: {}    contrast: {}".format(optimized_velocity[0] * 1e6, optimized_velocity[1] * 1e6, objective_loss))
        if objective_loss > highest_variance:
            highest_variance = objective_loss
            best_velocity = optimized_velocity
        initial_velocity = optimized_velocity
    return best_velocity, highest_variance

def find_best_velocity_advanced(sensor_size: Tuple[int, int], events: numpy.ndarray, opt_range:int, iterations: int, previous_velocities: typing.List[tuple[float, float]]):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity = None
    highest_variance = float('-inf')
    DISTANCE_THRESHOLD = 0 #0.01  # Adjust as needed
    PENALTY = 0 #0.5  # Adjust as needed

    for _ in range(iterations):
        initial_velocity   = random_velocity(opt_range)
        optimized_velocity = optimize_local(sensor_size=sensor_size,
                                            events=events,
                                            initial_velocity=initial_velocity,
                                            heuristic_name="variance",
                                            tau=10000,
                                            method="Nelder-Mead",
                                            callback=None)
        objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
        
        # Penalty for being close to previous velocities
        for prev_velocity in previous_velocities:
            dist = numpy.linalg.norm(numpy.array(optimized_velocity) - numpy.array(prev_velocity))
            if dist < DISTANCE_THRESHOLD:
                objective_loss -= PENALTY
        
        print("iter. vx: {}    iter. vy: {}    contrast: {}".format(optimized_velocity[0] * 1e6, optimized_velocity[1] * 1e6, objective_loss))
        
        if objective_loss > highest_variance:
            highest_variance = objective_loss
            best_velocity = optimized_velocity
            previous_velocities.append(optimized_velocity)  # Update previous velocities list
        initial_velocity = optimized_velocity
    return best_velocity, highest_variance


def calculate_patch_variance(sensor_size, events, x_start, y_start, window_size, optimized_velocity):
    """
    Calculate the variance for a specific patch of events using a given velocity.
    
    Parameters:
    - events: The events data.
    - x_start: The starting x-coordinate of the patch.
    - y_start: The starting y-coordinate of the patch.
    - window_size: The size of the patch.
    - optimized_velocity: The velocity value to use.
    
    Returns:
    - The variance of the warped patch.
    """
    mask = (
        (events["x"] >= x_start) & (events["x"] < x_start + window_size) &
        (events["y"] >= y_start) & (events["y"] < y_start + window_size)
    )

    # Extract the patch of events
    patch_events = {
        "x": events["x"][mask],
        "y": events["y"][mask],
        "p": events["on"][mask],
        "t": events["t"][mask]
    }

    # Warp the patch using the optimized_velocity
    # (Assuming you have a warp function. Modify as needed.)
    warped_patch = accumulate(sensor_size, patch_events, optimized_velocity)
    
    # Calculate the variance of the warped patch
    variance = numpy.var(warped_patch.pixels)
    return variance



def dvs_sparse_filter_alex_conv(events):
    x = events['x']
    y = events['y']
    x_max, y_max = x.max() + 1, y.max() + 1

    # Create a 2D histogram of event counts
    event_count = numpy.zeros((x_max, y_max), dtype=int)
    numpy.add.at(event_count, (x, y), 1)

    kernels = [
        numpy.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    ]

    # Apply convolution with each kernel and calculate ratios
    shifted = [convolve(event_count, k, mode="constant", cval=0.0) for k in kernels]
    max_shifted = numpy.maximum.reduce(shifted)
    ratios = event_count / (max_shifted + 1.0)
    smart_mask = ratios < 2.0

    yhot, xhot = numpy.where(~smart_mask)
    label_hotpix = numpy.zeros(len(events), dtype=bool)
    for xi, yi in zip(xhot, yhot):
        label_hotpix |= (y == xi) & (x == yi)
    label_hotpix_binary = label_hotpix.astype(int)

    print(f'Number of detected noise event: {len(xhot)}')
    print(f'Events marked as hot pixels: {numpy.sum(label_hotpix)}')

    return label_hotpix_binary



def subdivide_events(events: numpy.ndarray, sensor_size: Tuple[int, int], level: int) -> List[numpy.ndarray]:
    """Divide the events into 4^n subvolumes at a given level of the hierarchy.
       Inspired by this paper: Event-based Motion Segmentation with Spatio-Temporal Graph Cuts
    """
    subvolumes = []
    num_subvolumes = 4**level
    subvolume_size = numpy.array(sensor_size) / (num_subvolumes**0.5)
    
    for i in range(int(num_subvolumes**0.5)):
        for j in range(int(num_subvolumes**0.5)):
            # Define the boundaries of the subvolume
            xmin = i * subvolume_size[0]
            xmax = (i + 1) * subvolume_size[0]
            ymin = j * subvolume_size[1]
            ymax = (j + 1) * subvolume_size[1]
            
            # Select the events within the subvolume
            ii = numpy.where((events["x"] >= xmin) & (events["x"] < xmax) & 
                          (events["y"] >= ymin) & (events["y"] < ymax))
            subvolumes.append(events[ii])
    return subvolumes

# def find_optimal_focus_time(timestamps):
#     stopping_threshold = 0.001
#     eventstart_time = timestamps[0]
#     eventend_time = timestamps[-1]
#     golden_search_range = eventend_time - eventstart_time

#     optimal_start_time = None
#     optimal_end_time = None
#     max_event_rate = 0

#     iteration = 0
#     fixed_delta_time = None

#     while golden_search_range > stopping_threshold:
#         print(golden_search_range)
#         idx = (timestamps >= eventstart_time) & (timestamps <= eventend_time)
#         t = timestamps[idx]

#         if iteration < 2:
#             time_range = t[-1] - t[0]
#             delta_time = time_range * 0.618
#             if iteration == 1:
#                 fixed_delta_time = delta_time
#         else:
#             delta_time = fixed_delta_time

#         fst_start_time = t[0]
#         second_start_time = t[0] + (time_range * 0.381 if iteration < 2 else delta_time)

#         event_rate = 0
#         event_rate_sec = 0

#         first_batch_event_rate = 0
#         second_batch_event_rate = 0

#         for i in range(len(t)):
#             ev_t = t[i]
#             if ev_t > fst_start_time and ev_t <= (fst_start_time + delta_time):
#                 event_rate += 1
#             elif ev_t > fst_start_time + delta_time:
#                 event_rate /= delta_time
#                 first_batch_event_rate = event_rate * delta_time
#                 event_rate = 0
#                 fst_start_time = ev_t

#             if ev_t > second_start_time and ev_t <= (second_start_time + delta_time):
#                 event_rate_sec += 1
#             elif ev_t > second_start_time + delta_time:
#                 event_rate_sec /= delta_time
#                 second_batch_event_rate = event_rate_sec * delta_time
#                 event_rate_sec = 0
#                 second_start_time = ev_t

#         if first_batch_event_rate > max_event_rate:
#             max_event_rate = first_batch_event_rate
#             optimal_start_time = fst_start_time - delta_time
#             optimal_end_time = fst_start_time
#         if second_batch_event_rate > max_event_rate:
#             max_event_rate = second_batch_event_rate
#             optimal_start_time = second_start_time - delta_time
#             optimal_end_time = second_start_time

#         golden_search_range = abs(optimal_end_time - optimal_start_time)
#         eventstart_time = optimal_start_time
#         eventend_time = optimal_end_time

#         iteration += 1

#     return optimal_start_time, optimal_end_time

def dvs_autofocustt(events: numpy.ndarray):
    """
    Find the timestamp where the focus was correct
    Re-implementation from this paper:
    Autofocus for Event Cameras, CVPR'22
    """
    timescale = 1
    image_height = numpy.max(events['y'])
    image_width = numpy.max(events['x'])

    y_o = events['y']
    x_o = events['x']
    pol_o = events['on'].astype(int)
    pol_o[pol_o == 0] = -1
    t_o = events['t'] / timescale
    t_o = t_o - t_o[0]

    ev_rate_both = []
    winner_index = []
    time_batch1 = []
    time_batch2 = []

    optimal_focus_time = numpy.finfo(numpy.float64).max
    prev_optimal_focus_time = 0
    updated_focus_moving_step = numpy.finfo(numpy.float64).max
    stopping_threshold = 0.001
    eventstart_time = t_o[0]
    eventend_time = t_o[-1]
    golden_search_range = t_o[-1] - t_o[0]

    max_event_rate = 0
    max_event_rate_start_time = None
    max_event_rate_end_time = None

    while golden_search_range > stopping_threshold:
        x = x_o.copy()
        y = y_o.copy()
        pol = pol_o.copy()
        t = t_o.copy()
        idx = (t >= eventstart_time) & (t <= eventend_time)
        y = y[idx]
        x = x[idx]
        pol = pol[idx]
        t = t[idx]

        event_rate = 0
        event_rate_sec = 0
        event_rate_list = []

        event_rate_ts_list = []
        time_range = t[-1] - t[0]
        golden_search_range = time_range
        delta_time = time_range * 0.618
        event_size = len(t)
        fst_start_time = t[0]
        second_start_time = t[0] + time_range * 0.381
        EV_FLAG = True
        EV_FLAG_SEC = True

        first_batch_event_rate = 0
        second_batch_event_rate = 0

        for i in range(event_size):
            ev_t = t[i]
            if ev_t > fst_start_time and ev_t <= (fst_start_time + delta_time) and EV_FLAG:
                event_rate += 1
            elif ev_t > fst_start_time + delta_time and EV_FLAG:
                event_rate = event_rate / delta_time
                event_rate_list.append(event_rate)

                time_batch1.append((fst_start_time, fst_start_time + delta_time))
                first_batch_event_rate = event_rate * delta_time

                event_rate_ts_list.append(fst_start_time + delta_time)
                event_rate = 0
                fst_start_time = ev_t
                EV_FLAG = False

            if ev_t > second_start_time and ev_t <= (second_start_time + delta_time) and EV_FLAG_SEC:
                event_rate_sec += 1
            elif ev_t > second_start_time + delta_time and EV_FLAG_SEC:
                event_rate_sec = event_rate_sec / delta_time
                event_rate_list.append(event_rate_sec)

                time_batch2.append((second_start_time, second_start_time + delta_time))
                second_batch_event_rate = event_rate_sec * delta_time

                event_rate_ts_list.append(second_start_time + delta_time)
                event_rate_sec = 0
                second_start_time = ev_t
                EV_FLAG_SEC = False

        ev_rate_both.append((first_batch_event_rate, second_batch_event_rate))
        if event_rate_list:
            max_ev_rate_index = numpy.argmax(event_rate_list)
            winner_index.append(max_ev_rate_index)
            optimal_focus_time = event_rate_ts_list[max_ev_rate_index]
            event_rate_output = event_rate_list[max_ev_rate_index] * delta_time
            updated_focus_moving_step = abs(optimal_focus_time - prev_optimal_focus_time)
            prev_optimal_focus_time = optimal_focus_time
            eventstart_time = optimal_focus_time - delta_time
            eventend_time = optimal_focus_time

            if event_rate_output > max_event_rate:
                max_event_rate = event_rate_output
                max_event_rate_start_time = eventstart_time
                max_event_rate_end_time = eventend_time
        else:
            return max_event_rate_start_time, max_event_rate_end_time

    return max_event_rate_start_time, max_event_rate_end_time


def dvs_autofocus(events: numpy.ndarray):
    """
    Find the timestamp where the focus was correct
    Re-implementation from this paper:
    Autofocus for Event Cameras, CVPR'22
    """
    timescale = 1
    image_height = numpy.max(events['y'])
    image_width = numpy.max(events['x'])

    y_o = events['y']
    x_o = events['x']
    pol_o = events['on'].astype(int)
    pol_o[pol_o == 0] = -1
    t_o = events['t'] / timescale
    t_o = t_o - t_o[0]

    ev_rate_both   = []
    winner_index   = []
    time_batch1    = []
    time_batch2    = []

    optimal_focus_time = numpy.finfo(numpy.float64).max
    prev_optimal_focus_time = 0
    updated_focus_moving_step = numpy.finfo(numpy.float64).max
    stopping_threshold = 1
    eventstart_time = t_o[0]
    eventend_time = t_o[-1]
    golden_search_range = t_o[-1] - t_o[0]
    counterss = 0
    save_time_boundary = []
    while golden_search_range > stopping_threshold:
        
        x   = x_o.copy()
        y   = y_o.copy()
        pol = pol_o.copy()
        t   = t_o.copy()
        idx = (t >= eventstart_time) & (t <= eventend_time)
        y   = y[idx]
        x   = x[idx]
        pol = pol[idx]
        t   = t[idx]

        event_rate = 0
        event_rate_sec = 0
        event_rate_list = []
        
        
        event_rate_ts_list = []
        time_range = t[-1] - t[0]
        golden_search_range = time_range
        delta_time = time_range * 0.618
        event_size = len(t)
        fst_start_time = t[0]
        second_start_time = t[0] + time_range * 0.381
        EV_FLAG = True
        EV_FLAG_SEC = True

        batch1_wind1 = fst_start_time
        batch1_wind2 = fst_start_time + delta_time
        batch2_wind1 = second_start_time
        batch2_wind2 = second_start_time + delta_time

        save_time_boundary.append((batch1_wind1/1e6,batch1_wind2/1e6,batch2_wind1/1e6,batch2_wind2/1e6,batch1_wind2/1e6-batch1_wind1/1e6,batch2_wind2/1e6-batch2_wind1/1e6))

        for i in range(event_size):
            ev_t = t[i]
            if ev_t > fst_start_time and ev_t <= (fst_start_time + delta_time) and EV_FLAG:
                event_rate += 1
            elif ev_t > fst_start_time + delta_time and EV_FLAG:
                event_rate = event_rate / delta_time
                event_rate_list.append(event_rate)

                time_batch1.append((fst_start_time, fst_start_time + delta_time))
                first_batch_event_rate = event_rate * delta_time

                event_rate_ts_list.append(ev_t)
                event_rate = 0
                fst_start_time = ev_t
                EV_FLAG = False

            if ev_t > second_start_time and ev_t <= (second_start_time + delta_time) and EV_FLAG_SEC:
                event_rate_sec += 1
            elif ev_t > second_start_time + delta_time and EV_FLAG_SEC:
                event_rate_sec = event_rate_sec / delta_time
                event_rate_list.append(event_rate_sec)

                time_batch2.append((second_start_time, second_start_time + delta_time))
                second_batch_event_rate = event_rate_sec * delta_time

                event_rate_ts_list.append(ev_t)
                event_rate_sec = 0
                second_start_time = ev_t
                EV_FLAG_SEC = False

        ev_rate_both.append((first_batch_event_rate,second_batch_event_rate))
        if event_rate_list:
            max_ev_rate_index = numpy.argmax(event_rate_list)
            winner_index.append(max_ev_rate_index)

            if max_ev_rate_index == 0:
                time_start = save_time_boundary[counterss][0]
                time_finish = save_time_boundary[counterss][1]
            else:
                time_start = save_time_boundary[counterss][2]
                time_finish = save_time_boundary[counterss][3]

            optimal_focus_time = event_rate_ts_list[max_ev_rate_index]
            event_rate_output = event_rate_list[max_ev_rate_index]*delta_time
            updated_focus_moving_step = abs(optimal_focus_time - prev_optimal_focus_time)
            prev_optimal_focus_time = optimal_focus_time
            eventstart_time = optimal_focus_time - delta_time
            eventend_time = optimal_focus_time
        else:
            return optimal_focus_time,save_time_boundary

        counterss+=1

    return optimal_focus_time,save_time_boundary

def dvs_autofocus_test(events: numpy.ndarray):

    """
    Find the timestamp where the focus was correct and return the event rate and the 
    time interval of the section that had the highest event rate.
    
    Re-implementation from this paper:
    Autofocus for Event Cameras, CVPR'22
    """
    timescale = 1
    image_height = numpy.max(events['y'])
    image_width = numpy.max(events['x'])

    event_rate_array = []
    y_o = events['y']
    x_o = events['x']
    pol_o = events['on'].astype(int)
    pol_o[pol_o == 0] = -1
    t_o = events['t'] / timescale
    t_o = t_o - t_o[0]

    optimal_focus_time = numpy.finfo(numpy.float64).max
    highest_event_rate = 0
    highest_rate_start_time = 0
    highest_rate_end_time = 0
    prev_optimal_focus_time = 0
    stopping_threshold = 0.001
    eventstart_time = t_o[0]
    eventend_time = t_o[-1]
    golden_search_range = t_o[-1] - t_o[0]

    while golden_search_range > stopping_threshold:
        x = x_o.copy()
        y = y_o.copy()
        pol = pol_o.copy()
        t = t_o.copy()
        idx = (t >= eventstart_time) & (t <= eventend_time)
        y = y[idx]
        x = x[idx]
        pol = pol[idx]
        t = t[idx]

        event_rate = 0
        event_rate_sec = 0
        event_rate_list = []
        event_rate_ts_list = []
        event_rate_times_list = []
        time_range = t[-1] - t[0]
        golden_search_range = time_range
        delta_time = time_range * 0.618
        event_size = len(t)
        fst_start_time = t[0]
        second_start_time = t[0] + time_range * 0.381
        EV_FLAG = True
        EV_FLAG_SEC = True

        for i in range(event_size):
            ev_t = t[i]
            if ev_t > fst_start_time and ev_t <= (fst_start_time + delta_time) and EV_FLAG:
                event_rate += 1
            elif ev_t > fst_start_time + delta_time and EV_FLAG:
                if delta_time > 0:
                    event_rate /= delta_time
                event_rate_list.append(event_rate)
                event_rate_ts_list.append(ev_t)
                event_rate_times_list.append((fst_start_time, ev_t))
                event_rate = 0
                fst_start_time = ev_t
                EV_FLAG = False

            if ev_t > second_start_time and ev_t <= (second_start_time + delta_time) and EV_FLAG_SEC:
                event_rate_sec += 1
            elif ev_t > second_start_time + delta_time and EV_FLAG_SEC:
                if delta_time > 0:
                    event_rate_sec /= delta_time
                event_rate_list.append(event_rate_sec)
                event_rate_ts_list.append(ev_t)
                event_rate_times_list.append((second_start_time, ev_t))
                event_rate_sec = 0
                second_start_time = ev_t
                EV_FLAG_SEC = False

        if event_rate_list:
            print(event_rate_ts_list)
            print(event_rate_list[0]*delta_time,event_rate_list[1]*delta_time)
            max_ev_rate_index = numpy.argmax(event_rate_list)
            optimal_focus_time = event_rate_ts_list[max_ev_rate_index]
            highest_event_rate = event_rate_list[max_ev_rate_index]*delta_time
            highest_rate_start_time, highest_rate_end_time = event_rate_times_list[max_ev_rate_index]
            updated_focus_moving_step = abs(optimal_focus_time - prev_optimal_focus_time)
            prev_optimal_focus_time = optimal_focus_time
            eventstart_time = optimal_focus_time - delta_time
            eventend_time = optimal_focus_time
            event_rate_array.append((highest_event_rate,optimal_focus_time))
        else:
            break

    return optimal_focus_time, highest_event_rate, highest_rate_start_time, highest_rate_end_time


######################## Sliding autofocus


def dvs_sliding_autofocus(events: numpy.ndarray,windowsize:int):
    """
    Find the timestamp where the focus was correct
    Re-implementation from this paper:
    Autofocus for Event Cameras
    """
    timescale = 1e6
    image_height = numpy.max(events['y'])
    image_width = numpy.max(events['x'])

    optimal_focus_time_array = []
    events['t'] = events['t'] - events['t'][0]
    for initial_window in tqdm(range(0,int(events['t'][-1] / timescale),windowsize)):
        final_window = initial_window + windowsize
        idx = (events['t']/timescale >= initial_window) & (events['t']/timescale <= final_window)

        y_o = events['y'][idx]
        x_o = events['x'][idx]
        pol_o = events['on'][idx].astype(int)
        pol_o[pol_o == 0] = -1
        t_o = events['t'][idx] / timescale
        # t_o = t_o - t_o[0]

        optimal_focus_time = numpy.finfo(numpy.float64).max
        prev_optimal_focus_time = 0
        stopping_threshold = 0.001
        eventstart_time = t_o[0]
        eventend_time = t_o[-1]
        golden_search_range = t_o[-1] - t_o[0]

        while golden_search_range > stopping_threshold:
            x = x_o.copy()
            y = y_o.copy()
            pol = pol_o.copy()
            t = t_o.copy()
            idx = (t >= eventstart_time) & (t <= eventend_time)
            y = y[idx]
            x = x[idx]
            pol = pol[idx]
            t = t[idx]

            event_rate = 0
            event_rate_sec = 0
            event_rate_list = []
            event_rate_ts_list = []
            time_range = t[-1] - t[0]
            golden_search_range = time_range
            delta_time = time_range * 0.618
            event_size = len(t)
            fst_start_time = t[0]
            second_start_time = t[0] + time_range * 0.381
            EV_FLAG = True
            EV_FLAG_SEC = True

            for i in range(event_size):
                ev_t = t[i]
                if ev_t > fst_start_time and ev_t <= (fst_start_time + delta_time) and EV_FLAG:
                    event_rate += 1
                elif ev_t > fst_start_time + delta_time and EV_FLAG:
                    event_rate = event_rate / delta_time
                    event_rate_list.append(event_rate)
                    event_rate_ts_list.append(ev_t)
                    event_rate = 0
                    fst_start_time = ev_t
                    EV_FLAG = False

                if ev_t > second_start_time and ev_t <= (second_start_time + delta_time) and EV_FLAG_SEC:
                    event_rate_sec += 1
                elif ev_t > second_start_time + delta_time and EV_FLAG_SEC:
                    event_rate_sec = event_rate_sec / delta_time
                    event_rate_list.append(event_rate_sec)
                    event_rate_ts_list.append(ev_t)
                    event_rate_sec = 0
                    second_start_time = ev_t
                    EV_FLAG_SEC = False

            if event_rate_list:
                max_ev_rate_index = numpy.argmax(event_rate_list)
                optimal_focus_time = event_rate_ts_list[max_ev_rate_index]
                updated_focus_moving_step = abs(optimal_focus_time - prev_optimal_focus_time)
                prev_optimal_focus_time = optimal_focus_time
                eventstart_time = optimal_focus_time - delta_time
                eventend_time = optimal_focus_time
            else:
                return optimal_focus_time

        optimal_focus_time_array.append(optimal_focus_time)

    contrast = []
    for fc_idx in range(len(optimal_focus_time_array)):
        ii = numpy.where(numpy.logical_and(events["t"] > optimal_focus_time_array[fc_idx]*1e6, events["t"] < (optimal_focus_time_array[fc_idx]+2)*1e6))
        selected_events = events[ii]
        warped_image = accumulate((image_width, image_height), selected_events, (0,0))
        var = variance_loss_calculator(warped_image.pixels)
        contrast.append(var)

    max_contrast_index = numpy.argmax(contrast)
    best_focus_time = optimal_focus_time_array[max_contrast_index]
        
    return best_focus_time
###########################################


def optimization(events, sensor_size, initial_linear_vel, initial_angular_vel, initial_zoom, max_iters, lr, lr_step, lr_decay):
    optimizer_name = 'Adam'
    optim_kwargs = dict()  # Initialize as empty dict by default

    # lr = 0.005
    # iters = 100
    lr_step = max(1, lr_step)  # Ensure lr_step is at least 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # linear_vel = torch.tensor(initial_linear_vel).float().to(device)
    # linear_vel.requires_grad = True
    linear_vel = torch.tensor(initial_linear_vel, requires_grad=True)
    print(linear_vel.grad)

    optimizer = optim.__dict__[optimizer_name]([linear_vel], lr=lr, **optim_kwargs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)

    print_interval = 1
    min_loss = float('inf')  # Use Python's float infinity
    best_poses = linear_vel.clone()  # Clone to ensure we don't modify the original tensor
    best_it = 0

    if optimizer_name == 'Adam':
        for it in range(max_iters):
            optimizer.zero_grad()
            poses_val = linear_vel.cpu().detach().numpy()
            
            if numpy.isnan(poses_val).any():  # Proper way to check for NaN in numpy
                print("nan in the estimated values, something wrong takes place, please check!")
                exit()

            # Use linear_vel directly in the loss computation
            loss = opt_loss_cpp(events, sensor_size, linear_vel, initial_angular_vel, initial_zoom)

            if it == 0:
                print('[Initial]\tloss: {:.12f}\tposes: {}'.format(loss.item(), poses_val))
            elif (it + 1) % print_interval == 0:
                print('[Iter #{}/{}]\tloss: {:.12f}\tposes: {}'.format(it + 1, max_iters, loss.item(), poses_val))
            
            # Store a copy of the best linear_vel tensor
            if loss < min_loss:
                best_poses = linear_vel.clone()
                min_loss = loss.item()
                best_it = it
            try:
                loss.requires_grad = True
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                
            except Exception as e:
                print(e)
                return poses_val, loss.item()
            
            print("Loss before step:", loss.item())
            optimizer.step()
            print("Loss after step:", loss.item())
            scheduler.step()
    else:
        print("The optimizer is not supported.")

    best_poses = best_poses.cpu().detach().numpy()
    print('[Final Result]\tloss: {:.12f}\tposes: {} @ {}'.format(min_loss, best_poses, best_it))
    if device == torch.device('cuda:0'):
        torch.cuda.empty_cache()
    
    return best_poses, min_loss


def correction(i: numpy.ndarray, j: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int):
    return {
        '1': (1, vx / width, vy / height),
        '2': vx / x[i, j],
        '3': vy / y[i, j],
        '4': vx / (-x[i, j] + width + vx),
        '5': vy / (-y[i, j] + height + vy),
        '6': (vx*vy) / (vx*y[i, j] + vy*width - vy*x[i, j]),
        '7': (vx*vy) / (vx*height - vx*y[i, j] + vy*x[i, j]),
    }


def alpha_1(warped_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int, edgepx: int):
    """
    Input:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx < w and vy < h. The conditions are designed based on the pixel's 
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions = [
        (x > vx) & (x < width) & (y >= vy) & (y <= height),
        (x > 0) & (x < vx) & (y <= height) & (y >= ((vy*x) / vx)),
        (x >= 0) & (x <= width) & (y > 0) & (y < vy) & (y < ((vy*x) / vx)),
        (x >= width) & (x <= width+vx) & (y >= vy) & (y <= (((vy*(x-width)) / vx) + height)),
        (x > vx) & (x < width+vx) & (y > height) & (y > (((vy*(x-width)) / vx) + height)) & (y < height+vy),
        (x > width) & (x < width+vx) & (y >= ((vy*(x-width)) / vx)) & (y < vy),
        (x > 0) & (x < vx) & (y > height) & (y <= (((vy*x) / vx) + height))
    ]

    for idx, condition in enumerate(conditions, start=1):
        i, j = numpy.where(condition)            
        correction_func = correction(i, j, x, y, vx, vy, width, height)
        if idx == 1:
            warped_image[i+1, j+1] *= correction_func[str(idx)][0]
        else:    
            warped_image[i+1, j+1] *= correction_func[str(idx)]

    warped_image[x > width+vx-edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height+vy-edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy*(x-width)) / vx) + edgepx] = 0
    warped_image[y > (((vy*x) / vx) + height) - edgepx] = 0
    warped_image[numpy.isnan(warped_image)] = 0
    return warped_image


def alpha_2(warped_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int, edgepx: int, section: int):
    """
    Input:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx > w and vy > h. The conditions are designed based on the pixel's 
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions_1 = [
        (x >= width) & (x <= vx) & (y >= (vy*x)/vx) & (y <= (vy/vx)*(x-width-vx)+vy+height), 
        (x > 0) & (x < width) & (y >= (vy*x)/vx) & (y < height), 
        (x > 0) & (x <= width) & (y > 0) & (y < (vy*x)/vx), 
        (x > vx) & (x < vx+width) & (y > vy) & (y <= (vy/vx)*(x-width-vx)+vy+height), 
        (x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width-vx)+vy+height) & (y < height+vy), 
        (x > width) & (x <= vx+width) & (y >= (vy*(x-width))/vx) & (y < vy) & (y < (vy*x)/vx) & (y > 0), 
        (x > 0) & (x <= vx) & (y < (vy/vx)*x+height) & (y >= height) & (y > (vy/vx)*(x-width-vx)+vy+height) 
    ]

    conditions_2 = [
        (x >= 0) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy) & (y > height) & (y < (vy*x)/vx), 
        (x >= 0) & (x <= vx) & (y > (vy*x)/vx) & (y < height),
        (x >= 0) & (x < width) & (y >= 0) & (y < (vy*x)/vx) & (y < height), 
        (x > width) & (x < vx+width) & (y <= ((vy*(x-width))/vx)+height) & (y > vy), 
        (x >= vx) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy+height) & (y > vy), 
        (x >= width) & (x <= vx+width) & (y > (vy/vx)*(x-width)) & (y < ((vy*(x-width))/vx)+height) & (y > 0) & (y <vy), 
        (x >= 0) & (x <= vx) & (y <= (vy/vx)*x+height) & (y > (vy/vx)*x) & (y > height) & (y <= height+vy) 
    ]

    conditions = [conditions_1, conditions_2]
    for idx, condition in enumerate(conditions[section-1], start=1):
        i, j = numpy.where(condition)
        correction_func = correction(i, j, x, y, vx, vy, width, height)
        if idx == 1:
            warped_image[i+1, j+1] *= correction_func[str(idx)][section]
        else:    
            warped_image[i+1, j+1] *= correction_func[str(idx)]

    warped_image[x > width+vx-edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height+vy-edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy*(x-width)) / vx) + edgepx] = 0
    warped_image[y > (((vy*x) / vx) + height) - edgepx] = 0
    warped_image[numpy.isnan(warped_image)] = 0
    return warped_image

def mirror(warped_image: numpy.ndarray):
    mirrored_image = []
    height, width = len(warped_image), len(warped_image[0])
    for i in range(height):
        mirrored_row = []
        for j in range(width - 1, -1, -1):
            mirrored_row.append(warped_image[i][j])
        mirrored_image.append(mirrored_row)
    return numpy.array(mirrored_image)

def intensity_weighted_variance(sensor_size: tuple[int, int],events: numpy.ndarray,velocity: tuple[float, float]):
    numpy.seterr(divide='ignore', invalid='ignore')
    t               = (events["t"][-1]-events["t"][0])/1e6
    edgepx          = t
    width           = sensor_size[0]
    height          = sensor_size[1]
    fieldx          = velocity[0] / 1e-6
    fieldy          = velocity[1] / 1e-6
    velocity        = (fieldx * 1e-6, fieldy * 1e-6)
    warped_image    = accumulate(sensor_size, events, velocity)
    vx              = numpy.abs(fieldx*t)
    vy              = numpy.abs(fieldy*t)
    x               = numpy.tile(numpy.arange(1, warped_image.pixels.shape[1]+1), (warped_image.pixels.shape[0], 1))
    y               = numpy.tile(numpy.arange(1, warped_image.pixels.shape[0]+1), (warped_image.pixels.shape[1], 1)).T
    corrected_iwe   = None
    var             = 0.0
    
    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height):
        corrected_iwe            = alpha_1(warped_image.pixels, x, y, vx, vy, width, height, edgepx)
        
    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_1(warped_image.pixels, x, y, vx, vy, width, height, edgepx)
        
    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        corrected_iwe            = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 1)

    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        corrected_iwe            = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 2)

    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 1)

    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 2)
    
    if corrected_iwe is not None:
        var = variance_loss_calculator(corrected_iwe)
    return var

def intensity_maximum(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return dvs_sparse_filter_extension.intensity_maximum(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )

def calculate_heuristic(self, velocity: Tuple[float, float]):
        if self.heuristic == "variance":
            return intensity_variance(
                (self.width, self.height), self.events, velocity
            )
        if self.heuristic == "variance_ts":
            return intensity_variance_ts(
                (self.width, self.height), self.events, velocity, self.tau
            )
        if self.heuristic == "weighted_variance":
            return intensity_weighted_variance(
                (self.width, self.height), self.events, velocity
            )
        if self.heuristic == "max":
            return intensity_maximum(
                (self.width, self.height), self.events, velocity
            )
        raise Exception("unknown heuristic")

def optimize_local(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    initial_velocity: tuple[float, float],  # px/µs
    tau: int,
    heuristic_name: str,  # max or variance
    method: str,  # Nelder-Mead, Powell, L-BFGS-B, TNC, SLSQP
    # see Constrained Minimization in https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    callback: typing.Callable[[numpy.ndarray], None],
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "variance_ts":
            return -intensity_variance_ts(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3), 
                tau=tau)
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unknown heuristic name "{heuristic_name}"')

    if method == "Nelder-Mead":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] * 1e3, initial_velocity[1] * 1e3],
            method=method,
            bounds=scipy.optimize.Bounds([-1.0, -1.0], [1.0, 1.0]),
            options={'maxiter': 100},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    elif method == "BFGS":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] / 1e2, initial_velocity[1] / 1e2],
            method=method,
            options={'ftol': 1e-9,'maxiter': 50},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    elif method == "Newton-CG":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] / 1e2, initial_velocity[1] / 1e2],
            method=method,
            jac=True,
            options={'ftol': 1e-9,'maxiter': 50},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    else:
        raise Exception(f'unknown optimisation method: "{method}"')

def optimize_cma(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    initial_velocity: tuple[float, float],
    initial_sigma: float,
    heuristic_name: str,
    iterations: int,
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unknown heuristic name "{heuristic_name}"')

    optimizer = cmaes.CMA(
        mean=numpy.array(initial_velocity) * 1e3,
        sigma=initial_sigma * 1e3,
        bounds=numpy.array([[-1.0, 1.0], [-1.0, 1.0]]),
    )
    best_velocity: tuple[float, float] = copy.copy(initial_velocity)
    best_heuristic = numpy.Infinity
    for _ in range(0, iterations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = heuristic((x[0] / 1e3, x[1] / 1e3))
            solutions.append((x, value))
        optimizer.tell(solutions)
        velocity_array, heuristic_value = sorted(
            solutions, key=lambda solution: solution[1]
        )[0]
        velocity = (velocity_array[0] / 1e3, velocity_array[1] / 1e3)
        if heuristic_value < best_heuristic:
            best_velocity = velocity
            best_heuristic = heuristic_value
    return (float(best_velocity[0]), float(best_velocity[1]))


def calculate_stats(events, labels):
    # Flatten labels to match event coordinates
    labels_flat = labels.flatten()
    
    # Combine x and y coordinates into a single array of tuples for uniqueness
    xy_pairs = numpy.array(list(zip(events["x"].flatten(), events["y"].flatten())))
    
    # Filter xy_pairs with label=1
    unique_xy_label_1 = numpy.unique(xy_pairs[labels_flat == 1], axis=0)
    
    # Count of unique pixels with label=1
    unique_pixels_label_1_count = len(unique_xy_label_1)
    
    # Percentage of events with label=1
    percent_label_1 = (numpy.sum(labels_flat == 1) / len(labels_flat)) * 100
    
    # Percentage of events with label=0
    percent_label_0 = (numpy.sum(labels_flat == 0) / len(labels_flat)) * 100
    
    return unique_pixels_label_1_count, percent_label_1, percent_label_0