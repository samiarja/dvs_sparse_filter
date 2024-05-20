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
from matplotlib.patches import RegularPolygon, Ellipse, Circle
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

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

class SpatioTemporalCorrelationFilter:
    def __init__(self, size_x, size_y, subsample_by=1):
        self.num_must_be_correlated = 3  # k (constant)
        self.shot_noise_correlation_time_s = 0.01  # tau (variable for ROC)
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
        dt = int(round(self.shot_noise_correlation_time_s * 1e6))
        filtered_events = []
        for event in events:
            ts, x, y, on = event
            x >>= self.subsample_by
            y >>= self.subsample_by
            if not (0 <= x <= self.ssx and 0 <= y <= self.ssy):
                continue
            if self.timestamp_image[x, y] == DEFAULT_TIMESTAMP:
                self.store_timestamp_polarity(x, y, ts, on)
                continue
            ncorrelated = self.count_correlated_events(x, y, ts, dt)
            if ncorrelated < self.num_must_be_correlated:
                continue
            if self.filter_alternative_polarity_shot_noise_enabled and self.test_filter_out_shot_noise_opposite_polarity(x, y, ts, on):
                continue
            filtered_events.append(event)
            self.store_timestamp_polarity(x, y, ts, on)
        return numpy.array(filtered_events, dtype=events.dtype)

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


def plot_roc_curve(fpr,tpr):
    # auc = roc_auc_score(y_true,y_score)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # plt.title('roc curve' + str(auc))
    plt.plot(fpr,tpr,color='b',linewidth=1)
    plt.plot([0,1],[0,1],'r--')
    # plt.savefig(prefix + '_roccurve.pdf')
    # plt.clf()
    plt.show()


def roc_val(detected_noise, ground_truth):
    """
    Calculate the true positive (TP), true negative (TN), false positive (FP), and false negative (FN) rates.

    Parameters:
    label_hotpix_binary (numpy.array or list): Binary array or list where:
        - 0 indicates not detected noise event (considered as signal)
        - 1 indicates detected noise event
    ground_truth (numpy.array or list): Binary array or list where:
        - 0 indicates non-signal event (considered as noise)
        - 1 indicates signal event

    Returns:
    tuple: Normalized values of TP, TN, FP, and FN in the form (TP, TN, FP, FN)

    Definitions:
    - True Positive (TP): Signal event correctly labeled as signal (detected_noise == 0 and ground_truth == 1)
    - True Negative (TN): Noise event correctly labeled as noise (detected_noise == 1 and ground_truth == 0)
    - False Positive (FP): Noise event incorrectly labeled as signal (detected_noise == 0 and ground_truth == 0)
    - False Negative (FN): Signal event incorrectly labeled as noise (detected_noise == 1 and ground_truth == 1)
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
    
    # Compute precision, recall, and F1-score
    precision   = precision_score(ground_truth, 1 - detected_noise, pos_label=0)  # Set pos_label to 0 for noise
    recall      = recall_score(ground_truth, 1 - detected_noise, pos_label=0)
    f1          = f1_score(ground_truth, 1 - detected_noise, pos_label=0)
    
    print(f'TP: {TP} ({normalized_TP:.3f}), TN: {TN} ({normalized_TN:.3f}), FP: {FP} ({normalized_FP:.3f}), FN: {FN} ({normalized_FN:.3f})')
    print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}')

    return TP, TN, FP, FN, normalized_TP, normalized_TN, normalized_FP, normalized_FN, precision, recall, f1



def CrossConv(input_events, ground_truth, time_window):
    print("Start processing CrossConv filter...")
    ii = numpy.where(numpy.logical_and(input_events["t"] > time_window[0], input_events["t"] < time_window[1]))
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
    smart_mask = ratios < 2.0 #this should be 3 ideally

    yhot, xhot      = numpy.where(~smart_mask)
    label_hotpix    = numpy.zeros(len(input_events), dtype=bool)
    for xi, yi in zip(xhot, yhot):
        label_hotpix |= (y == xi) & (x == yi)
    label_hotpix_binary = label_hotpix.astype(int)

    jj              = numpy.where(label_hotpix_binary==0)[0]
    output_events   = input_events[jj]

    print(f'Number of detected hot pixels: {len(xhot)}')
    print(f'Events marked as hot pixels: {numpy.sum(label_hotpix)}')

    detected_noise  = label_hotpix_binary
    ground_truth    = ground_truth[ii]

    TP, TN, FP, FN, normalized_TP, normalized_TN, normalized_FP, normalized_FN, precision, recall, f1_score  = roc_val(detected_noise, ground_truth)
    performance     = [TP, TN, FP, FN, normalized_TP, normalized_TN, normalized_FP, normalized_FN, precision, recall, f1_score]

    return output_events, performance

def STCF(input_events, ground_truth, time_window):
    print("Start processing STCF filter...")

    ii = numpy.where(numpy.logical_and(input_events["t"] > time_window[0], input_events["t"] < time_window[1]))
    input_events = input_events[ii]
    ground_truth = ground_truth[ii]

    x_max, y_max = input_events['x'].max() + 1, input_events['y'].max() + 1

    # Create filter instance
    filter_instance = SpatioTemporalCorrelationFilter(size_x=x_max, size_y=y_max)

    # Prepare events
    events = numpy.zeros((len(input_events), 4), dtype=int)
    events[:, 0] = input_events['t']
    events[:, 1] = input_events['x']
    events[:, 2] = input_events['y']
    events[:, 3] = input_events['on']

    # Filter events
    filtered_events = filter_instance.filter_packet(events)

    # Extract filtered event indices
    filtered_indices = numpy.isin(input_events['t'], filtered_events[:, 0])

    # Prepare output events
    output_events = input_events[filtered_indices]

    # Create label_hotpix_binary for detected noise
    label_hotpix_binary = numpy.ones(len(input_events), dtype=int)
    label_hotpix_binary[filtered_indices] = 0

    print(f'Number of detected hot pixels: {len(input_events) - len(filtered_events)}')
    print(f'Events marked as hot pixels: {numpy.sum(label_hotpix_binary)}')

    detected_noise = label_hotpix_binary

    TP, TN, FP, FN, normalized_TP, normalized_TN, normalized_FP, normalized_FN, precision, recall, f1_score = roc_val(detected_noise, ground_truth)
    performance = [TP, TN, FP, FN, normalized_TP, normalized_TN, normalized_FP, normalized_FN, precision, recall, f1_score]

    return output_events, performance

def MLPF(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "MLPF performance metrics" 
    return output_events, performance

def deFEAST(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "deFEAST performance metrics" 
    return output_events, performance

def DWF(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "DWF performance metrics" 
    return output_events, performance

def TS(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "TS performance metrics" 
    return output_events, performance

def YNoise(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "YNoise performance metrics" 
    return output_events, performance

def RED(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "RED performance metrics" 
    return output_events, performance

def KNoise(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "KNoise performance metrics" 
    return output_events, performance

def EvFlow(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "EvFlow performance metrics" 
    return output_events, performance

def BAF(input_events, ground_truth, time_window):
    output_events = input_events  # Replace with actual processing
    performance = "BAF performance metrics" 
    return output_events, performance



def filter_events(input_events, ground_truth, time_window, method="STCF", save_performance=True):
    # Get the method function based on the method name
    method_function = globals().get(method)
    
    if method_function is None:
        raise ValueError(f"Method {method} not found")
    
    # Execute the method function
    start_time = time.time()
    output_events, performance = method_function(input_events, ground_truth, time_window)
    end_time = time.time()
    
    # Save performance metrics if requested
    if save_performance:
        performance += f" | Execution time: {end_time - start_time} seconds"
    
    return output_events, performance if save_performance else output_events

###########################################################################

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
    smart_mask = ratios < 3.0

    yhot, xhot = numpy.where(~smart_mask)
    label_hotpix = numpy.zeros(len(events), dtype=bool)
    for xi, yi in zip(xhot, yhot):
        label_hotpix |= (y == xi) & (x == yi)
    label_hotpix_binary = label_hotpix.astype(int)

    print(f'Number of detected hot pixels: {len(xhot)}')
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
    stopping_threshold = 0.001
    eventstart_time = t_o[0]
    eventend_time = t_o[-1]
    golden_search_range = t_o[-1] - t_o[0]

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
            optimal_focus_time = event_rate_ts_list[max_ev_rate_index]
            event_rate_output = event_rate_list[max_ev_rate_index]*delta_time
            updated_focus_moving_step = abs(optimal_focus_time - prev_optimal_focus_time)
            prev_optimal_focus_time = optimal_focus_time
            eventstart_time = optimal_focus_time - delta_time
            eventend_time = optimal_focus_time
        else:
            return optimal_focus_time

    return optimal_focus_time

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