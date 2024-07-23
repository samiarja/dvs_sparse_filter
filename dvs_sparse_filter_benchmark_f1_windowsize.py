import numpy as np
import scipy.io as sio
import os
from scipy.linalg.blas import ddot
from scipy.linalg.lapack import ztzrzf
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import typing
import pathlib
import dvs_sparse_filter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.io import savemat

'''
How well it removes hot pixels?
How well it removes background activity?
How well it preserves the sparse star signal?
'''

# recording_name                  = "2022-03-29T11-57-46Z_speed_survey_191.930378_-59.688764_0.00048828125"
# width, height, input_events     = dvs_sparse_filter.read_es_file(f"./data/{recording_name}.es")
# labeled_events                  = sio.loadmat(f'./output/F1/{recording_name}/{recording_name}_star_class.mat')
# ground_truth                    = labeled_events["star_class"].flatten()
# sensor_size                     = (width, height)

recording_name  = f"2024-06-11T12-38-52.910110Z_balanced_speed_43339_spacetrack" #f"2024-06-09T00-43-42.357645Z_balanced_speed_44850_spacetrack"
parent_folder   = f"/media/samiarja/VERBATIM HD/Dataset/EBSIST/recordings"

filter_idx  = 1
filter_type = ["STCF",
               "CrossConv",
               "DFWF",
               "TS",
               "YNoise",
               "KNoise",
               "EvFlow",
               "IETS",
               "STDF",
               "MLPF", #sep.
               "AEDNet", #sep.
               "deFEAST", #sep.
               "deODESSA", #sep.
               ]

width, height, events   = dvs_sparse_filter.read_es_file(f"{parent_folder}/{recording_name}/{recording_name}_psee400.es")
sensor_size             = (width, height)
labels                  = np.load(f"{parent_folder}/{recording_name}/labelled_events_v2.0.0.npy")
events_labels           = np.zeros(len(events["x"]), dtype=int)

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
events_with_labels = np.zeros(len(events), dtype=[('t', '<u8'),
                                                  ('x', '<u2'),
                                                  ('y', '<u2'),
                                                  ('on', '?'),
                                                  ('label', '<i2')])
events_with_labels['t'] = events['t']
events_with_labels['x'] = events['x']
events_with_labels['y'] = events['y']
events_with_labels['on'] = events['on']
events_with_labels['label'] = events_labels


kk = np.where(events_with_labels["label"]==0)[0]
label_array = dvs_sparse_filter.remove_hot_pixels_percentile(events_with_labels[kk], 0.005)

events_with_labels['label'][kk] = np.where(label_array == 1, -500, events_with_labels['label'][kk])
events_with_labels['label'][events_with_labels['label'] > 0] = 3  # Stars
events_with_labels['label'][events_with_labels['label'] == -500] = 2  # Hot pixels
events_with_labels['label'][events_with_labels['label'] < 0] = 1  # Satellites

ii = np.where(np.logical_and(events_with_labels["label"] < 3,
                             events_with_labels["t"] > 10e6, 
                             events_with_labels["t"] < events_with_labels["t"][-1]))

input_events = events_with_labels[ii]
ground_truth = input_events['label']
input_events['t'] = input_events['t'] - input_events['t'][0]

if not os.path.exists(f"output/roc_eval/{recording_name}"):
    os.mkdir(f"output/roc_eval/{recording_name}")

reduction_factor = 1  # increase factor per iteration
total_duration = input_events["t"][-1]
current_duration = total_duration * reduction_factor

final_performance = []
while current_duration <= total_duration:
    performance_metric = []
    start_time = 0
    iteration  = 0
    end_time   = current_duration
    
    dvs_sparse_filter.print_message(f"Current window: {current_duration/1e6}", color="red", style="bold")

    while end_time <= total_duration:
        time_window = [start_time, end_time]
        dvs_sparse_filter.print_message(f"Iteration {iteration} Window size: {[time_window[0]/1e6, time_window[1]/1e6]}", color="yellow", style="bold")

        output_events, performance, detected_noise = dvs_sparse_filter.filter_events(
            input_events, ground_truth, time_window, method=filter_type[filter_idx], save_performance=True
        )
        
        # Increment the start and end time for the next window
        start_time += current_duration * reduction_factor
        end_time   += current_duration * reduction_factor

        iteration += 1
        performance_metric.append(performance)

    performance_data = [row[:11] for row in performance_metric]
    performance_data = np.array(performance_data, dtype=float)
    performance_means = np.mean(performance_data, axis=0)
    final_performance.append((performance_means,current_duration/1e6))

    # Increase the window size by a factor
    current_duration += total_duration * reduction_factor
    
    peformance_path = f"output/roc_eval/{recording_name}/"
    
    # sio.savemat(f'./output/roc_eval/{recording_name}/method_{filter_type[filter_idx]}_performance_rate_{reduction_factor}.mat', {'final_performance': final_performance})
    
    ##################################################################################
    vx_velocity = np.zeros((len(input_events["x"]), 1)) + 0 / 1e6
    vy_velocity = np.zeros((len(input_events["y"]), 1)) + 0 / 1e6
    dd = np.where(np.logical_or(input_events["label"]==2,input_events["label"]==0))
    label_visualisation = input_events["label"]
    label_visualisation[dd]=0
    cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events,
                                                          label_visualisation.astype(np.int32),
                                                          (vx_velocity,vy_velocity))
    warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
    warped_image_segmentation_raw.save(f"output/roc_eval/{recording_name}/{filter_type[filter_idx]}_image_ground_truth.png")
    
    cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events,
                                                          detected_noise.astype(np.int32),
                                                          (vx_velocity,vy_velocity))
    warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
    warped_image_segmentation_raw.save(f"output/roc_eval/{recording_name}/{filter_type[filter_idx]}_image_detected_noise_and_signal.png")
    
    cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events[detected_noise==0],
                                                            detected_noise[detected_noise==0].astype(np.int32),
                                                            (vx_velocity[detected_noise==0],vy_velocity[detected_noise==0]))
    warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
    warped_image_segmentation_raw.save(f"output/roc_eval/{recording_name}/{filter_type[filter_idx]}_image_detected_noise_only_noise.png")
    
    cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events[detected_noise==1],
                                                            detected_noise[detected_noise==1].astype(np.int32),
                                                            (vx_velocity[detected_noise==1],vy_velocity[detected_noise==1]))
    warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
    warped_image_segmentation_raw.save(f"output/roc_eval/{recording_name}/{filter_type[filter_idx]}_image_detected_noise_only_signal.png")
    #################################################################################

with open(f'./output/roc_eval/{recording_name}/{filter_type[filter_idx]}_performance_rate_{reduction_factor}.txt', 'w') as file:
    for index, (array_data, value) in enumerate(final_performance):
        file.write(f"{index}:\n")
        array_str = np.array2string(array_data, separator=', ')
        file.write(f"{array_str}\n")
        file.write(f"{value}\n")
        file.write("\n")