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

# evaluation_metrics_all = np.load("/media/samiarja/VERBATIM HD/Denoising/CrossConv/evaluation_metrics_all.npy")


filter_idx  = 0
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

recording_name  = f"2024-06-11T12-38-52.910110Z_balanced_speed_43339_spacetrack"
parent_folder   = f"/media/samiarja/VERBATIM HD/Dataset/EBSIST/recordings"

recording_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
recording_names   = recording_folders[:10]

for recording_name in recording_names:
    print(f"Processing file: {recording_name}")
    folder_path = os.path.join(parent_folder, recording_name)
            
    es_file_path = None
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".es"):
            es_file_path = os.path.join(folder_path, file_name)
            break
    
    width, height, events = dvs_sparse_filter.read_es_file(es_file_path)
    sensor_size = (width, height)        
    
    if os.path.exists(f"{parent_folder}/{recording_name}/labelled_events_v2.0.0.npy"):
        labels = np.load(f"{parent_folder}/{recording_name}/labelled_events_v2.0.0.npy")
    else:
        labels = np.load(f"{parent_folder}/{recording_name}/labelled_events_v2.5.0.npy")
    
    events_labels = np.zeros(len(events["x"]), dtype=int)

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
    events_with_labels['t']     = events['t']
    events_with_labels['x']     = events['x']
    events_with_labels['y']     = events['y']
    events_with_labels['on']    = events['on']
    events_with_labels['label'] = events_labels


    kk = np.where(events_with_labels["label"]==0)[0]
    label_array = dvs_sparse_filter.remove_hot_pixels_percentile(events_with_labels[kk], 0.005)

    events_with_labels['label'][kk] = np.where(label_array == 1, -500, events_with_labels['label'][kk])
    events_with_labels['label'][events_with_labels['label'] > 0] = 3  # Stars
    events_with_labels['label'][events_with_labels['label'] == -500] = 2  # Hot pixels
    events_with_labels['label'][events_with_labels['label'] < 0] = 1  # Satellites

    ii = np.where(np.logical_and(events_with_labels["t"] > 10e6, 
                                events_with_labels["t"] < events_with_labels["t"][-1]))

    input_events = events_with_labels[ii]
    ground_truth = input_events['label']
    input_events['t'] = input_events['t'] - input_events['t'][0]

    if filter_type[filter_idx] == "CrossConv":
        if not os.path.exists(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}"):
            os.mkdir(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}")
        
        if not os.path.exists(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}"):
            os.mkdir(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}")
            
        ratio_par_range   = np.arange(0.1,8.1,0.1)
        evaluation_metric = []
        
        count = 0
        for ratio_par in ratio_par_range:
            output_events, detected_noise  = dvs_sparse_filter.CrossConv_HotPixelFilter(input_events,ratio_par)
            precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA  = dvs_sparse_filter.roc_val(
                                                                                                                input_events, 
                                                                                                                detected_noise, 
                                                                                                                ground_truth)
            
            np.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/output_events_detected_noise_ratio_par_{ratio_par:.3f}", output_events, detected_noise)
            
            HP_no_idx = np.where(np.logical_or(ground_truth==0, ground_truth==1))[0] #only noise
            input_events_selected = input_events[HP_no_idx]
            
            ground_truth_selected = ground_truth[HP_no_idx]
            detected_noise_selected = detected_noise[HP_no_idx]
            
            # how many satellites events were predicted as satellites and they were actually satellites
            TP = (np.sum((detected_noise_selected == 1) & (ground_truth_selected == 1)))/len(np.where(ground_truth_selected==1)[0])
            # an event was predicted as satellite but it was actually noise
            FP = (np.sum((detected_noise_selected == 1) & (ground_truth_selected == 0)))/len(np.where(ground_truth_selected==0)[0])
            
            dvs_sparse_filter.print_message(f"ratio_par: {ratio_par} True Positive: {TP:.3f} False Positive: {FP:.3f}", color="yellow", style="bold")
            
            evaluation_metric.append((ratio_par, TP, FP, precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA))
            
            ss = np.where(detected_noise_selected==0)[0]
            vx_velocity = np.zeros((len(input_events["x"]), 1)) + 0 / 1e6
            vy_velocity = np.zeros((len(input_events["y"]), 1)) + 0 / 1e6
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected[ss],
                                                                    detected_noise_selected[ss].astype(np.int32),
                                                                    (vx_velocity[ss],vy_velocity[ss]))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/detected_noise_{ratio_par:.3f}.png")
            
            nn = np.where(detected_noise_selected==1)[0]
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected[nn],
                                                                    detected_noise_selected[nn].astype(np.int32),
                                                                    (vx_velocity[nn],vy_velocity[nn]))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/detected_signal_{ratio_par:.3f}.png")
            
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected,
                                                                    detected_noise_selected.astype(np.int32),
                                                                    (vx_velocity,vy_velocity))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/detected_signal_and_noise_{ratio_par:.3f}.png")
            
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected,
                                                                    ground_truth_selected.astype(np.int32),
                                                                    (vx_velocity,vy_velocity))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/groundtruth_signal_and_noise.png")
            
            print("...")
            count += 1

    if filter_type[filter_idx] == "STCF":
        if not os.path.exists(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}"):
            os.mkdir(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}")
        
        if not os.path.exists(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}"):
            os.mkdir(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}")
            
        ratio_par_range   = np.arange(0.1,8.1,0.1)
        evaluation_metric = []
        
        count = 0
        for ratio_par in ratio_par_range:
            filter_instance  = dvs_sparse_filter.SpatioTemporalCorrelationFilter(size_x=width, 
                                                                                size_y=height, 
                                                                                num_must_be_correlated=2, 
                                                                                shot_noise_correlation_time_s=0.1)
            boolean_mask, output_events = filter_instance.filter_packet(input_events)
            detected_noise = boolean_mask.astype(int)
            
            precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA  = dvs_sparse_filter.roc_val(
                                                                                                                    input_events, 
                                                                                                                    detected_noise, 
                                                                                                                    ground_truth)
                
            np.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/output_events_detected_noise_ratio_par_{ratio_par:.3f}", output_events, detected_noise)
                
            
            HP_no_idx = np.where(np.logical_or(ground_truth==0, ground_truth==1))[0] #only noise
            input_events_selected = input_events[HP_no_idx]
            
            ground_truth_selected = ground_truth[HP_no_idx]
            detected_noise_selected = detected_noise[HP_no_idx]
            
            ss = np.where(detected_noise_selected==0)[0]
            vx_velocity = np.zeros((len(input_events["x"]), 1)) + 0 / 1e6
            vy_velocity = np.zeros((len(input_events["y"]), 1)) + 0 / 1e6
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected[ss],
                                                                    detected_noise_selected[ss].astype(np.int32),
                                                                    (vx_velocity[ss],vy_velocity[ss]))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/detected_noise_{ratio_par:.3f}.png")
            
            nn = np.where(detected_noise_selected==1)[0]
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected[nn],
                                                                    detected_noise_selected[nn].astype(np.int32),
                                                                    (vx_velocity[nn],vy_velocity[nn]))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/detected_signal_{ratio_par:.3f}.png")
            
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected,
                                                                    detected_noise_selected.astype(np.int32),
                                                                    (vx_velocity,vy_velocity))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/detected_signal_and_noise_{ratio_par:.3f}.png")
            
            cumulative_map_object, seg_label = dvs_sparse_filter.accumulate_cnt_rgb((width, height),input_events_selected,
                                                                    ground_truth_selected.astype(np.int32),
                                                                    (vx_velocity,vy_velocity))
            warped_image_segmentation_raw    = dvs_sparse_filter.rgb_render_white(cumulative_map_object, seg_label)
            warped_image_segmentation_raw.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/groundtruth_signal_and_noise.png")
            
        
        
    ratio_par, TP, FP, precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = zip(*evaluation_metric)
    plt.figure()
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve')
    plt.plot(FP, TP, color='b', linewidth=1, marker='o', markersize=5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xticks([i * 0.1 for i in range(11)])
    plt.yticks([i * 0.1 for i in range(11)])
    plt.savefig(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/{filter_type[filter_idx]}_roc_curve.png")

    # dvs_sparse_filter.plot_roc_curve(FP, TP, filter_type[filter_idx], recording_name)
    np.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{recording_name}/evaluation_metrics", evaluation_metric)
    print("Plot ROC curve")
    
        
## plot the ROC for all recording on a particular method, Use mean
method_folder = f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}" 
processed_folders = [f for f in os.listdir(method_folder) if os.path.isdir(os.path.join(method_folder, f))]

TPR = []
FPR = []
for recording_name in processed_folders:
    load_eval = np.load(f"{method_folder}/{recording_name}/evaluation_metrics.npy")
    ratio_par, TP, FP, precision_noise, recall_noise, f1_noise, precision_hp, recall_hp, f1_hp, SR, NR, HPR, DA, HDA = zip(*load_eval)
    TPR.append(TP)
    FPR.append(FP)

TPR_array = np.array(TPR)
mean_TPR  = np.mean(TPR_array, axis=0)
FPR_array = np.array(FPR)
mean_FPR  = np.mean(FPR_array, axis=0)

# Print out SR, NR, HPR, DA, AUC
# Save them too

plt.figure()
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve')
plt.plot(mean_FPR, mean_TPR, color='b', linewidth=1, marker='o', markersize=5)
plt.plot([0, 1], [0, 1], 'r--')
plt.xticks([i * 0.1 for i in range(11)])
plt.yticks([i * 0.1 for i in range(11)])
plt.savefig(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/{filter_type[filter_idx]}_roc_curve.png")
np.save(f"/media/samiarja/VERBATIM HD/Denoising/{filter_type[filter_idx]}/evaluation_metrics_all", TPR,FPR)


