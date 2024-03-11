import loris
import numpy as np
import matplotlib.pyplot as plt 
import os
import fnmatch
from tqdm import tqdm
import scipy.io as sio

FILENAME = "psee400"
myPath = "/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-54-02Z_speed_survey_191.930378_-59.688764_0.0078125/" + FILENAME + ".es"
events_stream = []
my_file = loris.read_file(myPath)
events = my_file['events']
for idx in tqdm(range(len(events))):
    event = -1*np.ones(4,dtype=np.uint16)
    event[0] = events[idx][0]
    event[1] = events[idx][1]
    event[2] = events[idx][2]
    event[3] = events[idx][3]
    events_stream.append(event)

sio.savemat("/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-54-02Z_speed_survey_191.930378_-59.688764_0.0078125/" + FILENAME + ".mat",{'events':np.asarray(events_stream)})
print(FILENAME+" file's saved")