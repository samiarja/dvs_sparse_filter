import loris
import numpy as np
import matplotlib.pyplot as plt 
import os
import fnmatch
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import event_stream

events = []
FILENAME = "2022-03-31T11-53-34Z_speed_survey_191.930378_-59.688764_0.015625"
mat = sio.loadmat('/home/samiarja/Desktop/PhD/Code/hot_pixel_filter/data/' + FILENAME + '.mat')

events = mat['td']
print(events[0][0]["x"].shape)

matX  =  events[0][0]["x"]
matY  =  events[0][0]["y"]
matP  =  events[0][0]["p"]
matTs =  events[0][0]["ts"]

# nEvents = events[0][0]["x"].shape[1]
nEvents = matX.shape[0]
x  = matX.reshape((nEvents, 1))
y  = matY.reshape((nEvents, 1))
p  = matP.reshape((nEvents, 1))
ts = matTs.reshape((nEvents, 1))

events = np.zeros((nEvents,4))

events = np.concatenate((ts,x, y, p),axis=1).reshape((nEvents,4))

finalArray = np.asarray(events)
print(finalArray)
# finalArray[:,0] -= finalArray[0,0]

ordering = "txyp"
loris.write_events_to_file(finalArray, "/home/samiarja/Desktop/PhD/Code/hot_pixel_filter/data/" + FILENAME + ".es",ordering)
print("File: " + FILENAME + "converted to .es")
