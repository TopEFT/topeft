import numpy as np
import numba

@numba.njit
def maxHistoryPDGID(id_array, mom_array, counts):     #id_array is the pdgID array of the gen particle, mom_array is the gen Part idx of the mothers, counts is an array of the number of gen particles
    maxPDGID_array = np.ones(len(id_array),np.int32)*-9

    #offset is the starting index for this event
    offset = 0
    #i is the event number. len(counts) is the total number of events
    for i in range(len(counts)):
        #j is the gen particle within a given event i
        for j in range(counts[i]):
            maxPDGID_array[offset+j] = id_array[offset+j]
            idx = mom_array[offset+j]  #genPart index of the mother
            while idx != -1:  #if a mother exists
                maxPDGID_array[offset+j] = max(id_array[offset+idx], maxPDGID_array[offset+j]) #perhaps the most important part of the code
                idx = mom_array[offset+idx]
        offset += counts[i]     #we do this because we are done with this event. Move to next event

    return maxPDGID_array
