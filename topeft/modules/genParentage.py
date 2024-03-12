import numpy as np
import numba

@numba.njit
def maxHistoryPDGID(id_array, mom_array, counts):
    maxPDGID_array = np.ones(len(id_array),np.int32)*-9  

    #offset is the starting index for this event
    offset = 0
    #i is the event number
    for i in range(len(counts)):
        #j is the gen particle within event i
        for j in range(counts[i]):
            maxPDGID_array[offset+j] = id_array[offset+j]
            idx = mom_array[offset+j] 
            while idx != -1:
                maxPDGID_array[offset+j] = max(id_array[offset+idx], maxPDGID_array[offset+j])
                idx = mom_array[offset+idx]
        offset += counts[i]
        
    return maxPDGID_array
