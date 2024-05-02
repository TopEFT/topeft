#Relevant for Zgamma CR studies
#The function returns integer from 0 to 5 for each event. If for a given event, the integer is 4, it means that event has exactly 2 jets and 2 bjets, etc.

import awkward as ak

def multiplicityOfJetsAndbJets(jet_collection, bjet_collection):
    num_jets = ak.num(jet_collection)
    num_bjets = ak.num(bjet_collection)
    
    jets1_bjets1 = ((num_jets==1) & (num_bjets==1))
    jets2_bjets1 = ((num_jets==2) & (num_bjets==1))
    jetsgeq3_bjets1 = ((num_jets>=3) & (num_bjets==1))
    jets2_bjets2 = ((num_jets==2) & (num_bjets==2))
    jetsgeq3_bjets2 = ((num_jets>=3) & (num_bjets==2))


    return 1*jets1_bjets1 + 2*jets2_bjets1 + 3*jetsgeq3_bjets1 + 4*jets2_bjets2 + 5*jetsgeq3_bjets2
    
