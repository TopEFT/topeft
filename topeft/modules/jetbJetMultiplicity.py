#Relevant only for Zgamma CR studies where we have 5 njet_bjet categories from Top-21-004
#For completeness, there are other low jet_bjet categories as well
#The function returns integer from 1 to 9 for each event. If for a given event, the integer is 8, it means that event has exactly 2 jets and 2 bjets, etc.

def multiplicityOfJetsAndbJets(num_jets, num_bjets):
    jets0_bjets0 = ((num_jets==0) & (num_bjets==0))
    jets1_bjets0 = ((num_jets==1) & (num_bjets==0))
    jets2_bjets0 = ((num_jets==2) & (num_bjets==0))
    jetsgeq3_bjets0 = ((num_jets>=3) & (num_bjets==0))
    jets1_bjets1 = ((num_jets==1) & (num_bjets==1))
    jets2_bjets1 = ((num_jets==2) & (num_bjets==1))
    jetsgeq3_bjets1 = ((num_jets>=3) & (num_bjets==1))
    jets2_bjets2 = ((num_jets==2) & (num_bjets==2))
    jetsgeq3_bjets2 = ((num_jets>=3) & (num_bjets==2))

    return 1*jets0_bjets0 + 2*jets1_bjets0 + 3*jets2_bjets0 + 4*jetsgeq3_bjets0 + 5*jets1_bjets1 + 6*jets2_bjets1 + 7*jetsgeq3_bjets1 + 8*jets2_bjets2 + 9*jetsgeq3_bjets2
