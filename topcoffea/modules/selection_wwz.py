import numpy as np
import awkward as ak
from mt2 import mt2
from coffea.nanoevents.methods import vector

import topcoffea.modules.selection as sel
from topcoffea.modules.GetValuesFromJsons import get_param


# 4l selection # SYNC
def add4lmask_wwz(events, year, isData):

    # Leptons and padded leptons
    leps = events.l_wwz_t
    leps_padded = ak.pad_none(leps,4)

    # Filters
    filter_flags = events.Flag
    filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & (((year == "2016")|(year == "2016APV")) | filter_flags.ecalBadCalibFilter) & (isData | filter_flags.eeBadScFilter)

    # Lep multiplicity
    nlep_4 = (ak.num(leps) == 4)

    # Check if the leading lep associated with Z has pt>25
    on_z = ak.fill_none(sel.get_Z_peak_mask(leps_padded[:,0:4],pt_window=10.0),False)

    # Remove low mass resonances
    cleanup = (events.min_mll_afos > 12)

    mask = filters & nlep_4 & on_z & cleanup
    events['is4lWWZ'] = ak.fill_none(mask,False)


# Takes as input the lep collection
# Finds SFOS pair that is closest to the Z peak
# Returns object level mask with "True" for the leptons that are part of the Z candidate and False for others
def get_z_candidate_mask(lep_collection):

    # Attach the local index to the lepton objects
    lep_collection['idx'] = ak.local_index(lep_collection, axis=1)

    # Make all pairs of leptons
    ll_pairs = ak.combinations(lep_collection, 2, fields=["l0","l1"])
    ll_pairs_idx = ak.argcombinations(lep_collection, 2, fields=["l0","l1"])

    # Check each pair to see how far it is from the Z
    dist_from_z_all_pairs = abs((ll_pairs.l0+ll_pairs.l1).mass - 91.2)

    # Mask out the pairs that are not SFOS (so that we don't include them when finding the one that's closest to Z)
    # And then of the SFOS pairs, get the index of the one that's cosest to the Z
    sfos_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    dist_from_z_sfos_pairs = ak.mask(dist_from_z_all_pairs,sfos_mask)
    sfos_pair_closest_to_z_idx = ak.argmin(dist_from_z_sfos_pairs,axis=-1,keepdims=True)

    # Construct a mask (of the shape of the original lep array) corresponding to the leps that are part of the Z candidate
    mask = (lep_collection.idx == ak.flatten(ll_pairs_idx.l0[sfos_pair_closest_to_z_idx]))
    mask = (mask | (lep_collection.idx == ak.flatten(ll_pairs_idx.l1[sfos_pair_closest_to_z_idx])))
    mask = ak.fill_none(mask, False)

    return mask


# Get the pair of leptons that are the Z candidate, and the W candidate leptons
# Basicially this function is convenience wrapper around get_z_candidate_mask()
def get_wwz_candidates(lep_collection):

    z_candidate_mask = get_z_candidate_mask(lep_collection)

    # Now we can grab the Z candidate leptons and the non-Z candidate leptons
    leps_from_z_candidate = lep_collection[z_candidate_mask]
    leps_not_z_candidate = lep_collection[~z_candidate_mask]

    leps_from_z_candidate_ptordered = leps_from_z_candidate[ak.argsort(leps_from_z_candidate.pt, axis=-1,ascending=False)]
    leps_not_z_candidate_ptordered  = leps_not_z_candidate[ak.argsort(leps_not_z_candidate.pt, axis=-1,ascending=False)]

    return [leps_from_z_candidate,leps_not_z_candidate]


# Do WWZ pre selection, construct event level mask
# Convenience function around get_wwz_candidates() and get_z_candidate_mask()
def attach_wwz_preselection_mask(events,lep_collection):

    leps_z_candidate_ptordered, leps_w_candidate_ptordered = get_wwz_candidates(lep_collection)

    # Pt requirements (assumes lep_collection is pt sorted and padded)
    pt_mask = ak.fill_none((lep_collection[:,0].pt > 25) & (lep_collection[:,1].pt > 15),False)

    # Build an event level mask for OS requirements for the W candidates
    os_mask = ak.any(((leps_w_candidate_ptordered[:,0:1].pdgId)*(leps_w_candidate_ptordered[:,1:2].pdgId)<0),axis=1) # Use ak.any() here so that instead of e.g [[None],None,...] we have [False,None,...]
    os_mask = ak.fill_none(os_mask,False) # Replace the None with False in the mask just to make it easier to think about

    # Build an event level mask for same flavor W lepton candidates
    sf_mask = ak.any((abs(leps_w_candidate_ptordered[:,0:1].pdgId) == abs(leps_w_candidate_ptordered[:,1:2].pdgId)),axis=1) # Use ak.any() here so that instead of e.g [[None],None,...] we have [False,None,...]
    sf_mask = ak.fill_none(sf_mask,False) # Replace the None with False in the mask just to make it easier to think about

    # Build an event level mask that checks if the z candidates are close enough to the z
    z_mass = (leps_z_candidate_ptordered[:,0:1]+leps_z_candidate_ptordered[:,1:2]).mass
    z_mass_mask = (abs((leps_z_candidate_ptordered[:,0:1]+leps_z_candidate_ptordered[:,1:2]).mass-91.2) < 10.0)
    z_mass_mask = ak.fill_none(ak.any(z_mass_mask,axis=1),False) # Make sure None entries are false

    # Build an event level mask to check the iso and sip3d for leps from Z and W
    leps_z_e = leps_z_candidate_ptordered[abs(leps_z_candidate_ptordered.pdgId)==11] # Just the electrons
    leps_w_e = leps_w_candidate_ptordered[abs(leps_w_candidate_ptordered.pdgId)==11] # Just the electrons
    iso_mask_z_e = ak.fill_none(ak.all((leps_z_e.pfRelIso03_all < get_param("wwz_z_iso")),axis=1),False) # This requirement is just on the electrons
    iso_mask_w_e = ak.fill_none(ak.all((leps_w_e.pfRelIso03_all < get_param("wwz_w_iso")),axis=1),False) # This requirement is just on the electrons
    id_mask_z = ak.fill_none(ak.all((leps_z_candidate_ptordered.sip3d < get_param("wwz_z_sip3d")),axis=1),False)
    id_mask_w = ak.fill_none(ak.all((leps_w_candidate_ptordered.sip3d < get_param("wwz_w_sip3d")),axis=1),False)
    id_iso_mask = (id_mask_z & id_mask_w & iso_mask_z_e & iso_mask_w_e)

    # The final preselection mask
    wwz_presel_mask = (os_mask & pt_mask & id_iso_mask)

    # Attach to the lepton objects
    events["wwz_presel_sf"] = (wwz_presel_mask & sf_mask)
    events["wwz_presel_of"] = (wwz_presel_mask & ~sf_mask)


# Get MT2 for WW
def get_mt2(w_lep0,w_lep1,met):

    # Construct misspart vector, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7 (but pass 0 not pi/2 for met eta)
    nevents = len(np.zeros_like(met))
    misspart = ak.zip(
        {
            "pt": met.pt,
            "eta": 0,
            "phi": met.phi,
            "mass": np.full(nevents, 0),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    # Do the boosts, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7
    rest_WW = w_lep0 + w_lep1 + misspart
    beta_from_miss_reverse = rest_WW.boostvec
    beta_from_miss = beta_from_miss_reverse.negative()
    w_lep0_boosted = w_lep0.boost(beta_from_miss)
    w_lep1_boosted = w_lep1.boost(beta_from_miss)
    misspart_boosted = misspart.boost(beta_from_miss)

    # Get the mt2 variable, use the mt2 package: https://pypi.org/project/mt2/
    mt2_var = mt2(
        w_lep0.mass, w_lep0_boosted.px, w_lep0_boosted.py,
        w_lep1.mass, w_lep1_boosted.px, w_lep1_boosted.py,
        misspart_boosted.px, misspart_boosted.py,
        np.zeros_like(met.pt), np.zeros_like(met.pt),
    )

    return mt2_var
