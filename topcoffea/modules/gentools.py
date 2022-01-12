import awkward as ak

####################### Getting children of top #######################

# Get final copies of children of gen particles
def get_final_children(genpart_children_candidates,innermost_dim):
    last_copy_flag_mask = genpart_children_candidates.hasFlags(["isLastCopy"])  # Works in most cases
    no_children_mask    = (ak.num(genpart_children_candidates.children,axis=innermost_dim)==0) # For the cases where there are no children but isLastCopy is False
    a_children_mask     = ((ak.num(genpart_children_candidates.children,axis=innermost_dim)==1) & ak.all(abs(genpart_children_candidates.children.pdgId)==22,axis=innermost_dim)) # For the inexplicable q->gamma
    e_children_mask     = ((ak.num(genpart_children_candidates.children,axis=innermost_dim)==2) & ak.all(abs(genpart_children_candidates.children.pdgId)==11,axis=innermost_dim)) # For the inexplicable q->ee
    last_copy_mask      = ak.where((~last_copy_flag_mask & (no_children_mask | a_children_mask | e_children_mask)), ~last_copy_flag_mask, last_copy_flag_mask)
    hard_proc_mask      = genpart_children_candidates.hasFlags(["fromHardProcess"])
    final_children_mask = (last_copy_mask & hard_proc_mask)
    return genpart_children_candidates[final_children_mask]


# Get a collection of the children (b,W) of the tops
def get_t_children(genpart):

    # Get the top children
    is_top = ((abs(genpart.pdgId)==6) & (genpart.hasFlags(["isLastCopy","fromHardProcess"])))
    tops = genpart[is_top]
    tops_children = get_final_children(tops.distinctChildren,innermost_dim=3)

    # In the weird cases where the b splits and both end up as final particles, only keep the one with lastCopy flag
    num_b_quarks = (ak.num(tops_children[abs(tops_children.pdgId)==5],axis=2))
    tops_children_rm_spurious_b  = ak.where((num_b_quarks>1),tops_children[tops_children.hasFlags(["isLastCopy"])],tops_children)

    return tops_children_rm_spurious_b


# Check the children_candidates for particles of pdg_id, and apply lastCopy to those which show up more than once
def rm_spurious_particles(children_candidates,innermost_axis,pdg_id):
    # Find the W decay products that include  more than one of the same particle
    n_particles_of_interest = ak.num(children_candidates[abs(children_candidates.pdgId)==pdg_id],axis=innermost_axis)
    # If there's more than one of the same particle in W's children, something is wrong
    is_spurious = ak.flatten(n_particles_of_interest>1,axis=-1)
    # We assume one of the two has a last copy flag, and the other does not, so get a mask of isLastCopy
    # Note we need this mask to be the same shape as the array, so we've introduces some None values when applying the mask
    w_children_last_copy_flag = ak.mask(children_candidates,children_candidates.hasFlags(["isLastCopy"]))
    # Here this is a nested ak.where(), it does this:
    # For W decay products that have more than one particle, apply an isLastCopy mask to those particles
    w_children_rm_spurious = ak.where(is_spurious,(ak.where(abs(children_candidates.pdgId)==pdg_id,w_children_last_copy_flag,children_candidates)),children_candidates)
    # Get rid of unnecessary level of nestedness
    w_children_rm_spurious = ak.flatten(w_children_rm_spurious,axis=innermost_axis)
    # Remove the None values from when we applied the isLastCopy mask 
    w_children_rm_spurious = (w_children_rm_spurious[~ak.is_none(w_children_rm_spurious,axis=innermost_axis-1)])
    # Put back original level of nestedness
    #print("\nin  ",children_candidates.pdgId)
    #print("out ",w_children_rm_spurious.pdgId)
    #w_children_rm_spurious = ak.singletons(w_children_rm_spurious)
    #print("out ",w_children_rm_spurious.pdgId)
    return w_children_rm_spurious


# Get the decay products of the t (b,q,q or b,l,nu)
def get_t_decay_products(t_children):

    is_b_mask = abs(t_children.pdgId) == 5
    is_w_mask = abs(t_children.pdgId) == 24

    # Need the children of the W
    w_children = get_final_children(t_children[is_w_mask].distinctChildren,innermost_dim=4)

    # Remove the spurious particles from the final state
    w_children_rm_spurious = w_children
    #for check_id in [3,4]:
    for check_id in [4]:
        # Right now can only loop once because of the nestendess of the returned array...
        w_children_rm_spurious = rm_spurious_particles(w_children_rm_spurious,innermost_axis=3,pdg_id=check_id)
    #w_children_rm_spurious = ak.flatten(w_children_rm_spurious,axis=3) # Don't need this additional level of nesting
    #print("w after",w_children_rm_spurious.pdgId)

    # Put w decay products back together with the b
    t_children_b = t_children[is_b_mask]
    t_children_decayed = ak.concatenate([w_children_rm_spurious,t_children_b],axis=2)

    return t_children_decayed

# Get a mask of top decays that are hadronic
def is_had_top(top_decay_products):

    # Check the decay products to see if they look like hadronic tops
    is_b_from_t     = ((abs(top_decay_products.distinctParent.pdgId)==6)  & (abs(top_decay_products.pdgId)==5))
    is_q_from_W     = ((abs(top_decay_products.distinctParent.pdgId)==24) & (top_decay_products.pdgId>=1) & (top_decay_products.pdgId<=5))
    is_qbar_from_W  = ((abs(top_decay_products.distinctParent.pdgId)==24) & (top_decay_products.pdgId<=-1) & (top_decay_products.pdgId>=-5))
    has_b_from_t     = (ak.count_nonzero(ak.values_astype(is_b_from_t,int),axis=-1)==1)
    has_q_from_W     = (ak.count_nonzero(ak.values_astype(is_q_from_W,int),axis=-1)==1)
    has_qbar_from_W  = (ak.count_nonzero(ak.values_astype(is_qbar_from_W,int),axis=-1)==1)

    #print("The decay product checks:")
    #print("\tis_q_f_W:    ",is_q_from_W)
    #print("\tis_qbar_f_W: ",is_qbar_from_W)
    #print("\tis_b_f_t     ",is_b_from_t)
    #print("\thas_q_f_W:   ",has_q_from_W)
    #print("\thas_qbar_f_W:",has_qbar_from_W)
    #print("\thas_b_f_t    ",has_b_from_t)

    # Look to see if something looks wrong with our bqq from the top
    q_consistency = (has_q_from_W != has_qbar_from_W)
    b_consistency = (~has_b_from_t)
    q_nissue = ak.count_nonzero(q_consistency)
    b_nissue = ak.count_nonzero(b_consistency)
    #print("q consistency:",q_consistency)
    #print("b consistency:",b_consistency)
    print("q number of issues:",q_nissue)
    print("b number of issues:",b_nissue)

    ok_decay_products = ((has_q_from_W == has_qbar_from_W) & (has_b_from_t))
    has_hadtop = (has_q_from_W & has_qbar_from_W & has_b_from_t & ok_decay_products)

    return has_hadtop


# Get mask of hadtop that we can match to jets
def get_matchable_mask(hadtop):
    pt_mask = ak.all(hadtop.pt>30.0,axis=2)
    eta_mask = ak.all(abs(hadtop.eta)<2.5,axis=2)
    return (pt_mask & eta_mask)


####################### Match children of top to jets #######################

# Find the jets that best match the bqq genparticles
def get_bqq_jets(had_tops,jets,btag_wp):

    # Get each particle from the genparticles bqq array 
    gen_b    = had_tops[abs(had_tops.distinctParent.pdgId)==6]
    gen_q    = had_tops[((abs(had_tops.distinctParent.pdgId)==24) & (had_tops.pdgId>0))]
    gen_qbar = had_tops[((abs(had_tops.distinctParent.pdgId)==24) & (had_tops.pdgId<0))]

    # Find combinations of each gen bqq particle and jets
    b_combos    = ak.cartesian({"gen_b":gen_b,"jets":jets})
    q_combos    = ak.cartesian({"gen_q":gen_q,"jets":jets})
    qbar_combos = ak.cartesian({"gen_qbar":gen_qbar,"jets":jets})

    # The dr values for each particle
    dr_b    = b_combos["gen_b"].delta_r(b_combos["jets"])
    dr_q    = q_combos["gen_q"].delta_r(q_combos["jets"])
    dr_qbar = qbar_combos["gen_qbar"].delta_r(qbar_combos["jets"])

    # Mask the non b jets
    dr_b = ak.mask(dr_b,(b_combos["jets"].btagDeepFlavB>btag_wp))

    # Get arg of best b match
    dr_b_argmin = ak.argmin(dr_b,axis=-1)

    # Get arg of best q match (remove b match from jets)
    dr_q = ak.mask(dr_q,(ak.local_index(q_combos["jets"],axis=1)!=dr_b_argmin))
    dr_q_argmin = ak.argmin(dr_q,axis=-1)

    # Get arg of best qbar match (remove b and q match from jets)
    dr_qbar = ak.mask(dr_qbar,((ak.local_index(qbar_combos["jets"],axis=1)!=dr_b_argmin) & (ak.local_index(qbar_combos["jets"],axis=1)!=dr_q_argmin)))
    dr_qbar_argmin = ak.argmin(dr_qbar,axis=-1)

    # The jets with best dr vals
    jets_b_drmin = b_combos["jets"][ak.singletons(dr_b_argmin)]
    jets_q_drmin = q_combos["jets"][ak.singletons(dr_q_argmin)]
    jets_qbar_drmin = qbar_combos["jets"][ak.singletons(dr_qbar_argmin)]

    # The dr vals themselves
    b_drmin = b_combos["gen_b"].delta_r(b_combos["jets"])[ak.singletons(dr_b_argmin)]
    q_drmin = q_combos["gen_q"].delta_r(q_combos["jets"])[ak.singletons(dr_q_argmin)]
    qbar_drmin = qbar_combos["gen_qbar"].delta_r(qbar_combos["jets"])[ak.singletons(dr_qbar_argmin)]
    qq_drmin = ak.concatenate([q_drmin,qbar_drmin],axis=1)
    bqq_drmin = ak.concatenate([qq_drmin,b_drmin],axis=1)

    # Put the matched jets back together
    jets_matched_qq  = ak.concatenate([jets_q_drmin,jets_qbar_drmin],axis=1)
    jets_matched_bqq = ak.concatenate([jets_matched_qq,jets_b_drmin],axis=1)

    #jets_matched_bqq_sum = jets_matched_bqq.sum()
    #jets_matched_qq_sum = jets_matched_qq.sum()

    # Maybe should have seperate functions for these
    return [jets_matched_bqq,jets_matched_qq,bqq_drmin,qq_drmin]


###############################################

def main():

    '''
    ### Get the top decay products ###

    # Get the b,W from the t
    t_children = get_t_children(genpart)
    print("\nt_children:")
    print(t_children.pdgId)

    # Get the b,q,q or b,l,nu from the b,W
    t_decay_products = get_t_decay_products(t_children)
    print("\nt_decay_products:")
    print(t_decay_products.pdgId)

    is_had_top_mask = is_had_top(t_decay_products)
    print("\nis_had_top_mask",is_had_top_mask)

    has_matchable_decay_products = get_matchable_mask(t_decay_products)
    print("\nhas_matchable_decay_products",has_matchable_decay_products)

    #print("\nt_decay_products:")
    #for i,x in enumerate(t_decay_products):
    #    if i > 10: break
    #    print(i,x.pdgId,x.pt,x.eta)
    #
    #print("\nis_had_top_mask:")
    #for i,x in enumerate(is_had_top_mask):
    #    if i > 10: break
    #    print(i,x)
    #
    #print("\nhas_matchable_decay_products:")
    #for i,x in enumerate(has_matchable_decay_products):
    #    if i > 10: break
    #    print(i,x)


    # Just grab the first hadronic top for now
    had_tops = ak.flatten(t_decay_products[is_had_top_mask&has_matchable_decay_products][:,:1],axis=2)


    ### Match to jets ###

    print ("\nJets:")
    jets = events.Jet
    jets = jets[jets.pt>30.0]
    for i,x in enumerate(jets):
        #if i > 10: break
        #print(i,x,x.pt)
        if i == 52: print(i,x,x.pt,x.eta,x.phi)

    # TMP: Use nearest to check things
    nearest_jet,dr = had_tops.nearest(jets,return_metric=True)
    nearest_jet_sum = nearest_jet.sum()
    print("\nnearest:")
    for i,x in enumerate(nearest_jet):
        if i > 100: break
        if len(x.pt) == 0: continue
        print(i,dr[i],x.pt,x.eta,x.phi)
    print("\nnearest_jet_sum:",nearest_jet_sum)
    for i,x in enumerate(nearest_jet_sum):
        if i > 100: break
        if x.mass == 0: continue
        print(i,x.mass,x.pt,x.eta,x.phi)


    # Find jets that match to the genparticles from the hadronic top
    jets_matched_bqq, jets_matched_qq, bqq_drmin, qq_drmin = get_bqq_jets(had_tops,jets)
    '''

    from coffea.nanoevents import NanoEventsFactory
    events = NanoEventsFactory.from_root("/hadoop/store/user/kmohrman/FullProduction/FullR2/UL17/Round1/Batch1/naodOnly_step/v4/nAOD_step_ttlnuJet_all22WCsStartPtCheckdim6TopMay20GST_run0/NAOD-00000_10508.root",entry_stop=1000).events()
    #events = NanoEventsFactory.from_root("central_ttH.root",entry_stop=1000).events()
    #events = NanoEventsFactory.from_root("central_ttH.root").events()
    genpart = events.GenPart

    jets = events.Jet
    jets = jets[jets.pt>30.0]


    # Gen matching
    t_children = get_t_children(genpart)
    t_decay_products = get_t_decay_products(t_children)
    is_had_top_mask = is_had_top(t_decay_products)
    has_matchable_decay_products = get_matchable_mask(t_decay_products)
    had_tops = ak.flatten(t_decay_products[is_had_top_mask&has_matchable_decay_products][:,:1],axis=2) # Just grab the first hadronic top for now
    had_reco_mask = ak.firsts((is_had_top_mask&has_matchable_decay_products)[:,:1])
    print("had_reco_mask",had_reco_mask)
    print("Number of had tops:",ak.count_nonzero(had_reco_mask,axis=-1),"/",ak.num(had_reco_mask,axis=-1))

    jets_matched_bqq, jets_matched_qq, bqq_drmin, qq_drmin = get_bqq_jets(had_tops,jets,0.0532)
    print("jets_matched_bqq",jets_matched_bqq)
    print("jets_matched_qq",jets_matched_qq)
    print("bqq_drmin",bqq_drmin)
    print("qq_drmin",qq_drmin)

    print("\nhad_tops",had_tops)
    for i,x in enumerate(had_tops[had_reco_mask]):
        print("\t",i,x.pt)

    # Note that we need to apply this mask (it can be with an arbitrarily high threshold) to avoid crashes
    # There are cases where no jets pass the btag wp, so the mass and dr for the event is None
    # Also, note we want this mask to have false values, not None
    ok_dr_mask = ak.fill_none((ak.max(bqq_drmin,axis=-1))<10,False)

    jets_matched_bqq_mass = (jets_matched_bqq.sum()).mass
    jets_matched_qq_mass = (jets_matched_qq.sum()).mass

    bqq_maxdr = ak.max(bqq_drmin,axis=-1)
    qq_maxdr = ak.max(qq_drmin,axis=-1)

    print("\njets_matched_qq_mass",jets_matched_qq_mass)
    for i,x in enumerate(jets_matched_qq_mass[had_reco_mask&ok_dr_mask]):
        print("\t",i,x)
    exit()

    print("\nbqq_maxdr",bqq_maxdr)
    for i,x in enumerate(bqq_maxdr[had_reco_mask]):
        print("\t",i,x)

    print("\nqq_maxdr",qq_maxdr)
    for i,x in enumerate(qq_maxdr[had_reco_mask]):
        print("\t",i,x)


###############################################

if __name__ == "__main__":
    main()

