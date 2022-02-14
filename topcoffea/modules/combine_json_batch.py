import json
import re

'''
Combine event information from multiple batches
Sums `nEvents`, `nGenEvents`, and `nSumOfWeights`
Does not update the list of files
'''
def combine_json_batch(batch_name):
    if '_b' not in batch_name:
        raise Exception(f"The file {batch_name} does not contain '_b' in the filename!")

    jsonFile = open(batch_name,'r')
    batch = json.load(jsonFile) # Load the JSON information for the batch file

    nom_name = batch_name.replace('_b1','')
    nom_name = re.sub(r'_b\d*', '_b1', batch_name) # '_b[0-9]' to '_b1' in the file name
    try:
        jsonFile = open(nom_name,'r')
    except FileNotFoundError:
        print(f'{nom_name} not found!')
        return
    nom = json.load(jsonFile) # Load the JSON information for the nominal file

    for l in batch['files']:
        nom['files'].append(l)

    ''' Load the event counts, sum of weights, etc. '''
    nom['nEvents']         = nom['nEvents']       + batch['nEvents']
    nom['nGenEvents']      = nom['nGenEvents']    + batch['nGenEvents']
    nom['nSumOfWeights']   = nom['nSumOfWeights'] + batch['nSumOfWeights']

    ''' This section is borrowed from topcoffea/modules/createJSON.py '''
    with open(nom_name, 'w') as outfile:
        json.dump(nom, outfile, indent=2)
        print('>> Updating json file: %s'%nom_name)
