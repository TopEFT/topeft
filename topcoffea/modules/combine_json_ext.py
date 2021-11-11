import json

def combine_json_ext(ext_name):
    if '_ext' not in ext_name:
        raise Exception(f"The file {ext_name} does not contain '_ext' in the filename!")

    jsonFile = open(ext_name,'r')
    ext = json.load(jsonFile) # Load the JSON information for the ext file

    nom_name = ext_name.replace('_ext','') # Remove '_ext' from the file name
    try:
        jsonFile = open(nom_name,'r')
    except FileNotFoundError:
        return
    nom = json.load(jsonFile) # Load the JSON information for the nominal file

    if any(['_ext' in sample for sample in nom['files']]):
        raise Exception(f"The file {nom_name} already contains '_ext' files!")

    for l in ext['files']:
        nom['files'].append(l)

    ''' Load the event counts, sum of weights, etc. '''
    nom['nEvents'] = nom['nEvents'] + ext['nEvents']
    nom['nGenEvents'] = nom['nGenEvents'] + ext['nGenEvents']
    nom['nSumOfWeights'] = nom['nSumOfWeights'] + ext['nSumOfWeights']

    ''' This section is borrowed from topcoffea/modules/createJSON.py '''
    with open(nom_name, 'w') as outfile:
        json.dump(nom, outfile, indent=2)
        print('>> Replacing json file: %s'%nom_name)
