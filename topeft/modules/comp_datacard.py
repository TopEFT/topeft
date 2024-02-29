import argparse
import re
import numpy as np

tolerance = 0.5e-5

'''
Open a datacard txt file and load the contents into a dictionary
'''

def strip(fname='ttx_multileptons-2lss_p_2b.txt'):
    f = open(fname, 'r')
    #f = open('sergio/ttx_multileptons-4l_2b.txt', 'r')
    fin = f.readlines()
    process = []
    rate = []
    systs = []
    systs_name = []
    for line in fin:
        if 'process' in line:
            # Skip process number lines
            if not any([p in line for p in ['sm','lin','quad','quad_mixed']]): continue
            line = line.split()[1:]
            #if line[0] == '0': continue
            process = line
        elif 'rate' in line:
            line = line.split()[1:]
            line = [float(l) for l in line]
            rate = line
        elif ('shape' in line or 'lnN' in line or any([re.findall(r'\.\d*', str(l)) for l in line])) and '*' not in line and '#' not in line:
            name = line.split()[0]
            line = line.split()[2:]
            if len(line) == 0: continue
            line = [str(l) for l in line]
            systs.append(line)
            systs_name.append(name)
    return [dict(zip(process, rate)), systs, systs_name]

'''
Look at a pair of datacard contents from strip
'''

def comp_datacard_dict(wc1, wc2, s1, s2, s1_n, s2_n):
    names = list(set([str(w) for w in wc1] + [str(w) for w in wc2]))

    def collect(s_n, wcs, sys):
        systs = {}
        for iline,line in enumerate(list(zip(wcs.keys(), s)) for s in sys):
            sname = s_n[iline]
            for term,syst in line:
                if '/' in syst:
                    syst = [float(x) for x in syst.split('/')]
                elif '-' not in syst:
                    syst = float(syst)
                #print(name, term, syst)
                if sname in systs:
                    systs[sname][term] = syst
                else:
                    systs[sname] = {term: syst}
        return systs

    systs1 = collect(s1_n, wc1, s1)
    systs2 = collect(s2_n, wc2, s2)
    bad = 0
    missing = 0
    total = np.sum([len(x) for x in s1])
    for syst, terms in systs1.items():
        if syst not in systs2:
            print(f'{syst} not found in file2!')
            continue
        for term,s1_val in terms.items():
            if 'FF' in syst and 'fake' not in term:
                continue
            tmp = term.split('_')
            tmp1 = tmp[-1]
            tmp[-1] = tmp[-2]
            tmp[-2] = tmp1
            tmp = '_'.join(tmp)
            if term not in systs2[syst] and tmp not in systs2[syst]:
                print(f'{syst} {term} not found in file2!')
                missing += 1
                continue
            if term not in systs2[syst]:
                if s1_val != systs2[syst][tmp]:
                    print(f'{syst} {term} {s1_val} does not match {systs2[syst][term]}!')
                    bad += 1
                    continue
            else:
                if s1_val != systs2[syst][term]:
                    print(f'{syst} {term} {s1_val} does not match {systs2[syst][term]}!')
                    bad += 1
                    continue
    print(f'Missing systematics:\t{missing} / {total}')
    print(f'Incorrect systematics:\t{bad} / {total}')
    for name in names:
        if name in wc1 and name in wc2:
            if wc1[name] == 0:
                print(f'{name} is empty, skipping!')
                continue
            diff = abs(wc1[name] - wc2[name]) / wc1[name]
            if diff > tolerance: print(f'{name} : [{round(wc1[name],2)}, {round(wc2[name],2)} {round(diff*100,2)}% difference!]')
            if diff > tolerance: return False
        elif name in wc1 and name not in wc2:
            pass
        elif name in wc2 and name not in wc1:
            if 'mixed' in name:
                tmp = name.split('_')
                tmp1 = tmp[-1]
                tmp[-1] = tmp[-2]
                tmp[-2] = tmp1
                tmp = '_'.join(tmp)
                if tmp in wc1:
                    if wc1[tmp] == 0:
                        print(f'{tmp} is empty, skipping!')
                        continue
                    if tmp not in wc1:
                        print(f'{tmp} missing from file1!')
                        continue
                    diff = abs(wc1[tmp] - wc2[name]) / wc1[tmp]
                    if diff > tolerance: print(f'{tmp} : ')
                    if diff > tolerance: print(f'{tmp} : [{round(wc1[tmp],2)}, {round(wc2[name],2)} {round(diff*100,2)}% difference!]')
                    if diff > tolerance: return False
                    continue
            print('{} is missing from the new list!'.format(name))
        else: pass
    return True

def comp_datacard(fin1, fin2):
    wc1,s1,s1_n = strip(fin1)
    wc2,s2,s2_n = strip(fin2)

    return comp_datacard_dict(wc1,wc2,s1,s2,s1_n,s2_n)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='You can select which file to run over')
    parser.add_argument('fin1'           , nargs='?', default=''           , help = 'First input file')
    parser.add_argument('fin2'           , nargs='?', default=''           , help = 'Second input file')
    args = parser.parse_args()

    comp_datacard(args.fin1, args.fin2)
