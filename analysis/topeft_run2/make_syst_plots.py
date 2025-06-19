'''
This script makes ratio plots of all the available systematics.
It runs over the datacards and produces PNG and PDF files.
The `group` option allows you to specify how many terms (x2 for up/down)
are included in a single plot. The default is 5.
If `group>1` the file names will contain the channel and a unique number.
If `group==1` the file names will contain the channel and systematic.
Example run:
    `python make_syst_plots.py /scratch365/byates2/3l_fwd/ptz-lj0pt_withSys/ -o ~/www/EFT/systs/group5/ --group 5 -C`
'''
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import argparse
import uproot
import glob
import subprocess
import os
import shutil
import copy

from topcoffea.scripts.make_html import make_html

CMS_COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
var_lookup = {'lj0pt': r'$\it{p}_{\rm{T}} (\ell j)_{\rm{max}}$', 'ptz': r'$\it{p}_{\rm{T}}(\rm{Z})$', 'lt': r'$\it{L}_{\rm{T}}$', 'photon_pt': r'$\gamma \it{p}_{\rm{T}}$'}

parser = argparse.ArgumentParser()
parser.add_argument("input_path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
parser.add_argument("-n", "--name_str", default="ttx_multileptons-", help = "The path to the pkl file")
parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
parser.add_argument("--group",             default="5", help = "The path the output files should be saved to")
parser.add_argument("--condor","-C",action="store_true",help="Split up the channels into multiple condor jobs")
parser.add_argument("--ch-lst","-c",default=[],action="extend",nargs="+",help="Specify a list of channels to process.")

args = parser.parse_args()

hep.style.use("CMS")
hep.cms.label('Preliminary Simulation', lumi='138', data=True)

def condor_submit(files):
    conda_path = subprocess.check_output(['which', 'python']).strip().decode('utf-8')
    paths = conda_path.split('envs')
    home_dir_os = os.path.expanduser("~")
    conda_type = [p for p in paths[0].split('/') if p != ''][-1]
    conda_path = 'miniconda3' if 'conda' in conda_type else 'micromamba'
    env_path = [p for p in paths[1].split('/') if p != ''][0]
    condor_dir = os.path.join(os.getcwd(),"job_logs")
    if not os.path.exists(condor_dir):
        print(f"Making condor output directory {condor_dir}")
        os.mkdir(condor_dir)
    if args.condor and not os.path.samefile(os.getcwd(),condor_dir):
        shutil.copy("make_syst_plots.py",condor_dir)

    sub_fragment = """\
    universe   = vanilla
    executable = condor.sh
    arguments  = "{in_path} {usr_dir} {out_dir} {condor_dir} '{conda_path}' '{conda_type}' '{env_path}' {group} '{channel}'"
    output = {condor_dir}/{channel}_$(Process).out
    error  = {condor_dir}/{channel}_$(Process).err
    log    = {condor_dir}/{channel}_$(Process).log

    request_cpus = 1
    request_memory = 8192
    request_disk = 1024

    transfer_input_files = make_syst_plots.py
    should_transfer_files = yes
    transfer_executable = true

    getenv = true
    queue 1
    """
    #print(sub_fragment.format(
    #            in_path=os.path.realpath(args.input_path),
    #            usr_dir=os.path.expanduser("~"),
    #            out_dir=os.path.realpath(args.output_path),
    #            condor_dir=condor_dir,
    #            conda_path=conda_path,
    #            conda_type=conda_type,
    #            env_path=env_path,
    #            ))
    #
    sh_fragment = r"""#!/bin/sh
    IN_PATH=${1}
    USR_DIR=${2}
    OUT_DIR=${3}
    CONDOR_DIR=${4}
    CONDA_PATH=${5}
    CONDA_TYPE=${6}
    ENV_PATH=${7}
    GROUP=${8}
    CHANNEL=${9}

    echo "IN_PATH: ${IN_PATH}"
    echo "USR_DIR: ${USR_DIR}"
    echo "OUT_DIR: ${OUT_DIR}"
    echo "CONDOR_DIR: ${CONDOR_DIR}"
    echo "CONDA_PATH: ${CONDA_PATH}"
    echo "CONDA_TYPE: ${CONDA_TYPE}"
    echo "ENV_PATH: ${ENV_PATH}"
    echo "GROUP: ${GROUP}"
    echo "CHANNEL: ${CHANNEL}"

    source ${USR_DIR}/${CONDA_PATH}/etc/profile.d/${CONDA_TYPE}.sh
    unset PYTHONPATH
    ${CONDA_TYPE} activate ${CONDA_DEFAULT_ENV}

    echo python make_syst_plots.py ${IN_PATH} -o ${OUT_DIR} --group ${GROUP} --ch-lst ${CHANNEL}
    python make_syst_plots.py ${IN_PATH} -o ${OUT_DIR} --group ${GROUP} --ch-lst ${CHANNEL}
    """
    #print(sh_fragment)
    #print(f'{os.path.realpath(args.input_path)=}')
    #print(f'{os.path.expanduser("~")=}')
    #print(f'{os.path.realpath(args.output_path)=}')
    #print(f'{condor_dir=}')
    #print(f'{conda_path=}')
    #print(f'{conda_type=}')

    condor_exe_fname = os.path.join(condor_dir,"condor.sh")
    with open(condor_exe_fname,"w") as f:
        f.write(sh_fragment)
    home = os.getcwd()
    for file_name in files:
        if not os.path.exists(condor_dir+f'/{file_name}'):
            print(f"Making condor output directory {condor_dir}/{file_name}")
            os.mkdir(condor_dir+f'/{file_name}')
        condor_submit_fname = os.path.join(condor_dir,f"condor.{file_name}.sub")
        with open(condor_submit_fname,"w") as f:
            f.write(sub_fragment.format(
                    in_path=os.path.realpath(args.input_path),
                    usr_dir=os.path.expanduser("~"),
                    out_dir=condor_dir,
                    #out_dir=condor_dir+f'/{file_name}',
                    #out_dir=os.path.realpath(args.output_path),
                    condor_dir=condor_dir,
                    conda_path=conda_path,
                    conda_type=conda_type,
                    env_path=env_path,
                    group=int(args.group),
                    channel=file_name,
                    ))
        cmd = ["condor_submit",condor_submit_fname]
        print(f"{'':>5} Condor command: {' '.join(cmd)}")
        os.chdir(condor_dir)
        p = subprocess.run(cmd)
        os.chdir(home)

def load_hists_from_root_files(path, name_str=''):
    files = glob.glob(f'{path}/{name_str}*.root')
    terms = {}
    if args.ch_lst != []:
        files = [file_name for file_name in files if any(ch in file_name for ch in args.ch_lst)]
        print(f'Running over {", ".join(files)}')
    #else:
    #    print(f'Submitting over {", ".join(files)}')
    #    files = [files[0]] #FIXME
    for fname in files:
        chan = fname.split('-')[-1].replace('.root', '') # Strip of `ttxmultilepton-` and `.root`
        terms[chan] = {}
        with uproot.open(fname) as fin:
            for key in fin.keys():
                terms[chan][key.split(';1')[0]] = fin[key].to_hist()
    return terms


def plot_hists(hists, path, group=1):
    def new_fig():
        return plt.subplots(figsize=figsize)

    for chan in hists:
        if not os.path.exists(f'{path}'):
            os.mkdir(f'{path}')
        if not os.path.exists(f'{path}/{chan}'):
            os.mkdir(f'{path}/{chan}')

        sm_procs = [p for p in hists[chan] if p.endswith('sm')]
        syst_procs = [p for p in hists[chan] if p.endswith('Up')]
        figsize = (12, 10)
        #fig,ax = plt.subplots(figsize=figsize)
        fig,ax = new_fig()
        fig2,ax2 = new_fig()
        label='Preliminary'
        hep.cms.label(label, lumi='138', ax=ax) 
        hep.cms.label(label, lumi='138', ax=ax2) 
        ax.axhline(y=1, color='gray')
        iplot = 0
        ymin = 1
        ymax = 0
        ymin2 = 1
        ymax2 = 0
        #nom_hists = [v for k,v in hists[chan].items() if 'Up' not in k and 'Down' not in k and 'data_obs' not in k])
        nom_hist = None
        for key,histo in hists[chan].items():
            if 'Up'  in key or 'Down'  in key or 'data_obs'  in key: continue
            if nom_hist is None:
                nom_hist = copy.copy(histo.values())
            else:
                nom_hist += histo.values()
        #syst_procs = [s for s in syst_procs if 'sm_FSR' in s]
        syst_procs = [s for s in syst_procs if 'FSR' in s]
        for isyst,syst in enumerate(syst_procs):
            nom  = hists[chan][syst.split('sm')[0]+'sm'].values()
            up   = hists[chan][syst].values()
            down = hists[chan][syst.replace('Up', 'Down')].values()
            hep.histplot(up/nom,    ax=ax, color=CMS_COLORS[isyst%group%5], label=f'{syst.replace("Up", "")}')
            hep.histplot(down/nom,  ax=ax, color=CMS_COLORS[isyst%group%5], ls='--')
            hep.histplot(up/nom_hist,    ax=ax2, color=CMS_COLORS[isyst%group%5], label=f'{syst.replace("Up", "")}')
            hep.histplot(down/nom_hist,  ax=ax2, color=CMS_COLORS[isyst%group%5], ls='--')
            #ymin = max(0, np.nan_to_num(min(np.min(up/nom), np.min(down/nom))))
            #ymax = min(2, np.nan_to
            #if ymax == 0.0: ymax = 1.
            ymin = min(ymin, np.min(np.nan_to_num(np.stack((up/nom, down/nom)))))
            ymax = max(ymax, np.max(np.nan_to_num(np.stack((up/nom, down/nom)))))
            ax.set_ylim(ymin*0.97, ymax*1.05)
            ch_lookup = chan.split('_')[-1]
            if 'photon_pt' in chan:
                ch_lookup = 'photon_pt'
            ax.set_xlabel(chan.replace('_'+ch_lookup, '') + '   ' + var_lookup[ch_lookup])
            ax.set_ylabel('Systematic / Nominal')
            ymin2 = min(ymin2, np.min(np.nan_to_num(np.stack((up/nom_hist, down/nom_hist)))))
            ymax2 = max(ymax2, np.max(np.nan_to_num(np.stack((up/nom_hist, down/nom_hist)))))
            ax2.set_ylim(ymin2*0.97, ymax2*1.1)
            ax2.set_xlabel(('_'.join(ch_lookup) if isinstance(ch_lookup, list) else ch_lookup) + '   ' + var_lookup[ch_lookup])
            #ax2.set_xlabel('_'.join(chan.split('_')[:-1]) + '   ' + var_lookup[chan.split('_')[-1]])
            ax2.set_ylabel('Systematic / Total')
            if 'fake' in syst:
                plt.ylim(0.8, 1.2)
            if group == 1:
                ax.legend()
                ax2.legend()
                fig.savefig(f'{path}/{chan}/{chan}_{iplot}.png')
                fig.savefig(f'{path}/{chan}/{chan}_{iplot}.pdf')
                fig2.savefig(f'{path}/{chan}/{chan}_{iplot}_rel.png')
                fig2.savefig(f'{path}/{chan}/{chan}_{iplot}_rel.pdf')
                plt.close()
                fig,ax = plt.subplots(figsize=figsize)
                hep.cms.label(label, lumi='138',ax=ax)
                fig2,ax2 = plt.subplots(figsize=figsize)
                hep.cms.label(label, lumi='138',ax=ax2)
                ax.axhline(y=1, color='gray')
                ymin = 1
                ymax = 0
                ymin2 = 1
                ymax2 = 0
            elif group <= 0:
                raise Exception(f'Invalid {group=}! Please use 1 or greater')
            elif isyst > 0 and isyst % group == 0:
                ax.legend(ncols=2, loc='upper left', fontsize='xx-small', )
                ax2.legend(ncols=2, loc='upper left', fontsize='xx-small', )
                fig.tight_layout()
                fig2.tight_layout()
                fig.savefig(f'{path}/{chan}/{chan}_{iplot}.png')
                fig.savefig(f'{path}/{chan}/{chan}_{iplot}.pdf')
                fig2.savefig(f'{path}/{chan}/{chan}_{iplot}_rel.png')
                fig2.savefig(f'{path}/{chan}/{chan}_{iplot}_rel.pdf')
                iplot += 1
                plt.close()
                fig,ax = plt.subplots(figsize=figsize)
                hep.cms.label(label, lumi='138', ax=ax)
                fig2,ax2 = plt.subplots(figsize=figsize)
                hep.cms.label(label, lumi='138',ax=ax2)
                ax.axhline(y=1, color='gray')
                ymin = 1
                ymax = 0
                ymin2 = 1
                ymax2 = 0
        make_html(f'{path}/{chan}')

hists = load_hists_from_root_files(args.input_path, args.name_str)
if args.condor:
    condor_submit(hists)
else:
    plot_hists(hists, args.output_path, group=int(args.group))
