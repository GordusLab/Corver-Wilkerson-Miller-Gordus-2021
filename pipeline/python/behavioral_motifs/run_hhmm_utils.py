#
# Utility script to import HHMM model and export likelihoods
#

import os, sys, time, subprocess, glob, joblib as jl, regex as re, pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    overwrite = False

    if len(sys.argv) <= 1:
        # List files to process
        fnames = glob.glob('Y:/wavelet/hhmm-results/*regimes_12minrun_manuallabels_5fold.pickle')
        fnames = [x for x in fnames if not 'idxmapping' in x]
        # Start jobs
        NEXT_NUMA_NODE = 0
        for fn in fnames:
            cmd = '{} {} {}'.format(
                os.path.abspath(os.path.join(os.__file__, os.pardir, os.pardir, 'python.exe')),
                __file__, '"{}"'.format(fn.replace('/', '\\')))
            print("Starting shell....")
            print(cmd)

            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            _p = subprocess.Popen('start /node {} '.format(NEXT_NUMA_NODE) + cmd, shell=True, close_fds=True,
                                  creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
            # For the next process, use the opposite NUMA core. This equally divides processes across NUMA cores.
            NEXT_NUMA_NODE = 1 - NEXT_NUMA_NODE
    else:
        try:
            # Get input string
            fname = sys.argv[1]
            time.sleep(3)

            # Does the output already exist?
            fnameOut = fname.replace('.pickle', '') + '.likelihoods.pickle'
            if os.path.exists(fnameOut) or overwrite:
                exit(0)
            else:
                # Load HHMM model
                print('Loading model data for {}'.format(fname))
                data = jl.load(fname)
                print('Finished loading model data.')

                # Get likelihoods
                logLs = []
                for modelRepIdx in tqdm(range(len(data['models']))):
                    for modelRecIdx in tqdm(range(len(data['models'][modelRepIdx]['model-fit-states'])), leave=False):
                        nameShort = re.search('^Z:/.*/(.*)/.*/.*$',
                            data['fnames'][modelRecIdx][0].replace('\\', '/')).group(1)
                        logL = data['models'][modelRepIdx]['model'].log_probability(
                            data['models'][modelRepIdx]['model-fit-states'][modelRecIdx])
                        logLs.append((modelRepIdx, modelRecIdx,
                                      data['numRegimesHHMM'],
                                      modelRecIdx in data['models'][modelRepIdx]['fold'][0],
                                      data['models'][modelRepIdx]['model-fit-states'][modelRecIdx].size,
                                      nameShort,
                                      logL))

                # Export
                logLsDf = pd.DataFrame(logLs, columns=['rep', 'rec',
                    'numRegimes', 'inTrainingSample', 'N', 'fname', 'logL'])
                logLsDf.to_pickle(fnameOut)
        except Exception as e:
            print(e)
            time.sleep(36000)