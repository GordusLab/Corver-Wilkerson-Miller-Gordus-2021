

# ======================================================================================================================
# ....
# ======================================================================================================================

import os, glob, gc, regex as re, numpy as np, pandas as pd, joblib as jl, json, time, itertools, rolling, miniball, scipy.ndimage
from scipy.stats import mode
from tqdm import tqdm

# Source: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
flatten = lambda *n: (e for a in n
    for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

os.getpid()

# ======================================================================================================================
# ....
# ======================================================================================================================

def loadPickle(fn):
    try:
        gc.disable()
        return jl.load(fn)
        gc.enable()
    except Exception as e:
        print(fn, str(e))
        return None


def loadPositionData(data, modelRecIdx):
    def loadJSON(x):
        if os.path.exists(x):
            with open(x, 'r') as f:
                return json.load(f)
        else:
            return None

    # Load position/orientation data
    fnamePos = glob.glob(os.path.abspath(os.path.join(os.path.dirname(
        data['fnames'][modelRecIdx][0]), '../croprot/*dlc_position_orientation.npy')))[0]

    # Load recording info
    fnameRecInfo = os.path.join(os.path.dirname(os.path.dirname(fnamePos)), 'recording.json')
    recordingInfo = loadJSON(fnameRecInfo)

    # Create short name
    fnamesShort = re.search('^Z:/.*/(.*)/.*/.*$', fnamePos.replace('\\', '/')).group(1)

    # Fill in missing stage information, if necessary
    s = recordingInfo
    s['fname'] = fnamePos

    # Does this recording.json file specify stage ranges, or starting points?
    for st in s['stages']:
        if s['stages'][st] == []:
            s['stages'][st] = []
        elif not isinstance(s['stages'][st][0], list):
            s['stages'][st] = [s['stages'][st], ]

    # Convert to indices used in analysis
    arrIdx = np.load(fnamePos.replace('_position_orientation.npy', '_abs_filt_interp_mvmt_noborder.idx.npy'))
    for st in s['stages']:
        for k in range(len(s['stages'][st])):
            for m in range(2):
                s['stages'][st][k][m] = np.argmin(np.abs(np.argwhere(arrIdx).T[0] - s['stages'][st][k][m]))

    # Load original data
    arr = np.load(fnamePos)

    # Subset by index
    arrIdx = np.load(fnamePos.replace('_position_orientation.npy',
                                      '_abs_filt_interp_mvmt_noborder.idx.npy'))
    arr = arr[arrIdx, :]

    # Done
    return arr, recordingInfo


def computeData(fname, data, modelRepIdx, modelRecIdx):
    # Load position data
    arr, recordingInfo = loadPositionData(data, modelRecIdx)

    # Load index mapping
    arrIdxMapping = jl.load(fname.replace(re.search('([.0-9]*\\.pickle)$',
        fname).group(0), '.idxmapping.pickle').replace('.resave', '').replace('.1', ''))

    # Load raw regime probabilities
    try:
        d = data['models'][modelRepIdx]['statesPredProb'][modelRecIdx].copy()
    except:
        d = data['models'][modelRepIdx]['model'].predict_log_proba(
            data['models'][modelRepIdx]['model-fit-states'][modelRecIdx]).copy()
    regimeIDs = np.array([(int(x.group(1)) if x is not None else -1) for x in [
        re.search('^r([0-9]*)_', x.name) for x in data['models'][modelRepIdx]['model'].states]])

    probRegimes = np.zeros((d.shape[0], data['numRegimesHHMM']))
    for regimeID in range(probRegimes.shape[1]):
        probRegimes[:, regimeID] = np.nanmax(d[:, np.argwhere(regimeIDs == regimeID)[:, 0]], axis=1)
    probRegimes = np.exp(probRegimes)

    # Reshape array to re-introduce duplicate states using the index loaded above
    probRegimes = np.array([(probRegimes[i, :] if i >= 0 else np.full(
        probRegimes.shape[1], np.nan)) for i in arrIdxMapping[modelRecIdx]])

    for regimeID in range(probRegimes.shape[1]):
        probRegimes[:, regimeID] = pd.DataFrame(probRegimes[:, regimeID]).fillna(
            method='ffill').fillna(method='bfill').values[:, 0]

    # Smooth and stack regime probabilities
    probRegimesSmoothed = probRegimes.copy()
    for j in range(probRegimes.shape[1]):
        probRegimesSmoothed[:, j] = pd.DataFrame(probRegimes[:, j]).rolling(window=3000).max().values[:, 0]
    probRegimesSmoothed /= np.sum(probRegimesSmoothed, axis=1)[:, np.newaxis]

    probRegimesSmoothedStacked = np.zeros((probRegimesSmoothed.shape[0], probRegimesSmoothed.shape[1] + 1))
    for j in range(probRegimesSmoothed.shape[1]):
        probRegimesSmoothedStacked[:, j + 1] = \
            probRegimesSmoothedStacked[:, j] + probRegimesSmoothed[:, j]

    # Determine center
    if 'center' in recordingInfo:
        center = np.array(recordingInfo['center'])
    else:
        print(recordingInfo)
        raise Exception('Center of web should be manually specified')
    # Convert position to polar coordinates relative to approximate center
    r = np.linalg.norm(arr[:, 0:2] - center[np.newaxis, :], axis=1)
    a = np.arctan2(arr[:, 0] - center[np.newaxis, 0], arr[:, 1] - center[np.newaxis, 1])
    arrPolar = np.hstack((r[:, np.newaxis], a[:, np.newaxis], arr[:, 2, np.newaxis]))
    # Remove noise
    isNoise = np.linalg.norm(arr[:, 0:2] - np.roll(arr, -1, axis=0)[:, 0:2], axis=1) > 50
    arr[isNoise, :] = np.nan
    arrPolar[isNoise, :] = np.nan
    # Compute velocities
    arrPolarVel = np.roll(arrPolar, -25, axis=0) - arrPolar
    # Wrap rotations into -pi / +pi
    for k in [1, 2]:
        for i in range(3):
            arrPolarVel[arrPolarVel[:, k] < -np.pi, k] += 2 * np.pi
        for i in range(3):
            arrPolarVel[arrPolarVel[:, k] > np.pi, k] -= 2 * np.pi
    # Filter out
    isNoise |= (np.abs(arrPolarVel[:, 0]) > 300) & (np.abs(arrPolarVel[:, 1]) > np.pi)
    #
    arr[isNoise, :] = np.nan
    arrPolar[isNoise, :] = np.nan
    arrPolarVel = np.roll(arrPolar, -25, axis=0) - arrPolar
    # Wrap rotations into -pi / +pi
    for k in [1, 2]:
        for i in range(3):
            arrPolarVel[arrPolarVel[:, k] < -np.pi, k] += 2 * np.pi
        for i in range(3):
            arrPolarVel[arrPolarVel[:, k] > np.pi, k] -= 2 * np.pi

    # Done
    return arr, arrPolar, probRegimes, probRegimesSmoothedStacked, recordingInfo

def processJob(fname, jobid):
    # -- Each predicted group can be assigned a given manual group
    #    So 5 options for each predicted group, so 5^Npred total options, filtered by constraints
    numRegimes = int(re.search('([0-9]{1,2})regimes', fname).group(1))

    # Load data
    print('Loading: {}'.format(fname))
    data = loadPickle(fname)
    Nrecs = len(data['fnames'])
    numRegimesHHMM = numRegimes #data['numRegimesHHMM']

    dataIntermediate = []
    # -- How does each assignment perform?
    #    Measure recall/precision for each "model x (manual) stage" combinations
    for modelRepIdx in tqdm(range(jobid * 5, jobid * 5 + 5), leave=False):
        for modelRecIdx in tqdm(range(Nrecs), leave=False):
            # Get data
            arr, arrPolar, probRegimes, probRegimesSmoothedStacked, recordingInfo = \
                computeData(fname, data, modelRepIdx, modelRecIdx)
            # Out-of-sample?
            outofsample = modelRecIdx in data['models'][modelRepIdx]['fold'][1]
            # Misc.
            dirName = os.path.dirname(os.path.dirname(data['fnames'][modelRecIdx][0]))
            dataIntermediate.append((numRegimesHHMM, modelRepIdx, modelRecIdx, arr, arrPolar, probRegimes,
                probRegimesSmoothedStacked, recordingInfo, outofsample, dirName, Nrecs))

    jl.dump(dataIntermediate, 'Z:/Abel/data-tmp/F1_intermediate_{}_{}.pickle'.format(
        numRegimesHHMM, jobid))

def processJobPt2(fnameIntermediate, modelRepIdxTarget, mergeProtoradii = False):
    _data = jl.load(fnameIntermediate)

    table = []

    numRegimes = _data[0][0]
    regPredicted = np.arange(numRegimes)
    predToManualMappings = []
    if len(regPredicted) < 5:
        manualToPredMappings = list(itertools.product(np.arange(len(regPredicted)), repeat=5))
        predToManualMappings = [[np.argwhere(np.array(m)==i)[:,0].tolist() for i in range(len(regPredicted))] for m in manualToPredMappings]
        # Every predicted-to-manual should be used
        predToManualMappings = [x for x in predToManualMappings if np.min([len(y) for y in x]) > 0]

    if len(regPredicted) >= 4:
        _predToManualMappings = list(itertools.product(np.arange(5), repeat=len(regPredicted)))
        _predToManualMappings = [x for x in tqdm(_predToManualMappings, leave=False) if np.max(
            np.unique(x, return_counts=True)[1]) < min(len(regPredicted) - 2, 6) and np.unique(x).size >= 4]
        predToManualMappings += _predToManualMappings

    STAGES = ['protoweb', 'radii', 'spiral_aux', 'spiral_cap', 'stabilimentum']

    for numRegimesHHMM, modelRepIdx, modelRecIdx, arr, arrPolar, probRegimes, \
            probRegimesSmoothedStacked, recordingInfo, outofsample, dirName, Nrecs in _data:
        if modelRepIdx != modelRepIdxTarget:
            continue

        # Compute manual states
        statesManual = np.full((probRegimes.shape[0], 5), False, dtype=np.bool)
        for st in recordingInfo['stages']:
            for m in recordingInfo['stages'][st]:
                statesManual[m[0]:m[1], STAGES.index(st)] = True
        # Crop recording
        s1 = 0
        for i in range(statesManual.shape[1]):
            a = np.argwhere(statesManual[:, i])[:, 0]
            if a.size > 0:
                s1 = max(s1, np.max(a))
        statesManual = statesManual[:s1, :]
        probRegimes = probRegimes[:s1, :]
        arr = arr[:s1, :]
        # Compute long pauses
        arr2 = arr[:,0:2].copy()
        arr2[scipy.ndimage.binary_dilation(np.linalg.norm(pd.DataFrame(
            arr[:, 0:2]).diff(1).values, axis=1) > 20, iterations=50), :] = np.nan
        arr2 = pd.DataFrame(arr2).fillna(method='ffill').fillna(method='bfill').values
        isNotLongPause = np.array([np.sqrt(miniball.Miniball(arr2[max(0, i - 750):(
                i + 750), 0:2]).squared_radius()) > 10 for i in tqdm(range(0, arr.shape[0], 50))])
        isNotLongPause = np.repeat(isNotLongPause, 50)[:arr.shape[0]]
        # Compute metrics
        probRegimesArgmax = np.argmax(probRegimes, axis=1)
        probRegimesArgmaxSmooth = pd.DataFrame(np.argmax(probRegimes, axis=1)).rolling(window=1500).apply(lambda x: mode(x).mode).values[:,0]
        for istage in (range(len(STAGES)) if not mergeProtoradii else [STAGES.index('protoweb'),]):
            for imapping, mapping in tqdm(list(enumerate(predToManualMappings)), leave=False):
                # Get true (manual) indices
                manualIdx = statesManual[:, istage]
                if mergeProtoradii:
                    manualIdx = statesManual[:, STAGES.index('protoweb')] | statesManual[:, STAGES.index('radii')]
                # Get predicted indices
                predIdx = np.full(manualIdx.size, False, dtype=np.bool)
                predIdxSmooth = np.full(manualIdx.size, False, dtype=np.bool)
                if not mergeProtoradii:
                    if isinstance(mapping[0], list):
                        predIdx = np.isin(probRegimesArgmax,
                                          [i for i, m in enumerate(mapping) if istage in m])
                        predIdxSmooth = np.isin(probRegimesArgmaxSmooth,
                                          [i for i, m in enumerate(mapping) if istage in m])
                    else:
                        if np.argwhere(np.array(mapping) == istage).size > 0:
                            predIdx = np.isin(probRegimesArgmax,
                                              np.argwhere(np.array(mapping) == istage)[:, 0])
                            predIdxSmooth = np.isin(probRegimesArgmaxSmooth,
                                              np.argwhere(np.array(mapping) == istage)[:, 0])
                else:
                    if isinstance(mapping[0], list):
                        predIdx = np.isin(probRegimesArgmax,
                                          [i for i, m in enumerate(mapping) if STAGES.index('protoweb') in m or STAGES.index('radii') in m])
                        predIdxSmooth = np.isin(probRegimesArgmaxSmooth,
                                          [i for i, m in enumerate(mapping) if STAGES.index('protoweb') in m or STAGES.index('radii') in m])
                    else:
                        if np.sum(np.isin(mapping, [STAGES.index('protoweb'), STAGES.index('radii')])) > 0:
                            predIdx = np.isin(probRegimesArgmax,
                                              np.argwhere(np.isin(mapping, [STAGES.index('protoweb'), STAGES.index('radii')]))[:, 0])
                            predIdxSmooth = np.isin(probRegimesArgmaxSmooth,
                                              np.argwhere(np.isin(mapping, [STAGES.index('protoweb'), STAGES.index('radii')]))[:, 0])
                # Compute precision / recall
                _precision = 0.0
                if np.sum(predIdx) == 0:
                    _precision = np.nan
                else:
                    _precision = np.nansum(manualIdx & predIdx & isNotLongPause) / np.sum(predIdx & isNotLongPause)
                _recall = 0.0
                if np.sum(manualIdx) == 0:
                    _recall = np.nan
                else:
                    _recall = np.nansum(manualIdx & predIdx & isNotLongPause) / np.sum(manualIdx & isNotLongPause)
                _F1 = 2 * (_precision * _recall) / (_precision + _recall)
                # Compute smoothed data
                _precisionSmooth = 0.0
                if np.sum(predIdxSmooth) == 0:
                    _precisionSmooth = np.nan
                else:
                    _precisionSmooth = np.nansum(manualIdx & predIdxSmooth & isNotLongPause) / np.sum(predIdxSmooth & isNotLongPause)
                _recallSmooth = 0.0
                if np.sum(manualIdx) == 0:
                    _recallSmooth = np.nan
                else:
                    _recallSmooth = np.nansum(manualIdx & predIdxSmooth & isNotLongPause) / np.sum(manualIdx & isNotLongPause)
                _F1Smooth = 2 * (_precisionSmooth * _recallSmooth) / (_precisionSmooth + _recallSmooth)
                # Store
                mappingStr = ''
                if not isinstance(mapping[0], list):
                    mappingStr = ''.join([str(x) for x in mapping])
                else:
                    mappingStr = ','.join([''.join([str(a) for a in x]) for x in mapping])
                table.append((modelRepIdx, modelRecIdx, numRegimesHHMM,
                              outofsample, _precision, _recall, _F1, _precisionSmooth, _recallSmooth, _F1Smooth,
                              dirName, mappingStr, STAGES[istage] if not mergeProtoradii else 'protoweb+radii'))
            if mergeProtoradii:
                break
        # Save intermediate and final results
        tablePd = pd.DataFrame(table, columns=['rep', 'recid', 'numregimes',
            'outofsample', 'precision', 'recall', 'f1', 'precisionS', 'recallS', 'f1S', 'fname', 'mapping', 'stage'])
        tablePd.to_pickle('Z:/Abel/data-tmp/F1_v2_{}_{}{}.pickle'.format(numRegimesHHMM, modelRepIdx, '_protoradii' if mergeProtoradii else ''))

# =====================================================================================
# CLI Command Handler
# =====================================================================================

def handleCommand():
    import sys

    # No commands to process?
    if len(sys.argv) <= 1:
        return False

    # Sleep for a while, to give a chance for debugger to be attached
    print("Started shell... PID={}... waiting 15 seconds before proceeding".format(os.getpid()))
    time.sleep(15)

    # Try to parse JSON
    cmd = json.loads(' '.join(sys.argv[1:]))

    # Print command
    print('\nReceived command: ' + ' '.join(sys.argv[1:]) + '\n\n')

    try:
        processJob(cmd['fname'], cmd['jobid'])

        # Return True to indicate no further processing is required
        return True
    except Exception as e:
        import traceback
        print(e)
        traceback.print_exception(e)
        print('Unable to parse command.')
        time.sleep(100)
        return True

def runPt1():
    if not handleCommand():
        # List models
        fnames = glob.glob('Y:/wavelet/hhmm-results/*regimes_12minrun_manuallabels_5fold.pickle')
        fnames = [x for x in fnames if not 'idxmapping' in x]
        fnames = sorted(fnames, key=lambda x: 10 if '10' in x else 0)

        fnames = [x for x in fnames if '2regimes' in x or '3regimes' in x or '4regimes' in x or '5regimes' in x or '6regimes' in x]
        print(fnames)
        #fnames = [x for x in fnames if '4regimes' in x or '5regimes' in x or '6regimes' in x]

        # Process in parallel
        tasks = [(fn, jobid) for fn in fnames for jobid in range(10)]

        # Start tasks as independent processes
        # Otherwise, fire a new commmand prompt for every task
        import subprocess
        NEXT_NUMA_NODE = 0
        for fn, jobid in tasks:
            cf = {'fname': fn, 'jobid': jobid}
            cmd = '{} {} {}'.format(
                os.path.abspath(os.path.join(os.__file__, os.pardir, os.pardir, 'python.exe')),
                __file__,
                json.dumps(cf).replace("'", '"').replace('"', '\\"'))
            print("Starting shell....")
            print(cmd)

            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            _p = subprocess.Popen('start /node {} '.format(NEXT_NUMA_NODE) + cmd, shell=True, close_fds=True,
                                  creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
            # For the next process, use the opposite NUMA core. This equally divides processes across NUMA cores.
            NEXT_NUMA_NODE = 1 - NEXT_NUMA_NODE

# ======================================================================================================================
# Pt 2
# ======================================================================================================================

def runPt2():
    for numRegimes in [5, 6, 3, 4]:
        fnamesIntermediate = glob.glob('Z:/Abel/data-tmp/F1_intermediate_{}_*.pickle'.format(numRegimes))
        jl.Parallel(n_jobs=35)(jl.delayed(processJobPt2)(fn, modelRepIdxTarget, mergeProtoradii=mergeProtoradii) for \
                              fn in fnamesIntermediate for modelRepIdxTarget in range(50) for mergeProtoradii in [True, False])

# ======================================================================================================================
# Entry point
# ======================================================================================================================

if __name__ == "__main__":
    #runPt1()
    runPt2()