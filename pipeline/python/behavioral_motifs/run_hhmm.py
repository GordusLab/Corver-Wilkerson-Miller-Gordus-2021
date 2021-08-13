
# ======================================================================================================================
# Imports
# ======================================================================================================================

import os, time, glob, numpy as np, pandas as pd, pomegranate as pg, joblib as jl, psutil
from tqdm import tqdm
from itertools import groupby

DEBUG = False

# ======================================================================================================================
# Helper functions
# ======================================================================================================================

def toNumber(x):
    try:
        return int(x)
    except:
        return -1

def loadLabels(fnameLabels):
    txtClusterLabels = ''
    with open(fnameLabels, 'r') as f:
        txtClusterLabels = f.read()
    clusterLabels = {}
    curLabel = ''
    for line in txtClusterLabels.split('\n'):
        if ':' in line:
            curLabel = line[:line.find(':')]
        elif len(line.strip()) > 0:
            clusterLabels[curLabel] = [toNumber(x) for x in line.split(',') if toNumber(x) >= 0]
    return clusterLabels

# ======================================================================================================================
# Only keep stable cluster states
# ======================================================================================================================

def filterClusters(cl, clusterLabels, clusterLabelsUnique, minrun=12):
    runs = []
    for c in cl:
        if clusterLabels is not None and clusterLabelsUnique is not None:
            if len([k for k in clusterLabels if c in clusterLabels[k]]) == 0:
                c = -1
            else:
                c = clusterLabelsUnique.index([k for k in clusterLabels if c in clusterLabels[k]][0])

        if len(runs) == 0:
            runs.append([c, ])
        elif runs[-1][-1] == c:
            runs[-1].append(c)
        else:
            runs.append([c, ])
    return np.hstack([np.array(x if len(x) >= minrun else [np.nan, ] * len(x), dtype=np.float64) for x in runs])

# ======================================================================================================================
# Fit HMM
# ======================================================================================================================

def fitHHMM(states, numStates=2, numRegimes=2, fold = None):
    MAX_ITERATIONS = 250

    t0 = time.time()

    model = pg.HiddenMarkovModel()

    # Add states
    statesObj = []
    for regime in range(numRegimes):
        statesObj.append(pg.State(None, name='r{:03d}'.format(regime)))
        for state in range(numStates):
            statesObj.append(pg.State(pg.DiscreteDistribution({
                i: (1 if i == state else 0) for i in range(numStates)
            }), name='r{:03d}_s{:03d}'.format(regime, state)))
    model.add_states(*statesObj)

    # Sequence can start at and end from any top-level state
    for s in statesObj:
        if 'r' in s.name and not 's' in s.name:
            model.add_transition(model.start, s, 1)
            model.add_transition(s, model.end, 1)

    # Allow full transitions within regime
    for regime in range(numRegimes):
        st = [x for x in statesObj if 'r{:03d}_'.format(regime) in x.name]
        for s1 in st:
            for s2 in st:
                model.add_transition(s1, s2, np.random.random())

    # Every state can transition to any other top-level regime state except itself
    for s in statesObj:
        if 's' in s.name:
            for s2 in [x for x in statesObj if 'r' in x.name and not 's' in x.name and x.name != s.name]:
                model.add_transition(s, s2, np.random.random())

    # The top-level regimes can transition only to their within-regime states
    for regime in range(numRegimes):
        st = [x for x in statesObj if 'r{:03d}_'.format(regime) in x.name]
        for s1 in [x for x in statesObj if x.name == 'r{:03d}'.format(regime)]:
            for s2 in st:
                model.add_transition(s1, s2, np.random.random())

    # Finalize model topology
    model.bake(merge=None)

    # Fit HMM
    statesToFit = states
    if fold is not None:
        statesToFit = [states[i] for i in range(len(states)) if i in fold[0]]

    print('Started fitting model!')
    model, hist = model.fit(statesToFit,
                            algorithm='baum-welch', min_iterations=250 if not DEBUG else 10,
                            max_iterations=MAX_ITERATIONS if not DEBUG else 10, stop_threshold=1e-5,
                            return_history=True, verbose=True)

    t1 = time.time()

    # Predict states
    print('Finished fitting model!')
    statesPred = [model.predict(x, algorithm='map') for x in states]

    print('Finished predicting model!')
    return {'model': model, 'model-fit-states': states, 'fithistory': hist, 'statesPred': statesPred, 'time-elapsed': t1 - t0, 'fold': fold}

# ======================================================================================================================
# Main Function
# ======================================================================================================================

def run(outDir, fnames, fnamesLabels, numRegimesHHMM = 4, numFitsHHMM = 1, minClusterRun = 12, crossValidation = '5fold'):
    # Settings
    useManualLabels = (fnamesLabels is not None)
    minUnifiedEvents = 50

    os.makedirs(outDir, exist_ok=True)
    fnameOut = os.path.join(outDir, '{}regimes_{}minrun_{}_{}.pickle'.format(
        numRegimesHHMM, minClusterRun, 'manuallabels' if useManualLabels else 'nomanuallabels', crossValidation))

    # Load labels
    if useManualLabels:
        clusterLabels = (loadLabels(fnamesLabels[0]), loadLabels(fnamesLabels[1]))
        clusterLabelsUnique = list(set(list(clusterLabels[0].keys()) + list(clusterLabels[1].keys())))
    else:
        clusterLabels = [None, None]
        clusterLabelsUnique = None

    # Load clusters
    arrClusters = [(np.load(fnameA), np.load(fnameP)) for fnameA, fnameP in tqdm(fnames, leave=False)]

    # Map clusters to manual labels# Convert single clusters into the shared space
    arrClustersMapped = [[None, None] for i in range(len(arrClusters))]

    for i in tqdm(range(len(arrClusters)), leave=False):
        for k in range(2):
            cl = arrClusters[i][k][:,0].astype(int)
            # Merge clusters of the same label before filtering
            cl = [(a[0] if len(a) == 1 else -1) for a in
                [[clusterLabels[k][x][0] for x in clusterLabels[k] if c in clusterLabels[k][x]] for c in cl]]
            arrClustersMapped[i][k] = filterClusters(cl, clusterLabels[k], clusterLabelsUnique, minrun = minClusterRun)

    # Determine all cluster IDs
    a = np.hstack([x[0] for x in arrClustersMapped] + [x[1] for x in arrClustersMapped])
    clEx = np.unique(a[~np.isnan(a)])
    clEx = clEx[clEx >= 0].astype(int)

    # Determine unified labels
    dCooc = pd.DataFrame({"clusterA": np.hstack([x[0] for x in arrClustersMapped]).astype(int),
                          "clusterP": np.hstack([x[1] for x in arrClustersMapped]).astype(int)})
    dCooc.loc[:, 'count'] = 1
    dCooc = dCooc[(dCooc.clusterA >= 0) & (dCooc.clusterP >= 0)]
    dCooc = dCooc.groupby(['clusterA', 'clusterP']).sum().reset_index()

    if useManualLabels:
        clusterLabelsUniqueUnified = sorted(['{}/{}'.format(clusterLabelsUnique[y], clusterLabelsUnique[x]) \
              for x in clEx for y in clEx if x != y and \
                    (dCooc.loc[(dCooc.clusterA == x) & (dCooc.clusterP == y), 'count'].shape[0] > 0) and \
                    (dCooc.loc[(dCooc.clusterA == x) & (dCooc.clusterP == y), 'count'].values[0] > minUnifiedEvents) and \
                    'stabilimentum' not in [clusterLabelsUnique[x], clusterLabelsUnique[y]]] + \
                        np.array(clusterLabelsUnique)[clEx].tolist())
    else:
        # Todo: Only feasible if we aggregate clusters while maintaining transition matrix (see. Gordon Behrman et al)
        # ...
        clusterLabelsUniqueUnified = sorted(['{}/{}'.format(y, x) \
              for x in clEx for y in clEx if x != y and \
                    (dCooc.loc[(dCooc.clusterA == x) & (dCooc.clusterP == y), 'count'].shape[0] > 0) and \
                    (dCooc.loc[(dCooc.clusterA == x) & (dCooc.clusterP == y), 'count'].values[0] > minUnifiedEvents)])

    # Unify labels
    arrClustersMappedUnified = []
    for arrClustersMappedA, arrClustersMappedP in tqdm(arrClustersMapped):
        tmp = []
        for a, p in zip(arrClustersMappedA.astype(int), arrClustersMappedP.astype(int)):
            if useManualLabels:
                # Determine manual label
                lbl = ''
                if a >= 0 and p >= 0 and clusterLabelsUnique[a] in 'stationary-anterior' and clusterLabelsUnique[p] in 'stationary-posterior':
                    lbl = 'stationary'
                else:
                    if p >= 0 and clusterLabelsUnique[p] == 'stabilimentum':
                        p = -1  # Don't allow posterior marker to signal stabilimentum, as it's less accurate than anterior marker
                    if a >= 0 and clusterLabelsUnique[a] in ['stationary-anterior', 'noisy']:
                        a = -1 # Noisy and stationary-anterior states are ignored for the purposes of transitions
                    if p >= 0 and clusterLabelsUnique[p] in ['stationary-posterior', 'noisy']:
                        p = -1 # Noisy and stationary-posterior states are ignored for the purposes of transitions
                    if (a >= 0 and clusterLabelsUnique[a] == 'stationary') or (p >= 0 and clusterLabelsUnique[p] == 'stationary'):
                        lbl = 'stationary' # If either A/P detects stationary posture, merge to overall 'stationary'
                    elif a >= 0 and p >= 0:
                        lbl = '{}/{}'.format(clusterLabelsUnique[p], clusterLabelsUnique[a])
                    elif max(a, p) >= 0:
                        lbl = clusterLabelsUnique[max(a, p)]
            else:
                lbl = '{}/{}'.format(p, a)
            # Transform to label
            if lbl in clusterLabelsUniqueUnified:
                tmp.append(clusterLabelsUniqueUnified.index(lbl))
            else:
                tmp.append(-1)
        # Store mapped-to-manual array
        arrClustersMappedUnified.append(np.array(tmp).astype(int))

    # Remove self-transitions
    arrClustersNoRep = []
    arrNorepToOrgs = []
    for idx in range(len(arrClustersMappedUnified)):
        d = arrClustersMappedUnified[idx][arrClustersMappedUnified[idx] >= 0]
        d = np.array([k for k, g in groupby(d) if k != 0])
        arrClustersNoRep.append(d)

        # For each file, save the mapping from no-rep to the original
        arrNorepToOrg = np.full(arrClusters[idx][0].shape[0], -1, dtype=int)
        for _regime, _idxs in [(a, [y[0] for y in b]) for (a, b) in zip(np.arange(arrClustersNoRep[idx].size, dtype=int), [
                list(g) for k, g in groupby(
                zip(np.argwhere(arrClustersMappedUnified[idx] >= 0).T[0],
                    arrClustersMappedUnified[idx][arrClustersMappedUnified[idx] >= 0]), key=lambda x: x[1]) if k != 0])]:
            arrNorepToOrg[_idxs] = _regime
        # Save regime estimates
        arrNorepToOrgs.append(arrNorepToOrg)

    # Save this mapping array
    fnameOutMapping = fnameOut.replace('.pickle', '') + '.idxmapping.pickle'
    jl.dump(arrNorepToOrgs, fnameOutMapping, compress=False)

    # Fit HHMM
    results = {
        'fnames': fnames,
        'fnamesLabels': fnamesLabels,
        'clusterLabelsUnique': clusterLabelsUnique,
        'clusterLabelsUniqueUnified': clusterLabelsUniqueUnified,
        'numRegimesHHMM': numRegimesHHMM,
        'minClusterRun': minClusterRun,
        'useManualLabels': useManualLabels,
        'models': []
    }

    from sklearn.model_selection import KFold
    kfolds = list(KFold(2 if DEBUG else (5 if crossValidation == '5fold' else 1)).split(np.arange(len(fnames))))
    modelResults = jl.Parallel(n_jobs=min(numFitsHHMM, 1 if DEBUG else 20), prefer='threads')(jl.delayed(fitHHMM)(arrClustersNoRep,
            numStates = np.max(np.hstack(arrClustersNoRep)) + 1,
            numRegimes = numRegimesHHMM, fold=fold) for it, fold in [(it, fold) for it in range(numFitsHHMM) for fold in kfolds])
    print('Num. models: {}'.format(len(modelResults)))

    for modelResult in modelResults:
        # Obtain transition matrix
        mtxTrans = modelResult['model'].dense_transition_matrix()

        # Convert state labels to corresponding regimes
        arrRegimes = []
        for idx in range(len(modelResult['statesPred'])):
            try:
                regime = np.array([int(modelResult['model'].states[x].name[1:4]) for x in modelResult['statesPred'][idx]])

                arrRegime = np.full(arrClusters[idx][0].shape[0], -1, dtype=int)
                for _regime, _idxs in [(a, [y[0] for y in b]) for (a, b) in zip(regime, [list(g) for k, g in groupby(
                        zip(np.argwhere(arrClustersMappedUnified[idx] >= 0).T[0],
                            arrClustersMappedUnified[idx][arrClustersMappedUnified[idx] >= 0]), key=lambda x: x[1]) if k != 0])]:
                    arrRegime[_idxs] = _regime
                # Save regime estimates
                arrRegimes.append(arrRegime)
            except Exception as e:
                arrRegimes.append(None)
                print(e)

        # Save in results
        results['models'].append({**{
            'mtxTrans': mtxTrans,
            'arrRegimes': arrRegimes
        }, **modelResult})

    # Save intermediate results!
    jl.dump(results, fnameOut, compress=True)

# ======================================================================================================================
# Entry Point
# ======================================================================================================================

if __name__ == "__main__":
    # Set priority to lowest value so it only takes up spare CPU cycles
    p = psutil.Process(os.getpid())
    p.nice(psutil.IDLE_PRIORITY_CLASS)

    # Specify base files to use
    fnamesA = glob.glob('Z:/behavior/*/wavelet/rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.clusters.npy')

    # Find accompanying posterior embedding files
    fnamesP = [x.replace('00000010001000000010001000', '00000000010001000000010001').replace(
        '_dlc_euclidean', '_dlc_euclidean-midline') for x in fnamesA]

    # Prepare A/P file array
    fnames = [(fnA, fnP) for fnA, fnP in zip(fnamesA, fnamesP) if os.path.exists(fnA) and os.path.exists(fnP) and \
              '190528' not in fnA and '4-17-19-a' not in fnA]

    if DEBUG:
        fnames = fnames[:2]

    # Label files to use
    fnameClusterLabelsA = '\\\\?\\Y:\\wavelet\\clips\\rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd\\cluster_names.txt'
    fnameClusterLabelsP = '\\\\?\\Y:\\wavelet\\clips\\rawmvmt_dlc_euclidean-midline_no-abspos_no-vel_00000000010001000000010001_60_16_meansub_scalestd\\cluster_names.txt'
    fnamesLabels = (fnameClusterLabelsA, fnameClusterLabelsP)

    print(fnamesLabels)

    # Start process
    _numRegimes = np.arange(1, 12)
    np.random.shuffle(_numRegimes)
    jl.Parallel(n_jobs=1 if DEBUG else 12, prefer='processes')(jl.delayed(run)(
        'Y:/wavelet/hhmm-results/', fnames,
        fnamesLabels, numRegimesHHMM = numRegimesHHMM, numFitsHHMM = 10, minClusterRun = 12) for numRegimesHHMM in _numRegimes)