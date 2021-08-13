
import glob, joblib as jl, regex as re, numpy as np
from tqdm import tqdm

DEBUG = False
MAX_ITERATIONS = 250

def trainModel(model, states, fold):

    # Fit HMM
    statesToFit = states
    if fold is not None:
        statesToFit = [states[i] for i in range(len(states)) if i in fold[0]]

    model, hist = model.fit(statesToFit,
        algorithm='baum-welch', min_iterations=100 if not DEBUG else 10,
        max_iterations=MAX_ITERATIONS if not DEBUG else 10, stop_threshold=1e-5,
        return_history=True, verbose=True)

    # Predict states
    statesPred = [model.predict(x, algorithm='map') for x in states]
    statesPredProb = [model.predict_log_proba(x) for x in states]

    # Done
    return model, hist, statesPred, statesPredProb

def run(fname):
    fnameBase = fname
    g = re.search('(\.([0-9]*))*\.pickle', fname).groups()
    runID = 0
    if g[1] is not None:
        fnameBase = fname.replace(g[0], '')
        runID = int(g[1])
    fnameOut = fnameBase.replace('.pickle', '.{}.pickle'.format(runID + 1))
    print('Outputting retrained model to {}'.format(fnameOut))

    # Load data
    data = jl.load(fname)
    print('Finished loading pickled model.')

    #
    results = jl.Parallel(n_jobs=min(len(data['models']), 1 if DEBUG else 10), prefer='threads')(jl.delayed(trainModel)(
        data['models'][i]['model'],
        data['models'][i]['model-fit-states'],
        data['models'][i]['fold']) for i in range(len(data['models'])))

    # Update dictionary
    for i, (model, hist, statesPred, statesPredProb) in enumerate(results):
        data['models'][i]['model'] = model
        data['models'][i]['statesPred'] = statesPred
        data['models'][i]['statesPredProb'] = statesPredProb
        fH = data['models'][i]['fithistory'] if isinstance(data['models'][i]['fithistory'], list) else [data['models'][i]['fithistory'], ]
        data['models'][i]['fithistory'] = fH + [hist, ]

    # Save data
    jl.dump(data, fnameOut)

if __name__ == "__main__":
    # Get current results
    fnames = [x for x in glob.glob('Y:/wavelet/hhmm-results/*.pickle') if not 'idxmapping' in x]

    # Get latest re-train for each model set
    fnamesUse = {}
    for fname in fnames:
        g = re.search('(\.([0-9]*))*\.pickle', fname).groups()
        if g[0] is not None:
            fname = fname.replace(g[0], '')
        runID = 0
        if g[1] is not None:
            runID = int(g[1])
        if fname not in fnamesUse:
            fnamesUse[fname] = runID
        else:
            fnamesUse[fname] = max(fnamesUse[fname], runID)

    fnamesUse = [x.replace('.pickle', '{}.pickle'.format('.{}'.format(fnamesUse[x]) if \
        fnamesUse[x] > 0 else '')) for x in fnamesUse]

    np.random.shuffle(fnamesUse)

    # Retrain each model
    jl.Parallel(n_jobs=1 if DEBUG else 10, prefer='processes')(jl.delayed(run)(fname) for fname in tqdm(fnames))