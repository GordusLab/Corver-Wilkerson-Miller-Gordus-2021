import run_wavelet, run_wavelet_cluster, logging, gc
from tqdm import tqdm as tqdm

FNAME_BASE = 'C:/DATA/'

def _getWaveletFilename(settings):
    fn = run_wavelet.getWaveletFilename('', settings)
    fn = os.path.basename(fn)
    fn = fn.replace('.npy','')

    return fn

BATCH_SIZE = 50000
MAX_PARALLEL = 1

# =====================================================================================
# Imports
# =====================================================================================

import joblib as jl, numpy as np, time, os, glob, psutil

# =====================================================================================
# Main function
# =====================================================================================

def getNewDataWaveletFilename(settings):
    a = run_wavelet_cluster.getDimReducFilename(settings)
    FNAME_DIMREDUC_MODEL = a.replace('.png','.pickle')

    a = os.path.basename(FNAME_DIMREDUC_MODEL)
    FNAME_NEWDATA_WAV_BASE = a[:a.find('_hipow')] + '.npy'

    return FNAME_NEWDATA_WAV_BASE

def getDimReducModelFilename(settings, fnameDir = ''):
    a = run_wavelet_cluster.getDimReducFilename(settings)
    FNAME_DIMREDUC_MODEL = a.replace('.png', '.pickle')

    # Some old files have 'nopca' instead of 'no-pca' in the filename
    for fixNoPCA in [True, False]:
        fnameAlt = FNAME_DIMREDUC_MODEL
        if fixNoPCA:
            fnameAlt = fnameAlt.replace('no-pca','nopca')

        if os.path.exists(os.path.join(fnameDir, fnameAlt)):
            return fnameAlt

    return '\\\\?\\' + FNAME_DIMREDUC_MODEL.replace('/', '\\')

def getNewDataDimreducFilename(settings, fnameDir = ''):
    fname = os.path.basename(getDimReducModelFilename(settings)).replace('.pickle','.npy')

    # Some old files have 'nopca' instead of 'no-pca' in the filename
    for fixNoPCA in [True, False]:
        # If this file does not exist, check if a '.pickle.npy' file exists (due to a typo in this code some
        # incorrectly named files were produced.)
        for fixPickle in [True, False]:
            fnameAlt = fname
            if fixPickle:
                fnameAlt = fnameAlt.replace('.npy','.pickle.npy')
            if fixNoPCA:
                fnameAlt = fnameAlt.replace('no-pca','nopca')

            if os.path.exists(os.path.join(fnameDir, fnameAlt)):
                return fnameAlt
    return fname

def run(d, settings, maxIntervalSize = 0, batchSize = 0, overwrite = True):

    # Report PID for debugging purposes and to allow user to change CPU resource allocation manually
    p = psutil.Process()
    pid = p.pid
    logging.info('Embedding new data (PID={})...'.format(pid))

    # Get filenames
    FNAME_DIMREDUC_MODEL = getDimReducModelFilename(settings, fnameDir = d)

    FNAME_NEWDATA_WAV_BASE = getNewDataWaveletFilename(settings)

    FNAME_NEWDATA_DIMREDUC_BASE = getNewDataDimreducFilename(settings, fnameDir = d)

    # Set priority to lowest value so it only takes up spare CPU cycles
    p = psutil.Process(os.getpid())
    p.nice(psutil.IDLE_PRIORITY_CLASS)

    # Determine filenames
    FNAME_NEWDATA_WAV      = '\\\\?\\' + os.path.join(d, FNAME_NEWDATA_WAV_BASE).replace('/', '\\')

    # If a basename is specified, save the clustered files in this directory
    if 'fname_base' in settings:
        FNAME_NEWDATA_DIMREDUC = os.path.join(settings['fname_base'], FNAME_NEWDATA_DIMREDUC_BASE)
    else:
        FNAME_NEWDATA_DIMREDUC = os.path.join(d, FNAME_NEWDATA_DIMREDUC_BASE)
    if not FNAME_NEWDATA_DIMREDUC.startswith('\\\\?\\'):
        FNAME_NEWDATA_DIMREDUC = '\\\\?\\' + FNAME_NEWDATA_DIMREDUC.replace('/', '\\')

    if os.path.exists(FNAME_NEWDATA_DIMREDUC) and not settings['overwrite']:
        logging.warn('Skipping {}'.format(FNAME_NEWDATA_DIMREDUC))
        return

    if not os.path.exists(FNAME_DIMREDUC_MODEL):
        raise Exception('Dimensionality Reduction Model Not Found!')

    if 'rawmvmt' in FNAME_NEWDATA_WAV:
        from pipeline.python.behavioral_motifs.run_wavelet_cluster_rawmvmt import computeAllEmbeddingData
        computeAllEmbeddingData(FNAME_NEWDATA_WAV, settings)
    if not os.path.exists(FNAME_NEWDATA_WAV):
        raise Exception('Wavelet Data Not Found!')

    # Load model
    model = jl.load(FNAME_DIMREDUC_MODEL)

    # Load wavelet data
    wav = np.load(FNAME_NEWDATA_WAV)

    # Perform PCA or NMF on wavelet prior to clustering, if required
    fnamePCAmodel = FNAME_NEWDATA_DIMREDUC_BASE[:FNAME_NEWDATA_DIMREDUC_BASE.rfind('_hipow_') + len('_hipow_')] + 'pca.pickle'
    fnamePCAmodel = os.path.join(os.path.dirname(FNAME_DIMREDUC_MODEL), fnamePCAmodel)
    if settings['use_pca'] not in ['no-pca','nopca']:
        pcaModel = jl.load(fnamePCAmodel)
        wavPCA = pcaModel.transform(wav)
        wav = wavPCA

    # Normalize
    if settings['histogram_normalize'] and not 'rawmvmt' in FNAME_NEWDATA_DIMREDUC:
        wav /= np.repeat(np.sum(wav, axis=1)[:, np.newaxis], wav.shape[1], axis=1)

    gc.collect()

    # Save the embedding model state
    embedding = None

    # Use same perplexity for new data
    perplexity = settings['perplexity_newdata']
    n_iter     = settings['n_iter_newdata']

    while True:
        # Embed in chunks of 10k at a time, choosing spacing in order to minimize gap in data
        if not os.path.exists(FNAME_NEWDATA_DIMREDUC) or overwrite:
            #arrEmbedded = np.memmap(FNAME_NEWDATA_DIMREDUC, mode='w+', shape=(wav.shape[0], 2), dtype=np.float32)
            #arrEmbedded[:, :] = np.nan

            arrEmbedded = np.full((wav.shape[0], 2), np.nan, dtype=np.float32)
        else:
            #arrEmbedded = np.memmap(FNAME_NEWDATA_DIMREDUC, mode='r+', shape=(wav.shape[0], 2), dtype=np.float32)
            arrEmbedded = np.load(FNAME_NEWDATA_DIMREDUC)

        # If this is the first set of points to be embedded, spread them evenly
        timeStart = time.time()
        Npoints = 0

        # Report largest interval
        largestInterval = -1

        if np.all(np.isnan(arrEmbedded)):

            step = maxIntervalSize
            if batchSize != 0:
                step = arrEmbedded.shape[0] / batchSize

            idx = np.arange(0, arrEmbedded.shape[0], step, dtype=np.int64)
            Npoints = len(idx)
            print('Started embedding {} additional points into the space.'.format(Npoints))

            embedding = None
            if 'umap' in str(type(model)).lower():
                embedding = model.transform(wav[idx, :])
            elif 'tsne' in str(type(model)).lower():
                model.n_jobs = 10
                embedding = model.transform(wav[idx, :], perplexity=perplexity, n_iter=n_iter)
            elif settings['method'] == 'agglom':
                from pipeline.python.behavioral_motifs.run_wavelet_cluster_agglom import runNewData
                embedding = runNewData(model, FNAME_NEWDATA_WAV, FNAME_DIMREDUC_MODEL, FNAME_NEWDATA_DIMREDUC, settings)
            else:
                raise Exception('Could not cluster new data, model not recognized.')

            if embedding is None:
                break

            arrEmbedded[idx, :] = embedding

            # Flush to disk
            np.save(FNAME_NEWDATA_DIMREDUC, arrEmbedded)
            del arrEmbedded

            # Status report
            timeEnd = time.time()
            print('Embedding {} additional points into the space took {} seconds.'.format(Npoints, timeEnd - timeStart))

            # Done!
            break

        else:
            # Get intervals for this batch
            idxInBatch = []

            # Keep finding indices (split largest intervals first)
            while len(idxInBatch) < BATCH_SIZE:
                nan = np.isnan(arrEmbedded[:, 0]).astype(np.int32)
                nan = np.pad(np.diff(nan), (1, 0), 'constant')

                # Every range between 1 and -1 is an interval of NaNs
                start = np.argwhere(nan == 1).T[0]
                end = np.argwhere(nan == -1).T[0]

                # End array might have a fewer item, in which we take "end of array" as the end
                if len(end) < len(start):
                    end = end.tolist()
                    end.append(nan.shape[0])
                    end = np.array(end)

                # Iterate through pairs
                sizes = []
                for i, (st, en) in enumerate(zip(start, end)):
                    sizes.append((i, en - st))

                if len(sizes) > 0:
                    break

                # Now add points to the biggest intervals
                sizes = sorted(sizes, key=lambda x: -x[1])

                largestInterval = max([x[1] for x in sizes])

                intIdx = [x[0] for x in sizes[0:BATCH_SIZE - len(idxInBatch)]]

                # Choose centers of each of these intervals
                for i in intIdx:
                    idxInBatch.append(int(0.5 * (start[i] + end[i])))

            # Status report
            Npoints = len(idxInBatch)
            timeEnd = time.time()
            print('Embedding {} additional points into the space took {} seconds.'.format(Npoints, timeEnd - timeStart))

            if largestInterval < maxIntervalSize and maxIntervalSize > 0:
                # Done! Reached sufficiently fine spacing
                del arrEmbedded
                break

            if len(idxInBatch) == 0:
                break

            # Now add points to this process
            embedding = model.transform(wav[idxInBatch, :], perplexity=perplexity, n_iter=n_iter)
            arrEmbedded[idxInBatch, :] = embedding

            # Flush to disk
            np.save(FNAME_NEWDATA_DIMREDUC, arrEmbedded)
            del arrEmbedded

# =====================================================================================
# Version of run() method safe for parallel processing (i.e. doesn't throw errors)
# =====================================================================================

def runSafe(*c):
    try:
        run(*c)
    except Exception as e:
        print(e)

# =====================================================================================
# List directories
# =====================================================================================

def findUnprocessedRecordings(rootpath, settings, overwrite=True):
    dirs = glob.glob('{}/*/wavelet/{}*'.format(rootpath, getNewDataWaveletFilename(settings)))

    # Make sure the dirs contain the file pattern
    if 'filename_pattern' in settings:
        dirs = [x for x in dirs if np.any([p in x for p in settings['filename_pattern']])]

    return list(set([os.path.dirname(d) for d in dirs]))

# =====================================================================================
# Entry Point 1
# =====================================================================================

def run1():
    _allSettings = []
    parallel = False

    _allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean-midline',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel
    })

    _allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel
    })

    _allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean-midline',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel,
        'legvars': {
            'samplingFreq': 50,
            'omega0': 5,
            'numPeriods': 20,
            'maxF': 25,
            'minF': 1,
            'stack': True
        }
    })

    _allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel,
        'legvars': {
            'samplingFreq': 50,
            'omega0': 5,
            'numPeriods': 20,
            'maxF': 25,
            'minF': 1,
            'stack': True
        }
    })

    # Run all settings for a single recording?
    for x in range(len(_allSettings)):
        rec = 'Z:/behavior/6-23-19-e'
        _allSettings[x]['fname_base'] = os.path.join(rec, 'embeddings/')
        _allSettings[x]['recordings'] = [rec, ]
        _allSettings[x]['highpower_rows_per_file'] = 200000

    allSettings = []
    for settings in _allSettings:
        if 'fname_base' in settings:
            dirs = [os.path.abspath(os.path.join(settings['fname_base'], '../wavelet'))]
        else:
            dirs = findUnprocessedRecordings('Z:/behavior/', _allSettings[0], overwrite=_allSettings[0]['overwrite'])
        for d in dirs:
            logging.info(d)
            allSettings.append((d, settings, 2))  # maxIntervalSize = 2

    logging.info('Processing {} configurations.'.format(len(allSettings)))

    r = jl.Parallel(n_jobs=min(len(allSettings), MAX_PARALLEL))(jl.delayed(runSafe)(*c) for c in tqdm(allSettings))

    logging.info('Done processing {} configurations.'.format(len(allSettings)))

# =====================================================================================
# Entry Point 2
# =====================================================================================

def run2():
    _allSettings = []
    parallel = False

    _allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22, 9, 13, 21, 25],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel,
        'legvars': {
            'samplingFreq': 50,
            'omega0': 5,
            'numPeriods': 20,
            'maxF': 25,
            'minF': 1,
            'stack': True
        }
    })

    # Run all settings for a single recording?
    for x in range(len(_allSettings)):
        rec = 'Z:/behavior/6-23-19-e'
        _allSettings[x]['fname_base'] = os.path.join(rec, 'embeddings/')
        _allSettings[x]['recordings'] = [rec, ]
        _allSettings[x]['highpower_rows_per_file'] = 200000

    allSettings = []
    for settings in _allSettings:
        if 'fname_base' in settings:
            dirs = [os.path.abspath(os.path.join(settings['fname_base'], '../wavelet'))]
        else:
            dirs = findUnprocessedRecordings('Z:/behavior/', _allSettings[0], overwrite=_allSettings[0]['overwrite'])
        for d in dirs:
            logging.info(d)
            allSettings.append((d, settings, 2))  # maxIntervalSize = 2

    logging.info('Processing {} configurations.'.format(len(allSettings)))

    r = jl.Parallel(n_jobs=min(len(allSettings), MAX_PARALLEL))(jl.delayed(runSafe)(*c) for c in tqdm(allSettings))

    logging.info('Done processing {} configurations.'.format(len(allSettings)))


# =====================================================================================
# Entry Point 1
# =====================================================================================

def run3():
    _allSettings = []
    parallel = False

    _allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean-midline',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel
    })

    _allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel
    })

    _allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean-midline',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel,
        'legvars': {
            'samplingFreq': 50,
            'omega0': 5,
            'numPeriods': 20,
            'maxF': 25,
            'minF': 1,
            'stack': True
        }
    })

    _allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': False,
        # Coords
        'coords': 'euclidean',
        'use_pca': 'no-pca',
        'metric': 'euclidean',
        'include_absolute_position': 'no-abspos',
        'include_velocities': 'no-vel',
        'tsne_perplexities': [100, ],  # 100, 500
        'perplexity': 100,
        'perplexity_newdata': 100,
        'n_iter': 2000,
        'n_iter_newdata': 500,
        'maxN': 200000,
        'method': 'tsne',  # 'agglom',
        'parallel': parallel,
        'legvars': {
            'samplingFreq': 50,
            'omega0': 5,
            'numPeriods': 20,
            'maxF': 25,
            'minF': 1,
            'stack': True
        }
    })

    allSettings = []
    for settings in _allSettings:
        if 'fname_base' in settings:
            dirs = [os.path.abspath(os.path.join(settings['fname_base'], '../wavelet'))]
        else:
            dirs = findUnprocessedRecordings('Z:/behavior/', _allSettings[0], overwrite=_allSettings[0]['overwrite'])
        dirs = ['Z:/behavior\\190528_RIG2_SPIDER0002~\\wavelet\\', ]
        for d in dirs:
            logging.info(d)
            allSettings.append((d, settings, 2))  # maxIntervalSize = 2

    #Z:\behavior\190528_RIG2_SPIDER0002~

    logging.info('Processing {} configurations.'.format(len(allSettings)))

    r = jl.Parallel(n_jobs=min(len(allSettings), MAX_PARALLEL))(jl.delayed(runSafe)(*c) for c in tqdm(allSettings))

    logging.info('Done processing {} configurations.'.format(len(allSettings)))

# =====================================================================================
# Entry Point
# =====================================================================================

if __name__ == "__main__":
    #run1()
    #run2()
    run3()