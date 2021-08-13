# =====================================================================================
# Globals & Imports
# =====================================================================================

import numpy as np, matplotlib.pyplot as plt, sys, os, gc, pandas as pd, logging, regex as re, \
    datashader as ds, datashader.transfer_functions as tf, umap, imageio, joblib as jl, time, copy, json, numba as nb
from sklearn.decomposition.pca import PCA
from openTSNE import TSNE
from tqdm import tqdm as tqdm

# Import the function library
import run_wavelet

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
except:
    p.nice(20)

# When we run processes, we keep track of the NUMA core it was run on, to equally divide work across NUMA cores.
NEXT_NUMA_NODE = 0

# =====================================================================================
# Run wavelet analyses
# =====================================================================================

FNAME_BASE = lambda settings: 'Y:/wavelet/' if not 'fname_base' in settings else settings['fname_base']

def _getWaveletFilename(settings):
    if not 'rawmvmt' in settings or settings['rawmvmt'] is not True:
        fn = run_wavelet.getWaveletFilename('', settings)
    else:
        from pipeline.python.behavioral_motifs.run_wavelet_cluster_rawmvmt import getWaveletFilename as rawmvmt_getWaveletFilename
        fn = rawmvmt_getWaveletFilename(settings)

    fn = os.path.basename(fn)
    fn = fn.replace('.npy','')

    return fn

FNAME_DATA      = lambda settings: '{}{}/{}.npy'.format          (FNAME_BASE(settings), _getWaveletFilename(settings), _getWaveletFilename(settings))
FNAME_HIPOW     = lambda settings: '{}{}/{}_hipow.npy'.format    (FNAME_BASE(settings), _getWaveletFilename(settings), _getWaveletFilename(settings))
FNAME_HIPOW_IDX = lambda settings: '{}{}/{}_hipow_idx.npy'.format(FNAME_BASE(settings), _getWaveletFilename(settings), _getWaveletFilename(settings))
FNAME_PCA       = lambda settings: '{}{}/{}_hipow_{}.npy'.format (FNAME_BASE(settings), _getWaveletFilename(settings), _getWaveletFilename(settings), settings['use_pca'])
FNAME_POWER     = lambda settings: '{}{}/{}_pow.npy'.format      (FNAME_BASE(settings), _getWaveletFilename(settings), _getWaveletFilename(settings))
FNAME_IDX       = lambda settings: '{}{}/{}_idx.npy'.format      (FNAME_BASE(settings), _getWaveletFilename(settings), _getWaveletFilename(settings))
ACTIVITY_THRESHOLD = 0 # Power is now filtered at the coordinate level, so probably no need to filter again based on wavelet power

def runWavelet(settings):
    if 'recordings' in settings:
        fnames = settings['recordings']
    else:
        fnames = run_wavelet.listAvailableRecordings()

    # Run transform
    for fname in tqdm(fnames):
        ampl, f = run_wavelet.runWaveletTransform(fname, settings, loadWaveletIfExists = True)

        # Generate plot
        if ampl is not None:
            run_wavelet.plotOverview(fname, settings, ampl, f)

# =====================================================================================
# Merge wavelets
# =====================================================================================

def getMmapShape(f):
    A = np.load(f, mmap_mode='r')
    s = A.shape
    del A
    gc.collect()
    return s

def runWaveletMerge(settings):
    # Allow user to specify recordings to use (useful for e.g. testing simulated datasets)
    if 'recordings' in settings:
        fnames = settings['recordings']
    else:
        fnames = run_wavelet.listAvailableRecordings()

    # Check settings
    settings = run_wavelet.checkSettings(settings)

    # Concatenate the wavelet data
    fnamesWav = [run_wavelet.getWaveletFilename(d, settings) for d in fnames]

    # Create shortened filename
    fnamesWavShort = [re.search('/behavior/([^/]*)/', x.replace('\\','/')).group(1) for x in fnamesWav]

    # Obtain number of rows/cols without reading data
    shapes = [getMmapShape(f) for f in tqdm(fnamesWav, leave=False)]

    # Merge into single file
    nrows = np.sum([x[0] for x in shapes])
    ncols = shapes[0][1]

    if not os.path.exists(FNAME_DATA(settings)) or settings['overwrite']:
        os.makedirs(os.path.dirname(FNAME_DATA(settings)), exist_ok=True)

        # Merge data into single file
        data = None

        # Determine maximum filename length
        fnameMaxLen = max([len(f) for f in fnamesWavShort])

        # Keep index of filenames
        idx = np.zeros(nrows, dtype={'names': ('file', 'index'), 'formats': (np.dtype(('U', fnameMaxLen)), np.int64)})

        # Keep track of power
        pw = np.zeros(nrows, dtype=np.float32)

        # Save all data into this array
        curr = 0
        for fi, f in tqdm(enumerate(fnamesWav), total=len(fnamesWav), leave=False):
            # Merge data
            print('Processing: {}'.format(f))
            mtx = np.load(f, mmap_mode=None)
            gc.collect()

            # Compute power (in blocks)
            # Note: There is a strange bug (?) that causes the power computation to fail on certain large files
            #       (e.g. 4-13-19-a) unless it is done in blocks.
            print('4c: {}'.format(f))
            BLOCK_SIZE = 100000
            for i in range(0, mtx.shape[0], BLOCK_SIZE):
                pw[(curr + i):(curr + min(i + BLOCK_SIZE, mtx.shape[0]))] = \
                    np.nansum(np.square(mtx[i:(i+BLOCK_SIZE),:]), axis=1)

            # Keep index
            _idx = []
            print('2: {}'.format(f))
            for _i in np.arange(mtx.shape[0], dtype=np.int64): _idx.append( (fnamesWavShort[fi], _i) )
            print('3: {}'.format(f))
            idx[curr:(curr + mtx.shape[0])] = _idx

            # Option 1: Scale by mean
            #pw[curr:(curr + mtx.shape[0])] *= 1000.0 / np.nanmean(pw[curr:(curr + mtx.shape[0])])

            # Option 2: Scale by Nth percentile... This should ensure the same number of rows are sampled from
            #           each dataset.
            print('5: {}'.format(f))
            pw[curr:(curr + mtx.shape[0])] *= 1000.0 / np.percentile(pw[curr:(curr + mtx.shape[0])], ACTIVITY_THRESHOLD)

            print('1: {}'.format(f))
            if data is None:
                data = np.memmap(filename=FNAME_DATA(settings), mode='w+', shape=(nrows, ncols), dtype=np.float32)
            data[curr:(curr + mtx.shape[0]), :] = mtx

            # Next
            curr += mtx.shape[0]

            # Clear memory
            print('6: {}'.format(f))
            del mtx
            gc.collect()
            print('7: {}'.format(f))

        # Save power and index
        np.save(FNAME_POWER(settings), pw)
        np.save(FNAME_IDX(settings), idx)

        # Delete data object
        del data
        del pw
        gc.collect()

# =====================================================================================
# Filter by power (from wavelet power)
# =====================================================================================

def runWaveletHighPower(settings):
    """
    This function filters the wavelet data based on the power of the wavelet transform output.
    """
    # Check settings
    settings = run_wavelet.checkSettings(settings)

    # Open merged files
    pw = np.load(FNAME_POWER(settings))
    idx = np.load(FNAME_IDX(settings), mmap_mode='r')

    fnameData = FNAME_DATA(settings)
    ampl = np.memmap(filename=fnameData, mode='r', dtype=np.float32)
    nrows = pw.shape[0]
    ncols = int(ampl.size / nrows)
    del ampl
    ampl = np.memmap(filename=fnameData, mode='r', shape=(nrows, ncols), dtype=np.float32)

    # Only filter by power
    if not os.path.exists(FNAME_HIPOW(settings)) or settings['overwrite']:

        # Determine what rows to keep
        subset = pw >= np.percentile(pw, ACTIVITY_THRESHOLD)

        # Get max rows to keep per file
        rowsPerFile = 10000
        if 'highpower_rows_per_file' in settings:
            rowsPerFile = settings['highpower_rows_per_file']

        # Randomly sample a fixed number of frames (10k) from each video, to make sure no one video dominates the
        # dimensionality reduction space
        idxNames = np.array([x[0] for x in tqdm(idx,desc='preparing name index',leave=False)])
        subsetNew = subset.copy()
        subsetNew[:] = False
        for _fn in tqdm(np.unique(idxNames), desc='randomly subsetting each file', leave=False):
            subsetRandom = np.random.choice(np.argwhere(idxNames==_fn).T[0],
                min(rowsPerFile, np.argwhere(idxNames==_fn).T[0].shape[0]), replace=False)
            subsetNew[subsetRandom] = True
        subset = subsetNew

        # New number of rows
        nr = np.sum(subset)
        logging.info('Keeping {} rows from file {}'.format(nr, fnameData))

        # Save subsetted dataset
        amplHiPow = np.memmap(filename=FNAME_HIPOW(settings), mode='w+', shape=(nr, ncols), dtype=np.float32)
        amplHiPow[:, :] = ampl[subset]
        del amplHiPow
        gc.collect()

        # Save subsetted index
        idxHiPow = idx[subset]
        np.save(FNAME_HIPOW_IDX(settings), idxHiPow)

# =====================================================================================
# PCA
# =====================================================================================

def runPCA(settings):
    # Check settings
    settings = run_wavelet.checkSettings(settings)

    # Already processed?
    fnamePCA = FNAME_PCA(settings)
    if os.path.exists(fnamePCA) and not settings['overwrite']:
        logging.info('Skipping PCA computation for: {}'.format(fnamePCA))
        return
    else:
        logging.info('Running PCA on: {}'.format(FNAME_PCA(settings)))

    # Open merged files
    idx = np.load(FNAME_HIPOW_IDX(settings), mmap_mode='r')

    ampl = np.memmap(filename=FNAME_HIPOW(settings), mode='r', dtype=np.float32)
    nrows = idx.shape[0]
    ncols = int(ampl.size / nrows)
    del ampl
    ampl = np.memmap(filename=FNAME_HIPOW(settings), mode='r', shape=(nrows, ncols), dtype=np.float32)

    # Generate a dataset where each of the columns (wavelets) is shuffled in time (so variance stays the same
    # but correlations are removed)
    amplShuffled = ampl.copy()
    for i in tqdm(range(0, amplShuffled.shape[1], 10), leave=False):
        amplShuffled[:, i:(i + 10)] = amplShuffled[np.random.permutation(amplShuffled.shape[0]), i:(i + 10)]
    gc.collect()

    if settings['use_pca'] == 'nmf':
        # Perform NMF with increasing rank, and keep track of RSS
        from sklearn.decomposition import NMF
        rss = []
        rssShuff = []

        amplTest         = ampl        [np.random.choice(np.arange(ampl.shape[0]), 50000, replace=False),:]
        amplShuffledTest = amplShuffled[np.random.choice(np.arange(ampl.shape[0]), 50000, replace=False),:]

        for r in tqdm(range(1, 50), desc='Computing NMFs'):
            rss.append     ( NMF(n_components=r).fit(amplTest        ).reconstruction_err_ )
            rssShuff.append( NMF(n_components=r).fit(amplShuffledTest).reconstruction_err_ )

        # Determine number of components to keep
        #rss = np.array(rss); rssShuff = np.array(rssShuff)
        #_nComponents = np.argwhere(rssShuff > rss)[0, 0]
        _nComponents = 40

        # This is producing weird plots...
        fnamePlotNMF = fnamePCA.replace('.npy','') + '_components.png'
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        ax.plot(rss     , color='blue')
        ax.plot(rssShuff, color='red' )
        fig.savefig(fnamePlotNMF)

        # Perform NMF
        amplNMF = NMF(n_components=_nComponents).fit_transform(ampl)

        # Save PCA values
        np.save(fnamePCA, amplNMF)
        logging.info('Saved {}'.format(fnamePCA))

    # THIS CODE PERFORMS PCA, BUT CREATED NEGATIVE VALUES THAT ARE INCOMPATIBLE WITH JENSEN-SHANNON/KL DISTANCE
    elif settings['use_pca'] == 'pca':
        # Fit PCA to both
        pca = PCA().fit(ampl)
        pcaShuff = PCA().fit(amplShuffled)

        # The number of components to keep
        _nComponents = np.argwhere(pcaShuff.singular_values_ > pca.singular_values_)[0, 0]

        # Make sure the number of components is not less than 20
        nComponents = max(_nComponents, 20)
        logging.info('PCA coordinates to keep based on shuffling is {}, but kept {}'.format(
            _nComponents, nComponents))

        # This is producing weird plots...
        fnamePlotPCA = fnamePCA.replace('.npy','') + '_components.png'
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        ax.plot(pca.singular_values_     , color='blue')
        ax.plot(pcaShuff.singular_values_, color='red' )
        fig.savefig(fnamePlotPCA)

        # Now re-fit PCA with the correct number of components
        pca = PCA(n_components=nComponents).fit(ampl)

        # Transform, filter, and transform back
        amplPCA = pca.transform(ampl)

        # DEPRECATED: Instead of taking truncated PCA as output, we previously transformed data back to original space.
        #amplPCA[:, nComponents:] = 0
        #amplPCAorg = pca.inverse_transform(amplPCA)
        #logging.info('PCA-transformed data (#components={}), and transformed back (#features={})'.format(
        #    nComponents, amplPCAorg.shape[1]))

        logging.info('PCA-transformed data (#components={})'.format(
                nComponents))

        # Save PCA model so more data can be embedded
        fnamePCAmodel = fnamePCA.replace('.npy','') + '.pickle'
        jl.dump(pca, fnamePCAmodel)
        logging.info('Saved {}'.format(fnamePCAmodel))

        # Save PCA values
        #np.save(fnamePCA, amplPCAorg)
        np.save(fnamePCA, amplPCA)
        logging.info('Saved {}'.format(fnamePCA))

# =====================================================================================
# Clustering (UMAP or t-SNE)
# =====================================================================================

#def runDimensionalityReduction(settings):
#    from dask.distributed import get_client
#    from dask.diagnostics import ResourceProfiler
#    client = get_client()
#
#    with ResourceProfiler(dt=0.5) as rprof:
#        x = client.submit(_runDimensionalityReduction, settings).result()
#        r = rprof.results

@nb.jit(nopython=True, fastmath=True)
def metricKL(x, y):
    d = 0
    # KL(x|y) + KL(y|x) [should be divided by 2 to get true average KL, but this measure need only be relative]
    for i in range(x.shape[0]):
        d += (x[i] - y[i]) * (np.log2( x[i] ) - np.log2( y[i] ))
    return d

# Jensen-Shannon
@nb.jit(nopython=True, fastmath=True)
def metricJS(x, y):
    d = 0
    for i in range(x.shape[0]):
        a = np.log2( 0.5 * y[i] + 0.5 * x[i] )
        d += x[i] * (np.log2( x[i] ) - a)
        d += y[i] * (np.log2( y[i] ) - a)
    return d

def getDimReducFilename(settings):
    settings = run_wavelet.checkSettings(settings)

    if not 'method' in settings:
        settings['method'] = 'tsne'

    if not 'fname' in settings:
        settings['fname'] = FNAME_PCA(settings) if settings['use_pca'] in ['pca','nmf'] else FNAME_HIPOW(settings)

    fnameData = settings['fname']
    fname = '{}_{}'.format(fnameData.replace('.npy', ''), settings['method'])

    if settings['method'] == 'tsne':
        return fname.replace('.npy', '') + '_{}_perplexity_{}_{}_{}_{}{}.png'.format(
            settings['use_pca'],
            settings['perplexity'],
            settings['maxN'],
            settings['n_iter'],
            settings['metric'],
            '_nohistogramnorm' if not settings['histogram_normalize'] else '')

    elif settings['method'] == 'umap':
        return fname.replace('.npy', '') + '_{}_mindist_{}_neighbors_{}_{}_{}_{}.png'.format(
            settings['use_pca'],
            int(100 * settings['min_dist']),
            settings['n_neighbors'],
            settings['maxN'],
            settings['n_iter'],
            settings['metric'],
            '_nohistogramnorm' if not settings['histogram_normalize'] else '')
    elif settings['method'] == 'agglom':
        return fname.replace('.npy', '') + '_ward.png'
    else:
        raise Exception("Unknown method specifies.")

def runDimensionalityReduction(settings):
    # Get filename info
    fnameData = settings['fname']
    fname = '{}_{}'.format(fnameData.replace('.npy',''), settings['method'])

    # Determine output filename
    fnamePlot = '\\\\?\\' + getDimReducFilename(settings).replace('/','\\')

    # Plot exists?
    if os.path.exists(fnamePlot) and not settings['overwrite']:
        logging.warn('Skipping {}'.format(fnamePlot))
        return

    if 'rawmvmt' in settings and settings['rawmvmt'] is True:
        from pipeline.python.behavioral_motifs.run_wavelet_cluster_rawmvmt import getEmbeddingData as rawmvmt_getEmbeddingData
        amplSS = rawmvmt_getEmbeddingData(settings)
    else:
        # Open file
        idx = np.load(FNAME_HIPOW_IDX(settings), mmap_mode='r')

        ampl = None
        # First try np.load, if the data was saved in this format (e.g. the PCA output)
        try:
            ampl = np.load(fnameData)
        except:
            # Try memory-mapping
            ampl = np.memmap(filename=fnameData, mode='r', dtype=np.float32)
            nrows = idx.shape[0]
            ncols = int(ampl.size / nrows)
            del ampl
            ampl = np.memmap(filename=fnameData, mode='r', shape=(nrows, ncols), dtype=np.float32)

        # Take random subset
        amplSS = ampl[np.random.choice(np.arange(0, ampl.shape[0], dtype=np.int64),
            min(ampl.shape[0], settings['maxN']), replace=False),:].copy()

        # Normalize each timepoint (see Berman et al)
        if settings['histogram_normalize']:
            amplSS /= np.repeat(np.sum(amplSS, axis=1)[:,np.newaxis], amplSS.shape[1], axis=1)
        elif settings['metric'] in ['KL','JS']:
            raise Exception('KL and JS metrics require histogram normalization.')

        del ampl
        gc.collect()

    # Prepare metric function
    # Note: For custom Numba metric functions, make sure the function does not throw errors before running all code!
    #       The error messages might otherwise not be very clear about the source of the error.
    fnMetric = settings['metric']

    if 'rawmvmt' in settings and settings['rawmvmt'] is True:
        from pipeline.python.behavioral_motifs.run_wavelet_cluster_rawmvmt import getMetricFunction as rawmvmt_getMetricFunction
        fnMetric = rawmvmt_getMetricFunction(settings)
        logging.info('Using custom rawmvmt metric.')
    else:
        logging.info('Using metric: {}'.format(fnMetric))
        if fnMetric == 'KL':
            fnMetric = metricKL
        elif fnMetric == 'JS':
            fnMetric = metricJS

    # Measure time it takes UMAP to complete
    st = time.time()

    dr, drarr = None, None
    err = ''

    try:
        if settings['method'] == 'tsne':

            # Quick t-SNE of random subset
            dr = TSNE(n_components=2,
                      perplexity=settings['perplexity'],
                      n_iter=settings['n_iter'],
                      metric=fnMetric).fit(amplSS)

            drarr = dr.transform(amplSS)

        elif settings['method'] == 'umap':

            # Quick UMAP of random subset
            dr = umap.UMAP(
                min_dist=settings['min_dist'],
                n_neighbors=settings['n_neighbors'],
                n_components=2,
                metric=fnMetric
            ).fit(amplSS)

            drarr = dr.transform(amplSS)

        elif settings['method'] == 'agglom':


            # Agglomerative clustering
            import fastcluster

            # Determine filenames
            fnameDistMtx = fnamePlot.replace('.png', '') + '_distmtx.npy'
            fnameDistMtxCompr = fnameDistMtx.replace('.npy', '') + '_compr.npy'

            DISTMTX_SHAPE = (amplSS.shape[0], amplSS.shape[0])

            if not os.path.exists(fnameDistMtxCompr):
                arrDistDst = np.memmap(fnameDistMtx, mode='w+', shape=DISTMTX_SHAPE, dtype=np.float32)
                del arrDistDst

                # Compute distance matrix
                @nb.njit(nogil=True, fastmath=True, parallel=False)
                def computeDistMtxBlock(arrDist, arr, x0, x1, y0, y1):
                    for i in range(x0, x1):
                        for j in range(y0, y1):
                            arrDist[i - x0, j - y0] = fnMetric(arr[i, :], arr[j, :])

                def computeDistMtxBlockJob(x0, x1, y0, y1):
                    try:
                        # Load output array
                        arrDist = np.full((x1 - x0, y1 - y0), np.nan, dtype=np.float32)
                        # Load input array
                        arr = np.load(fnameData, mmap_mode='r')
                        # Compute output
                        computeDistMtxBlock(arrDist, arr, x0, x1, y0, y1)
                        # Flush to disk
                        arrDistDst = np.memmap(fnameDistMtx, mode='r+', shape=DISTMTX_SHAPE, dtype=np.float32)
                        arrDistDst[x0:x1, y0:y1] = arrDist
                        del arrDistDst
                        del arrDist
                        del arr
                        # Done
                        return True
                    except Exception as e:
                        print(e)
                        return False

                result = jl.Parallel(n_jobs=40)(jl.delayed(computeDistMtxBlockJob)(
                    i * 1000, (i + 1) * 1000, 0, DISTMTX_SHAPE[1]) for i in tqdm(range(DISTMTX_SHAPE[0] // 1000)))

                # Create compressed distance matrix
                arrDist = np.memmap(fnameDistMtx, mode='r', shape=DISTMTX_SHAPE, dtype=np.float32)
                arrDistCompr = np.memmap(fnameDistMtxCompr, mode='w+',
                                         shape=(DISTMTX_SHAPE[0] * (DISTMTX_SHAPE[1] - 1) // 2,), dtype=np.float32)

                c = 0
                for i in tqdm(range(DISTMTX_SHAPE[0])):
                    _n = (DISTMTX_SHAPE[1] - i) - 1
                    arrDistCompr[c:(c + _n)] = arrDist[i, (i + 1):]
                    c += _n

                del arrDist;
                del arrDistCompr

            # Compute linkage
            arrDistCompr = np.memmap(fnameDistMtxCompr, mode='r', dtype=np.float32)
            dr = fastcluster.linkage(arrDistCompr, method='complete', preserve_input=True)
            drarr = None

    except Exception as e:
        import traceback
        err = str(e)
        tb = traceback.format_exc()
        err += ' ' + tb

    # Measure time it takes UMAP to complete
    et = time.time()

    # Plot
    if drarr is not None:
        cvs = ds.Canvas(plot_width=200, plot_height=200)
        agg = cvs.points(pd.DataFrame(drarr, columns=['x', 'y']), 'x', 'y')
        img = tf.shade(agg, how='eq_hist')

        # Rescale img
        imgr = np.array(img.to_pil()).astype(np.double)
        mask = (imgr[:, :, 3] == 0)
        imgr = imgr[:, :, 0]
        imgr[mask] = 255
        imgr -= imgr.min()
        imgr /= (imgr.max() / 255.0)
        imgr = imgr.astype(np.uint8)

        imageio.imsave(fnamePlot, imgr)

    # Aggregate and save settings / log information
    duration = int(et - st)

    log = {}
    log['settings'] = settings
    log['duration'] = duration
    log['wavelet_shape'] = list(amplSS.shape)

    if len(err) > 0:
        log['error'] = err

    with open(fnamePlot.replace('.png','') + '_log.json', 'w') as f:
        json.dump(log, f, sort_keys=True, indent=4)

    # Save dimensionality reduction result
    if drarr is not None:
        np.save(fnamePlot.replace('.png', '') + '.npy', drarr)

    # Save reducer state (so more data can be transformed without re-computing tSNE/UMAP model)
    jl.dump(dr, fnamePlot.replace('.png', '') + '.pickle')

    # Get profile information
    #from dask.distributed import get_client, get_worker
    #client = get_client()
    #pr = client.profile(key='runDimensionalityReduction', start=st, stop=et,
    #                    filename='C:/Users/acorver/Desktop/test.html')
    #get_worker().close()

# =====================================================================================
# Main clustering function
# =====================================================================================

def runClustering(settings):

    settings = run_wavelet.checkSettings(settings)

    # Run UMAP
    configs = []

    SETTINGS_METHOD     = ['umap',] #'tsne',
    SETTINGS_N          = [200000,] # [1000, 10000, 100000, 1000000, 5000000]
    SETTINGS_PERPLEXITY = [100, 500] #, 1000]
    SETTINGS_MINDIST    = [0.1,]
    SETTINGS_NEIGHBORS  = [20, 100, 500]

    USE_DASK = False

    if 'tsne_perplexities' in settings:
        SETTINGS_PERPLEXITY = settings['tsne_perplexities']

    n_iter = 5000
    if 'n_iter' in settings:
        n_iter = settings['n_iter']

    if 'method' in settings:
        if isinstance(settings['method'], list):
            SETTINGS_METHOD = settings['method']
        else:
            SETTINGS_METHOD = [settings['method'],]

    if 'maxN' in settings:
        SETTINGS_N = [settings['maxN'],]

    for N in SETTINGS_N:
        s = copy.copy(settings)
        s['fname'] = FNAME_PCA(s) if s['use_pca']!='no-pca' else FNAME_HIPOW(s)
        s['maxN'] = N
        s['n_iter'] = n_iter

        logging.info('Using file {}'.format(s['fname']))

        for method in SETTINGS_METHOD:
            s['method'] = method

            if method == 'umap':
                for min_dist in SETTINGS_MINDIST:
                    for neighbors in SETTINGS_NEIGHBORS:
                        s['min_dist']  = min_dist
                        s['n_neighbors'] = neighbors
                        configs.append( copy.copy(s) )

            elif method == 'tsne':

                for perplexity in SETTINGS_PERPLEXITY:
                    s['perplexity'] = perplexity
                    configs.append(copy.copy(s))

            elif method == 'agglom':
                configs.append(s)

    # DEBUG
    #configs = [[x for x in configs if 'dlc' in x['algorithm']][0],]
    #configs = [configs[0],]

    # Randomize execution order to balance task sizes
    np.random.shuffle(configs)

    # If only one task is present, run it in this process (makes debugging easier)
    if (len(configs) == 1 and ('parallel' not in settings or not settings['parallel'])) or ('parallel' in settings and not settings['parallel']):
        for c in configs:
            runDimensionalityReduction(c)

    elif False and USE_DASK:
        # If multiple tasks are present, dispatch them to the Dask server, which will continue running these tasks
        # and exit this script. This allows multiple jobs to be dispatched from this same script.
        from dask.distributed import Client, fire_and_forget
        client = Client('localhost:8786')
        for c in configs:
            fire_and_forget(client.submit(runDimensionalityReduction, c))

    else:
        # Otherwise, fire a new commmand prompt for every task
        import subprocess
        for c in configs:
            cf = { 'command': 'runDimensionalityReduction', 'params': c }
            cmd = '{} {} {}'.format(
                os.path.abspath(os.path.join(os.__file__, os.pardir, os.pardir, 'python.exe')),
                __file__,
                json.dumps(cf).replace("'",'"').replace('"','\\"') )
            print("Starting shell....")
            print(cmd)

            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            global NEXT_NUMA_NODE
            _p = subprocess.Popen('start /node {} '.format(NEXT_NUMA_NODE) + cmd, shell=True, close_fds=True,
                             creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
            # For the next process, use the opposite NUMA core. This equally divides processes across NUMA cores.
            NEXT_NUMA_NODE = 1 - NEXT_NUMA_NODE
            _tmp = 1 + 1

    # To-Do: Find a way to monitor and save per-task CPU/memory info
    #from dask.diagnostics import ResourceProfiler
    #with ResourceProfiler(dt=0.25) as rprof:
    #f.result()
    #x = rprof.results

# =====================================================================================
# CLI Command Handler
# =====================================================================================

def handleCommand():
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
        if cmd['command'] == 'runDimensionalityReduction':
            s = cmd['params']
            runDimensionalityReduction(s)

        # Return True to indicate no further processing is required
        return True
    except Exception as e:
        import traceback
        print(e)
        traceback.print_exception(e)
        print('Unable to parse command.')
        time.sleep(100)
        return True

# =====================================================================================
# Entry Point 1
# =====================================================================================

def run1():

    allSettings = []

    parallel = True
    overwrite = False

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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
    for rec in ['Z:/behavior/6-23-19-e', 'Z:/behavior/4-12-19-b', 'Z:/behavior/4-9-19-a']:
        for x in range(len(allSettings)):
            allSettings[x]['fname_base'] = os.path.join(rec, 'embeddings/')
            allSettings[x]['recordings'] = [rec, ]
            allSettings[x]['highpower_rows_per_file'] = 200000

        allSettings = allSettings[::-1]

        # Check input
        print('Num. settings: {}'.format(len(allSettings)))
        if not handleCommand():
            # Run each of the settings
            for settings in tqdm(allSettings):
                try:
                    if not 'rawmvmt' in settings or settings['rawmvmt'] is False:
                        # Run wavelet transforms
                        runWavelet(settings)

                        # Merge wavelets
                        runWaveletMerge(settings)

                        # Filter by high power
                        runWaveletHighPower(settings)

                    # Dimensionality reduction
                    #runPCA(settings)

                    # Dimensionality reduction
                    runClustering(settings)
                except Exception as e:
                    print(e)
                    print(settings)
                    # Continue
                    pass

# =====================================================================================
# Entry Point 2
# =====================================================================================

def run2():

    allSettings = []

    parallel = True
    overwrite = False

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 0,
        'rawmvmt_meansubtract': 'nomeansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 0,
        'rawmvmt_meansubtract': 'nomeansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'nomeansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'nomeansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 0,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 0,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'noscalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'noscalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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
    """
    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 60,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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
    """;

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 100,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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

    allSettings.append({
        'rawmvmt': True,
        'rawmvmt_duration': 100,
        'rawmvmt_maxoffset': 16,
        'rawmvmt_meansubtract': 'meansub',
        'rawmvmt_scalestd': 'scalestd',
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22],
        # Overwrite?
        'overwrite': overwrite,
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

    # Check input
    print('Num. settings: {}'.format(len(allSettings)))
    if not handleCommand():
        # Run each of the settings
        for settings in tqdm(allSettings):
            try:
                if not 'rawmvmt' in settings or settings['rawmvmt'] is False:
                    # Run wavelet transforms
                    runWavelet(settings)

                    # Merge wavelets
                    runWaveletMerge(settings)

                    # Filter by high power
                    runWaveletHighPower(settings)

                # Dimensionality reduction
                #runPCA(settings)

                # Dimensionality reduction
                runClustering(settings)
            except Exception as e:
                print(e)
                print(settings)
                # Continue
                pass


# =====================================================================================
# Entry Point 3
# =====================================================================================

def run3():

    allSettings = []

    parallel = True
    overwrite = False

    allSettings.append({
        'rawmvmt': False,
        'algorithm': 'dlc',
        # Include only front and back legs
        'joints': [6, 10, 18, 22, 9, 13, 21, 25],
        # Overwrite?
        'overwrite': overwrite,
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
    for rec in ['Z:/behavior/6-23-19-e', 'Z:/behavior/4-12-19-b', 'Z:/behavior/4-9-19-a']:
        for x in range(len(allSettings)):
            allSettings[x]['fname_base'] = os.path.join(rec, 'embeddings/')
            allSettings[x]['recordings'] = [rec, ]
            allSettings[x]['highpower_rows_per_file'] = 200000

        allSettings = allSettings[::-1]

        # Check input
        print('Num. settings: {}'.format(len(allSettings)))
        if not handleCommand():
            # Run each of the settings
            for settings in tqdm(allSettings):
                try:
                    if not 'rawmvmt' in settings or settings['rawmvmt'] is False:
                        # Run wavelet transforms
                        runWavelet(settings)

                        # Merge wavelets
                        runWaveletMerge(settings)

                        # Filter by high power
                        runWaveletHighPower(settings)

                    # Dimensionality reduction
                    #runPCA(settings)

                    # Dimensionality reduction
                    runClustering(settings)
                except Exception as e:
                    print(e)
                    print(settings)
                    # Continue
                    pass

# =====================================================================================
# Entry Point
# =====================================================================================

if __name__ == "__main__":
    #run1()
    #run2()
    run3()