
import os, gc, glob, numpy as np, joblib as jl, imageio
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

def filterClusters(cl, minrun = 12):
    runs = []
    for c in cl:
        if len(runs) == 0:
            runs.append([c, ])
        elif runs[-1][-1] == c:
            runs[-1].append(c)
        else:
            runs.append([c, ])
    return np.hstack([(x if len(x) >= minrun else [np.nan,] * len(x)) for x in runs])


# Sample event points that are at least 2 seconds apart
def sampleEvent(arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames, fnameIdx, clusterID, numEvents=10):
    eventOccurrences = np.array([], dtype=int)
    eventOccCandidates = np.argwhere(arrClustersFilt[fnameIdx] == clusterID).T[0]

    if eventOccCandidates.size == 0:
        return np.zeros((50, 120, 120), dtype=np.uint8)

    for i in range(10000):
        if len(eventOccurrences) > numEvents:
            break
        else:
            e = np.random.choice(eventOccCandidates, 1)
            if len(eventOccurrences) == 0 or np.min(np.abs(eventOccurrences - e)) > 100:
                eventOccurrences = np.append(eventOccurrences, e)

    eventOccurrencesAll = np.array([], dtype=int)
    for eo in eventOccurrences:
        eventOccurrencesAll = np.hstack((eventOccurrencesAll, np.arange(eo - 30, eo + 30)))

    _idx = np.load(fnamesCroprotIdx[fnameIdx])

    if arrClustersFilt[fnameIdx].shape[0] != np.sum(_idx):
        print(fnamesCroprotIdx[fnameIdx])
        print(fnames[fnameIdx])
        print(np.sum(_idx), arrClustersFilt[fnameIdx].shape)
        raise Exception('Files not right size?')

    eventOccurrencesAll = np.clip(eventOccurrencesAll, 0, np.sum(_idx) - 1)

    _idxs = np.argwhere(_idx)[eventOccurrencesAll, 0]
    isCluster = (arrClustersFilt[fnameIdx][eventOccurrencesAll] == clusterID)

    mm = np.memmap(fnamesCroprot[fnameIdx], mode='r', dtype=np.uint8, shape=(_idx.size, 200, 200))
    if mm.shape[0] != _idx.size:
        raise Exception('Croprot not right size?')

    # _idxs = np.clip(_idxs, 0, mm.shape[0] - 1)
    # import pdb; pdb.set_trace()

    return mm[_idxs, 40:160, 40:160] * (0.7 + 0.3 * isCluster[:, np.newaxis, np.newaxis])

def sampleEventVideo(arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames, clusterID, numEvents=50, grid=True):
    anis = [sampleEvent(arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames,
                        fnameIdx, clusterID=clusterID, numEvents=numEvents) for \
            fnameIdx in tqdm(range(len(fnames)), leave=False)]

    if not grid:
        return anis
    else:
        nrows = 5
        ncols = 5
        vid = np.zeros((np.max([x.shape[0] for x in anis]), nrows * 120, ncols * 120), dtype=np.uint8)
        for i in range(len(anis)):
            nrow = int(i // ncols)
            ncol = int(i % ncols)
            vid[:anis[i].shape[0], (nrow * 120):(nrow * 120 + 120), (ncol * 120):(ncol * 120 + 120)] = anis[i]
        vid = np.clip(np.repeat(vid[:, :, :, np.newaxis], 3, axis=3).astype(int) * 2, 0, 255).astype(np.uint8)

        return vid

# Export all clusters
def saveEventVideo(arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames,
                   dirname, fnamesShort, fnameIdx, clusterID, numEvents = 50):
    vid = sampleEvent(arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames,
                      fnameIdx, clusterID=clusterID, numEvents = numEvents)
    if vid.shape[0] < 200:
        return
    fnameOut = 'Y:/wavelet/clips/{}/{}_{}.mp4'.format(
        dirname, fnamesShort[fnameIdx], clusterID)
    os.makedirs(os.path.dirname(fnameOut), exist_ok=True)
    wr = imageio.get_writer(fnameOut, fps = 50)
    for i in range(vid.shape[0]):
        wr.append_data(np.repeat(vid[i,:,:,np.newaxis], 3, axis=2))
    wr.close()


def sampleEventVideoToFile(arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames, dirname, clusterID, numEvents=50):
    if '/wavelet/' in fnameBase:
        fnameOut = 'Y:/wavelet/clips/{}/cluster_{}.mp4'.format(dirname, clusterID)
    else:
        fnameOut = 'Y:/wavelet/clips/{}/cluster_{}.mp4'.format(dirname, clusterID)

    os.makedirs(os.path.dirname(fnameOut), exist_ok=True)
    if not os.path.exists(fnameOut):
        wr = imageio.get_writer(fnameOut, fps=50)
        vid = sampleEventVideo(arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames,
                               clusterID=clusterID, numEvents=numEvents)
        for i in range(vid.shape[0]):
            wr.append_data(vid[i, :, :, :])
        wr.close()

def run(fnameBase, subsetModulo = (0, 5)):
    fnames = glob.glob(fnameBase)
    fnames = [x for x in fnames if '190528' not in x]

    arrClusters = [np.load(fname) for fname in tqdm(fnames, leave=False)]

    # Convert single clusters into the shared space
    arrClustersFilt = []
    for i in tqdm(range(len(arrClusters)), leave=False):
        arrClustersFilt.append(filterClusters(arrClusters[i][:, 0].astype(int), 12))

    del arrClusters
    gc.collect()

    clEx = np.unique(np.hstack(arrClustersFilt)[~np.isnan(np.hstack(arrClustersFilt))]).astype(int)

    fnamesCroprotIdx = [glob.glob(os.path.abspath(os.path.join(os.path.dirname(fname),
        '../croprot/*_dlc_abs_filt_interp_mvmt_noborder.idx.npy')))[0] for fname in fnames]

    fnamesCroprot = [glob.glob(os.path.abspath(os.path.join(os.path.dirname(fname),
        '../croprot/*_img.npy')))[0] for fname in fnames]

    fnamesShort = [y[:y.find('/')] for y in [x[len('Z:/behavior/'):].replace('\\', '/') for x in fnames]]

    dirname = os.path.basename(fnameBase)
    dirname = dirname[:dirname.find('_hipow')]

    clExShuff = clEx.astype(int).copy()
    np.random.shuffle(clExShuff)

    # Subset
    if subsetModulo is not None:
        clExShuff = [x for x in clExShuff if (x % subsetModulo[1]) == subsetModulo[0]]
        print('Subsetted to {}: {}'.format(len(clExShuff), clExShuff))

    jl.Parallel(n_jobs=1)(jl.delayed(sampleEventVideoToFile)(
        arrClustersFilt, fnamesCroprotIdx, fnamesCroprot, fnames, dirname,
        clusterID, 100) for clusterID in tqdm(clExShuff))

if __name__ == "__main__":
    fnameBases = [
        'Z:/behavior/*/wavelet/rawmvmt_dlc_euclidean-midline_no-abspos_no-vel_00000000010001000000010001_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.clusters.npy',
        'Z:/behavior/*/wavelet/rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.clusters.npy'
    ]

    Njobs = 20

    for fnameBase in fnameBases:
        jl.Parallel(Njobs, prefer='processes')(jl.delayed(run)(
            fnameBase, subsetModulo=(k, Njobs)) for k in range(Njobs))