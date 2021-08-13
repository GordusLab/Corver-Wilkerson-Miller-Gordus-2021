

# ======================================================================================================================
# Choose recording and temporal offset
# ======================================================================================================================

#recID, stage, plotStartIdx = 4, 'spiral_cap', 35000
recID, stage, plotStartIdx = 4, 'radii', 10000

aniDur = 3000
aniStep = 2

# ...
colorTime = False
T = ''

# ======================================================================================================================
# Imports
# ======================================================================================================================

import pandas as pd, numpy as np, matplotlib.pyplot as plt, gc, joblib as jl, imageio, os, skimage.draw
from matplotlib import cm
import matplotlib as mpl
from tqdm import tqdm
import colorcet, glob, json, copy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import scipy.signal

# ======================================================================================================================
# Export npy to avi?
# ======================================================================================================================

fnameNpy = 'C:/Users/acorver/Desktop/animation_{}_{}_{}.npy'.format(stage, recID, plotStartIdx)
fnameMp4 = 'C:/Users/acorver/Desktop/animation_{}_{}_{}.mp4'.format(stage, recID, plotStartIdx)
if os.path.exists(fnameNpy) and not os.path.exists(fnameMp4):
    imgs = np.load(fnameNpy)
    wr = imageio.get_writer(fnameMp4, fps=25)
    for img in imgs:
        wr.append_data(img)
    wr.close()
    exit(0)

# ======================================================================================================================
# Load data
# ======================================================================================================================

def loadJSON(x):
    with open(x, 'r') as f:
        return json.load(f)

recordingInfo = [{**loadJSON(x), **{'fnameJSON': x}} for x in glob.glob('Z:/behavior/*/recording.json')]
recordingInfo = [x for x in recordingInfo if x['web_complete'] and x['tracking_successful'] and 'stages' in x]

# Does this recording.json file specify stage ranges, or starting points?
s = copy.deepcopy(recordingInfo[recID])
if isinstance(s['stages']['protoweb'], list):
    for st in s['stages']:
        try:
            if not isinstance(s['stages'][st][0], list):
                s['stages'][st] = [s['stages'][st], ]
        except:
            s['stages'][st] = []
else:
    # Add the end of the recording
    a = np.load(s['fname'], mmap_mode='r')
    s['stages']['end'] = a.shape[0]

    if 'stabilimentum' in s['stages']:
        if s['stages']['stabilimentum'] >= 0:
            pass
        else:
            s['stages']['stabilimentum'] = s['stages']['end']
    else:
        s['stages']['stabilimentum'] = s['stages']['end']

    # Now convert to ranges
    s['stages']['protoweb'] = [[s['stages']['protoweb'], s['stages']['radii']], ]
    s['stages']['radii'] = [[s['stages']['radii'], s['stages']['spiral_aux']], ]
    s['stages']['spiral_aux'] = [[s['stages']['spiral_aux'], s['stages']['spiral_cap']], ]
    s['stages']['spiral_cap'] = [[s['stages']['spiral_cap'], s['stages']['stabilimentum']], ]
    s['stages']['stabilimentum'] = [[s['stages']['stabilimentum'], s['stages']['end']], ]
    del s['stages']['end']

# Load original data
fnamePos = glob.glob(os.path.abspath(os.path.join(os.path.dirname(
    recordingInfo[recID]['fnameJSON']), 'croprot/*dlc_position_orientation.npy')))[0]
arr = np.load(fnamePos)

# Convert to indices used in analysis
arrIdx = np.load(fnamePos.replace('_position_orientation.npy', '_abs_filt_interp_mvmt_noborder.idx.npy'))
for st in s['stages']:
    for k in range(len(s['stages'][st])):
        for m in range(2):
            s['stages'][st][k][m] = np.argmin(np.abs(np.argwhere(arrIdx).T[0] - s['stages'][st][k][m]))

recInfo = s

# Subset by index
arrIdx = np.load(fnamePos.replace('_position_orientation.npy',
                                  '_abs_filt_interp_mvmt_noborder.idx.npy'))
arr = arr[arrIdx, :]

fnameTsneP = '\\\\?\\Y:\\wavelet\\rawmvmt_dlc_euclidean-midline_no-abspos_no-vel_00000000010001000000010001_60_16_meansub_scalestd\\rawmvmt_dlc_euclidean-midline_no-abspos_no-vel_00000000010001000000010001_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.npy'

#fnameTsneP = os.path.abspath(os.path.join(os.path.dirname(recInfo['fnameJSON']),
#    'wavelet/rawmvmt_dlc_euclidean-midline_no-abspos_no-vel_00000000010001000000010001_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.filtered2.npy'))

fnameTsneA = '\\\\?\\Y:\\wavelet\\rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd\\rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.npy'

#fnameTsneA = os.path.abspath(os.path.join(os.path.dirname(recInfo['fnameJSON']),
#    'wavelet/rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.filtered2.npy'))

arrTsneP = np.load(fnameTsneP)[:,0:2]
arrTsneP = arrTsneP[~np.any(np.isnan(arrTsneP), axis=1)]
arrTsneP -= np.min(arrTsneP, axis=0)
arrTsneP /= np.max(arrTsneP, axis=0)
imgDensityP = np.histogram2d(arrTsneP[:,0], arrTsneP[:,1], bins=(200,200), range=((0,1),(0,1)))[0]
imgDensityP = np.clip(imgDensityP, 0, 80)

arrTsneA = np.load(fnameTsneA)[:,0:2]
arrTsneA = arrTsneA[~np.any(np.isnan(arrTsneA), axis=1)]
arrTsneA -= np.min(arrTsneA, axis=0)
arrTsneA /= np.max(arrTsneA, axis=0)
imgDensityA = np.histogram2d(arrTsneA[:,0], arrTsneA[:,1], bins=(200,200), range=((0,1),(0,1)))[0]
imgDensityA = np.clip(imgDensityA, 0, 80)

fnameP = 'rawmvmt_dlc_euclidean-midline_no-abspos_no-vel_00000000010001000000010001_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.clusters.npy'
fnameA = 'rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd_hipow_tsne_no-pca_perplexity_100_200000_2000_euclidean.clusters.npy'

arrClusters = [np.load(os.path.abspath(os.path.join(os.path.dirname(
    fnamePos), '../wavelet/', x))) for x in [fnameP, fnameA]]

stageIdx = np.full(arr.shape[0], -1, dtype=int)
STAGES = ['protoweb', 'radii', 'spiral_aux', 'spiral_cap', 'stabilimentum']
for st in recInfo['stages']:
    for i0, i1 in recInfo['stages'][st]:
        stageIdx[i0:i1] = STAGES.index(st)


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

ANTERIOR = 1
POSTERIOR = 0

fnameClusterLabelsA = '\\\\?\\Y:\\wavelet\\clips\\rawmvmt_dlc_euclidean_no-abspos_no-vel_00000010001000000010001000_60_16_meansub_scalestd\\cluster_names.txt'
fnameClusterLabelsP = '\\\\?\\Y:\\wavelet\\clips\\rawmvmt_dlc_euclidean-midline_no-abspos_no-vel_00000000010001000000010001_60_16_meansub_scalestd\\cluster_names.txt'
fnamesLabels = [None, None]
fnamesLabels[ANTERIOR] = fnameClusterLabelsA
fnamesLabels[POSTERIOR] = fnameClusterLabelsP

clusterLabels = (loadLabels(fnamesLabels[0]), loadLabels(fnamesLabels[1]))
clusterLabelsUnique = list(set(list(clusterLabels[0].keys()) + list(clusterLabels[1].keys())))
clusterLabelsUnique = [x for x in clusterLabelsUnique if x not in ['noisy', ]]

def filterClusters(cl, clusterLabels, minrun=12):
    runs = []
    for c in cl:
        if len([k for k in clusterLabels if c in clusterLabels[k]]) == 0:
            c = -1
        else:
            _k = [k for k in clusterLabels if c in clusterLabels[k]][0]
            if _k in clusterLabelsUnique:
                c = clusterLabelsUnique.index(_k)
            else:
                c = -1

        if len(runs) == 0:
            runs.append([c, ])
        elif runs[-1][-1] == c:
            runs[-1].append(c)
        else:
            runs.append([c, ])
    return np.hstack([(x if len(x) >= minrun else [np.nan, ] * len(x)) for x in runs])

arrClustersMappedP = filterClusters(arrClusters[POSTERIOR][:,0].astype(int), clusterLabels[0], 12)
arrClustersMappedA = filterClusters(arrClusters[ANTERIOR][:,0].astype(int), clusterLabels[1], 12)

allData = pd.DataFrame({
    'x': arr[:,0],
    'y': arr[:,1],
    'theta': arr[:,2],
    'tsne_a_x': arrClusters[ANTERIOR][:, -5],
    'tsne_a_y': arrClusters[ANTERIOR][:, -4],
    'tsne_p_x': arrClusters[POSTERIOR][:, -5],
    'tsne_p_y': arrClusters[POSTERIOR][:, -4],
    'cluster_a': arrClusters[ANTERIOR][:, 0],
    'cluster_p': arrClusters[POSTERIOR][:, 0],
    'stage': stageIdx,
    'clusterManual_a': [(clusterLabelsUnique[int(x)] if x>=0 else np.nan) for x in arrClustersMappedA],
    'clusterManual_p': [(clusterLabelsUnique[int(x)] if x>=0 else np.nan) for x in arrClustersMappedP],
})
d = allData[(~pd.isnull(allData.tsne_a_x))|(~pd.isnull(allData.tsne_p_x))]

_fn = (lambda a: np.min(a[a >= 0]) if np.any(a >= 0) else np.nan)
_fn2 = (lambda a: -np.max(a[a <= 0]) if np.any(a <= 0) else np.nan)

# Detect radii starts
xy = d.loc[:, ['x', 'y']].fillna(method='ffill').values
polyPerim = Polygon([(0,0), (0,1024), (1024,1024), (1024, 0)])
perimDist = np.array([polyPerim.exterior.distance(Point(x, y)) for x, y in xy])
idxPeak, _ = scipy.signal.find_peaks(perimDist, distance=50, height=100)
idxPeak = idxPeak[scipy.signal.peak_prominences(perimDist, idxPeak)[0] > 100]
radiusIdxs = d.index[idxPeak]

d.loc[:,'timeSinceRadius'] = [_fn(x-np.array(radiusIdxs)) for x in tqdm(d.index, leave=False)]
d.loc[:,'timeToRadius'] = [_fn2(x-np.array(radiusIdxs)) for x in tqdm(d.index, leave=False)]
d.loc[:,'eventDurationRadius'] = d.timeSinceRadius + d.timeToRadius

# Abdomen events
abdomenIdxs = d.index[[('bend-abdomen' in str(x)) for x in d.clusterManual_a]]
abdomenIdxs1 = abdomenIdxs[d.tsne_a_x[abdomenIdxs] < 0.5]
abdomenIdxs2 = abdomenIdxs[d.tsne_a_x[abdomenIdxs] > 0.5]
d.loc[:,'timeSinceAttachline1'] = [_fn(x-np.array(abdomenIdxs1)) for x in tqdm(d.index, leave=False)]
d.loc[:,'timeSinceAttachline2'] = [_fn(x-np.array(abdomenIdxs2)) for x in tqdm(d.index, leave=False)]
d.loc[:,'timeToAttachline'] = [_fn2(x-np.array(abdomenIdxs)) for x in tqdm(d.index, leave=False)]
d.loc[:,'eventDuration'] = np.minimum(d.timeSinceAttachline1, d.timeSinceAttachline2) + d.timeToAttachline

clusterManual_a_unique = d.clusterManual_a.unique().tolist()
clusterManual_a_unique = [x for x in clusterManual_a_unique if not pd.isnull(x)]

if T == '1':
    d.loc[:,'timeSinceAttachline'] = d.timeSinceAttachline1
    d = d[d.timeSinceAttachline1 < d.timeSinceAttachline2]
elif T == '2':
    d.loc[:,'timeSinceAttachline'] = d.timeSinceAttachline2
    d = d[d.timeSinceAttachline2 < d.timeSinceAttachline1]
else:
    d.loc[:,'timeSinceAttachline'] = np.minimum(d.timeSinceAttachline1, d.timeSinceAttachline2)

# Get movie frames
fnameCroprot    = fnamePos.replace('_dlc_position_orientation.npy', '_img.npy')
fnameCroprotIdx = fnamePos.replace('_dlc_position_orientation.npy',
                                   '_dlc_abs_filt_interp_mvmt_noborder.idx.npy')
idxCroprot = np.load(fnameCroprotIdx)
imgCroprot = np.memmap(fnameCroprot, mode='r', dtype=np.uint8, shape=(idxCroprot.size, 200, 200))

def copyFrame(fr):
    img = np.zeros(fr.shape, dtype=fr.dtype)
    img[:,:] = fr
    return img

CLUSTER_COLORS = {
    'right-leg': '#a7dcfa',
    'left-leg': '#f0c566',
    'one-leg-after-other': '#7a988c',
    'both-legs': '#a0718b',
    'walk': '#66a3d3',
    'bend-abdomen': '#e69e66',
    'stabilimentum': '#e0afca',
    'extrude': '#66c5ab',
    'extrude-slow': '#f6ef8e',
    'stationary': '#999999',
    'stationary-anterior': '#c2c2c2',
    'stationary-posterior': '#c2c2c2'
}

CLUSTER_LABELS = {
    'right-leg': 'Left Leg',
    'left-leg': 'Right Leg',
    'one-leg-after-other': 'Alternating Legs',
    'both-legs': 'Both Legs (Rotation)',
    'walk': 'Walk',
    'bend-abdomen': 'Anchor',
    'stabilimentum': 'Stabilimentum',
    'extrude': 'Silk pull (Fast)',
    'extrude-slow': 'Silk pull (Slow)',
    'stationary': 'Stationary',
    'stationary-anterior': 'Stationary Anterior',
    'stationary-posterior': 'Stationary Posterior'
}

CIRCLE_MASK = np.full((200, 200), False, dtype=np.bool)
rr, cc = skimage.draw.circle(100, 100, 80, shape=(200, 200))
CIRCLE_MASK[rr, cc] = True

def plot(idxStart, idxDur, imgSpider, d, stage, clusterManual_a_unique,
         imgCroprot, idxCroprot, colorTime=False, aniStart=0, aniEnd=0):
    # Compute X,Y bounding box of entire animation
    idxAni = d.index[aniStart:aniEnd]
    xyAni = d.loc[idxAni, ['x', 'y']].values
    xyMin = np.nanmin(xyAni, axis=0) - 10
    xyMax = np.nanmax(xyAni, axis=0) + 10

    # Idx
    idx = d.index[idxStart:(idxStart + idxDur)]

    tsneAx = d.loc[idx, 'tsne_a_x'].fillna(method='ffill').values
    tsneAy = d.loc[idx, 'tsne_a_y'].fillna(method='ffill').values
    tsnePx = d.loc[idx, 'tsne_p_x'].fillna(method='ffill').values
    tsnePy = d.loc[idx, 'tsne_p_y'].fillna(method='ffill').values

    # Plot
    colsA, colsP = None, None
    if colorTime:
        cols = None
        if stage != 'radii':
            cols = ["#{0:02x}{1:02x}{2:02x}".format(x[0], x[1], x[2]) for x in cm.jet(
                d.loc[idx, 'timeSinceAttachline'] / d.loc[idx, 'eventDuration'], bytes=True)]
        else:
            cols = ["#{0:02x}{1:02x}{2:02x}".format(x[0], x[1], x[2]) for x in cm.jet(
                d.loc[idx, 'timeSinceRadius'] / d.loc[idx, 'eventDurationRadius'], bytes=True)]
        colsA = cols
        colsP = cols
    else:
        colsA = [(CLUSTER_COLORS[x] if x in CLUSTER_COLORS else '#222222') for \
                 x in d.clusterManual_a[idx]]
        colsP = [(CLUSTER_COLORS[x] if x in CLUSTER_COLORS else '#222222') for \
                 x in d.clusterManual_p[idx]]

    fig, ax = plt.subplots(2, 3, figsize=(20, 12), facecolor='black')
    fig.subplots_adjust(hspace=0, wspace=0)

    # tSNE A
    ax[1][0].plot(tsneAy, tsneAx, color='black', alpha=0.5)
    ax[1][0].scatter(tsneAy, tsneAx, color=colsA, alpha=0.6, s=25, zorder=100)
    ax[1][0].scatter(tsneAy[-1], tsneAx[-1], color='red', s=700, marker='*', zorder=100)
    ax[1][0].imshow(np.flip(imgDensityA, axis=0), cmap='gray', extent=(0, 1, 0, 1), vmax=np.max(imgDensityA) * 1.5)
    ax[1][0].set_xlim(0.15, 0.85)
    ax[1][0].set_ylim(0.15, 0.85)
    ax[1][0].set_axis_off()

    # tSNE P
    ax[1][1].plot(tsnePy, tsnePx, color='black', alpha=0.5)
    ax[1][1].scatter(tsnePy, tsnePx, color=colsP, alpha=0.6, s=25, zorder=100)
    ax[1][1].scatter(tsnePy[-1], tsnePx[-1], color='red', s=700, marker='*', zorder=100)
    ax[1][1].imshow(np.flip(imgDensityP, axis=0), cmap='gray', extent=(0, 1, 0, 1), vmax=np.max(imgDensityP) * 1.5)
    ax[1][1].set_xlim(0.15, 0.85)
    ax[1][1].set_ylim(0.15, 0.85)
    ax[1][1].set_axis_off()

    # Web Anterior
    x = d.loc[idx, 'x'].values
    y = d.loc[idx, 'y'].values
    ax[0][0].scatter(x, y, color=colsA, s=10)
    ax[0][0].scatter(x[-1], y[-1], color='red', s=700, marker='*')
    # ax[0][0].plot(x, y, color='black', alpha=0.5)
    ax[0][0].set_xlim(xyMin[0], xyMax[0])
    ax[0][0].set_ylim(xyMin[1], xyMax[1])
    ax[0][0].set_axis_off()

    # Web Posterior
    x = d.loc[idx, 'x'].values
    y = d.loc[idx, 'y'].values
    ax[0][1].scatter(x, y, color=colsP, s=10)
    ax[0][1].scatter(x[-1], y[-1], color='red', s=700, marker='*')
    # ax[0][1].plot(x, y, color='black', alpha=0.5)
    ax[0][1].set_xlim(xyMin[0], xyMax[0])
    ax[0][1].set_ylim(xyMin[1], xyMax[1])
    ax[0][1].set_axis_off()

    # Spider
    circleSpider = imgSpider.copy()
    circleSpider[~CIRCLE_MASK] = 0
    ax[0][2].imshow(circleSpider, cmap='gray')
    ax[0][2].set_axis_off()

    # Set titles, etc.
    ax[0][0].text(0.5, 0.95, 'Trajectory (Anterior Clusters)',
                  color='white', ha='center', fontsize=22, transform=ax[0][0].transAxes)
    ax[0][1].text(0.5, 0.95, 'Trajectory (Posterior Clusters)',
                  color='white', ha='center', fontsize=22, transform=ax[0][1].transAxes)
    ax[0][2].text(0.5, 0.95, 'Cropped/Rotated Image',
                  color='white', ha='center', fontsize=22, transform=ax[0][2].transAxes)
    ax[0][2].text(0.5, 0.05, '(ventral view)', alpha=0.5,
                  color='white', ha='center', fontsize=18, transform=ax[0][2].transAxes)
    ax[1][0].text(0.5, 0.95, 'Anterior t-SNE Embedding',
                  color='white', ha='center', fontsize=22, transform=ax[1][0].transAxes)
    ax[1][1].text(0.5, 0.95, 'Posterior t-SNE Embedding',
                  color='white', ha='center', fontsize=22, transform=ax[1][1].transAxes)

    # Legend
    clInUse = [d.clusterManual_a[idx].values[-1], d.clusterManual_p[idx].values[-1]]
    patches = [mpatches.Patch(facecolor=CLUSTER_COLORS[x],
                              label=CLUSTER_LABELS[x], edgecolor='white' if x in clInUse else None,
                              linewidth=5, linestyle='--') for x in [
                   'right-leg', 'left-leg', 'one-leg-after-other', 'both-legs',
                   'walk', 'bend-abdomen', 'extrude-slow', 'extrude', 'stabilimentum',
                   'stationary', 'stationary-anterior', 'stationary-posterior'
               ]]
    lgnd = ax[1][2].legend(handles=patches,
                           prop={'size': 24}, frameon=False, bbox_to_anchor=[0.8, 1])
    for text in lgnd.get_texts():
        text.set_color("white")
    ax[1][2].set_axis_off()

    # Done
    fig.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    del fig
    gc.collect()
    return data

imgs = []

# Get indices to plot
idxStart = np.argwhere(d.stage==STAGES.index(stage))[plotStartIdx, 0]
idxDur = 3000

# Generate images
imgs = jl.Parallel(n_jobs=1)(jl.delayed(plot)(
    idxStart + i, idxDur,
    copyFrame(imgCroprot[np.argwhere(idxCroprot)[d.index[idxStart + i + idxDur - 1], 0]].copy()),
    d, stage, clusterManual_a_unique, imgCroprot, idxCroprot,
    colorTime=colorTime, aniStart=idxStart, aniEnd=idxStart+aniDur+idxDur) for i in tqdm(range(0, aniDur, aniStep)))

# First export to raw as backup if mp4 fails
np.save('C:/Users/acorver/Desktop/animation_{}_{}_{}.npy'.format(stage, recID, plotStartIdx), imgs)

wr = imageio.get_writer('C:/Users/acorver/Desktop/animation_{}_{}_{}.mp4'.format(stage, recID, plotStartIdx), fps=25)
for img in imgs:
    wr.append_data(img)
wr.close()