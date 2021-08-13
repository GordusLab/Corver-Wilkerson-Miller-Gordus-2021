
# =====================================================================================
#
# This script takes a set of AVI files as input, and crops/rotates the spider.
#
# =====================================================================================

# Settings: overwrite recordings, or individually processed chunks?
OVERWRITE_CHUNK     = True

# Settings: Process multiple files in parallel?
FILES_PARALLEL = True

# Settings: Enable debugging? (turns of parallelism, Numba, etc.)
DEBUG = False

DEBUG_FILES = None #['a_2019-04-09-164951-0085.avi',]

# Settings: Numba configuration
USE_NUMBA = not DEBUG
NUMBA_PARALLEL = False
NOGIL = False

# Settings: Batch/job size for parallel processing
BATCH_SIZE = 200
NJOBS_PER_FILE = 1
NJOBS_PER_SEQ  = 60

NUM_BG = 10000

ALTERNATIVE_FLIPFIX = False

# =====================================================================================
# Imports & Globals
# =====================================================================================

# Import main libraries
import os, regex as re, itertools, numpy as np, imageio, warnings, logging, gc, joblib as jl

# Import progress bar library
from tqdm import tqdm as tqdm

# Import libraries for high-performance and parallel computing
import joblib
from joblib import Parallel, delayed
from numba import jit, njit, prange

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
except:
    p.nice(20)

# =====================================================================================
# Helper Functions
# =====================================================================================

# Chunk an iterarable
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

# This is a Numba-compatible version of np.mgrid
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def mgrid(nrows, ncols):
    y_indices = np.zeros((nrows, ncols))
    x_indices = np.zeros((nrows, ncols))
    for i in range(nrows):
        y_indices[i, :] = i
    for j in range(ncols):
        x_indices[:, j] = j
    return y_indices, x_indices

# See:
# https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def raw_moment(data, i_order, j_order):
  nrows, ncols = data.shape
  y_indices, x_indicies = mgrid(nrows, ncols)
  return (data * x_indicies**i_order * y_indices**j_order).sum()

@jit(nopython=USE_NUMBA, nogil=NOGIL)
def moments_cov(data):
  data_sum = data.sum()

  if data_sum < 0.000001:
      return np.array([[np.nan, np.nan], [np.nan, np.nan]])

  m10 = raw_moment(data, 1, 0)
  m01 = raw_moment(data, 0, 1)
  x_centroid = m10 / data_sum
  y_centroid = m01 / data_sum
  u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
  u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
  u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
  cov = np.array([[u20, u11], [u11, u02]])
  return cov

#new crop doesn't change shape of image just turns bright edge pixels to 0
#I guess its not technically a crop. Rename?
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def crop(bw, frame, dim):
    # The current arguments are currently hardwired (Numba doesn't appear to support optional arguments)
    left = 1; top = 1; bot = 1; right = 1;

    while np.sum(bw[dim[1]-bot,:]) > 10:
        if bot > 50:
            break
        bw[dim[1]-bot,:] = 0
        bot = bot + 2

    while np.sum(bw[:,dim[0]-right]) > 10:
        if right > 50:
            break
        bw[:,dim[0]-right] = 0
        right = right + 2

    while np.sum(bw[top,:]) > 10:
        if top > 50:
            break
        bw[top,:] = 0
        top = top + 2

    while np.sum(bw[:,left]) > 10:
        if left > 50:
            break
        bw[:,left] = 0
        left = left + 2

    frame[:,dim[0]-right] = 0
    frame[dim[1]-bot,:] = 0
    frame[:, left] = 0
    frame[top,:] = 0

    return bw, frame

# =====================================================================================
# Main spider detection function
# =====================================================================================

# Custom function to erode binary array ('k' specifies number of erosions to perform)
# Note: Wrote custom function to make this Numba (optimization) compatible
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def erosion(img, k = 1):
    imgout = img.copy()
    h, w = img.shape

    for _k in range(k):
        for y in range(h):
            for x in range(w):
                if img[y, x] > 0:
                    imgout[y, x] = img[y-1, x] & img[y+1, x] & img[y, x-1] & img[y, x+1]
        img = imgout.copy()
    return imgout

# Custom function to dilate binary array ('k' specifies number of dilations to perform)
# Note: Wrote custom function to make this Numba (optimization) compatible
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def dilation(img, k = 1):
    imgout = img.copy()
    h, w = img.shape

    for _k in range(k):
        for y in range(w):
            for x in range(h):
                imgout[y, x] = img[y-1, x] | img[y+1, x] | img[y, x-1] | img[y, x+1]
        img = imgout.copy()
    return imgout


@jit(nopython=USE_NUMBA, nogil=NOGIL)
def erosiondilation(img, k = 1):
    return dilation(erosion(img, k), k)

# This function replaces skimage.label, making it Numba-compatible
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def label(d):
    # Variable that controls whether a local pixel is included as a cluster (quickly excludes isolated pixels
    # for a slight speedup)
    localThreshold = 1
    # Create output array
    labels = np.zeros(d.shape, dtype=np.uint16)
    numclusters = 0
    # First pass
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if d[y, x] > 0:
                # If a neighbor (up, down, left or right) is present with a label, copy that label number
                if y > 0 and labels[y - 1, x] > 0:
                    labels[y, x] = labels[y - 1, x]
                elif x > 0 and labels[y, x - 1] > 0:
                    labels[y, x] = labels[y, x - 1]
                elif y < labels.shape[0] - 1 and labels[y + 1, x] > 0:
                    labels[y, x] = labels[y + 1, x]
                elif x < labels.shape[1] - 1 and labels[y, x + 1] > 0:
                    labels[y, x] = labels[y, x + 1]
                else:
                    # No clusters nearby?
                    # -- Ensure this cluster is big enough before proceeding
                    if np.sum(d[(y - 1):(y + 2), (x - 1):(x + 2)]) > localThreshold:
                        numclusters += 1
                        labels[y, x] = numclusters

    # Second pass
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            if labels[y, x] > 0:
                # Ensure that any neighboring pixels have the same label identity
                nv = labels[y, x]
                if y > 0 and d[y - 1, x] > 0:
                    v = labels[y - 1, x]
                    # Search-and-replace label
                    if nv != v:
                        for a in range(max(0, x-100), min(labels.shape[1]-1, x+100)):
                            for b in range(max(0, y-100), min(labels.shape[0]-1, y+100)):
                                if labels[b,a] == v or labels[b,a] == nv:
                                    labels[b,a] = min(v, nv)
                if x > 0 and d[y, x - 1] > 0:
                    v = labels[y, x - 1]
                    # Search-and-replace label
                    if nv != v:
                        for a in range(max(0, x-100), min(labels.shape[1]-1, x+100)):
                            for b in range(max(0, y-100), min(labels.shape[0]-1, y+100)):
                                if labels[b,a] == v or labels[b,a] == nv:
                                    labels[b,a] = min(v, nv)
    return labels

# Image rotation function ('theta_' in degrees)
# Note:
#    This function replaces the use of scikit-image's 'rotate' function, such that is can be
#    incorporated into the optimized Numba-only code
# Adapted from:
#    https://github.com/SpikingNeuron/Python-Cython-Numba-CUDA/blob/master/Algo_RotateLinear.pyx
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def rotate(image, theta_):
    #
    src = image.astype(np.uint16)
    theta = np.pi / 180 * theta_
    ix = image.shape[0]
    iy = image.shape[1]

    # Image size is fixed
    oy, ox = image.shape[0], image.shape[1]

    # create array for return
    dst = np.zeros((ox, oy), dtype=np.uint16)

    # populate the destination image
    cx = ox / 2
    cy = oy / 2

    # rotation logic
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    for index_x in range(ox):
        for index_y in range(oy):
            index_y_new = int(((index_x - cx) * sin_t + (index_y - cy) * cos_t) + ix / 2)
            index_x_new = int(((index_x - cx) * cos_t - (index_y - cy) * sin_t) + iy / 2)
            index_y_new_p1 = index_y_new + 1
            index_x_new_p1 = index_x_new + 1
            if 0 <= index_x_new_p1 < ix and 0 <= index_y_new_p1 < iy:
                dst[index_x, index_y] = (
                        (
                                src[index_x_new, index_y_new] +
                                src[index_x_new, index_y_new_p1] +
                                src[index_x_new_p1, index_y_new] +
                                src[index_x_new_p1, index_y_new_p1]
                        ) / 4.0)

    return dst.astype(np.uint8)

# This is the main spider detection function. It operates by finding blobs of pixels above a certain threshold and
# of a certain size, and then finding the moment of this blob to determine the spider's position and orientation. Rotation aligns
# the long axis of the spider, but leaves it pointing either left or right. We next use the fact that the abdomen is
# generally bigger/brighter in this rotated/cropped image to always have the spider pointing right.
@jit(nopython=USE_NUMBA, nogil=NOGIL)
def detectSpider(f, mask):
    # Check if this is a full-zero frame
    if np.sum(f) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8)

    # For now, use a fixed threshold of 30
    THRESHOLD = 30
    useZscore = False

    # Detect main spider (thorax/abdomen) blob
    bw = f > THRESHOLD
    if useZscore:
        bw = ((f.astype(np.float64) - mask[:,:,0]) / mask[:,:,1]) > THRESHOLD
    bw, f = crop(bw, f, f.shape)
    d = erosiondilation(bw, 7)

    # Label blobs
    dl = label(d)
    # Determine blob size
    ls = [np.sum(dl == li) for li in range(0, np.max(dl) + 1)]
    # Only keep blobs in plausible range
    blobids = [i for i in range(len(ls)) if 100 < ls[i] < 2000]
    # If no blob is found, reduce the threshold
    minsize = 100
    while len(blobids) == 0:
        minsize -= 2
        if minsize <= 30:
            break
        blobids = [i for i in range(len(ls)) if minsize < ls[i] < 2000]
    if minsize == 0:
        print('No blobs found among candidates: ', ls)
    elif 0 < minsize < 100:
        print('Had to reduce minimum blob size:', len(ls), minsize)
    # If multiple blobs pass this test, look for further characteristics
    if len(blobids) > 0:
        pass

    # check for no spider
    if len(blobids) < 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8)

    # Now for the first plausible spider blob (we're assuming here there'll be one blob left
    # at this point), compute the long axis
    li = blobids[0]
    blob = np.zeros(d.shape, dtype=d.dtype)
    _ = np.nonzero(dl == li)
    y = _[0]; x = _[1];
    for i in range(len(y)):
        blob[y[i],x[i]] = d[y[i],x[i]]

    # Rotate
    if np.sum(blob) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8)

    cov = moments_cov(blob)
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]

    theta = 0
    if np.abs(x_v1) < 0.000001:
        pass
        #return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
        #       np.nan, np.nan, np.nan, np.zeros((200, 200), dtype=np.uint8)
    else:
        # Determine angle of long axis
        theta = np.arctan((y_v1) / (x_v1)) / np.pi * 180

    # Determine position as center of the blob bounding box (works better than the mean)
    _xy = np.nonzero(blob); _y = _xy[0]; _x = _xy[1]

    if len(_x) == 0 or len(_y) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8)

    px = 0.5 * (np.max(_x) + np.min(_x))
    py = 0.5 * (np.max(_y) + np.min(_y))

    # Crop spider
    _f = np.zeros((f.shape[0] + 200, f.shape[1] + 200), dtype=f.dtype)
    _f[100:100+f.shape[0], 100:100+f.shape[1]] = f
    yi = int(py) + 100
    xi = int(px) + 100
    fsp = _f[(yi - 100):(yi + 100), (xi - 100):(xi + 100)].copy()

    # Define axes for possible plotting
    scale = 20
    xL1 = x_v1 * -scale * 2 + px
    xL2 = x_v1 * scale * 2 + px
    yL1 = y_v1 * -scale * 2 + py
    yL2 = y_v1 * scale * 2 + py

    xS1 = x_v2 * -scale * 2 + px
    xS2 = x_v2 * scale * 2 + px
    yS1 = y_v2 * -scale * 2 + py
    yS2 = y_v2 * scale * 2 + py

    # Rotate
    frot = rotate(fsp, theta)

    # Check if image is inverted...
    # In the rotated, thresholded image, label the blobs
    # Then take the min and max of the bounding box of the blob that contains the center pixel (X,Y=100,100)
    if useZscore:
        # Crop mask
        _f[100:100 + f.shape[0], 100:100 + f.shape[1]] = mask[:, :, 0]
        fspMask = _f[(yi - 100):(yi + 100), (xi - 100):(xi + 100)].copy()
        _f[100:100 + f.shape[0], 100:100 + f.shape[1]] = mask[:, :, 1]
        fspMaskStd = _f[(yi - 100):(yi + 100), (xi - 100):(xi + 100)].copy()
        # Rotate mask
        maskRot = rotate(fspMask, theta)
        maskStdRot = rotate(fspMaskStd, theta)
        # Label Z-scored image
        lbl = label(((frot.astype(np.float64) - maskRot) / maskStdRot) > THRESHOLD)
    else:
        lbl = label(frot > THRESHOLD)

    nz = np.nonzero(lbl[100, 100] == lbl)[1]
    if len(nz) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, \
               np.nan, np.nan, np.nan, np.zeros((200, 200), dtype=np.uint8), np.zeros((200, 200), dtype=np.uint8)

    # Extent of right side should be larger than extent of left side. If this is not the case (not b > a), rotate the image 180 degrees.
    if 100 - np.min(nz) > np.max(nz) - 100:
        # Rotate by 180 degrees
        theta += 180
        frot = rotate(fsp, theta)

    return px, py, xL1, xL2, yL1, yL2, xS1, xS2, yS1, yS2, theta, frot, fsp

# Call the detectSpider function on the 3D matrix of frames in this frame batch
# This should be optimized by Numba
@jit(nopython=USE_NUMBA, parallel=NUMBA_PARALLEL, cache=USE_NUMBA)
def detectSpider_batch(frames, mask):
    results = []
    for i in prange(frames.shape[0]):
        results.append( detectSpider(frames[i,:,:], mask) )
    return results

# =====================================================================================
# Run detectSpider in batch, then cache results
# =====================================================================================

# Each file is processed and then written to cache, to reduce RAM needs.
# After all file chunks are processed, they can be merged together into the final output file.
def detectSpiderToCache(fname, frames, dirCache, mask, overwrite=False):
    # Don't do anything if input is None
    if frames is None:
        return

    # Determine output name
    fnameCache = os.path.join(dirCache, '{}_{}.pickle'.format(os.path.basename(fname), frames[0][0]))

    # Already exists?
    # Note: This can happen when a processing session crashes and we restart it
    if os.path.exists(fnameCache) and not overwrite:
        print("Skipping file, already exists: {}".format(fname))
        return

    # Run 'detectSpider' for every frame...
    _frames = np.array([x[1] for x in frames]) # Will already be type uint8, as required...
    results = detectSpider_batch(_frames, mask)

    # Save the output to cache (with frame IDs)
    joblib.dump([(frames[i][0], results[i]) for i in range(len(results))], fnameCache)

# =====================================================================================
# Determine if spider moves
# =====================================================================================

#
# Note: Currently only the first and last frames are being compared, but this is a bit tricky. Especially towards 
#       the later stages of web building, there are examples where the spider hangs out in the middle, makes a few 
#       trips outwards, but then comes back to almost exactly the same position, which would trick this function 
#       into thinking there was no movement. In the future, either process all videos, or check movement in ~10 second 
#       blocks.
#

def didSpiderMove(frames, mask):
    
    distance_thresh = 5 #number of pixels required for spider to have considered to have moved

    x1, y1, xL1, xL2, yL1, yL2, xS1, xS2, yS1, yS2, theta, frot, fsp = detectSpider(frames[0], mask)
    x2, y2, xL1, xL2, yL1, yL2, xS1, xS2, yS1, yS2, theta, frot, fsp = detectSpider(frames[-1], mask)

    if np.any(np.isnan([x1, y1, x2, y2])):
        return True
    else:
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])

        p1[p1 < 0] = np.nan
        p2[p2 < 0] = np.nan

        # Compute distance from first to last frame
        distance = np.linalg.norm( p1 - p2 )

        # Did the spider move? (if distance is NaN, we don't know, so assume it did...)
        return (distance > distance_thresh or np.isnan(distance))

# =====================================================================================
# Step 1: Analyze entire video & write to cache
# =====================================================================================

# Process an entire recording file
def processMovie_1(fname, processIndex, dirCache, mask, maxNumFrames = 1000000, streaming=True):

    # Ensure cache directory exists
    os.makedirs(dirCache, exist_ok=True)

    logging.info('Processing: {}'.format(fname))

    # Call relevant handler
    if fname.endswith('.avi'):
        processMovie_1_AVI(fname, processIndex, dirCache, mask, maxNumFrames = maxNumFrames, streaming = streaming)
    else:
        raise NotImplementedError('Filetype not supported')

def processMovie_1_UFMF(fname, processIndex, dirCache, mask, maxNumFrames = 1000000, streaming=True):
    # Open file
    # -- Import relevant library
    import os, sys;
    libpath = os.getcwd()[:os.getcwd().rfind('spider-behavior') + len('spider-behavior')];
    libpath = os.path.join(libpath, 'libraries')
    if not libpath in sys.path:
        sys.path.append(libpath)
    from motmot.SpiderMovie import SpiderMovie
    # -- Open actual file
    vid = SpiderMovie(fname)
    # -- Don't add background back in (we want to use the background-subtracted image)... For now, set vid._bg to None...
    #    To-Do: Have a nicer flag for this to pass to SpiderMovie
    vid._bg = None

    # Frame iterator
    def iterFrames(batchSize = BATCH_SIZE):
        L = len(vid)
        for i in range(int(np.ceil(L / batchSize))):
            # Already exists?
            # Note: This can happen when a processing session crashes and we restart it
            fnameCache = os.path.join(dirCache, '{}_{}.pickle'.format(os.path.basename(fname), i * batchSize))
            if os.path.exists(fnameCache) and not OVERWRITE_CHUNK:
                print("Skipping file, already exists: {}".format(fname))
                yield None
            else:
                # Otherwise, yield entire batch of frames
                batch = [(i * batchSize + j, vid.get_frame(i * batchSize + j)[0]) for j in range(min(batchSize, L - i * batchSize))]
                # Not sure why any frames should be None if its frame is in range?
                # For now, yield an empty frame
                batch = [((y[0], np.zeros((200,200), dtype=np.uint8)) if y[1] is None else y) for y in batch]
                yield batch

    # Detect spider in parallel
    njobs = 60 if not DEBUG else 1
    if njobs == 1:
        for f in tqdm(iterFrames(),
                  position=processIndex, leave=False, desc='detecting spider (batch={})'.format(BATCH_SIZE)):
            detectSpiderToCache(fname, f, dirCache, overwrite=OVERWRITE_CHUNK)
    else:
        Parallel(n_jobs=njobs)(
            delayed(detectSpiderToCache)(fname, f, dirCache, overwrite=OVERWRITE_CHUNK) for \
            f in tqdm(iterFrames(),
                      position=processIndex, leave=False, desc='detecting spider (batch={})'.format(BATCH_SIZE)))

def processMovie_1_AVI(fname, processIndex, dirCache, mask, maxNumFrames = 1000000, streaming=True):
    try:
        # Frame array
        vidAVI, frames = None, None

        def iterFrames():
            if frames is not None:
                for f in frames:
                    yield f
            elif vidAVI is not None:
                c = 0
                for f in vidAVI.iter_data():
                    if c >= maxNumFrames:
                        return
                    else:
                        fr = f[:,:,0]
                        #if mask is not None:
                        #    fr[~mask] = 0
                        yield fr
                        c += 1

        # Load AVI (Either buffered or streaming)
        if fname.endswith('.avi'):
            vidAVI = imageio.get_reader(fname)
            if not streaming:
                frames = [f for f in tqdm(iterFrames(), desc='Reading AVI', leave=False)]
        else:
            raise NotImplementedError('Unsupported input file type')

        # Only spend time detecting spider if it indeed moved...
        spiderMoved = False
        if streaming:
            f1 = vidAVI.get_data(0)[:,:,0]
            f2 = vidAVI.get_data(vidAVI.get_length()-1)[:,:,0]
            #if mask is not None:
            #    f1[~mask] = 0
            #    f2[~mask] = 0
            spiderMoved = didSpiderMove([f1, f2], mask)
        elif frames is not None and len(frames) >= 2:
            spiderMoved = didSpiderMove([frames[0], frames[-1]], mask)

        if spiderMoved:
            # Run detectSpider in parallel #, max_nbytes=100000000, mmap_mode=None, batch_size=100, pre_dispatch=2000
            if NJOBS_PER_FILE == 1:
                for f in tqdm(grouper(BATCH_SIZE, enumerate(iterFrames())), position=processIndex, leave=False, desc='detecting spider...'):
                    detectSpiderToCache(fname, f, dirCache, mask, overwrite=OVERWRITE_CHUNK)
            else:
                Parallel(n_jobs=NJOBS_PER_FILE if not DEBUG else 1)(delayed(detectSpiderToCache)(fname, f, dirCache, mask, overwrite=OVERWRITE_CHUNK) for \
                    f in tqdm(grouper(BATCH_SIZE, enumerate(iterFrames())), position=processIndex, leave=False, desc='detecting spider...'))
        else:
            print('Skipping file: {}'.format(fname))

        if vidAVI is not None:
            vidAVI.close()
    except Exception as e:
        print(e)

# =====================================================================================
# Step 2: Read in cached files and concatenate into final output format
# =====================================================================================

# Get all files in this sequence
def getAllCachedFilesInSequence(fnameBaseAll):
    allFiles = [os.path.join(os.path.dirname(fnameBaseAll), x) for x in os.listdir(
        os.path.dirname(fnameBaseAll)) if os.path.basename(fnameBaseAll) in x]

    # Sort chunks by both their file and chunk ID
    def fileOrder(f):
        try:
            try:
                # Try AVI pattern first:
                fileIDs = [int(x) for x in re.search('-([0-9]+)\\.avi_([0-9]+)\\.pickle', f).groups()]
                return fileIDs[0] * 1000000 + fileIDs[1]
            except:
                return int(re.search('\\.ufmf_([0-9]+)\\.pickle', f).group(1))
        except:
            return None

    files = sorted([x for x in allFiles if fileOrder(x) is not None], key=fileOrder)
    return files

# Discover the prefixes/base names of all recording sequences in the cache directory
def getAllRecordingSequences(dirCache):
    def _prefix(x):
        try:
            return re.search('^(.*)((-[0-9]+)\\.avi|\\.ufmf)_[0-9]+\\.pickle', x).groups()[0]
        except:
            return None
    return list(set([os.path.join(dirCache, _prefix(x)) for x in os.listdir(dirCache) if _prefix(x) is not None]))

# Process and save movie chunks, outputting them as numpy arrays that can be directly fed
# to the LEAP/DeepLabCut scripts.
# Note:
#   * The input file 'fname' can be any file in the relevant sequence. This function will
#     automatically detect the remaining file chunks and process them in the correct order.
def processMovie_2(fnameBaseAll, dirCache = None):

    # If no recording prefix is specified, choose the first one available...
    if fnameBaseAll is None:
        try:
            fnameBaseAll = getAllRecordingSequences(dirCache)[0]
        except:
            return None

    print('Processing recording w/ prefix {}'.format(fnameBaseAll))

    # Get all input files (from cache)
    files = getAllCachedFilesInSequence(fnameBaseAll)

    # Output filesnames
    fnameMatOut = fnameBaseAll + '_mat.npy'
    fnameImgOut = fnameBaseAll + '_img.npy'

    # Matrix for position, rotation, etc.
    N = len(files) * BATCH_SIZE
    mat = np.memmap(filename=fnameMatOut, dtype=np.double, mode='w+', shape=(N, 4))
    # Matrix for rotated images (b/c of the large filesize, memory map this array rather than keeping it in RAM)
    img = np.memmap(filename=fnameImgOut, dtype=np.uint8, mode='w+' if not os.path.exists(fnameImgOut) else 'r+', shape=(N, 200, 200))

    i = 0
    for fname in tqdm(files, leave=True, desc='concatenating data to output matrices'):
        d = joblib.load(fname)
        for r in d:
            # Save variables
            mat[i, 0] = r[0]     # Extract frame ID
            mat[i, 1] = r[1][0]  # Extract X coord
            mat[i, 2] = r[1][1]  # Extract Y coord
            mat[i, 3] = r[1][10] # Extract theta
            # Save rotated image
            if not isinstance(r[1][11], float):
                _img = None
                if np.max(r[1][11]) > 1:
                    _img = r[1][11]
                else:
                    _img = (r[1][11] * 255).astype(np.uint8)

                if ALTERNATIVE_FLIPFIX:
                    if np.sum(_img[30:170,30:100] > 20) > np.sum(_img[30:170,100:170] > 20):
                        _img = np.flip(np.flip(_img, axis=1), axis=0)
                        mat[i, 3] = r[1][10] + 180

                img[i, :, :] = _img
            # Increase position in array
            i += 1

    # Close the files & write to memory
    del img
    del mat

    # Now delete all the cached files
    for file in files:
        pass
        #DEBUG: For now don't remove cached files until we can confirm this works...
        #os.remove(file)

# Determine whether recording processing is already completed
def isRecordingProcessed(fnameBaseAll, dirCache = None):
    # If no recording prefix is specified, choose the first one available...
    if fnameBaseAll is None:
        try:
            fnameBaseAll = getAllRecordingSequences(dirCache)[0]
        except:
            return None

    # Output filesnames
    fnameMatOut = fnameBaseAll + '_mat.npy'
    fnameImgOut = fnameBaseAll + '_img.npy'

    # Already processed?
    return (os.path.exists(fnameMatOut) and os.path.exists(fnameImgOut))

# =====================================================================================
# Compute mask (used to remove static background)
# Note: UFMF will already have background removed
# =====================================================================================

@njit(fastmath=True, nogil=True)
def _characterizeBackground(frames):
    bgDistr = np.zeros((frames.shape[1], 1024, 2), dtype=np.float64)
    for i in range(0, bgDistr.shape[0]):
        if (i%16) == 0:
            pass #print(i)
        for j in range(0, 1024):
            px = frames[:, i, j]
            if np.max(px) == 0:
                bgDistr[i, j, :] = 0
            else:
                _y, x = np.histogram(px, range=(0, 255), bins=255)
                y = np.zeros(256, dtype=np.int64)
                for k in range(1, 256 - 2):
                    y[k] = np.median(_y[(k - 1):(k + 2)])
                am = np.argmax(y)
                newDistr = np.hstack((px[px <= am], am - (px[px < am] - am)))
                if newDistr.size == 0:
                    bgDistr[i, j, :] = 0
                else:
                    m, s = np.mean(newDistr), np.std(newDistr)
                    bgDistr[i, j, 0] = m
                    bgDistr[i, j, 1] = s
    return bgDistr

def characterizeBackground(frames):
    # Process
    bgDistr = jl.Parallel(n_jobs=32, prefer='threads')(jl.delayed(_characterizeBackground)(
        frames[:, (32 * i):(32 * (i + 1)), :]) for i in range(32))
    bgDistr = np.vstack(bgDistr)
    gc.collect()

    # Done
    return bgDistr

def computeMask(fnames, dirCache):

    # Init
    bg = None

    # Determine output filename
    fnameBg = fnames[0][1].replace('.avi', '') + '_bg.npy'

    # Make sure all files are AVI's
    if len(fnames) == 0:
        raise Exception('No files passed')

    elif fnames[0][1].endswith('.avi'):

        if os.path.exists(fnameBg):
            bg = np.load(fnameBg)
        else:
            # Make sure all files have same .avi extension
            for _, fname in fnames:
                if not fname.endswith('.avi'):
                    raise Exception('Not all files are AVIs.')

            # Get lengths of AVI chunks
            filesAVI = [x[1] for x in fnames]

            def _getLenAVI(fname):
                for i in range(5):
                    try:
                        readerAVI = imageio.get_reader(fname)
                        l = readerAVI.get_length()
                        readerAVI.close()
                        del readerAVI
                        gc.collect()
                        return l
                    except Exception as e:
                        print(fname, e)
                return 0

            aviLens = jl.Parallel(n_jobs=10, prefer='processes')(jl.delayed(_getLenAVI)(fn) for fn in tqdm(filesAVI))

            # Get random subset of frames
            chooseFromFile = np.random.choice(np.arange(len(filesAVI), dtype=int),
                NUM_BG, replace=True,
                p=np.array(aviLens, dtype=float) / np.sum(aviLens))
            frames = np.zeros((chooseFromFile.shape[0], 1024, 1024), dtype=np.uint8)

            def _getFramesAVI(fname, idxs):
                if len(idxs) == 0:
                    return []
                else:
                    readerAVI = None
                    for i in range(5):
                        try:
                            readerAVI = imageio.get_reader(fname)
                        except Exception as e:
                            print(fname, str(e))
                            readerAVI = None
                    if readerAVI is None:
                        return []
                    fr = []
                    for _k, frameID in enumerate(np.sort(idxs)):
                        if (_k%100) == 0:
                            print('Loaded {} AVI frames...'.format(_k))
                        fr.append(np.mean(readerAVI.get_data(frameID), axis=2).astype(np.uint8))
                    readerAVI.close()
                    del readerAVI
                    gc.collect()
                    return fr

            print('Loading {} AVI frames: {}'.format(chooseFromFile.size, aviLens))
            frs = jl.Parallel(n_jobs=10, prefer='processes')(jl.delayed(_getFramesAVI)(
                filesAVI[i], np.random.choice(aviLens[i], min(aviLens[i], np.sum(chooseFromFile == i)), replace=False) if \
                    np.sum(chooseFromFile == i) > 0 else []) for i in tqdm(range(len(filesAVI))))

            print('Saving AVI frames into array')
            for ifname, fr in tqdm(zip(range(len(filesAVI)), frs)):
                if len(fr) > 0:
                    idx = np.argwhere(chooseFromFile == ifname)[:len(fr), 0]
                    for outID, frame in zip(idx, fr):
                        frames[outID, :, :] = frame

            del frs
            gc.collect()

            # Characterize mean/std of background
            print('Characterizing background')
            bg = characterizeBackground(frames)

            # Save the mask
            print('Saving mask')
            np.save(fnameBg, bg)
            print('Done saving mask')

        # Now create a mask based on the minimum background
        return bg

    else:
        raise Exception('Unknown recording filetype.')

# =====================================================================================
# Process an entire recording directory
# =====================================================================================

def processRecordingDir(dirpath, overwrite=False):
    # Ensure that an output directory exists
    dirOut = os.path.join(dirpath, 'croprot/')
    os.makedirs(dirOut, exist_ok=True)

    # Use output dir as cache dir also
    dirCache = dirOut

    # Already processed?
    if isRecordingProcessed(None, dirCache) and not overwrite:
        return

    # List files to be processed
    gen = []
    try:
        gen = list(enumerate(
            [os.path.join(os.path.join(dirpath, 'raw'), x) for x in os.listdir(os.path.join(dirpath, 'raw')) if \
                re.search('-[0-9]*\\.avi$', x) is not None]))
    except Exception as e:
        print(e)
        return
    
    # Filter files?
    if DEBUG_FILES is not None:
        print('{} files before debug subsetting'.format(len(gen)))
        gen = [x for x in gen if np.any([(y in x[1]) for y in DEBUG_FILES])]
        print('{} files after debug subsetting'.format(len(gen)))

    # No files to process?
    if len(gen) == 0:
        return

    # Compute mask across all files (This will black out background pixels and prevent misdetection of the spider blob)
    mask = computeMask(gen, dirCache)

    # Process all files in directory (Step 1: Write to cache)
    njobs = min(len(gen), NJOBS_PER_SEQ if (FILES_PARALLEL and not DEBUG) else 1)
    print('Processing {} files simultaneously'.format(njobs))

    # Process movie chunks in parallel
    if njobs == 1:
        for i, fname in gen:
            processMovie_1(fname, i, dirCache, mask)
    else:
        Parallel(n_jobs=njobs)(delayed(processMovie_1)(fname, i, dirCache, mask) for i, fname in gen)

    # Now concatenate all these files (Step 2)
    processMovie_2(None, dirCache)

    # Done!

def processRecordingDirSafe(dirpath, overwrite=False):
    try:
        processRecordingDir(dirpath, overwrite=overwrite)
    except Exception as e:
        warnings.warn('Error encountered in file {}: {}'.format(dirpath, str(e)))

# =====================================================================================
# Find directories that have not been processed
# =====================================================================================

def findUnprocessedRecordings(rootpath, overwrite=True):
    dirs = []
    for x in os.listdir(rootpath) + ['',]:
        fx = os.path.join(rootpath, x)

        isAlreadyProcessed = False
        if os.path.exists(os.path.join(fx, 'croprot/')):
            _f = os.listdir(os.path.join(fx, 'croprot/'))
            if len([x for x in _f if x.endswith('_img.npy')]) > 0:
                isAlreadyProcessed = True

        if isRecordingDir(fx) and (overwrite or not isAlreadyProcessed):
            dirs.append(fx)
    return dirs

def isRecordingDir(dirpath):
    return os.path.exists(os.path.join(dirpath, 'raw')) or os.path.exists(os.path.join(dirpath, 'ufmf'))

# =====================================================================================
# Run this script on an entire directory
# =====================================================================================

if __name__ == "__main__":
    from pipeline.python.misc import gui_misc
    selectedDir, OVERWRITE_RECORDING, nRec = gui_misc.askUserForDirectory(findUnprocessedRecordings)

    print(selectedDir)

    if nRec > 0:
        # Is this a recording dir?
        if isRecordingDir(selectedDir):
            # Process this directory
            processRecordingDir(selectedDir, overwrite=OVERWRITE_RECORDING)
        else:
            # Otherwise, this is a root directory with multiple behavior folders...
            # Automatically find the unprocessed behaviors
            dirs = findUnprocessedRecordings(selectedDir, overwrite=OVERWRITE_RECORDING)

            # Only process filenames with tildes
            dirs = [x for x in dirs if '~' in x]

            njobs = 1
            if njobs == 1:
                [processRecordingDirSafe(d, overwrite=OVERWRITE_RECORDING) for d in tqdm(dirs)]
            else:
                Parallel(n_jobs=njobs)(delayed(processRecordingDirSafe)(d, overwrite=OVERWRITE_RECORDING) for d in tqdm(dirs))
