
# ======================================================================================================================
# Imports
# ======================================================================================================================

import os, json, joblib as jl
from tqdm import tqdm

# ======================================================================================================================
# Set process priority
# ======================================================================================================================

# Set process priority to lowest so this script doesn't interfere with OS function
import psutil
p = psutil.Process()
try:
    p.nice(psutil.IDLE_PRIORITY_CLASS)
#    p.nice(psutil.NORMAL_PRIORITY_CLASS)
except:
    p.nice(20)
print(p.pid)

# ======================================================================================================================
# Helper function
# ======================================================================================================================

def isRecordingValid(fn):
    try:
        if os.path.exists(fn):
            with open(fn, 'r') as f:
                js = json.load(f)
                if js['web_complete'] and js['tracking_successful']:
                    if 'needs_review' in js:
                        if js['needs_review'] is not False:
                            return False
                    return True
        return False
    except Exception as e:
        print(fn, e)
        return False

def safeClusterNewData(c):
    try:
        import run_wavelet_cluster_newdata
        run_wavelet_cluster_newdata.run(*c)
    except Exception as e:
        print(e)

# ======================================================================================================================
# ...
# ======================================================================================================================

if __name__ == "__main__":
    fnames = [y for y in [os.path.join('Z:/behavior/', x) for x in os.listdir('Z:/behavior/')] \
              if isRecordingValid(os.path.join(y, 'recording.json'))]

    RECS = [x for x in fnames if '6-3-19-e' in x]
    RECS = [x for x in fnames]
    print('Num recordings: {}'.format(len(RECS)))

    j1 = [6, 10, 18, 22, 9, 13, 21, 25]
    j2 = [10, 22, 13, 25]
    j3 = []

    allSettings = []
    for rec in RECS:
        for joints in [j3, j1]:  # [j3, j1, j2]
            for metric in ['euclidean', ]:  # 'JS'
                for numPeriods in [20, ]:  # [20, 40]:
                    for algorithm in ['dlc', ]:  # 'rawpca']:
                        for coords in ['euclidean', ]:  # 'polar',
                            for include_absolute_position in ['no-abspos', ]:  # 'with-abspos']:
                                for vel in ['no-vel' if len(joints) > 0 else 'with-vel', ]:  # 'no-vel'
                                    for use_pca in ['no-pca']:  # ['no-pca','pca','nmf',]
                                        for histNorm in [True, ]:  # , False
                                            for method in ['tsne', ]:
                                                settings = {
                                                    'algorithm': algorithm,
                                                    # Include only front and back legs
                                                    'joints': joints,
                                                    # Overwrite?
                                                    'overwrite': True,
                                                    # Coords
                                                    'coords': coords,
                                                    'use_pca': use_pca,
                                                    'metric': metric,
                                                    'include_absolute_position': include_absolute_position,
                                                    'include_velocities': vel,
                                                    'numPeriods': numPeriods,
                                                    'tsne_perplexities': [100, ],  # 25, 500,
                                                    'parallel': True,
                                                    'histogram_normalize': histNorm,
                                                    'fname_base': os.path.join(rec, 'embeddings/'),
                                                    'recordings': [rec, ],
                                                    'n_iter': 2000,
                                                    'highpower_rows_per_file': 100000,
                                                    'perplexity': 100,  # 25, 500
                                                    'perplexity_newdata': 100,
                                                    'n_iter_newdata': 500,
                                                    'min_dist': 0.1,
                                                    'n_neighbors': 100,
                                                    'maxN': 200000,
                                                    'method': method,
                                                    'legvars': {
                                                        'samplingFreq': 50,
                                                        'omega0': 5,
                                                        'numPeriods': numPeriods,
                                                        'maxF': 25,
                                                        'minF': 1,  # 0.1
                                                        'stack': True,
                                                        'numProcessors': 4
                                                    }
                                                }
                                                allSettings.append(settings)

    allSettings = []

    for include_velocities in ['no-vel',]: #['with-vel-polar', 'no-vel']:
        refC = {
            'duration': 60,
            'maxoffset': 16,
            'meansubtract': 'meansub', # meansuball
            'scalestd': 'scalestd', #'scalestd',
            'coords': 'polar',
            'include_velocities': include_velocities
        }

        for duration in [100, 60]:
            for maxoffset in [0, 16]:
                for meansubtract in (['meansub', 'nomeansub', 'meansuball'] if refC['include_velocities'] == 'no-vel' else ['meansub',]):
                    for scalestd in ['scalestd',]: #'scalestd', 'noscalestd', 'scaletest']:
                        for coords in (['polar', 'euclidean'] if refC['include_velocities'] == 'no-vel' else ['euclidean',]):
                            # Only one property can deviate from the reference configuration, to avoid exponential increase
                            # of configurations
                            numDiff = 0
                            if duration != refC['duration']: numDiff += 1
                            if maxoffset != refC['maxoffset']: numDiff += 1
                            if scalestd != refC['scalestd']: numDiff += 1
                            if coords != refC['coords']: numDiff += 1
                            if meansubtract != refC['meansubtract']: numDiff += 1
                            if numDiff > 0:
                                continue

                            for rec in RECS:
                                for jnts in [[6, 10, 18, 22], [9, 13, 21, 25]]:
                                    allSettings.append({
                                        'rawmvmt': True,
                                        'rawmvmt_duration': duration,
                                        'rawmvmt_maxoffset': maxoffset,
                                        'rawmvmt_meansubtract': meansubtract,
                                        'rawmvmt_scalestd': scalestd,
                                        'algorithm': 'dlc',
                                        # Include only front and back legs
                                        'joints': [] if refC['include_velocities'] != 'no-vel' else jnts,
                                        # Overwrite?
                                        'overwrite': False,
                                        # Coords
                                        'coords': coords,
                                        'use_pca': 'no-pca',
                                        'metric': 'euclidean',
                                        'include_absolute_position': 'no-abspos',
                                        'include_velocities': refC['include_velocities'],
                                        'tsne_perplexities': [100, ],  # 100, 500
                                        'perplexity': 100,
                                        'perplexity_newdata': 100,
                                        'n_iter': 2000,
                                        'n_iter_newdata': 500,
                                        'maxN': 200000,
                                        'method': 'tsne', #'agglom'
                                        'fname_base': os.path.join(rec, 'embeddings/'),
                                        'recordings': [rec, ],
                                        'parallel': True
                                    })

    print(len(allSettings))
    import run_wavelet_cluster, psutil, time

    #for settings in allSettings:
    #    try:
    #        # run_wavelet.runWaveletTransform(settings['recordings'][0], settings)
    #        # run_wavelet_cluster.runWaveletMerge(settings)
    #        # run_wavelet_cluster.runWaveletHighPower(settings)
    #        # run_wavelet_cluster.runPCA(settings)
    #        run_wavelet_cluster.runClustering(settings)
    #        pass
    #    except Exception as e:
    #        print(e)

    for settings in tqdm(allSettings):
        while True:
            if psutil.virtual_memory().percent < 80:
                run_wavelet_cluster.runClustering(settings)
                time.sleep(300)
                break

    # Embed new datapoints
    #allSettings2 = []
    #for s in allSettings:
    #    s['parallel'] = False
    #    allSettings2.append(s)
    #configs = [(os.path.join(s['recordings'][0], 'wavelet'), s, 2) for s in allSettings2]
    #jl.Parallel(n_jobs=min(len(configs), 10))(jl.delayed(safeClusterNewData)(c) for c in tqdm(configs))

