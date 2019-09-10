import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
import pandas as pd
from scipy import special,stats

def _get_connvex_neigh(seedvex, faces, mask = None, masklabel = 1):
    """
    Function to get neighbouring vertices of a seed that satisfied overlap with mask
    """
    if mask is not None:
        assert mask.shape[0] == np.max(faces) + 1 ,"mask need to have same vertex number with faces connection relationship"
    assert isinstance(seedvex, (int, np.integer)), "only allow input an integer as seedvex"

    raw_faces, _ = np.where(faces == seedvex)
    rawconnvex = np.unique(faces[raw_faces])
    if mask is not None:
        connvex = set()
        list_connvex = [i for i in rawconnvex if mask[i] == masklabel]
        connvex.update(list_connvex)
    else:
        connvex = set(rawconnvex)
    connvex.discard(seedvex)
    return connvex


def get_n_ring_neighbor(vertx, faces, n=1, ordinal=False):
    """
    Get n ring neighbour from faces array
    Parameters:
    ---------
    vertex: vertex number
    faces : the array of shape [n_triangles, 3]
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor
    Return:
    ---------
    ringlist: array of ring nodes of each vertex
              The format of output will like below
              [{i1,j1,k1,...}, {i2,j2,k2,...}, ...]
			  each index of the list represents a vertex number
              each element is a set which includes neighbors of corresponding vertex
    Example:
    ---------
    >>> ringlist = get_n_ring_neighbour(24, faces, n)
    """
    if isinstance(vertx, int):
        vertx = [vertx]
    nth_ring = [set([vx]) for vx in vertx]
    nring = [set([vx]) for vx in vertx]
    while n != 0:
        n = n - 1
        for idx, neighbor_set in enumerate(nth_ring):
            neighbor_set_tmp = [_get_connvex_neigh(vx, faces) for vx in neighbor_set]
            neighbor_set_tmp = set().union(*neighbor_set_tmp)
            neighbor_set_tmp.difference_update(nring[idx])
            nth_ring[idx] = neighbor_set_tmp
            nring[idx].update(nth_ring[idx])
    if ordinal is True:
        return nth_ring
    else:
        return nring
        

def mask_localmax(data, mask):
    """
    Get vertex number of the point with maximum values.
    Parameters:
    -----------
    data [array, vertex*nsubj]: source data, it could be activation map, structural maps, etc. Note that the spatial dimension is in shape 0.
    mask [array, vertex*1]: mask with several ROIs.
    
    Returns:
    --------
    locmax_vx [array, nsubj*masklabel]: vertex number of the point with maximum values. 
             
    Examples:
    ---------
    >>> locmax_vx = mask_localmax(actdata, mask)
    """
    assert data.shape[0] == mask.shape[0], "Maps are unmatched."
    masklabel = np.unique(mask[mask!=0])
    nsubj = data.shape[1]
    locmax_vx = np.zeros((nsubj, len(masklabel)))
    for i, masklbl in enumerate(masklabel):
        mask_tmp = (mask==masklbl)
        mask_tmp = np.tile(mask_tmp, (1,nsubj))
        data_mask = data*mask_tmp
        data_mask[~mask_tmp] = np.min(data)
        locmax_vx_rg = np.argmax(data_mask,axis=0)
        locmax_vx[:,i] = locmax_vx_rg
    return locmax_vx
    
def threshold_by_rggrow(seedvx, vxnum, faces, scalarmap, option='descend', restrictedROI=None):
    """
    Threshold scalarmap with specific vertex number (vxnum) by region growing algorithm.
    Parameters:
    ------------
        seedvx: seed vertex
        vxnum: number of vertices to generate
        faces: relationship of geometry connection
        scalarmap: scalar map, e.g. activation map.
        option: 'descend', selected vertices with value smaller than seedvx.
                'ascend', selected vertices with value larger than seedvx.
        restrictedROI: ROI to limit range of region growing.
    Returns:
    --------
    rg_scalar: a new scalar map generated from the region growing algorithm
    vxpack: packed vertices
    Example:
    --------
    >>> rg_scalar, vxpack = threshold_by_rggrow(24, 300, faces, scalarmap)
    """
    assert np.ndim(scalarmap) == 1, "Please flatten scalarmap first."
    if option == 'ascend':
        actdata = -1.0*scalarmap
    else:
        actdata = 1.0*scalarmap 
    if restrictedROI is None:
        restrictedROI = np.ones_like(actdata)
    vxpack = set()
    vxpack.add(seedvx)
    backupvx = set()
    seed_neighbor = get_n_ring_neighbor(seedvx, faces, 1, ordinal=True)[0]
    backupvx.update(seed_neighbor.difference(vxpack))
    while (len(vxpack)<vxnum+1):
        # print('{} vertices contained'.format(len(vxpack)))
        backupvx = backupvx.difference(vxpack)
        if len(backupvx)==0:
            break
        array_backupvx = np.array(list(backupvx))
        array_vxpack = np.array(list(vxpack))
        seed_bp = int(array_backupvx[np.argmax(actdata[array_backupvx])])
        if restrictedROI[seed_bp] != 0:
            vxpack.add(seed_bp)
            seed_bp_neigh = get_n_ring_neighbor(seed_bp, faces, 1, ordinal=True)[0]
            backupvx.update(seed_bp_neigh)
        else:
            # Outside to restrictedROI 
            backupvx.discard(seed_bp)
            continue
    rg_scalar = np.zeros_like(scalarmap)
    rg_scalar[np.array(list(vxpack))] = 1
    rg_scalar = rg_scalar*scalarmap
    return rg_scalar, array_vxpack
 
 
def get_signals(atlas, mask, roilabels, method = 'mean'):
    """
    Extract brain activation from ROI
    
    Parameters:
    ------------
    brainimg[array]: A 4D brain image array with the first dimension correspond to timeseries and the rest 3D correspond to brain images
    mask[array]: A 3D brain image array with the same size as the rest 3D of brainimg.
    roilabels[list/array]: ROI labels
    method[str]: method to integrate activation from each ROI, by default is 'mean'.
    
    Returns:
    ---------
    roisignals[list]: Extracted brain activation. 
                      Each element in the list is the extracted activation of the roilabels.
                      Due to different label may contain different number of activation voxels, 
                      the output activation could not stored as numpy array list.
   
    Example:
    -------
    >>> signals = get_signals(atlas, mask, [1,2,3,4], 'mean')
    """
    if method == 'mean':
        calc_way = partial(np.mean, axis=1)
    elif method == 'std':
        calc_way = partial(np.std, axis=1)
    elif method == 'max':
        calc_way = partial(np.max, axis=1)
    elif method == 'voxel':
        calc_way = np.array
    else:
        raise Exception('We haven''t support this method, please contact authors to implement.')
    
    assert atlas.shape[1:] == mask.shape, "atlas and mask are mismatched."
    roisignals = []    
    for i, lbl in enumerate(roilabels):
        roisignals.append(calc_way(atlas[:,mask==lbl]))
    return roisignals
    

def pearsonr(A, B):
    """
    A broadcasting method to compute pearson r and p
    -----------------------------------------------
    Parameters:
        A: matrix A, (i*k)
        B: matrix B, (j*k)
    Return:
        rcorr: matrix correlation, (i*j)
        pcorr: matrix correlation p, (i*j)
    Example:
        >>> rcorr, pcorr = pearsonr(A, B)
    """
    if isinstance(A,list):
        A = np.array(A)
    if isinstance(B,list):
        B = np.array(B)
    if np.ndim(A) == 1:
        A = A[None,:]
    if np.ndim(B) == 1:
        B = B[None,:]
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    rcorr = np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:,None], ssB[None]))
    df = A.T.shape[1] - 2   
    r_forp = rcorr*1.0
    r_forp[r_forp==1.0] = 0.0
    t_squared = rcorr.T**2*(df/((1.0-rcorr.T)*(1.0+rcorr.T)))
    pcorr = special.betainc(0.5*df, 0.5, df/(df+t_squared))
    return rcorr, pcorr

def anova_decomposition(Y):
    """
    Decompositing variance of dataset Y into Mean Square and its corresponded dof.
    The data Y are entered as a 'table' with subjects (targets) are in rows
    and repeated measure (judges) in columns
    Reference: P.E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428.
    Source of variance: SST = SSW + SSB; SSW = SSBJ + SSE
    """
    [n_subjects, n_conditions] = Y.shape
    dfbt = n_subjects - 1
    dfbj = n_conditions - 1
    dfwt = n_subjects*dfbj
    dfe = dfbt * dfbj
    # SST
    mean_Y = np.mean(Y)
    SST = ((Y-mean_Y)**2).sum()
    # WMS (within-target mean square)
    Avg_WithinTarg = np.tile(np.mean(Y, axis=1), (n_conditions, 1)).T
    SSW = ((Y - Avg_WithinTarg)**2).sum()
    WMS = 1.0*SSW/dfwt
    # BMS (between-target mean square)
    SSB = ((Avg_WithinTarg - mean_Y)**2).sum()
    BMS = 1.0*SSB/dfbt
    # BJMS 
    Avg_BetweenTarg = np.tile(np.mean(Y,axis=0), (n_subjects, 1))
    SSBJ = ((Avg_BetweenTarg - mean_Y)**2).sum()
    BJMS = 1.0*SSBJ/dfbj
    # EMS
    SSE = SST - SSBJ - SSB
    EMS = 1.0*SSE/dfe
    
    # Package variables
    Output = {}
    Output['WMS'] = WMS
    Output['BMS'] = BMS
    Output['BJMS'] = BJMS
    Output['EMS'] = EMS
    Output['dof_bt'] = dfbt
    Output['dof_wt'] = dfwt
    Output['dof_bj'] = dfbj
    Output['dof_e'] = dfe
     
    return Output


def icc(Y, methods='(1,1)'):
    """
    Intra-correlation coefficient.
    The data Y are entered as a 'table' with subjects (targets) are in rows,
    and repeated measure (judges) in columns
    
    Reference: P.E. Shrout & Joseph L. Fleiss (1979). "Intraclass Correlations: Uses in Assessing Rater Reliability". Psychological Bulletin 86 (2): 420-428.
    Parameters:
    -----------
    Y: Original dataset, with its rows are targets and columns are judges.
    methods: Please see attached reference for details.
             (1,1), One-random effects
                    Each target is rated by a different set of k judges, 
                    randomly selected from a larger population of judges.
             ML, Calculate ICC by ML estimation.
             ReML, Calculate ICC by ReML estimation.
             (2,1), Two-way random effects
                    A random sample of k judges is selected from a larger 
                    population, and each judge rates each target, that is,
                    each judge rates n targets altogether.
             (3,1), Two-way mixed model
                    Each target is rated by each of the same k judges, who
                    are only judges of interest.
    Return: 
    -------
    r: intra-class correlation
    """
    decomp_var = anova_decomposition(Y)
    [n_targs, n_judges] = Y.shape
    if methods == '(1,1)':
        r = (decomp_var['BMS'] - decomp_var['WMS'])/(decomp_var['BMS']+(n_judges-1)*decomp_var['WMS'])
        F = decomp_var['BMS']/decomp_var['WMS']
        p = stats.f.sf(F, n_targs-1, n_targs*(n_judges-1))
    elif methods == 'ML':
        N = n_targs * n_judges
        # Design matrix
        X = np.ones((N,1))
        Z = np.kron(np.eye(n_targs), np.ones((n_judges,1)))
        y = Y.reshape((N,1))
        # Estimate variance components using ReML
        s20 = [0.001, 0.1]
        dim = [1*n_targs]
        s2, b, u, Is2, C, loglik, loops = _mixed_model(y, X, Z, dim, s20, method=1)
        r = s2[0]/np.sum(s2)
        WMS = s2[1]/n_judges
        BMS = s2[0]+s2[1]/n_judges
        F = 1.0*BMS/WMS
        p = stats.f.sf(F, n_targs-1, n_targs*(n_judges-1))
    elif methods == 'ReML':
        N = n_targs * n_judges
        # Design matrix
        X = np.ones((N,1))
        Z = np.kron(np.eye(n_targs), np.ones((n_judges,1)))
        y = Y.reshape((N,1))
        # Estimate variance components using ReML
        s20 = np.array([0.001, 0.1])
        dim = [1*n_targs]
        s2, b, u, Is2, C, loglik, loops = _mixed_model(y, X, Z, dim, s20, method=2)
        r = s2[0]/np.sum(s2)
        WMS = s2[1]/n_judges
        BMS = s2[0]+s2[1]/n_judges
        F = 1.0*BMS/WMS
        p = stats.f.sf(F, n_targs-1, n_targs*(n_judges-1))
    elif methods == '(2,1)':
        r = (decomp_var['BMS'] - decomp_var['EMS'])/(decomp_var['BMS']+(n_judges-1)*decomp_var['EMS']+n_judges*(decomp_var['BJMS']-decomp_var['EMS'])/n_targs)
        F = decomp_var['BMS']/decomp_var['EMS']
        p = stats.f.sf(F, n_targs-1, (n_judges-1)*(n_targs-1))
    elif methods == '(3,1)':
        r = (decomp_var['BMS'] - decomp_var['EMS'])/(decomp_var['BMS']+(n_judges-1)*decomp_var['EMS'])
        F = decomp_var['BMS']/decomp_var['EMS']
        p = stats.f.sf(F, n_targs-1, (n_targs-1)*(n_judges-1))
    else:
        raise Exception('Not support this method.')
    return r, p
    

def regressoutvariable(rawdata, covariate, fit_intercept=True):
    """
    Regress out covariate variables from raw data
    -------------------------------------------------
    Parameters:
        rawdata: rawdata, as Nx1 series.
        covariate: covariate to be regressed out, as Nxn series.
        fit_intercept: whether to fit intercept or not.
                       By default is False
    Return:
        residue
    """
    if isinstance(rawdata, list):
        rawdata = np.array(rawdata)
    if isinstance(covariate, list):
        covariate = np.array(covariate)
    clf = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=True)
    clf.fit(covariate, rawdata)
    residue = rawdata - clf.predict(covariate)
    return residue


def rggrow_roi_by_activation(actdata, mask, faces, vxsize=100, hemisphere='left'):
    """
    Using region growing method to generate story/math ROIs in each subject
    
    Parameters:
    ------------
    actdata: activation brain image data
    mask: probabilistic activation map as restriction
    faces: geometry relationship in surface space
    vxsize: ROI size. by default is 100 vertices
    hemisphere: hemisphere, by default is left
    
    Returns:
    ---------
    outmask: output ROIs, with the same shape as actdata
    """
    nsubj = actdata.shape[0]
    masklabel = np.unique(mask[mask!=0])
    locmax = mask_localmax(actdata.T, mask.T)

    outmask = np.zeros((32492, nsubj))
    for i in range(nsubj):
        print('Subject {}'.format(i+1))
        for j, masklbl in enumerate(masklabel):
            # Left hemisphere, masklbl < len(masklabel)/2+1
            if hemisphere == 'left':
                if ~np.isnan(locmax[i,j]):
                    vxloc_tmp, vxpack = threshold_by_rggrow(int(locmax[i,j]), vxsize, faces, actdata[i,:].flatten(), restrictedROI=(mask==masklbl).flatten().astype('int'))
                    outmask[vxpack, i] = masklbl
            # Right hemisphere
            else:
                if ~np.isnan(locmax[i,int(j-np.max(masklabel)/2)]):
                    vxloc_tmp, vxpack = threshold_by_rggrow(int(locmax[i, int(j-np.max(masklabel)/2)]), vxsize, faces, actdata[i,:].flatten(), restrictedROI=(mask==masklbl).flatten().astype('int'))
                    outmask[vxpack, i] = masklbl
    return outmask


def extract_avg_signals(actdata, mask):
    """
    """
    nsubj = actdata.shape[0]
    masklabel = np.unique(mask[mask!=0])
    avg_act = np.zeros((nsubj, len(masklabel)))
    for i in range(nsubj):
        avg_act[i,:] = get_signals(actdata[i,:], mask[i,:], masklabel)
    return avg_act


def pca_decomposition(avg_act, pcamodel=None):
    """
    """
    if pcamodel:
        with open(pcamodel, 'r') as f:
            pcamodel = pickle.load(f)
    else:
        pcamodel = PCA()
        pcamodel.fit(avg_act)
    evr = pcamodel.explained_variance_ratio_
    pcacomp = pcamodel.transform(avg_act)
    corrmat, _ = pearsonr(pcacomp.T, avg_act.T)
    return evr, pcacomp, corrmat, pcamodel
    

def prepare_twin_csv(story_MZ1, story_DZ1, story_MZ2, story_DZ2, math_MZ1, math_DZ1, math_MZ2, math_DZ2, storyrgname, mathrgname, zscore=True):
    """
    """
    nsubj_MZ = len(story_MZ1)
    nsubj_DZ = len(story_DZ1)
    if story_MZ1.ndim == 1:
        story_MZ1 = story_MZ1[:,None]
    if story_MZ2.ndim == 1:
        story_MZ2 = story_MZ2[:,None]
    if story_DZ1.ndim == 1:
        story_DZ1 = story_DZ1[:,None]
    if story_DZ2.ndim == 1:
        story_DZ2 = story_DZ2[:,None]
    if math_MZ1.ndim == 1:
        math_MZ1 = math_MZ1[:,None]
    if math_MZ2.ndim == 1:
        math_MZ2 = math_MZ2[:,None]
    if math_DZ1.ndim == 1:
        math_DZ1 = math_DZ1[:,None]
    if math_DZ2.ndim == 1:
        math_DZ2 = math_DZ2[:,None]
    story_rgnum = story_MZ1.shape[1]
    math_rgnum = math_MZ1.shape[1]
    assert story_rgnum == len(storyrgname), "Region number is mismatch in story activation."
    assert math_rgnum == len(mathrgname), "Region number is mismatch in math activation."
    concate_mat = np.zeros((nsubj_MZ+nsubj_DZ, 2*(story_rgnum+math_rgnum)+1))
    if zscore:
        story_MZ1 = stats.zscore(story_MZ1)
        story_MZ2 = stats.zscore(story_MZ2)
        story_DZ1 = stats.zscore(story_DZ1)
        story_DZ2 = stats.zscore(story_DZ2)
        math_MZ1 = stats.zscore(math_MZ1)
        math_MZ2 = stats.zscore(math_MZ2)
        math_DZ1 = stats.zscore(math_DZ1)
        math_DZ2 = stats.zscore(math_DZ2)
    concate_mat[:,:-1] = np.hstack((
                             np.concatenate((story_MZ1, story_DZ1),axis=0),
                             np.concatenate((story_MZ2, story_DZ2), axis=0),
                             np.concatenate((math_MZ1, math_DZ1), axis=0),
                             np.concatenate((math_MZ2, math_DZ2), axis=0)
                                  ))
    concate_mat[:nsubj_MZ, -1] = 1.0
    concate_mat[nsubj_MZ:, -1] = 3.0
    column_story = [sn+tn for tn in ['1', '2'] for sn in storyrgname]
    column_math = [sn+tn for tn in ['1', '2'] for sn in mathrgname]
    column_name = column_story + column_math + ['zyg']
    concate_pd = pd.DataFrame(data=concate_mat, columns=column_name)
    return concate_pd


def extract_correlation(restact, story_mask, math_mask):
    """
    """
    lang_sig = get_signals(restact, story_mask, np.arange(1,9,1))
    math_sig = get_signals(restact, math_mask, np.arange(1,19,1))
    lang_sig = np.array(lang_sig)
    math_sig = np.array(math_sig)
    langmath_sig = np.concatenate((lang_sig,math_sig),axis=0)
    r_corr, _ = pearsonr(langmath_sig, langmath_sig)
    return r_corr

