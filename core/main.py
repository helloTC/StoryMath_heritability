from StoryMath_heritability.core import tools



# Outline
# 1) Prepare subject-specific ROIs
# 2) Extract brain activations
# 3) Generate PCA models
# 4) Get and extract the first components
# Control analysis
# 5) roi size
# 6) structure effect
# 7) gender & age effects


twin_act = pd.read_csv(pjoin(parpath, 'lang_math', 'csv_data', 'summary', 'twin_act.csv'))
unrelated_act = pd.read_csv(pjoin(parpath, 'lang_math', 'csv_data', 'summary', 'unrelated_act.csv'))
# 1) Prepare subject-specific ROIs
def prepare_subjROI(task='lang'):
    """
    """
    parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study'
    roisize = 100
    nsubj_MZ = 129
    nsubj_DZ = 73

    # Load activation data
    iocifti_unrelated = iofiles.make_ioinstance(pjoin(parpath, 'data', 'cohend', 'sm4', 'merge_lang_cohend_unrelated.dscalar.nii'))
    act_func_left, act_func_right = iocifti_unrelated.load_surface_data()
    if task == 'math':
        act_func_left = -1.0*act_func_left
        act_func_right = -1.0*act_func_right

    # Load mask
    iocifti_mask = iofiles.make_ioinstance(pjoin(parpath, 'lang_math', 'apm_ROIs', task+'_mask.dscalar.nii'))
    mask_left, mask_right = iocifti_mask.load_surface_data()

    # Load face
    faces_LH = nib.load('/nfs/p1/atlases/Yeo_templates/surface/fs_LR_32k/fsaverage.L.inflated.32k_fs_LR.surf.gii').darrays[1].data
    faces_RH = nib.load('/nfs/p1/atlases/Yeo_templates/surface/fs_LR_32k/fsaverage.R.inflated.32k_fs_LR.surf.gii').darrays[1].data


    subjmask_lh = tools.rggrow_roi_by_activation(act_func_left, mask_left, faces_LH, vxsize=roisize) 
    subjmask_rh = tools.rggrow_roi_by_activation(act_func_right, mask_right, faces_RH, vxsize=roisize, hemisphere='right')

#     Save subject specific ROIs
    outputpath_roi_lh = pjoin(parpath, 'lang_math', 'subjmask', 'mathROI', 'train', task+'_test_subj_vx'+str(roisize)+'_lh.func.gii')
   outputpath_roi_rh = pjoin(parpath, 'lang_math', 'subjmask', 'mathROI', 'train', task+'_test_subj_vx'+str(roisize)+'_rh.func.gii')
    iogii_lh = iofiles.make_ioinstance(outputpath_roi_lh) 
    iogii_lh.save(subjmask_lh, hemisphere='CortexLeft')
    iogii_rh = iofiles.make_ioinstance(outputpath_roi_rh) 
    iogii_rh.save(subjmask_rh, hemisphere='CortexRight')


# 2) Extract avg signals from unrelated participants
def extract_roisignal():
    """
    """
    actdata, _ = iocifti_unrelated.load_raw_data()
    iocifti_mask_story = iofiles.make_ioinstance(pjoin(parpath, 'lang_math', 'subjmask', 'langROI', 'train', 'lang_test_subj_vx'+str(roisize)+'.dscalar.nii'))
    iocifti_mask_math = iofiles.make_ioinstance(pjoin(parpath, 'lang_math', 'subjmask', 'mathROI', 'train', 'math_test_subj_vx'+str(roisize)+'.dscalar.nii'))
    subjmask_story, _ = iocifti_mask_story.load_raw_data()
    subjmask_math, _ = iocifti_mask_math.load_raw_data()
    avgact_story = tools.extract_avg_signals(actdata, subjmask_story)
    story_roi_name = ['L_story_roi1', 'L_story_roi2', 'L_story_roi3', 'L_story_roi4', 'R_story_roi1', 'R_story_roi2', 'R_story_roi3', 'R_story_roi4']
    avgact_story_pd = pd.DataFrame(avgact_story, columns=story_roi_name)
    avgact_math = tools.extract_avg_signals(actdata, subjmask_math)
    math_roi_name = ['L_math_roi1', 'L_math_roi2', 'L_math_roi3', 'L_math_roi4', 'L_math_roi5', 'L_math_roi6', 'L_math_roi7', 'L_math_roi8', 'L_math_roi9', 'R_math_roi1', 'R_math_roi2', 'R_math_roi3', 'R_math_roi4', 'R_math_roi5', 'R_math_roi6', 'R_math_roi7', 'R_math_roi8', 'R_math_roi9']
    avgact_math_pd = pd.DataFrame(avgact_math, columns=math_roi_name)


# 3) Generate PCA models
def generate_pca_model():
    """
    """
    evr_story, pcacomp_story, corrmat_story, pcamodel_story = tools.pca_decomposition(avgact_story)
    evr_math, pcacomp_math, corrmat_math, pcamodel_math = tools.pca_decomposition(avgact_math)
    # save pca models
    iopkl_story = iofiles.make_ioinstance(pjoin(parpath, 'lang_math', 'train_model', 'pcamodel_story_vx'+str(roisize)+'.pkl'))
    iopkl_math = iofiles.make_ioinstance(pjoin(parpath, 'lang_math', 'train_model', 'pcamodel_math_vx'+str(roisize)+'.pkl'))
    iopkl_story.save(pcamodel_story)
    iopkl_math.save(pcamodel_math)
    

# 4) Extract average signals and PC1
def extract_pc1():
    """
    """
    # Get PC1
    MZ1_story_pc1 = pcamodel_story.transform(avgact_story_MZ1)[:,0]
    MZ2_story_pc1 = pcamodel_story.transform(avgact_story_MZ2)[:,0]
    DZ1_story_pc1 = pcamodel_story.transform(avgact_story_DZ1)[:,0]
    DZ2_story_pc1 = pcamodel_story.transform(avgact_story_DZ2)[:,0]
    MZ1_math_pc1 = pcamodel_math.transform(avgact_math_MZ1)[:,0]
    MZ2_math_pc1 = pcamodel_math.transform(avgact_math_MZ2)[:,0]
    DZ1_math_pc1 = pcamodel_math.transform(avgact_math_DZ1)[:,0]
    DZ2_math_pc1 = pcamodel_math.transform(avgact_math_DZ2)[:,0]
    # Transfer to pandas.Dataframe
    syscomp_pd = prepare_twin_csv(MZ1_story_pc1, DZ1_story_pc1, MZ2_story_pc1, DZ2_story_pc1, MZ1_math_pc1, DZ1_math_pc1, MZ2_math_pc1, DZ2_math_pc1, ['story'], ['math'], zscore=True) 


# 7) Control gender & ages
def control_gender_age():
    """
    """
    parpath = '/nfs/h1/workingshop/huangtaicheng/hcp_test/twin_study/'
    twin_act = pd.read_csv(pjoin(parpath, 'lang_math', 'csv_data', 'summary', 'unrelated_act.csv'))

    twin_actval = np.array(twin_act)[:,3:].astype('float')
    twin_covariance = np.array(twin_act)[:,1:3]
    twin_covariance[twin_covariance == 'F'] = 1
    twin_covariance[twin_covariance == 'M'] = 0
    twin_covariance = twin_covariance.astype('int')
    twin_actval_reg = tools.regressoutvariable(twin_actval, twin_covariance)
   
   
# Extract resting stats signals and calculate correlation matrix
def extract_fc():
    """
    """
    parpath = '/home/ubuntu/workingdir/fcmat'
    hcp_parpath = '/home/ubuntu/s3/hcp'
    with open(pjoin(parpath, 'twinID'), 'r') as f:
        sessid_twins = f.read().splitlines()
    with open(pjoin(parpath, 'sessid_unrelated'), 'r') as f:
        sessid_unrelated = f.read().splitlines()
    # Read mask
    lang_twins, _ = cifti.read(pjoin(parpath, 'twins', 'lang_test_subj_vx100.dscalar.nii'))
    math_twins, _ = cifti.read(pjoin(parpath, 'twins', 'math_test_subj_vx100.dscalar.nii'))
    lang_unrelated, _ = cifti.read(pjoin(parpath, 'train', 'lang_test_subj_vx100.dscalar.nii'))
    math_unrelated, _ = cifti.read(pjoin(parpath, 'train', 'math_test_subj_vx100.dscalar.nii'))
    
    # Twins
    r_fc_twins = []
    for i, sid in enumerate(sessid_twins):
        print('subject {}'.format(sid))
        r_persubj_twins = []
        rest1_LR_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest1_LR_path):
            rest1_LR, _ = cifti.read(rest1_LR_path)
            r_tmp_twins = tools.extract_correlation(rest1_LR, lang_twins[i,...], math_twins[i,...])
            r_persubj_twins.append(r_tmp_twins)

        rest1_RL_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST1_RL', 'rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest1_LR_path):
            rest1_RL, _ = cifti.read(rest1_RL_path)
            r_tmp_twins = tools.extract_correlation(rest1_RL, lang_twins[i,...], math_twins[i,...])
            r_persubj_twins.append(r_tmp_twins)

        rest2_LR_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST2_LR', 'rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest2_LR_path):
            rest2_LR, _ = cifti.read(rest2_LR_path)
            r_tmp_twins = tools.extract_correlation(rest2_LR, lang_twins[i,...], math_twins[i,...])
            r_persubj_twins.append(r_tmp_twins)

        rest2_RL_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST2_RL', 'rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest2_RL_path):
            rest2_RL, _ = cifti.read(rest2_RL_path)
            r_tmp_twins = tools.extract_correlation(rest2_RL, lang_twins[i,...], math_twins[i,...])
            r_persubj_twins.append(r_tmp_twins)

        r_persubj_twins = np.array(r_persubj_twins)
        r_fc_twins.append(np.mean(r_persubj_twins,axis=0))
        time1 = time.time()
        print('time for 1 subject {}'.format(time1-time0))
    r_fc_twins = np.array(r_fc_twins)
    iopkl = iofiles.make_ioinstance('rest_fc_twins.pkl')
    iopkl.save(r_fc_twins)
    
    # Unrelated subjects
    r_fc_unrelated = []
    for i, sid in enumerate(sessid_unrelated):
        print('subject {}'.format(sid))
        r_persubj_unrelated = []

        rest1_LR_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest1_LR_path):
            rest1_LR, _ = cifti.read(rest1_LR_path)
            r_tmp_unrelated = tools.extract_correlation(rest1_LR, lang_unrelated[i,...], math_unrelated[i,...])
            r_persubj_unrelated.append(r_tmp_unrelated)

        rest1_RL_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST1_RL', 'rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest1_RL_path):
            rest1_RL, _ = cifti.read(rest1_RL_path)
            r_tmp_unrelated = tools.extract_correlation(rest1_RL, lang_unrelated[i,...], math_unrelated[i,...])
            r_persubj_unrelated.append(r_tmp_unrelated)

        rest2_LR_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST2_LR', 'rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest2_LR_path):
            rest2_LR, _ = cifti.read(rest2_LR_path)
            r_tmp_unrelated = tools.extract_correlation(rest2_LR, lang_unrelated[i,...], math_unrelated[i,...])
            r_persubj_unrelated.append(r_tmp_unrelated)

        rest2_RL_path = pjoin(hcp_parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST2_RL', 'rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii')
        if os.path.exists(rest2_RL_path):
            rest2_RL, _ = cifti.read(rest2_RL_path)
            r_tmp_twins = tools.extract_correlation(rest2_RL, lang_unrelated[i,...], math_unrelated[i,...])
            r_persubj_unrelated.append(r_tmp_unrelated)

        r_persubj_unrelated = np.array(r_persubj_unrelated)
        r_fc_unrelated.append(np.mean(r_persubj_unrelated,axis=0)) 
    r_fc_unrelated = np.array(r_fc_unrelated)
    iopkl = iofiles.make_ioinstance('rest_fc_unrelated.pkl')
    iopkl.save(r_fc_unrelated)
