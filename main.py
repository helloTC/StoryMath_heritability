from StoryMath_heritability.core import tools
from os.path import join as pjoin
import pickle
import numpy as np
import pandas as pd

# Pre-defined valuables
package_path = '/'.join(tools.__file__.split('/')[:-2])
parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study/'
nsubj_twins = 404
nsubj_unrelated = 436
# ---------------------------
# Extract activations
    # Load activation and covariances
unrelated_act = pd.read_csv(pjoin(package_path, 'data', 'unrelated_act.csv'))
twin_act = pd.read_csv(pjoin(package_path, 'data', 'twin_act.csv'))
unrelated_actval = np.array(unrelated_act)[:,3:].astype('float')
unrelated_covariance = np.array(unrelated_act)[:,1:3]
unrelated_covariance[unrelated_covariance=='F'] = 1
unrelated_covariance[unrelated_covariance=='M'] = 0
unrelated_covariance = unrelated_covariance.astype('int')
twin_actval = np.array(twin_act)[:,3:].astype('float')
twin_covariance = np.array(twin_act)[:,1:3]
twin_covariance[twin_covariance=='F'] = 1
twin_covariance[twin_covariance=='M'] = 0
twin_covariance = twin_covariance.astype('int')
    # Load get pcamodel/or just load model from data 
evr_act_story, pcacomp_act_story, corrmat_act_story, pcamodel_act_story = tools.pca_decomposition(unrelated_actval[:,:8])    
evr_act_math, pcacomp_act_math, corrmat_act_math, pcamodel_act_math = tools.pca_decomposition(-1.0*unrelated_actval[:,8:])    
    # Extract the first component from activation in Twins
pc1_storyact_MZ1 = pcamodel_act_story.transform(twin_actval[:129,:8])[:,0]
pc1_storyact_MZ2 = pcamodel_act_story.transform(twin_actval[129:2*129,:8])[:,0]
pc1_storyact_DZ1 = pcamodel_act_story.transform(twin_actval[2*129:2*129+73,:8])[:,0]
pc1_storyact_DZ2 = pcamodel_act_story.transform(twin_actval[2*129+73:,:8])[:,0]
pc1_mathact_MZ1 = pcamodel_act_math.transform(-1.0*twin_actval[:129,8:])[:,0]
pc1_mathact_MZ2 = pcamodel_act_math.transform(-1.0*twin_actval[129:2*129,8:])[:,0]
pc1_mathact_DZ1 = pcamodel_act_math.transform(-1.0*twin_actval[2*129:2*129+73,8:])[:,0]
pc1_mathact_DZ2 = pcamodel_act_math.transform(-1.0*twin_actval[2*129+73:,8:])[:,0]
    # Intra-class correlation
MZ_storyact_icc, _ = tools.icc(np.vstack((pc1_storyact_MZ1,pc1_storyact_MZ2)).T, methods='(3,1)')
DZ_storyact_icc, _ = tools.icc(np.vstack((pc1_storyact_DZ1,pc1_storyact_DZ2)).T, methods='(3,1)')
h2_storyact_falc = 2*(MZ_storyact_icc - DZ_storyact_icc)
MZ_mathact_icc, _ = tools.icc(np.vstack((pc1_mathact_MZ1,pc1_mathact_MZ2)).T, methods='(3,1)')
DZ_mathact_icc, _ = tools.icc(np.vstack((pc1_mathact_DZ1,pc1_mathact_DZ2)).T, methods='(3,1)')
h2_mathact_falc = 2*(MZ_mathact_icc - DZ_mathact_icc)
    # Generate pandas.Dataframe and stored it for ACE/AE model
pc1_pdact_system = tools.prepare_twin_csv(pc1_storyact_MZ1, pc1_storyact_DZ1, pc1_storyact_MZ2, pc1_storyact_DZ2, pc1_mathact_MZ1, pc1_mathact_DZ1, pc1_mathact_MZ2, pc1_mathact_DZ2, ['story'], ['math'])

# ---------------------------
# FC connectivity activation
with open(pjoin(parpath, 'lang_math', 'rest_timeseries', 'rest_fc_twins.pkl'), 'rb') as f:
    fc_corr = pickle.load(f, encoding='iso-8859-1')
    # Set element in diagonal as 0
for i in range(nsubj_twins):
     np.fill_diagonal(fc_corr[i,...],0)
story_mat_fc = fc_corr[:,:8,:8].reshape((nsubj_twins,-1))
math_mat_fc = fc_corr[:,8:,8:].reshape((nsubj_twins,-1))
    # PCA decomposition
evr_fc_story, pcacomp_fc_story, corrmat_fc_pcastory, pcamodel_fc_story = tools.pca_decomposition(story_mat_fc) 
evr_fc_math, pcacomp_fc_math, corrmat_fc_pcamath, pcamodel_fc_math = tools.pca_decomposition(math_mat_fc) 
    # extract the first component 
pc1_storyfc_MZ1 = pcamodel_fc_story.transform(story_mat_fc[:129,:])[:,0]
pc1_storyfc_MZ2 = pcamodel_fc_story.transform(story_mat_fc[129:2*129,:])[:,0]
pc1_storyfc_DZ1 = pcamodel_fc_story.transform(story_mat_fc[2*129:2*129+73,:])[:,0]
pc1_storyfc_DZ2 = pcamodel_fc_story.transform(story_mat_fc[2*129+73:,:])[:,0]
pc1_mathfc_MZ1 = pcamodel_fc_math.transform(math_mat_fc[:129,:])[:,0]
pc1_mathfc_MZ2 = pcamodel_fc_math.transform(math_mat_fc[129:2*129,:])[:,0]
pc1_mathfc_DZ1 = pcamodel_fc_math.transform(math_mat_fc[2*129:2*129+73,:])[:,0]
pc1_mathfc_DZ2 = pcamodel_fc_math.transform(math_mat_fc[2*129+73:,:])[:,0]
    # Intra-class correlation & falconer's heritability
MZ_storyfc_icc, _ = tools.icc(np.vstack((pc1_storyfc_MZ1,pc1_storyfc_MZ2)).T, methods='(3,1)')
DZ_storyfc_icc, _ = tools.icc(np.vstack((pc1_mathfc_DZ1,pc1_mathfc_DZ2)).T, methods='(3,1)')
h2_storyfc_falc = 2*(MZ_storyfc_icc - DZ_storyfc_icc)
MZ_mathfc_icc, _ = tools.icc(np.vstack((pc1_mathfc_MZ1,pc1_mathfc_MZ2)).T, methods='(3,1)')
DZ_mathfc_icc, _ = tools.icc(np.vstack((pc1_mathfc_DZ1,pc1_mathfc_DZ2)).T, methods='(3,1)')
h2_mathfc_falc = 2*(MZ_mathfc_icc - DZ_mathfc_icc)
    # Generate pandas.Dataframe and stored it for ACE/AE model
pc1_pdfc_system = tools.prepare_twin_csv(pc1_storyfc_MZ1, pc1_storyfc_DZ1, pc1_storyfc_MZ2, pc1_storyfc_DZ2, pc1_mathfc_MZ1, pc1_mathfc_DZ1, pc1_mathfc_MZ2, pc1_mathfc_DZ2, ['story'], ['math'])
# ---------------------------
