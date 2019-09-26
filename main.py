from StoryMath_heritability.core import tools
from os.path import join as pjoin
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pre-defined valuables
package_path = '/'.join(tools.__file__.split('/')[:-2])
parpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study/'
nsubj_twins = 404
nsubj_unrelated = 434
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
with open(pjoin(package_path, 'data', 'rest_fc_unrelated.pkl'), 'rb') as f:
    fc_corr_unrelated = pickle.load(f, encoding='iso-8859-1')
with open(pjoin(package_path, 'data', 'rest_fc_twins.pkl'), 'rb') as f:
    fc_corr_twins = pickle.load(f, encoding='iso-8859-1')

fc_corr_unrelated_story = fc_corr_unrelated[:,:8,:8]
fc_corr_unrelated_math = fc_corr_unrelated[:,8:,8:]
fc_corr_twins_story = fc_corr_twins[:,:8,:8]
fc_corr_twins_math = fc_corr_twins[:,8:,8:]
    # Get their values from upper triangular matrix
story_mat_fc_unrelated = tools.get_uppertriu_value(fc_corr_unrelated_story)
math_mat_fc_unrelated  = tools.get_uppertriu_value(fc_corr_unrelated_math)
story_mat_fc_twins = tools.get_uppertriu_value(fc_corr_twins_story)
math_mat_fc_twins = tools.get_uppertriu_value(fc_corr_twins_math)
    # PCA decomposition
evr_fc_story, pcacomp_fc_story, corrmat_fc_pcastory, pcamodel_fc_story = tools.pca_decomposition(story_mat_fc_unrelated) 
evr_fc_math, pcacomp_fc_math, corrmat_fc_pcamath, pcamodel_fc_math = tools.pca_decomposition(math_mat_fc_unrelated) 

evr_fc_btw, pcacomp_fc_btw, corrmat_fc_pcabtw, pcamodel_fc_btw = tools.pca_decomposition(fc_corr_unrelated[:,:8,8:].reshape((nsubj_unrelated,-1)))
    # extract the first component 
pc1_storyfc_MZ1 = -1.0*pcamodel_fc_story.transform(story_mat_fc_twins[:129,:])[:,0]
pc1_storyfc_MZ2 = -1.0*pcamodel_fc_story.transform(story_mat_fc_twins[129:2*129,:])[:,0]
pc1_storyfc_DZ1 = -1.0*pcamodel_fc_story.transform(story_mat_fc_twins[2*129:2*129+73,:])[:,0]
pc1_storyfc_DZ2 = -1.0*pcamodel_fc_story.transform(story_mat_fc_twins[2*129+73:,:])[:,0]
pc1_mathfc_MZ1 = pcamodel_fc_math.transform(math_mat_fc_twins[:129,:])[:,0]
pc1_mathfc_MZ2 = pcamodel_fc_math.transform(math_mat_fc_twins[129:2*129,:])[:,0]
pc1_mathfc_DZ1 = pcamodel_fc_math.transform(math_mat_fc_twins[2*129:2*129+73,:])[:,0]
pc1_mathfc_DZ2 = pcamodel_fc_math.transform(math_mat_fc_twins[2*129+73:,:])[:,0]

pc1_btwfc = pcamodel_fc_btw.transform(fc_corr_twins[:,:8,8:].reshape((nsubj_twins,-1)))[:,0]
pc1_btwfc_MZ1 = pc1_btwfc[:129]
pc1_btwfc_MZ2 = pc1_btwfc[129:2*129]
pc1_btwfc_DZ1 = pc1_btwfc[2*129:2*129+73]
pc1_btwfc_DZ2 = pc1_btwfc[2*129+73:]

    # Regress within-network connectivity from between-network connectivity
pc1_storyfc = np.concatenate((pc1_storyfc_MZ1, pc1_storyfc_MZ2, pc1_storyfc_DZ1, pc1_storyfc_DZ2))    
pc1_mathfc = np.concatenate((pc1_mathfc_MZ1, pc1_mathfc_MZ2, pc1_mathfc_DZ1, pc1_mathfc_DZ2))   
pc1_residuefc = tools.regressoutvariable(pc1_btwfc[:,None], np.vstack((pc1_storyfc, pc1_mathfc)).T)[:,0] 
pc1_residuefc_MZ1 = pc1_residuefc[:129]
pc1_residuefc_MZ2 = pc1_residuefc[129:129*2]
pc1_residuefc_DZ1 = pc1_residuefc[129*2:129*2+73]
pc1_residuefc_DZ2 = pc1_residuefc[129*2+73:]

    # Intra-class correlation & falconer's heritability
MZ_storyfc_icc, _ = tools.icc(np.vstack((pc1_storyfc_MZ1,pc1_storyfc_MZ2)).T, methods='(3,1)')
DZ_storyfc_icc, _ = tools.icc(np.vstack((pc1_mathfc_DZ1,pc1_mathfc_DZ2)).T, methods='(3,1)')
h2_storyfc_falc = 2*(MZ_storyfc_icc - DZ_storyfc_icc)
MZ_mathfc_icc, _ = tools.icc(np.vstack((pc1_mathfc_MZ1,pc1_mathfc_MZ2)).T, methods='(3,1)')
DZ_mathfc_icc, _ = tools.icc(np.vstack((pc1_mathfc_DZ1,pc1_mathfc_DZ2)).T, methods='(3,1)')
h2_mathfc_falc = 2*(MZ_mathfc_icc - DZ_mathfc_icc)

MZ_btwfc_icc, _ = tools.icc(np.vstack((pc1_btwfc_MZ1, pc1_btwfc_MZ2)).T, methods='(3,1)')
DZ_btwfc_icc, _ = tools.icc(np.vstack((pc1_btwfc_DZ1, pc1_btwfc_DZ2)).T, methods='(3,1)')
h2_btwfc_falc = 2*(MZ_btwfc_icc - DZ_btwfc_icc)

MZ_residuefc_icc, _ = tools.icc(np.vstack((pc1_residuefc_MZ1, pc1_residuefc_MZ2)).T, methods='(3,1)')
DZ_residuefc_icc, _ = tools.icc(np.vstack((pc1_residuefc_DZ1, pc1_residuefc_DZ2)).T, methods='(3,1)')
h2_residuefc_falc = 2*(MZ_residuefc_icc - DZ_residuefc_icc)

    # Generate pandas.Dataframe and stored it for ACE/AE model
pc1_pdfc_system = tools.prepare_twin_csv(pc1_storyfc_MZ1, pc1_storyfc_DZ1, pc1_storyfc_MZ2, pc1_storyfc_DZ2, pc1_mathfc_MZ1, pc1_mathfc_DZ1, pc1_mathfc_MZ2, pc1_mathfc_DZ2, ['story'], ['math'])

# ---------------------------
# Plot figures
# Read heritability estimating from AE models and plot figures
    # system/behavior heritability
      # system
sysbsp_comp = pd.read_csv(pjoin(package_path, 'data', 'AE_h2estimate_fc_vx100.csv'))
story_h2 = sysbsp_comp['story']
math_h2 = sysbsp_comp['math']
mean_story_h2 = np.mean(sysbsp_comp['story'])
std_story_h2 = np.std(sysbsp_comp['story'])
mean_math_h2 = np.mean(sysbsp_comp['math'])
std_math_h2 = np.std(sysbsp_comp['math'])
left, width = 0.1, 0.65
bottom, height = 0.1, 0.85
spacing = 0.005
rect_errorbar = [left, bottom, width, height]
plt.figure()
ax_errorbar = plt.axes(rect_errorbar)
ax_errorbar.errorbar([1], mean_story_h2, yerr=std_story_h2, fmt='ro')
ax_errorbar.errorbar([2], mean_math_h2, yerr=std_math_h2, fmt='bo')
plt.xticks(np.arange(4), ['','story','math',''])
plt.yticks(np.arange(0,1.1,0.1))
rect_histy = [left+width+spacing, bottom, 0.2, height]
ax_histy = plt.axes(rect_histy)
ax_histy.hist(np.array(story_h2), orientation='horizontal', color='r')
ax_histy.hist(np.array(math_h2), orientation='horizontal', color='b')
plt.yticks(np.arange(0,1.1,0.1), [])
plt.xticks([])
plt.legend(['story', 'math'])
plt.show()


