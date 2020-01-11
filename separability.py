# Streamline to compute separability

import cifti
import nibabel as nib
from os.path import join as pjoin
from ATT.algorithm import surf_tools, tools
from ATT.iofunc import iofiles
import numpy as np
import pandas as pd

class separability(object):
    """
    Class for calculating separability.
    """
    def __init__(self, parpath, sessid):
        self.sessid = sessid
        self.parpath = parpath
        self.subjnum = len(sessid)

    def load_mask(self, mask_parpath):
        """
        Load mask
        ----------
        mask_parpath[list]: two elements
        """
        # Load mask
        langroi, _ = cifti.read(mask_parpath[0])
        mathroi, _ = cifti.read(mask_parpath[1])
        langroi = langroi[:,:59412]
        mathroi = mathroi[:,:59412]
        if langroi.shape[0] == 1:
            self.langroi = np.tile(langroi, (self.subjnum, 1))
        else:
            self.langroi = langroi
        if mathroi.shape[0] == 1:
            self.mathroi = np.tile(mathroi, (self.subjnum, 1))
        else:
            self.mathroi = mathroi
        # Get mask labels 
        self.langlbl = np.unique(langroi[langroi!=0])
        self.mathlbl = np.unique(mathroi[mathroi!=0])
        self.region_num = len(self.langlbl) + len(self.mathlbl)

    def load_activation(self, actpath=None, outpath=None):
        """
        Method to load myelination and extract the averaged signals in each ROIs.
        Please call load_mask before it.

        Parameters:
        ------------
        myelinpath[str]: parent path of myelination
        """
        print('Activation extraction....')
        if actpath is None:
            actpath = self.parpath
        act_value = []
        for i, sj in enumerate(self.sessid):
            actmap_tmp, _ = cifti.read(pjoin(actpath, sj, 'MNINonLinear', 'Results', 'tfMRI_LANGUAGE', 'tfMRI_LANGUAGE_hp200_s4_level2_MSMAll.feat', 'GrayordinatesStats', 'cope3.feat', 'zstat1.dtseries.nii'))
            actmap_tmp = actmap_tmp[:,:59412]
            actlang_tmp = surf_tools.get_signals(-1.0*actmap_tmp, self.langroi[i,:], roilabels=self.langlbl)
            actlang_tmp = np.array(actlang_tmp)
            actmath_tmp = surf_tools.get_signals(actmap_tmp, self.mathroi[i,:], roilabels=self.mathlbl)
            actmath_tmp = np.array(actmath_tmp)
            actvalue_tmp = np.concatenate((actlang_tmp, actmath_tmp), axis=0)[:,0]
            act_value.append(actvalue_tmp)
        act_value = np.array(act_value)
        self.act_value = act_value
        if outpath is not None:
            np.save(pjoin(outpath, 'act_value.npy'), act_value)
        return  act_value

    def load_myelin(self, myelinpath=None, outpath=None):
        """
        Method to load myelination and extract the averaged signals in each ROIs.
        Please call load_mask before it.

        Parameters:
        ------------
        myelinpath[str]: parent path of myelination
        """
        print('Myelin extraction....')
        if myelinpath is None:
            myelinpath = self.parpath
        myelin_value = []
        for i, sj in enumerate(self.sessid):
            myelinmap_tmp, _ = cifti.read(pjoin(myelinpath, sj, 'MNINonLinear', 'fsaverage_LR32k', sj+'.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii'))
            myelinmap_tmp = myelinmap_tmp[:,:59412]
            myelinlang_tmp = surf_tools.get_signals(myelinmap_tmp, self.langroi[i,:], roilabels=self.langlbl)
            myelinlang_tmp = np.array(myelinlang_tmp)
            myelinmath_tmp = surf_tools.get_signals(myelinmap_tmp, self.mathroi[i,:], roilabels=self.mathlbl)
            myelinmath_tmp = np.array(myelinmath_tmp)
            myelinvalue_tmp = np.concatenate((myelinlang_tmp, myelinmath_tmp), axis=0)[:,0]
            myelin_value.append(myelinvalue_tmp)
        myelin_value = np.array(myelin_value)
        self.myelin_value = myelin_value
        if outpath is not None:
            np.save(pjoin(outpath, 'myelin_value.npy'), myelin_value)
        return  myelin_value

    def load_thickness(self, thickpath=None, outpath=None):
        """
        Method to load thickness and extract the averaged signals in each ROIs.
        
        """
        print('Thickness extraction....')
        if thickpath is None:
            thickpath = self.parpath
        thick_value = []
        for i, sj in enumerate(self.sessid):
            thickmap_tmp, _ = cifti.read(pjoin(thickpath, sj, 'MNINonLinear', 'fsaverage_LR32k', sj+'.thickness_MSMAll.32k_fs_LR.dscalar.nii'))
            thickmap_tmp = thickmap_tmp[:,:59412]
            thicklang_tmp = surf_tools.get_signals(thickmap_tmp, self.langroi[i,:], roilabels=self.langlbl)
            thicklang_tmp = np.array(thicklang_tmp)
            thickmath_tmp = surf_tools.get_signals(thickmap_tmp, self.mathroi[i,:], roilabels=self.mathlbl)
            thickmath_tmp = np.array(thickmath_tmp)
            thickvalue_tmp = np.concatenate((thicklang_tmp, thickmath_tmp), axis=0)[:,0]
            thick_value.append(thickvalue_tmp)
        thick_value = np.array(thick_value)
        self.thick_value = thick_value
        if outpath is not None:
            np.save(pjoin(outpath, 'thick_value.npy'), thick_value)
        return  thick_value

    def _load_rsMRI_tmseries(self, sid, parpath=None):
        """
        Load rsMRI time series of one participant
        """
        if parpath is None:
            parpath = self.parpath
        rsMRI = np.zeros((1200, 59412))
        indicator = 0
        # Load LR1
        try:
            rsMRI_tmp, _ = cifti.read(pjoin(parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST1_LR', 'rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii'))
        except IOError: 
            indicator+=1
        try:
            rsMRI+=rsMRI_tmp[:,:59412]
        except (ValueError,UnboundLocalError):
            pass
        print('.LR1')
        # Load RL1
        try:
            rsMRI_tmp, _ = cifti.read(pjoin(parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST1_RL', 'rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii'))
        except IOError:
            indicator+=1
            
        try:
            rsMRI+=rsMRI_tmp[:,:59412]
        except (ValueError,UnboundLocalError):
            pass
        print('.RL1')
        # Load LR2
        try:
            rsMRI_tmp, _ = cifti.read(pjoin(parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST2_LR', 'rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii'))
        except IOError:
            indicator+=1
        try:
            rsMRI+=rsMRI_tmp[:,:59412]
        except (ValueError,UnboundLocalError):
            pass
        print('.LR2')
        # Load RL2
        try:
            rsMRI_tmp, _ = cifti.read(pjoin(parpath, sid, 'MNINonLinear', 'Results', 'rfMRI_REST2_RL', 'rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'))
        except IOError:
            indicator+=1
        try:
            rsMRI+=rsMRI_tmp[:,:59412]
        except (ValueError,UnboundLocalError):
            pass
        print('.RL2')
        rsMRI = rsMRI/(4-indicator)
        return rsMRI

    def _seg_hemisphere(self, diff_mat):
        """
        """
        langnum_hemi = int(len(self.langlbl)/2) # 4
        mathnum_hemi = int(len(self.mathlbl)/2) # 9
        diff_mat_left = np.zeros((self.subjnum, int(self.region_num/2), int(self.region_num/2)))
        diff_mat_right = np.zeros_like(diff_mat_left)
        diff_mat_left[:,:langnum_hemi,:langnum_hemi] = diff_mat[:,:langnum_hemi,:langnum_hemi]
        diff_mat_left[:,langnum_hemi:,langnum_hemi:] = diff_mat[:,2*langnum_hemi:(2*langnum_hemi+mathnum_hemi),2*langnum_hemi:(2*langnum_hemi+mathnum_hemi)]
        diff_mat_left[:,langnum_hemi:,:langnum_hemi] = diff_mat[:,2*langnum_hemi:(2*langnum_hemi+mathnum_hemi),:langnum_hemi]
        diff_mat_left[:,:langnum_hemi,langnum_hemi:] = diff_mat[:,:langnum_hemi,2*langnum_hemi:(2*langnum_hemi+mathnum_hemi)]

        diff_mat_right[:,:langnum_hemi,:langnum_hemi] = diff_mat[:,langnum_hemi:2*langnum_hemi,langnum_hemi:2*langnum_hemi]
        diff_mat_right[:,langnum_hemi:,langnum_hemi:] = diff_mat[:,2*langnum_hemi:(2*langnum_hemi+mathnum_hemi),2*langnum_hemi:(2*langnum_hemi+mathnum_hemi)]
        diff_mat_right[:,langnum_hemi:,:langnum_hemi] = diff_mat[:,2*langnum_hemi:(2*langnum_hemi+mathnum_hemi),langnum_hemi:2*langnum_hemi]
        diff_mat_right[:,:langnum_hemi,langnum_hemi:] = diff_mat[:,langnum_hemi:2*langnum_hemi,(2*langnum_hemi+mathnum_hemi):]
        return diff_mat_left, diff_mat_right 

    def calc_rsMRI_fcdist(self, parpath=None, outpath=None):
        """
        Load and calculate functional connectivity
        """
        print('rsMRI time series extraction....')
        if parpath is None:
            parpath = self.parpath
        func_dist = [] 
        for i, sj in enumerate(self.sessid):
            print('{}: Subject {}...'.format(i+1, sj))
            rsMRI = self._load_rsMRI_tmseries(sj, parpath=parpath)
            # extract averaged timeseries
            avgtmseries_lang = surf_tools.get_signals(rsMRI, self.langroi[i,:], roilabels=self.langlbl)
            avgtmseries_math = surf_tools.get_signals(rsMRI, self.mathroi[i,:], roilabels=self.mathlbl)
            avgtmseries = np.concatenate((avgtmseries_lang, avgtmseries_math),axis=0)
            # calculate functional correlation
            r, _ = tools.pearsonr(avgtmseries, avgtmseries)
            func_dist.append(r)
        func_dist = 1-np.array(func_dist)
        self.func_dist_left, self.func_dist_right = self._seg_hemisphere(func_dist)
        if outpath is not None:
            np.save(pjoin(outpath, 'func_dist_left.npy'), self.func_dist_left) 
            np.save(pjoin(outpath, 'func_dist_right.npy'), self.func_dist_right)  
    def calc_absolutediff(self, outpath=None):
        """
        """
        langnum = len(self.langlbl)
        mathnum = len(self.mathlbl)
        if 'myelin_value' in self.__dict__.keys():
            # Get difference matrix
            myelin_diff = np.zeros((self.subjnum, self.region_num, self.region_num))
            for i in range(self.region_num):
                for j in range(i, self.region_num):
                    myelin_diff[:,i,j] = np.abs(self.myelin_value[:,i]-self.myelin_value[:,j])
            for i in range(self.subjnum):
                myelin_diff[i,...] = myelin_diff[i,...] + myelin_diff[i,...].T 
            self.myelin_diff_left, self.myelin_diff_right = self._seg_hemisphere(myelin_diff)
            if outpath is not None:
                np.save(pjoin(outpath, 'myelin_diff_left.npy'), self.myelin_diff_left)
                np.save(pjoin(outpath, 'myelin_diff_right.npy'), self.myelin_diff_right)

        if 'thick_value' in self.__dict__.keys():
            thick_diff = np.zeros((self.subjnum, self.region_num, self.region_num))
            for i in range(self.region_num):
                for j in range(i, self.region_num):
                    thick_diff[:,i,j] = np.abs(self.thick_value[:,i]-self.thick_value[:,j])
            for i in range(self.subjnum):
                thick_diff[i,...] = thick_diff[i,...] + thick_diff[i,...].T
            self.thick_diff_left, self.thick_diff_right = self._seg_hemisphere(thick_diff)
            if outpath is not None:
                np.save(pjoin(outpath, 'thick_diff_left.npy'), self.thick_diff_left)
                np.save(pjoin(outpath, 'thick_diff_right.npy'), self.thick_diff_right)

    def calc_segregation(self):
        """
        """
        print('Calculate segregation...')
        langrg_num = int(len(self.langlbl)/2)
        mathrg_num = int(len(self.mathlbl)/2)
        if 'myelin_diff_left' in self.__dict__.keys():
            self.segre_myelin_left = self._calc_segre(self.myelin_diff_left, langrg_num, mathrg_num)
        if 'myelin_diff_right' in self.__dict__.keys():
            self.segre_myelin_right = self._calc_segre(self.myelin_diff_right, langrg_num, mathrg_num)
        if 'thick_diff_left' in self.__dict__.keys():
            self.segre_thick_left = self._calc_segre(self.thick_diff_left, langrg_num, mathrg_num)
        if 'thick_diff_right' in self.__dict__.keys():
            self.segre_thick_right = self._calc_segre(self.thick_diff_right, langrg_num, mathrg_num)
        if 'func_dist_left' in self.__dict__.keys():
            self.segre_fc_left = self._calc_segre(self.func_dist_left, langrg_num, mathrg_num)
        if 'func_dist_right' in self.__dict__.keys():
            self.segre_fc_right = self._calc_segre(self.func_dist_right, langrg_num, mathrg_num)

    def _calc_segre(self, diff_mat, langrg_num, mathrg_num):
        """
        """
        segre_metric = []
        for i in range(self.subjnum):
            lang_within_mat = diff_mat[i,:langrg_num,:langrg_num]
            lang_within_tmp = lang_within_mat[np.triu_indices(langrg_num,1)]
            math_within_mat = diff_mat[i,langrg_num:,langrg_num:]
            math_within_tmp = math_within_mat[np.triu_indices(mathrg_num,1)]
            within_tmp = np.concatenate((lang_within_tmp, math_within_tmp))
            langmath_between_tmp = diff_mat[i,:langrg_num, langrg_num:].flatten()
            # Pooling STD
            pooled_std = np.sqrt(((len(within_tmp)-1)*within_tmp.var()+(len(langmath_between_tmp)-1)*langmath_between_tmp.var())/(len(langmath_between_tmp)+len(within_tmp)-2))
            segre_metric_tmp = (langmath_between_tmp.mean() - within_tmp.mean())/pooled_std
            segre_metric.append(segre_metric_tmp)
        segre_metric = np.array(segre_metric)
        return segre_metric

    def write_segre(self, outpath_csv, MZ_num=129, DZ_num=73):
        """
        Note that the arrangement of subjects is:
        MZ1 --- MZ2 --- DZ1 --- DZ2
        """
        assert 2*(MZ_num+DZ_num) == self.subjnum, "Subject mismatched."
        output_csv = {}
        if 'segre_fc_left' in self.__dict__.keys():
            segfc_left_MZ1 = self.segre_fc_left[:MZ_num]
            segfc_left_MZ2 = self.segre_fc_left[MZ_num:2*MZ_num]
            segfc_left_DZ1 = self.segre_fc_left[2*MZ_num:2*MZ_num+DZ_num]
            segfc_left_DZ2 = self.segre_fc_left[2*MZ_num+DZ_num:]
            output_csv['segre_fc_lh1'] = np.concatenate((segfc_left_MZ1, segfc_left_DZ1))
            output_csv['segre_fc_lh2'] = np.concatenate((segfc_left_MZ2, segfc_left_DZ2))
        if 'segre_fc_right' in self.__dict__.keys():
            segfc_right_MZ1 = self.segre_fc_right[:MZ_num]
            segfc_right_MZ2 = self.segre_fc_right[MZ_num:2*MZ_num]
            segfc_right_DZ1 = self.segre_fc_right[2*MZ_num:2*MZ_num+DZ_num]
            segfc_right_DZ2 = self.segre_fc_right[2*MZ_num+DZ_num:]
            output_csv['segre_fc_rh1'] = np.concatenate((segfc_right_MZ1, segfc_right_DZ1))
            output_csv['segre_fc_rh2'] = np.concatenate((segfc_right_MZ2, segfc_right_DZ2))
        if 'segre_myelin_left' in self.__dict__.keys():
            segml_left_MZ1 = self.segre_myelin_left[:MZ_num]
            segml_left_MZ2 = self.segre_myelin_left[MZ_num:2*MZ_num]
            segml_left_DZ1 = self.segre_myelin_left[2*MZ_num:2*MZ_num+DZ_num]
            segml_left_DZ2 = self.segre_myelin_left[2*MZ_num+DZ_num:]
            output_csv['segre_myelin_lh1'] = np.concatenate((segml_left_MZ1, segml_left_DZ1))
            output_csv['segre_myelin_lh2'] = np.concatenate((segml_left_MZ2, segml_left_DZ2))
        if 'segre_myelin_right' in self.__dict__.keys():
            segml_right_MZ1 = self.segre_myelin_right[:MZ_num]
            segml_right_MZ2 = self.segre_myelin_right[MZ_num:2*MZ_num]
            segml_right_DZ1 = self.segre_myelin_right[2*MZ_num:2*MZ_num+DZ_num]
            segml_right_DZ2 = self.segre_myelin_right[2*MZ_num+DZ_num:]
            output_csv['segre_myelin_rh1'] = np.concatenate((segml_right_MZ1, segml_right_DZ1))
            output_csv['segre_myelin_rh2'] = np.concatenate((segml_right_MZ2, segml_right_DZ2))
        if 'segre_thick_left' in self.__dict__.keys():
            segtk_left_MZ1 = self.segre_thick_left[:MZ_num]
            segtk_left_MZ2 = self.segre_thick_left[MZ_num:2*MZ_num]
            segtk_left_DZ1 = self.segre_thick_left[2*MZ_num:2*MZ_num+DZ_num]
            segtk_left_DZ2 = self.segre_thick_left[2*MZ_num+DZ_num:]
            output_csv['segre_thick_lh1'] = np.concatenate((segtk_left_MZ1, segtk_left_DZ1))
            output_csv['segre_thick_lh2'] = np.concatenate((segtk_left_MZ2, segtk_left_DZ2))
        if 'segre_thick_right' in self.__dict__.keys():
            segtk_right_MZ1 = self.segre_thick_right[:MZ_num] 
            segtk_right_MZ2 = self.segre_thick_right[MZ_num:2*MZ_num] 
            segtk_right_DZ1 = self.segre_thick_right[2*MZ_num:2*MZ_num+DZ_num] 
            segtk_right_DZ2 = self.segre_thick_right[2*MZ_num+DZ_num:]
            output_csv['segre_thick_rh1'] = np.concatenate((segtk_right_MZ1, segtk_right_DZ1))
            output_csv['segre_thick_rh2'] = np.concatenate((segtk_right_MZ2, segtk_right_DZ2))
        output_csv['zyg'] = np.concatenate((np.ones((MZ_num)), 3.0*np.ones((DZ_num))))
        output_csv = pd.DataFrame(output_csv)
        output_csv.to_csv(outpath_csv, index=False)


class SubjectROI(object):
    """
    """
    def __init__(self, parpath, sessid):
        self.parpath = parpath
        self.sessid = sessid
        self.subjnum = len(sessid)   

    def _load_surface_cifti(self, datapath):
        """
        """
        data, header = cifti.read(datapath)
        vxidx = header[1].get_element
        assert (data.shape[1] == 59412) | (data.shape[1] == 91282), "Not in fsLR_32k space."
        vxid_left = [vxidx(i)[1] for i in range(29696)]
        vxid_right = [vxidx(i)[1] for i in range(29696,59412)]
        data_left = np.zeros((data.shape[0], 32492))
        data_right = np.zeros((data.shape[0], 32492))
        data_left[:,vxid_left] = data[:,:29696]
        data_right[:,vxid_right] = data[:,29696:59412]
        return data_left, data_right     

    def load_actdata(self, act_parpath=None, cope=3):
        """
        """
        print('Loading activation data...')
        if act_parpath is None:
            act_parpath = self.parpath
        act_left = np.zeros((self.subjnum, 32492))
        act_right = np.zeros((self.subjnum, 32492))
        for i, sid in enumerate(self.sessid):
            act_tmp_left, act_tmp_right = self._load_surface_cifti(pjoin(act_parpath, sid, 'MNINonLinear', 'Results', 'tfMRI_LANGUAGE', 'tfMRI_LANGUAGE_hp200_s4_level2_MSMAll.feat', 'GrayordinatesStats', 'cope'+str(cope)+'.feat', 'zstat1.dtseries.nii'))
            act_left[i,...] = act_tmp_left.flatten()
            act_right[i,...] = act_tmp_right.flatten()
        self.act_left = act_left
        self.act_right = act_right

    def load_apmmask(self, mask_parpath):
        """
        """
        print('Loading probabilistic mask...')
        langroi_left, langroi_right = self._load_surface_cifti(mask_parpath[0])
        mathroi_left, mathroi_right = self._load_surface_cifti(mask_parpath[1])
        self.langroi_left = langroi_left
        self.langroi_right = langroi_right
        self.mathroi_left = mathroi_left
        self.mathroi_right = mathroi_right

    def load_faces(self, faces_parpath):
        """
        """
        print('Loading faces...')
        faces_left = nib.load(faces_parpath[0]).darrays[1].data
        faces_right = nib.load(faces_parpath[1]).darrays[1].data
        self.faces_left = faces_left
        self.faces_right = faces_right

    def _rggrow_roi_by_activation(self, actdata, apmmask, faces, vxsize=100, hemisphere='left'):
        """
        """
        masklabel = np.unique(apmmask[apmmask!=0])
        locmax = surf_tools.mask_localmax(actdata.T, apmmask.T)
        outmask = np.zeros((32492, self.subjnum))
        for i in range(self.subjnum):
            print('Subject {}'.format(i+1))
            for j, masklbl in enumerate(masklabel):
            # Left hemisphere, masklbl < len(masklabel)/2+1
                if hemisphere == 'left':
                    if ~np.isnan(locmax[i,j]):
                        vxloc_tmp, vxpack = surf_tools.threshold_by_rggrow(int(locmax[i,j]), vxsize, faces, actdata[i,:].flatten(), restrictedROI=(apmmask==masklbl).flatten().astype('int'))
                        outmask[vxpack, i] = masklbl
            # Right hemisphere
                else:
                    if ~np.isnan(locmax[i,int(j-np.max(masklabel)/2)]):
                        vxloc_tmp, vxpack = surf_tools.threshold_by_rggrow(int(locmax[i, int(j-np.max(masklabel)/2)]), vxsize, faces, actdata[i,:].flatten(), restrictedROI=(apmmask==masklbl).flatten().astype('int'))
                        outmask[vxpack, i] = masklbl
        return outmask

    def get_subjroi(self, act_parpath, apm_parpath, faces_parpath, vxsize=100):
        """
        """
        self.vxsize = vxsize
        self.load_actdata(act_parpath)
        self.load_apmmask(mask_parpath)
        self.load_faces(faces_parpath)
        print('Do region growing for language mask in left hemisphere...')
        langmask_left = self._rggrow_roi_by_activation(-1.0*self.act_left, self.langroi_left, self.faces_left, vxsize, 'left')
        print('Do region growing for language mask in right hemisphere...')
        langmask_right = self._rggrow_roi_by_activation(-1.0*self.act_right, self.langroi_right, self.faces_right, vxsize, 'right')
        print('Do region growing for math mask in left hemisphere...')
        mathmask_left = self._rggrow_roi_by_activation(self.act_left, self.mathroi_left, self.faces_left, vxsize, 'left')
        print('Do region growing for math mask in right hemisphere...')
        mathmask_right = self._rggrow_roi_by_activation(self.act_right, self.mathroi_right, self.faces_right, vxsize, 'right')
        self.langmask_left = langmask_left
        self.langmask_right = langmask_right
        self.mathmask_left = mathmask_left
        self.mathmask_right = mathmask_right

    def save_subjmask(self, outparpath):
        """
        """
        if 'langmask_left' in self.__dict__.keys():
            save_langlh = iofiles.make_ioinstance(pjoin(outparpath, 'lang_subjroi_vx'+str(self.vxsize)+'_lh.func.gii'))        
            save_langlh.save(self.langmask_left, hemisphere='CortexLeft')
        if 'langmask_right' in self.__dict__.keys():
            save_langrh = iofiles.make_ioinstance(pjoin(outparpath, 'lang_subjroi_vx'+str(self.vxsize)+'_rh.func.gii'))
            save_langrh.save(self.langmask_right, hemisphere='CortexRight')
        if 'mathmask_left' in self.__dict__.keys():
            save_mathlh = iofiles.make_ioinstance(pjoin(outparpath, 'math_subjroi_vx'+str(self.vxsize)+'_lh.func.gii'))
            save_mathlh.save(self.mathmask_left, hemisphere='CortexLeft')
        if 'mathmask_right' in self.__dict__.keys():
            save_mathrh = iofiles.make_ioinstance(pjoin(outparpath, 'math_subjroi_vx'+str(self.vxsize)+'_rh.func.gii'))
            save_mathrh.save(self.mathmask_right, hemisphere='CortexRight')

if __name__ == '__main__':
    parpath = '/nfs/m1/hcp'
    # Session ID
    with open('/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study/twinID', 'r') as f:
        sessid = f.read().splitlines()
    # sessid = sessid[:2]
    # sessid = ['248238']

    # Prepare subject ROIs
    # Load mask
    # Group ROI
    # mask_parpath = ['/nfs/h1/workingshop/huangtaicheng/hcp_test/twin_study/lang_math/apm_ROIs/lang_mask.dscalar.nii',
    #                 '/nfs/h1/workingshop/huangtaicheng/hcp_test/twin_study/lang_math/apm_ROIs/math_mask.dscalar.nii']
    # faces_parpath = ['/nfs/p1/atlases/Yeo_templates/surface/fs_LR_32k/fsaverage.L.inflated.32k_fs_LR.surf.gii', 
    #                 '/nfs/p1/atlases/Yeo_templates/surface/fs_LR_32k/fsaverage.R.inflated.32k_fs_LR.surf.gii']
    # outpath_subjroi = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study/code_twin/papercode/subjroi'
    # sr_cls = SubjectROI(parpath, sessid)
    # sr_cls.get_subjroi(act_parpath=parpath, apm_parpath=mask_parpath, faces_parpath=faces_parpath, vxsize=200)
    # sr_cls.save_subjmask(outpath_subjroi)


    # Compute separability
    sep_cls = separability(parpath, sessid)
    mask_parpath = ['/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study/code_twin/papercode/subjroi/lang_subjroi_vx100.dscalar.nii', 
                    '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study/code_twin/papercode/subjroi/math_subjroi_vx100.dscalar.nii']
    outpath = '/nfs/s2/userhome/huangtaicheng/hworkingshop/hcp_test/twin_study/code_twin/papercode/data/' 
    # Load mask
    sep_cls.load_mask(mask_parpath = mask_parpath)
    # Load fc corr
    sep_cls.calc_rsMRI_fcdist(parpath=parpath, outpath=outpath)
    # Load activation
    act_value = sep_cls.load_activation(actpath=parpath, outpath=outpath)
    # Load myelin
    myelin_value = sep_cls.load_myelin(myelinpath=parpath, outpath=outpath)
    # Load thickness
    thick_value = sep_cls.load_thickness(thickpath=parpath, outpath=outpath)
    sep_cls.calc_absolutediff(outpath=outpath)
    sep_cls.calc_segregation()
    sep_cls.write_segre(pjoin(outpath, 'separability_vx100.csv'))
