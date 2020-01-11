# Language ROI
wb_command -cifti-create-dense-from-template /nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii lang_subjroi_vx100.dscalar.nii -metric CORTEX_LEFT lang_subjroi_vx100_lh.func.gii -metric CORTEX_RIGHT lang_subjroi_vx100_rh.func.gii
# Math ROI
wb_command -cifti-create-dense-from-template /nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii math_subjroi_vx100.dscalar.nii -metric CORTEX_LEFT math_subjroi_vx100_lh.func.gii -metric CORTEX_RIGHT math_subjroi_vx100_rh.func.gii
