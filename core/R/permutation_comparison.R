require(OpenMx)
source('Heritability_model_function.R')

# Load Data
# data <- read.csv('data/summary/langmath_spec_vx50.csv')

# roiname_all <- c('L_AG', 'L_MTG', 'L_ATL', 'L_IFG',
#                  'R_AG', 'R_MTG', 'R_ATL', 'R_IFG')

# Load Data
# data <- read.csv('data/summary/')

# roiname_all <- c('L_aIPS', 'L_dIPS', 'L_SPL', 'L_MTL', 'L_latSMA', 'L_medSMA', 'L_MFG', 'L_INS', 'L_vlPFC',
#                  'R_aIPS', 'R_dIPS', 'R_SPL', 'R_MTL', 'R_latSMA', 'R_medSMA', 'R_MFG', 'R_INS', 'R_vlPFC')

n_bootstrap <- 1000
# Load Data
data <- read.csv('D:/code/pycharm_code/StoryMath_heritability/data/pc1_fc_system_vx100.csv')
roiname_all <- c('story', 'math')

# ACE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
# ACE_c2estimate_matric <- matrix(1, 3, length(roiname_all))
# ACE_e2estimate_matric <- matrix(1, 3, length(roiname_all))

AE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
# AE_e2estimate_matric <- matrix(1, 3, length(roiname_all))

h2_AEdistribute_raw <- matrix(0, 1, length(roiname_all))
h2_AEdistribute <- matrix(0, n_bootstrap, length(roiname_all))
# h2_ACEdistribute <- matrix(0, n_bootstrap, length(roiname_all))
# c2_ACEdistribute <- matrix(0, n_bootstrap, length(roiname_all))

for (i in seq_along(roiname_all))
{
  selVars <- c(paste(roiname_all[i],'1',sep=''), paste(roiname_all[i],'2',sep=''))
  
  # Select Data for Analysis
  mzData    <- subset(data, zyg==1, selVars)
  # mzData    <- data.frame(apply(mzData,2,scale))
  nsubj_mz <- nrow(mzData)
  dzData    <- subset(data, zyg==3, selVars)
  # dzData    <- data.frame(apply(dzData,2,scale))
  nsubj_dz <- nrow(dzData)
  AEmodelFit <- Twin_AEmodel(mzData, dzData, selVars) 
  h2_AEdistribute_raw[i] <- summary(AEmodelFit)$CI[1,2]
}
h2_AEdiff_raw <- h2_AEdistribute_raw[1] - h2_AEdistribute_raw[2]

j=1
while (j<n_bootstrap+1)
{
  data_tmp <- SwapTableColumns(data, c(1,2), c(3,4))
  for (i in seq_along(roiname_all))
  {
    selVars <- c(paste(roiname_all[i],'1',sep=''), paste(roiname_all[i],'2',sep=''))
    
    # Select Data for Analysis
    mzData    <- subset(data_tmp, zyg==1, selVars)
    # mzData    <- data.frame(apply(mzData,2,scale))
    nsubj_mz <- nrow(mzData)
    dzData    <- subset(data_tmp, zyg==3, selVars)
    # dzData    <- data.frame(apply(dzData,2,scale))
    nsubj_dz <- nrow(dzData)
    AEmodelFit <- Twin_AEmodel(mzData, dzData, selVars)
    # ACEmodelFit <- Twin_ACEmodel(mzData, dzData, selVars)
    AE_confint_tmp <- summary(AEmodelFit)$CI
    h2_AEdistribute[j,i] <- AE_confint_tmp[1,2]
  }
  j = j+1
}
h2_AEdiff <- h2_AEdistribute[,1] - h2_AEdistribute[,2]
