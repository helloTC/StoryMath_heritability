require(OpenMx)
source('Heritability_model_function.R')

# Load Data
# data <- read.csv('data/separate_pca/story_rgspec_vx100.csv')

# roiname_all <- c('L_AG', 'L_MTG', 'L_ATL', 'L_IFG',
#                  'R_AG', 'R_MTG', 'R_ATL', 'R_IFG')

# Load Data
# data <- read.csv('data/separate_pca/math_rgspec_vx100.csv')

# roiname_all <- c('L_aIPS', 'L_dIPS', 'L_SPL', 'L_MTL', 'L_latSMA', 'L_MFG', 'L_INS', 'L_medSMA', 'L_vlPFC',
#                  'R_aIPS', 'R_dIPS', 'R_SPL', 'R_MTL', 'R_latSMA', 'R_MFG', 'R_INS', 'R_medSMA', 'R_vlPFC')

# Load Data
data <- read.csv('D:/code/pycharm_code/StoryMath_heritability/data/pc1_act_system_vx100.csv')
roiname_all <- c('story', 'math')

ACE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
ACE_c2estimate_matric <- matrix(1, 3, length(roiname_all))
ACE_e2estimate_matric <- matrix(1, 3, length(roiname_all))

AE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
AE_e2estimate_matric <- matrix(1, 3, length(roiname_all))
for (i in seq_along(roiname_all))
{
  # Select Variables for Analysis
  selVars <- c(paste(roiname_all[i],'1',sep=''), paste(roiname_all[i],'2',sep=''))
  aceVars   <- c("A1","C1","E1","A2","C2","E2")
  
  # Select Data for Analysis
  mzData    <- subset(data, zyg==1, selVars)
  dzData    <- subset(data, zyg==3, selVars)
  
  AEmodelFit <- Twin_AEmodel(mzData, dzData, selVars)
  ACEmodelFit <- Twin_ACEmodel(mzData, dzData, selVars)
  print(ComparedModel <- ModelComparison(ACEmodelFit, AEmodelFit))
  AE_confint_tmp <- summary(AEmodelFit)$CI
  ACE_confint_tmp <- summary(ACEmodelFit)$CI
  
  ACE_h2estimate_matric[1:3, i] <- cbind(ACE_confint_tmp[1,1], ACE_confint_tmp[1,2], ACE_confint_tmp[1,3])
  ACE_c2estimate_matric[1:3, i] <- cbind(ACE_confint_tmp[2,1], ACE_confint_tmp[2,2], ACE_confint_tmp[2,3])
  ACE_e2estimate_matric[1:3, i] <- cbind(ACE_confint_tmp[3,1], ACE_confint_tmp[3,2], ACE_confint_tmp[3,3])
  
  AE_h2estimate_matric[1:3, i] <- cbind(AE_confint_tmp[1,1], AE_confint_tmp[1,2], AE_confint_tmp[1,3])
  AE_e2estimate_matric[1:3, i] <- cbind(AE_confint_tmp[3,1], AE_confint_tmp[3,2], AE_confint_tmp[3,3])
  
  # DataFrame
  ACE_h2estimate <- as.data.frame(ACE_h2estimate_matric)
  names(ACE_h2estimate) <- roiname_all
  ACE_c2estimate <- as.data.frame(ACE_c2estimate_matric)
  names(ACE_c2estimate) <- roiname_all
  ACE_e2estimate <- as.data.frame(ACE_e2estimate_matric)
  names(ACE_e2estimate) <- roiname_all
  AE_h2estimate <- as.data.frame(AE_h2estimate_matric)
  names(AE_h2estimate) <- roiname_all
  AE_e2estimate <- as.data.frame(AE_e2estimate_matric)
  names(AE_e2estimate) <- roiname_all
}




