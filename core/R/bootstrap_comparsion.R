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

# Load Data
data <- read.csv('D:/code/pycharm_code/StoryMath_heritability/data/pc1_fc_system_vx100.csv')
roiname_all <- c('story', 'math')

ACE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
ACE_c2estimate_matric <- matrix(1, 3, length(roiname_all))
ACE_e2estimate_matric <- matrix(1, 3, length(roiname_all))

AE_h2estimate_matric <- matrix(1, 3, length(roiname_all))
AE_e2estimate_matric <- matrix(1, 3, length(roiname_all))

n_bootstrap <- 1000
h2_AEdistribute <- matrix(0, n_bootstrap, length(roiname_all))
h2_ACEdistribute <- matrix(0, n_bootstrap, length(roiname_all))
c2_ACEdistribute <- matrix(0, n_bootstrap, length(roiname_all))
for (i in seq_along(roiname_all))
{
  message(sprintf('Region: %s', roiname_all[i]))
  # Select Variables for Analysis
  selVars <- c(paste(roiname_all[i],'1',sep=''), paste(roiname_all[i],'2',sep=''))
  aceVars   <- c("A1","C1","E1","A2","C2","E2")
  
  # Select Data for Analysis
  mzData    <- subset(data, zyg==1, selVars)
  nsubj_mz <- nrow(mzData)
  dzData    <- subset(data, zyg==3, selVars)
  nsubj_dz <- nrow(dzData)

  j = 1  
  while (j<n_bootstrap+1)
  {
    # Sample subset of mzData and dzData
    subidx_mz <- sample(seq(1,nsubj_mz), size=nsubj_mz, replace=TRUE)
    subidx_dz <- sample(seq(1,nsubj_dz), size=nsubj_dz, replace=TRUE)
    mzData_sub <- mzData[subidx_mz,]
    dzData_sub <- dzData[subidx_dz,]
    
    AEmodelFit <- Twin_AEmodel(mzData_sub, dzData_sub, selVars)
    ACEmodelFit <- Twin_ACEmodel(mzData_sub, dzData_sub, selVars)
    # ComparedModel <- ModelComparison(ACEmodelFit, AEmodelFit)
    AE_confint_tmp <- summary(AEmodelFit)$CI
    ACE_confint_tmp <- summary(ACEmodelFit)$CI
    
    if (AE_confint_tmp[1,2]>0.99)
    {
      next
    }
    else
    {
      h2_AEdistribute[j,i] = AE_confint_tmp[1,2]
      h2_ACEdistribute[j,i] = ACE_confint_tmp[1,2]
      # c2_ACEdistribute[j,i] = ACE_confint_tmp[2,2]
      j=j+1
    }
    # AE_h2estimate_matric[1:3, i] <- cbind(AE_confint_tmp[1,1], AE_confint_tmp[1,2], AE_confint_tmp[1,3])
    # AE_e2estimate_matric[1:3, i] <- cbind(AE_confint_tmp[3,1], AE_confint_tmp[3,2], AE_confint_tmp[3,3])
  }
}
# DataFrame
AE_h2estimate <- as.data.frame(h2_AEdistribute)
ACE_h2estimate <- as.data.frame(h2_ACEdistribute)
# ACE_c2estimate <- as.data.frame(c2_ACEdistribute)
names(AE_h2estimate) <- roiname_all
names(ACE_h2estimate) <- roiname_all
# names(ACE_c2estimate) <- roiname_all