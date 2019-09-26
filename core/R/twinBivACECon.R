# ------------------------------------------------------------------------------
# Program: twinMulAceCon.R  
#  Author: Hermine Maes
#    Date: 02 25 2012 
#Edited: Yu Luo Mar 22 2012
# Bivariate Twin Analysis model to estimate causes of variation
# Matrix style model - Raw data - Continuous data
# -------|---------|---------|---------|---------|---------|---------|

#Set Working Directory
#setwd('PUT THE PATH WHERE YOUR WORK IS LOCATED HERE') 
# Load Library
require(OpenMx)
require(psych)
source("GenEpiHelperFunctions.R")

# --------------------------------------------------------------------
# PREPARE DATA

# Load Data
testData <- read.csv('D:/code/pycharm_code/StoryMath_heritability/data/pc1_act_system_vx100.csv')
Vars <- c('story', 'math')
nv        <- 2       # number of variables
ntv       <- nv*2    # number of total variables
selVars   <- paste(Vars,c(rep(1,nv),rep(2,nv)),sep="")

# Select Data for Analysis
mzData    <- subset(testData, zyg==1, selVars)
dzData    <- subset(testData, zyg==3, selVars)
describe(mzData, skew=F)
describe(dzData, skew=F)
dim(mzData)
dim(dzData)

# Generate Descriptive Statistics
colMeans(mzData,na.rm=TRUE)
colMeans(dzData,na.rm=TRUE)
cov(mzData,use="complete")
cov(dzData,use="complete")
cor(mzData,use="complete")
cor(dzData,use="complete")

# ------------------------------------------------------------------------------
# PREPARE SATURATED MODEL

# Saturated Model
# Set Starting Values
meVals    <- c(0,0)         # start value for means
vaVal     <- 1.0                       # start value for variances
vaLBo     <- .0001                     # lower bounds for variances
vaValD    <- vech(diag(vaVal,ntv,ntv)) # start values for diagonal of covariance matrix
vaLBoD    <- diag(vaLBo,ntv,ntv)       # lower bounds for diagonal of covariance matrix
vaLBoD[lower.tri(vaLBoD)] <- -10       # lower bounds for below diagonal elements 
vaLBoD[upper.tri(vaLBoD)] <- -10       # lower bounds for above diagonal elements
vaUBo     <- 10

# Create Labels for Lower Triangular Matrices
meLabsMZ  <- paste("mMZ",1:ntv,sep="_")
meLabsDZ  <- paste("mDZ",1:ntv,sep="_")
vaLabsMZ  <- paste("vMZ",rev(ntv+1-sequence(1:ntv)),rep(1:ntv,ntv:1),sep="_")
vaLabsDZ  <- paste("vDZ",rev(ntv+1-sequence(1:ntv)),rep(1:ntv,ntv:1),sep="_")

# -------|---------|---------|---------|---------|---------|---------|
# Algebra for expected Mean Matrices in MZ & DZ twins
meanMZ    <- mxMatrix( type="Full", nrow=1, ncol=ntv, free=TRUE,
                       values=meVals, labels=meLabsMZ, name="expMeanMZ" )
meanDZ    <- mxMatrix( type="Full", nrow=1, ncol=ntv, free=TRUE,
                       values=meVals, labels=meLabsDZ, name="expMeanDZ" )

# Algebra for expected Variance/Covariance Matrices in MZ & DZ twins
covMZ     <- mxMatrix( type="Symm", nrow=ntv, ncol=ntv, free=TRUE,
                       values=vaValD, lbound=vaLBoD, ubound=vaUBo, labels=vaLabsMZ, name="expCovMZ" )
covDZ     <- mxMatrix( type="Symm", nrow=ntv, ncol=ntv, free=TRUE,
                       values=vaValD, lbound=vaLBoD, ubound=vaUBo, labels=vaLabsDZ, name="expCovDZ" )

# Data objects for Multiple Groups
dataMZ    <- mxData( observed=mzData, type="raw" )
dataDZ    <- mxData( observed=dzData, type="raw" )

# Objective objects for Multiple Groups
objMZ     <- mxExpectationNormal( covariance="expCovMZ", means="expMeanMZ",
                                  dimnames=selVars )
objDZ     <- mxExpectationNormal( covariance="expCovDZ", means="expMeanDZ",
                                  dimnames=selVars )

# Choose a fit function
fitFunction <- mxFitFunctionML()

# Combine Groups
modelMZ   <- mxModel( "MZ", meanMZ, covMZ, objMZ , fitFunction, dataMZ)
modelDZ   <- mxModel( "DZ", meanDZ, covDZ, objDZ , fitFunction, dataDZ)

minus2ll  <- mxAlgebra( MZ.objective+ DZ.objective, name="minus2sumloglikelihood" )
obj       <- mxFitFunctionAlgebra( "minus2sumloglikelihood" )
ciCov     <- mxCI( c('MZ.expCovMZ','DZ.expCovDZ' ))
ciMean    <- mxCI( c('MZ.expMeanMZ','DZ.expMeanDZ' ))
twinSatModel   <- mxModel( "twinSat", modelMZ, modelDZ, minus2ll, obj,
                           ciCov, ciMean )

# ------------------------------------------------------------------------------
# RUN SATURATED MODEL

# Run Saturated Model
twinSatFit     <- mxRun( twinSatModel, intervals=F )
twinSatSumm    <- summary( twinSatFit)
twinSatSumm

# Use Helper Functions
source("GenEpiHelperFunctions.R")
expectedMeansCovariances(twinSatFit)
# tableFitStatistics(twinSatFit)

# ------------------------------------------------------------------------------
# RUN SATURATED SUBMODELS

# Constrain expected Means and Variances to be equal across twin order
eqMVarsTwinModel    <- mxModel( twinSatModel, name="eqM&Vtwins" )
meLabsMZt2 <- paste("mMZ",nv+1:nv,sep="_")
meLabsMZt1 <- paste("mMZ",1:nv,sep="_")
meLabsDZt2 <- paste("mDZ",nv+1:nv,sep="_")
meLabsDZt1 <- paste("mDZ",1:nv,sep="_")
vaLabsMZt2 <- paste("vMZ",nv+1:nv,nv+1:nv,sep="_")
vaLabsMZt1 <- paste("vMZ",1:nv,1:nv,sep="_")
vaLabsDZt2 <- paste("vDZ",nv+1:nv,nv+1:nv,sep="_")
vaLabsDZt1 <- paste("vDZ",1:nv,1:nv,sep="_")

eqMVarsTwinModel    <- omxSetParameters( eqMVarsTwinModel,
                                         label=meLabsMZt2, free=TRUE, values=meVals, newlabels=meLabsMZt1 )
eqMVarsTwinModel    <- omxSetParameters( eqMVarsTwinModel,
                                         label=meLabsDZt2, free=TRUE, values=meVals, newlabels=meLabsDZt1 )
eqMVarsTwinModel    <- omxSetParameters( eqMVarsTwinModel,
                                         label=vaLabsMZt2, free=TRUE, values=vaVal, newlabels=vaLabsMZt1 )
eqMVarsTwinModel    <- omxSetParameters( eqMVarsTwinModel,
                                         label=vaLabsDZt2, free=TRUE, values=vaVal, newlabels=vaLabsDZt1 )

eqMVarsTwinFit      <- mxRun( eqMVarsTwinModel, intervals=F )
eqMVarsTwinSumm     <- summary( eqMVarsTwinFit )
# tableFitStatistics(twinSatFit, eqMVarsTwinFit)

# Constrain expected Means and Variances to be equal across twin order and zygosity
eqMVarsZygModel     <- mxModel( eqMVarsTwinModel, name="eqM&Vzyg" )
meLabsZt1 <- paste("mZ",1:nv,sep="_")
vaLabsZt1 <- paste("vZ",1:nv,1:nv,sep="_")

eqMVarsZygModel     <- omxSetParameters( eqMVarsZygModel,
                                         label=meLabsMZt1, free=TRUE, values=meVals, newlabels=meLabsZt1 )
eqMVarsZygModel     <- omxSetParameters( eqMVarsZygModel,
                                         label=meLabsDZt1, free=TRUE, values=meVals, newlabels=meLabsZt1 )
eqMVarsZygModel     <- omxSetParameters( eqMVarsZygModel,
                                         label=vaLabsMZt1, free=TRUE, values=vaVal, newlabels=vaLabsZt1 )
eqMVarsZygModel     <- omxSetParameters( eqMVarsZygModel,
                                         label=vaLabsDZt1, free=TRUE, values=vaVal, newlabels=vaLabsZt1 )

eqMVarsZygFit       <- mxRun( eqMVarsZygModel, intervals=F )
eqMVarsZygSumm      <- summary( eqMVarsZygFit )
eqMVarsZygSumm 
# tableFitStatistics(eqMVarsTwinFit, eqMVarsZygFit)

# Print Comparative Fit Statistics
SatNested <- list(eqMVarsTwinFit, eqMVarsZygFit)
# tableFitStatistics(twinSatFit, SatNested)

# ------------------------------------------------------------------------------
# PREPARE GENETIC MODEL

# ------------------------------------------------------------------------------
# Cholesky Decomposition ACE Model
# ------------------------------------------------------------------------------
# Set Starting Values
meVals    <- c(0,0)         # start value for means
meanLabs  <- paste(Vars,"mean",sep="") # create labels for means
paVal     <- .5                        # start value for path coefficient
paLBo     <- .0001                     # start value for lower bounds
paValD    <- vech(diag(paVal,nv,nv))   # start values for diagonal of covariance matrix
paLBoD    <- diag(paLBo,nv,nv)         # lower bounds for diagonal of covariance matrix
paLBoD[lower.tri(paLBoD)] <- -10       # lower bounds for below diagonal elements
paLBoD[upper.tri(paLBoD)] <- NA        # lower bounds for above diagonal elements

# Create Labels for Lower Triangular Matrices
aLabs     <- paste("a",rev(nv+1-sequence(1:nv)),rep(1:nv,nv:1),sep="_")
cLabs     <- paste("c",rev(nv+1-sequence(1:nv)),rep(1:nv,nv:1),sep="_")
eLabs     <- paste("e",rev(nv+1-sequence(1:nv)),rep(1:nv,nv:1),sep="_")

# Matrices declared to store a, c, and e Path Coefficients
pathA     <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=TRUE,
                       values=paValD, labels=aLabs, lbound=paLBoD, name="a" )
pathC     <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=TRUE,
                       values=paValD, labels=cLabs, lbound=paLBoD, name="c" )
pathE     <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=TRUE,
                       values=paValD, labels=eLabs, lbound=paLBoD, name="e" )
pathA	
# Matrices generated to hold A, C, and E computed Variance Components
covA      <- mxAlgebra( expression=a %*% t(a), name="A" )
covC      <- mxAlgebra( expression=c %*% t(c), name="C" ) 
covE      <- mxAlgebra( expression=e %*% t(e), name="E" )

# Algebra to compute total variances and standard deviations (diagonal only)
covP      <- mxAlgebra( expression=A+C+E, name="V" )
matI      <- mxMatrix( type="Iden", nrow=nv, ncol=nv, name="I")
invSD     <- mxAlgebra( expression=solve(sqrt(I*V)), name="iSD")

# Algebra for expected Mean and Variance/Covariance Matrices in MZ & DZ twins
meanG     <- mxMatrix( type="Full", nrow=1, ncol=nv, free=TRUE,
                       values=meVals, labels=meanLabs, name="Mean" )
meanT     <- mxAlgebra( expression= cbind(Mean,Mean), name="expMean" )
covMZ     <- mxAlgebra( expression=
                          rbind( cbind(A+C+E , A+C),
                                 cbind(A+C   , A+C+E)), name="expCovMZ" )
covDZ     <- mxAlgebra( expression=
                          rbind( cbind(A+C+E     , 0.5%x%A+C),
                                 cbind(0.5%x%A+C , A+C+E)), name="expCovDZ" )

# Data objects for Multiple Groups
dataMZ    <- mxData( observed=mzData, type="raw" )
dataDZ    <- mxData( observed=dzData, type="raw" )

# Objective objects for Multiple Groups
objMZ     <- mxExpectationNormal( covariance="expCovMZ", means="expMean",
                                  dimnames=selVars )
objDZ     <- mxExpectationNormal( covariance="expCovDZ", means="expMean",
                                  dimnames=selVars )

# Choose a fit function
fitFunction <- mxFitFunctionML()

# standardized estimates requiring CI
standardA   <- mxAlgebra(iSD %*% a, name="StPathA")                          
# standardized additive genetic path
standardC   <- mxAlgebra(iSD %*% c, name="StPathC")                           
# standardized shared environmental path
standardE   <- mxAlgebra(iSD %*% e, name="StPathE")    
# standardized nonshared environmental path
estA      <- mxAlgebra(A/V, name="PropVA")                          
# standardized additive genetic variance
estC      <- mxAlgebra(C/V, name="PropVC")                          
# standardized shared environmental variance
estE      <- mxAlgebra(E/V, name="PropVE")   
# standardized nonshared environmental variance
rA      <- mxAlgebra(solve(sqrt(I*A)) %&% A, name="corA")
# standardized genetic correlation
rC      <- mxAlgebra(solve(sqrt(I*C)) %&% C, name="corC")
# standardized genetic correlation
rE      <- mxAlgebra(solve(sqrt(I*E)) %&% E, name="corE")
# standardized genetic correlation
rP      <- mxAlgebra(solve(sqrt(I*V)) %&% V, name="corP")
# standardized phenotypic correlation

# Combine Groups
pars      <- list( pathA, pathC, pathE, covA, covC, covE, covP,
                   matI, invSD, meanG, meanT, standardA, standardC, standardE, 
                   estA, estC, estE, rA, rC, rE, rP)
modelMZ   <- mxModel( pars, covMZ, dataMZ, objMZ, fitFunction, name="MZ" )
modelDZ   <- mxModel( pars, covDZ, dataDZ, objDZ, fitFunction, name="DZ" )
minus2ll  <- mxAlgebra( expression=MZ.objective + DZ.objective, name="m2LL" )
obj       <- mxFitFunctionAlgebra( "m2LL" )
ci        <- mxCI(c("StPathA","StPathC","StPathE","PropVA", "PropVC", "PropVE",
                    "corA","corC","corE", "corP"))
CholAceModel  <- mxModel( "CholACE", pars, modelMZ, modelDZ, minus2ll, obj, ci )


# ------------------------------------------------------------------------------
# RUN GENETIC MODEL

# To change 'Default optimizer' globally, use, e.g.:
# mxOption(NULL, 'Default optimizer', 'NPSOL')

# Run Cholesky Decomposition ACE model
CholAceFit    <- mxRun(CholAceModel, intervals = T)
CholAceSumm   <- summary(CholAceFit)
CholAceSumm

round(CholAceFit@output$estimate,4)

# Generate Output with Functions
source("GenEpiHelperFunctions.R")
parameterSpecifications(CholAceFit)
expectedMeansCovariances(CholAceFit)
tableFitStatistics(eqMVarsZygFit,CholAceFit)

# Generate List of Parameter Estimates and Derived Quantities using formatOutputMatrices
# ACE Path Coefficients & Standardized Path Coefficients (pre-multiplied by inverse of standard deviations)
#ACEpathMatrices <- c("a","c","e","iSD","iSD %*% a","iSD %*% c","iSD %*% e")
#ACEpathLabels <- c("pathA","pathC","pathE","isd","stPathA","stPathC","stPathE")
#formatOutputMatrices(CholAceFit,ACEpathMatrices,ACEpathLabels,Vars,4)

# ACE Covariance Matrices & Proportions of Variance Matrices
#ACEcovMatrices <- c("A","C","E","V","A/V","C/V","E/V")
#ACEcovLabels <- c("covA","covC","covE","Var","stCovA","stCovC","stCovE")
#formatOutputMatrices(CholAceFit,ACEcovMatrices,ACEcovLabels,Vars,4)

# ACE Correlation Matrices 
#ACEcorMatrices <- c("solve(sqrt(I*A)) %&% A","solve(sqrt(I*C)) %&% C","solve(sqrt(I*E)) %&% E","solve(sqrt(I*V)) %&% V")
#ACEcorLabels <- c("corA","corC","corE", "corV")
#formatOutputMatrices(CholAceFit,ACEcorMatrices,ACEcorLabels,Vars,4)

# ------------------------------------------------------------------------------                                           
# FIT GENETIC SUBMODELS
paZero    <- vech(diag(0,nv,nv))   # start values for diagonal of covariance matrix

# Run Cholesky AE model
CholAeModel <- mxModel( CholAceFit, name="CholAE" )
pathC    <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=FALSE, values=paZero,
                      labels=cLabs, lbound=paLBoD, name="c" )#pathC is 2X2 matrix
CholAeModel$MZ$c <-pathC
CholAeModel$DZ$c <-pathC
CholAeModel$c <-pathC
rC    <- mxAlgebra(I*C, name="corC")
CholAeModel$corC   <- rC
CholAeModel$MZ$corC   <- rC
CholAeModel$DZ$corC   <- rC

CholAeFit     <- mxRun(CholAeModel, intervals = TRUE)
CholAeSumm   <- summary(CholAeFit)
CholAeSumm
round(CholAeFit@output$estimate,4)
expectedMeansCovariances(CholAeFit)
tableFitStatistics(CholAceFit,CholAeFit)

# Run Cholesky CE model
CholCeModel <- mxModel( CholAceFit, name="CholCE" )
pathA    <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=FALSE, values=paZero,
                      labels=aLabs, lbound=paLBoD, name="a" )#pathA is 2X2 matrix

CholCeModel$MZ$a <-pathA
CholCeModel$DZ$a <-pathA
CholCeModel$a <-pathA
rA    <- mxAlgebra(I*A, name="corA")
CholCeModel$corA   <- rA
CholCeModel$MZ$corA   <- rA
CholCeModel$DZ$corA   <- rA

CholCeFit    <- mxRun(CholCeModel, intervals = F)

CholCeSumm   <- summary(CholCeFit)
CholCeSumm
round(CholCeFit@output$estimate,4)
expectedMeansCovariances(CholCeFit)
tableFitStatistics(CholAceFit,CholCeFit)

# Generate List of Parameter Estimates and Derived Quantities using formatOutputMatrices
# ACE Path Coefficients & Standardized Path Coefficients (pre-multiplied by inverse of standard deviations)
#ACEpathMatrices <- c("a","c","e","iSD","iSD %*% a","iSD %*% c","iSD %*% e")
#ACEpathLabels <- c("pathA","pathC","pathE","isd","stPathA","stPathC","stPathE")
#formatOutputMatrices(CholDeFit,ACEpathMatrices,ACEpathLabels,Vars,4)

# ACE Covariance Matrices & Proportions of Variance Matrices
#ACEcovMatrices <- c("A","C","E","V","A/V","C/V","E/V")
#ACEcovLabels <- c("covA","covC","covE","Var","stCovA","stCovC","stCovE")
#formatOutputMatrices(CholDeFit,ACEcovMatrices,ACEcovLabels,Vars,4)

# ACE Correlation Matrices 
#ACEcorMatrices <- c("solve(sqrt(I*C)) %&% C","solve(sqrt(I*E)) %&% E", "solve(sqrt(I*V)) %&% V")
#ACEcorLabels <- c("corC","corE", "corV")
#formatOutputMatrices(CholDeFit,ACEcorMatrices,ACEcorLabels,Vars,4)

# # Run Cholesky E model
# CholEModel <- mxModel( CholAceFit, name="CholE" )
# pathA    <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=FALSE, values=paZero,
#                       labels=aLabs, lbound=paLBoD, name="a" )#pathA is 2X2 matrix
# pathC    <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=FALSE, values=paZero,
#                       labels=cLabs, lbound=paLBoD, name="c" )#pathA is 2X2 matrix
# CholEModel$MZ$a <-pathA
# CholEModel$DZ$a <-pathA
# CholEModel$a <-pathA
# CholEModel$MZ$c <-pathC
# CholEModel$DZ$c <-pathC
# CholEModel$c <-pathC
# rA    <- mxAlgebra(I*A, name="corA")
# CholEModel$corA   <- rA
# CholEModel$MZ$corA   <- rA
# CholEModel$DZ$corA   <- rA
# rC    <- mxAlgebra(I*C, name="corC")
# CholEModel$corC   <- rC
# CholEModel$MZ$corC   <- rC
# CholEModel$DZ$corC   <- rC
# 
# CholEFit     <- mxRun(CholEModel)
# CholESumm   <- summary(CholEFit)
# CholESumm
# 
# round(CholEFit@output$estimate,4)
# expectedMeansCovariances(CholEFit)
# tableFitStatistics(CholAceFit,CholEFit)
# 
# # Print Comparative Fit Statistics
# bivACENested <- list(CholAeFit, CholCeFit, CholEFit)
# tableFitStatistics(CholAceFit,bivACENested)
# 
# 
# #----------------------------------------------------------
# # fit asymmetrical submodel
# #----------------------------------------------------------
# 
# paFTval  <- c(T,F,  F,F)
# # Run Cholesky ACE-CE model
# Chol_ACECE_Model <- mxModel( CholAceFit, name="Chol_ACECE" )
# pathA    <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=paFTval, values=paZero,
#                       labels=aLabs, lbound=paLBoD, name="a" )#pathA is 2X2 matrix
# Chol_ACECE_Model$MZ$a <-pathA
# Chol_ACECE_Model$DZ$a <-pathA
# Chol_ACECE_Model$a <-pathA
# rA    <- mxAlgebra(I*A, name="corA")
# Chol_ACECE_Model$corA   <- rA
# Chol_ACECE_Model$MZ$corA   <- rA
# Chol_ACECE_Model$DZ$corA   <- rA
# 
# Chol_ACECE_Fit    <- mxRun(Chol_ACECE_Model, intervals = TRUE)
# 
# Chol_ACECE_Summ   <- summary(Chol_ACECE_Fit)
# Chol_ACECE_Summ
# round(CholCeFit@output$estimate,4)
# expectedMeansCovariances(CholCeFit)
# tableFitStatistics(CholAceFit,Chol_ACECE_Fit)
# tableFitStatistics(eqMVarsZygFit,Chol_ACECE_Fit)
# 
# pcFTval  <- c(F,F,  F,T)
# # Run Cholesky AE-CE model
# Chol_AECE_Model <- mxModel( Chol_ACECE_Fit, name="Chol_AECE" )
# pathC    <- mxMatrix( type="Lower", nrow=nv, ncol=nv, free=pcFTval, values=paZero,
#                       labels=cLabs, lbound=paLBoD, name="c" )#pathC is 2X2 matrix
# Chol_AECE_Model$MZ$c <-pathC
# Chol_AECE_Model$DZ$c <-pathC
# Chol_AECE_Model$c <-pathC
# rC    <- mxAlgebra(I*C, name="corC")
# Chol_AECE_Model$corC   <- rC
# Chol_AECE_Model$MZ$corC   <- rC
# Chol_AECE_Model$DZ$corC   <- rC
# 
# Chol_AECE_Fit    <- mxRun(Chol_AECE_Model, intervals = TRUE)
# 
# Chol_AECE_Summ   <- summary(Chol_AECE_Fit)
# Chol_AECE_Summ
# 
# bivACENested <- list(Chol_ACECE_Fit,Chol_AECE_Fit)
# tableFitStatistics(eqMVarsZygFit,bivACENested)
# parameterSpecifications(Chol_AECE_Fit)

