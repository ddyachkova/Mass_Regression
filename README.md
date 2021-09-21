# Mass_Regression

The goal of this project is to effectively perform a regression of mass values of boosted top quark jets. 
The first step was the data production, that was performed with Pythia 6 followed by selecting the particles of interest with MLAnalyzer - the software developed by the E2E research group at CERN. 
To account of the imbalance of the dataset (the distribution of the generated mass of the particles that passed the selection of the MLAnalyzer was not flat but rather Poisson), we inversed the bias of that distribution and re-ran the data production step with the bias hard-coded as the selection parameter, to receive a flat distribution at the end. 
Next, we developed a ResNet-based mass regressor that learns to predict the mass of the particle given its kinematics. 
