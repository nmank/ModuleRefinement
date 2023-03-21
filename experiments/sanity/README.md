# 1st Principal Angle Experiments

This directory contains the results where we tested the preformance of the first principal angle between subspaces of dimension > 1 for module refinement.

Our conclusions are that this produces comparable and even perhaps worse refined modules than using the 2nd principal angle.

## Bottom Lines

I re-ran the experiments using the 1st principal angle instead of the 2nd.

* 11 out of 20 of the re-run experiments ran to completion
* 2D data and 8D flag median results in only 1 cluster with all genes for all 5 folds 
* The highest SVM BSR relative gain gains were .01 below the highest relative gains from the original experiment (with 2nd principal angle)
* None of the trials resulted in a positive relative gain in GO significance



## Details
I re-ran 4D and 8D flag mean and flag median for 2D data representation within 5-fold cross validation. This results in modules for 5 folds for each of the 4 subspace configurations (e.g., 4D center and 2D data).

* For our experiments (these and those in the paper), we terminate LBG clustering if it did not converge after 20 iterations. Many of these trials 9/20 failed trials failed because LBG clustering did not converge after 20 iterations.
* The .csvs for the mean relative gains in SVM BSR and GO significance are attached. Our results with the 2nd principal angle are in the manuscript: https://www.overleaf.com/8358321547nxkrfxkpzxcv 