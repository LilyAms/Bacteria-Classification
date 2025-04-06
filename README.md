# Bacteria Classification from Long DNA Reads #

## Context ##
Identifying bacteria species requires long quality DNA sequences. 3rd Generation sequencers which produce those DNA sequences introduce noise that make it hard to recognize the bacteria at the species level. 

The aim of this project is to study the extent to which machine learning algorithms can classify noisy bacteria DNA sequences from third generation sequencers at the species level.

## About the project ##
In this project we : 
* Take Gold Standard full-length bacteria 16SrRNA sequences. 
* Simulate the output of a third-generation sequencer by introducing errors into the sequences (insertion - deletion - substitution) at different error rates. 
* Train non-neural ML models for bacteria classification on this synthetic data set, and evaluate their robustness to increasing error rates. 
* Test the model on real experimental data (noisy sequences) obtained from a third generation sequencer (MinION sequencer), and compare the results to the ones obtained with a regular biology software. 


## More Details ##
* We transform DNA sequences into feature vectors using kmer frequencies (subsequences of length k).
* We train a variety of models including logistic regression, SVMs, Random Forest, and a SGD classifier. 
* Dataset : from 17 base species, we simulate around 3000 mock sequences containing errors. Trials were also conducted with 100 and 200 base species.

## Results ##
* Best results were obtained with an SGD classifier, and with a 10% error rate, which approximates well the real error rate of third generation sequencers. 
* Improvements could be made by simulating mutations in DNA sequences that are closer to reality, and with further adequate feature extraction. 