#!/usr/bin/env python3

"""
Created on Tue May 11 16:46:56 2021

@author: Lily Amsellem
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import re 
import joblib
from joblib import dump, load
from os import listdir
from data_processing import seq2vec, load_data_as_df, pad_sequence, array_as_one_hot, extract_kmers
from data_generation import parse_fasta
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, plot_confusion_matrix


def ordinal_encoding(seq_df, code = {"A":1, "C":2, "G":3, "T":4}):
    """ 
    Encode sequences into ints/floats using a mapping from nucleotides to the wanted ints/floats. 
    
    Parameters
    ----------
    seq_df : a pandas dataframe with a 'sequence' column containing DNA strings.
    code: a mapping from nucleotide to float to encode the DNA sequences. Default is: {"A":1, "C":2, "G":3, "T":4}

    Returns
    -------
    data: an array containing the DNA sequences encoded with the code given

    """
    # Vectorise sequences
    vec_sequences = [seq2vec(sequence, nuc2int = code) for sequence in seq_df['sequence']]
    max_len = max([len(seq) for seq in vec_sequences])
    data = pad_sequence(vec_sequences, max_len)
    return data

def one_hot_encoding(seq_df):
    """ 
    Encode sequences from a dataframe into a one hot representation. 
    
    Parameters
    ----------
    seq_df : a pandas dataframe with a 'sequence' column containing DNA strings.

    Returns
    -------
    one_hot_array : a 3D numpy array of shape (num_sequences, max_seq_length,5)

    """
    
    # Encode sequences to integers
    enc_data = ordinal_encoding(seq_df)
    # Turn into a one-hot array
    one_hot_array = array_as_one_hot(enc_data)
    return one_hot_array

def transform_k_mers(seq_df, k):
    """ 
    Generate the overlapping k-mers (sequences of k consecutive nucleotides) for all sequences in the dataset.
    For each sequence, a corresponding sequence of all overlapping k-mers separated by a space ' ' will be
    written in a new column "{k}_mers". 

    Parameters
    ----------
    seq_df : a pandas dataframe with a 'sequence' column containing DNA strings.
    k : the length of the k-mers

    Returns
    -------
    seq_df : the same pandas dataframe with a new column "{k}_mers" containing a sequence of overlapping kmers. 

    """
    
    # Extract k-mers
    k_mers_seq = seq_df.apply(lambda x: extract_kmers(x['sequence'], k), axis = 1)
    
    # Join them into sequences
    k_mers_seq = k_mers_seq.apply(lambda x: ' '.join(x))
    seq_df['{}_mers'.format(k)] = k_mers_seq

    return seq_df

def encode_labels(labels):
    
    lb = LabelEncoder()
    y_tr = lb.fit_transform(labels)
    return y_tr, lb
    
def model_selection(models, err_rate, verbose = 0):
    """ 
    Train different models on sequence classification using cross-validation (default is 5-fold).
    

    Parameters
    ----------
    models : list containing tuples ('model_name', model) where model is an sklearn classifier or regressor, with
    the required parameters specified. 
    err_rate : error rate of the sequences on which the models are trained
    verbose : if set to 1, print for each error rate the mean accuracy and std of each model. Default is 0. 

    Returns
    -------
    train_scores : list containing the cross-validated score of each model.

    """
    # Perform model selection
    train_scores = []
    names = []
    
    for name,clf in models:
        train_score = cross_val_score(clf,X_train_tr, y_train)
        train_scores.append(train_score)
        names.append(name)
    
    if verbose:
        print("-------------------------")
        print("Error Rates: ", err_rate)
        for name,score in zip(names,train_scores):
            print("%s classifier: accuracy: %0.2f, std: %0.2f" % (name, score.mean(), score.std()))
    
    return train_scores
    
def model_grid_search(model, params, X_train, y_train, search = "regular", verbose = 0):
    """
    Perform hyperparameter grid search on given model, using 5-fold cross validation.

    Parameters
    ----------
    model : sklearn regressor/classsifier object. 
    
    params : grid of parameters to iterate over {'param1':(value1, value2, value3), 'param2':[value]}
    
    X_train : numpy array of training data
    y_train : numpy array of training labels
    search : default is "regular" to perform a gridsearch over the given parameters. If set to "random",
    will perform random search of hyperparameters
    verbose : if set to 1, will print the best parameters found by grid search.
    Default is 0, and it will only print the best score (mean corss-validated score of the best estimator 
                                                         found by grid search). 

    Returns
    -------
    best_score : cross-validated score of the best estimator found by grid seach
    best_estimator: best estimator refitted on the whole dataset, without CV, with the best hyperparameters. 

    """
    
    if search=='regular':
        # Regular grid search
        grid_search = GridSearchCV(model, params)
    else:
        # Randomized grid search
        grid_search = RandomizedSearchCV(model,param_distributions = params, n_iter = 10) 
        
    print("Performing Grid Search...")
    print()
    
        
    # Fit grid search
    grid_search.fit(X_train,y_train)
    
    # Find best score and params
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_estimator_.get_params()
    
    print("Best score: ")
    print(best_score)
    print()
    
    if verbose: 
        print("Best params found through gridsearch: ")
        for param in sorted(best_params.keys()):
            print("\t%s: %r" % (param,best_params[param]))
        print()
    
    return best_score, best_estimator

def test_model(model, X_test, y_test, report = False):
    """
    Test a fitted machine learning model on the test set. 

    Parameters
    ----------
    model : sklearn classifier or regressor object. The model must be fitted prior to using this function
    
    X_test : numpy array of sequence samples for testing. 
    y_test : numpy array of sequence labels
    report : if set to True, will print a classification report from sklearn with the main classification metrics 
    for each class (precision, recall, f1-score)

    Returns
    -------
    test_score : model score on the test set 
    y_pred : predicted labels for the test et

    """
        
    # Evaluate model on test data
    print("Evaluating model...")
    
    test_score = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    
    print("test score: ", test_score)
    
    if report:
        clf_report = classification_report(y_test,y_pred)
        print(clf_report)
        return test_score, y_pred, clf_report
    
    return test_score, y_pred
    
     
if __name__=='__main__':
    np.random.seed(0)
    start = time.time()
    
    loc_dir = '../simulated_data/eq_prop_mutations/'
    blast_loc_dir = '../simulated_data/blast_mutations/mutations_2_sequences/'
    record_paths = ['nb_it_100_in_0.133_del_0.133_sub_0.133']#sorted([f for f in listdir(loc_dir) if not f.startswith('.')
                          # and f.startswith('nb_it')])
    blast_record_paths = ['nb_it_100_in_0.133_del_0.133_sub_0.133'] #sorted([f for f in listdir(blast_loc_dir) if not f.startswith('.')
                                 #and f.startswith('nb_it')])

    val_scores = []
    test_scores = []
    error_rates = []
    #accs = []
    #test_scores = []
    # Iterate through each data file
    for record_path, blast_path in zip(*(record_paths, blast_record_paths)):
        # Get error rate from file name
        err_rates = [float(s) for s in re.findall('[\d]*[.][\d]+', record_path)]
        err_rate = round(np.sum(err_rates),2)
        error_rates.append(err_rate)
        
        data_file = loc_dir + record_path + "/sequences.fst" 
        blast_data_file = blast_loc_dir + blast_path + "/sequences.fst" 
        
        # -------------------------- DATA PREPROCESSING -----------------------------------------
        # Load data
        seq_df = parse_fasta(data_file)
        seq_df_blast = parse_fasta(blast_data_file)
        seq_df = seq_df.append(seq_df_blast, ignore_index = True)
        
        X = seq_df['Sequence'].values
        print("Total number of sequences in data set: ", len(X))
        #X = ordinal_encoding(seq_df, code =  {"A":0.25, "C":0.50, "G": 0.75, "T": 1})
        y = seq_df['Name'].values
        y, lb_enc = encode_labels(y)
        print('nb of species: ', len(np.unique(y)))
        
        # Plot the species distribution
        #seq_df['species'].value_counts().sort_index().plot().bar()
        
        # Split data set
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
        
        
        # Define Preprocessor : Bag of Words or TF-iDF
        #bow_vec = CountVectorizer(analyzer = 'char', ngram_range = (3,3))
        tf_idf_vec = TfidfVectorizer(analyzer = 'char', ngram_range = (5,5), use_idf=(False))
        
        start = time.time()
        # Fit it to the training data and transform. Get dense representation
        X_train_tr = tf_idf_vec.fit_transform(X_train).toarray()
        X_test_tr = tf_idf_vec.transform(X_test).toarray()
        
        # Standardize data
        std_scaler = StandardScaler()
        X_train_tr = std_scaler.fit_transform(X_train_tr)
        X_test_tr = std_scaler.transform(X_test_tr)
        
        
        # Save preprocessing steps
        k = 5
        joblib.dump(tf_idf_vec, blast_loc_dir + 'models/tf_idf_vec_{}mers.joblib'.format(k))
        joblib.dump(std_scaler,  blast_loc_dir + 'models/std_scaler_{}mers.joblib'.format(k))
        

        # ------------------------------------- MODEL SELECTION -----------------------------------------------------
        
        # models = [('LogisticRegression', LogisticRegression()), 
        #           ('SVC', SVC()),
        #           ('SGD', SGDClassifier()), 
        #           ('RandomForest', RandomForestClassifier()), 
        #           ('MLP', MLPClassifier(hidden_layer_sizes= (150,100,50), max_iter = 500)), 
        #           ('AdaBoost', AdaBoostClassifier())]
        
        #train_scores = model_selection(models, err_rate, verbose = 1)
        #accs.append(train_scores)
                
        #print("elapsed time", time.time() - start)

    
    # Plot model accuracy vs error rate
    #for i in range(len(models)):
        #model_accs = [acc[i].mean() for acc in accs]
        #plt.plot(error_rates, model_accs, label = models[i][0])
    #plt.show()
    
    # ------------------------------------ BEST MODEL TRAINING -----------------------------------------------------
        best_model = SVC()
    
        params = {'C':(1,0.5,0.1,0.05),
                  'kernel':('linear','poly','rbf'),
                  'degree':(2,3,4),
                  'gamma':['scale']}
        
        # Perform grid search to find the best hyperprameters
        # Get mean cross-validated score of the best estimator
        best_score, svm_model = model_grid_search(best_model, params, X_train_tr, y_train, verbose = 1)
        
        #best_params = svm_grid.best_params_
        val_scores.append(best_score)
        
        #best_params = {'C':1, 'kernel':'linear'}
        # svm_model = SVC(C= 1, kernel = 'linear', probability= True)
        # print('Training model...')
        # svm_model.fit(X_train_tr, y_train)
        # mean_val_score = np.mean(cross_val_score(svm_model, X_train_tr, y_train))
        # val_scores.append(mean_val_score)
        # print("Mean cross val score for error rate {}: {}".format(err_rate,mean_val_score))
        joblib.dump(svm_model, blast_loc_dir + 'models/svm_clf_err_rate_{}.joblib'.format(err_rate))
   # ---------------------------------- TEST MODEL ----------------------------------------------------------
     
       # Test the model
        test_score, y_pred = test_model(svm_model, X_test_tr, y_test)
        print("Mean accuracy on test data", test_score)
        test_scores.append(test_score)
        


# ---------------------------------- PLOT TRAINING CURVES ---------------------------------------------------
     
    #Sort error rates in ascending fashion - data files are not read in order    
    sorted_idx = np.argsort(error_rates)
    error_rates = [error_rates[i] for i in sorted_idx]
    val_scores = [val_scores[i] for i in sorted_idx]
    test_scores = [test_scores[i] for i in sorted_idx]
    plt.plot(error_rates,val_scores, label = "Mean val score")
    plt.plot(error_rates,test_scores, label = "Mean test score")
    plt.xlabel('Error rate')
    plt.ylabel('Performance')
    plt.title('SVC Performance on 4500 sequences from 45 species')
    plt.legend()
    plt.show()
        
            
        


