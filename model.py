# SVM + rm 0 - X corr + PCA 884 (0.89676 var)
import numpy as np
import pandas as pd
import timeit
from sklearn import preprocessing
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from joblib import dump, load
from numpy.random import seed
seed(1)

# highly correlated rows
to_drop = [0, 1, 2, 4, 5, 6, 7, 8, 12, 16, 17, 20, 21, 24, 28,
                    32, 36, 48, 64, 65, 68, 69, 80, 81, 84, 127, 189, 250, 251, 254, 265, 
                    266, 268, 269, 280, 310, 311, 314, 325, 326, 328, 427, 441, 465, 468, 
                    484, 513, 644, 647, 658, 684, 687, 688, 743, 813, 815, 827, 840, 850, 
                    851, 853, 888, 890, 891, 904, 905, 907, 916, 952, 987, 999, 1090, 1091, 
                    1112, 1139, 1147, 1148, 1149, 1152, 1155, 1246, 1248, 1373, 1375, 1391,
                    1392, 1393, 1403, 1404, 1411, 1490, 1491, 1499, 1508, 1509, 1514, 1515, 
                    1516, 1517, 1605, 1612, 1642, 1643, 1664, 1665, 1666, 1671, 1746, 1752, 
                    1771, 1772, 1818, 1820, 1845, 1846, 1847, 1852, 1858, 1859, 1861, 1862, 
                    1863, 1864, 1865, 1866, 1867, 1868, 1869, 1883, 1884, 1888, 1889, 1902, 
                    1903, 1906, 1907, 1912, 1916, 1918, 1920, 1921, 1922, 1923, 1924, 1925, 
                    1926, 1929, 1934, 1935, 1937, 1938, 1939, 1987, 1998, 2000, 2031, 2041, 
                    2049, 2050, 2056, 2062, 2063]

labels = np.array(['staphylococcus_aureus', 'staphylococcus_pyogenes',
       'mycobacterium_ulcerans', 'burkholderia_pseudomallei',
       'corynebacterium_ulcerans', 'decoy', 'corynebacterium_diphtheriae',
       'mycobacterium_tuberculosis', 'streptococcus_pneumoniae',
       'pseudomonas_aeruginosa', 'neisseria_gonorrhoeae',
       'salmonella_enterica_typhimurium', 'salmonella_enterica_paratyphi',
       'campylobacter_jejuni', 'klebsiella_pneumoniae',
       'listeria_monocytogenes', 'clostridioides_difficile',
       'acinetobacter_baumannii', 'vibrio_parahaemolyticus',
       'vibrio_cholerae', 'neisseria_meningitidis',
       'legionella_pneumophila', 'staphylococcus_pseudintermedius',
       'streptococcus_suis', 'citrobacter_freundii',
       'klebsiella_michiganensis', 'serratia_liquefaciens',
       'stenotrophomonas_maltophilia', 'streptococcus_agalactiae',
       'streptococcus_equi', 'yersinia_enterocolitica'], dtype=object)

le = preprocessing.LabelEncoder()
le.fit(labels)

def precision_per_patient(patient_id, preds):
    df_true = pd.read_csv('datasets/validation/patient{}_labels.txt'.format(patient_id))
    tp,fp, tp_labels=0,0, df_true['true_label'].shape[0]
    print('my prediction(s) for patient {}:'.format(patient_id))
    print(preds)
    print('true pathogen')
    print(df_true['true_label'].values)
    #if don't predict any pathogen, it means there is only decoy in the test dataset (your prediction)
    if len(preds) == 0:
        preds = ['decoy']
    for item in np.unique(preds):
        if item in df_true['true_label'].values:
            tp+=1
        else:
            fp+=1
    #you have to predict all labels correctly, but you are penalized for any false positive
    return tp/(tp_labels+fp)

#prediction for all patients
def run_test(model, preprocess, threshold=0.99):
    all_precision = []
    for patient_id in range(1,11):
        print('predicting for patient {}'.format(patient_id))
        
        starting_time = timeit.default_timer()
        with open('datasets/validation/patient{}.6mer.npy'.format(patient_id), 'rb') as read_file:
            df_test = np.load(read_file)
            df_test = pd.DataFrame(df_test)

        # preprocessing    
        df_test.drop(columns=to_drop, inplace=True)
        df_test = preprocess.transform(df_test)
        
        y_predprob = model.predict_proba(df_test)
        
        final_predictions = le.inverse_transform(np.unique([np.argmax(item) for item in y_predprob  if len(np.where(item>= threshold)[0]) >=1]
                                                    ))
        #my pathogens dectected, decoy will be ignored
        final_predictions = [item for item in final_predictions if item !='decoy']
        
        precision = precision_per_patient(patient_id, final_predictions)
        print('precision: {}'.format(precision))
        all_precision.append(precision)
        print("Time taken :", timeit.default_timer() - starting_time)

    # performance per patient and its final average
    print([f'patient {c}: {item}' for c, item in enumerate(all_precision, start=1)])
    print(f'avg: {np.mean(all_precision)}')
    return np.mean(all_precision)


def main():
    print("loading model")
    # load trained model
    clf = load('svm_svdfull_910.joblib')

    # load preprocess model
    with open('svd_n500.pkl', 'rb') as pickle_file: # PCA embeddings trained on full data
        preprocess=pkl.load(pickle_file) 

    # predicting
    run_test(model=clf, preprocess=preprocess, threshold=0.99)

if __name__=="__main__":
    main()