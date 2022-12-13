import calcom
import pandas as pd
import numpy as np

def returnTimeid(timebin):

	l = int(timebin.split('_')[0])
	u = int(timebin.split('_')[1])
	if l < 0: #for controls
		return np.arange(l,u)
	else:
		return np.arange(l,u+1)

def loadCalcomDataUsingStudy(dataPath,timeBin,study):
	print('Loading controls and shedder from time bin',timeBin,'. Study: ',study)
	# create an ccd onject
	ccd = calcom.io.CCDataSet(dataPath)
	#quary string for controls
	q0 = {'time_id': returnTimeid('-100_1'), 'StudyID':study}

	#quary string for shedders
	q1 = {'time_id': returnTimeid(timeBin),'StudyID':study,'shedding':True}
	
	tmpString = 'Shedder'+timeBin

	ccd.generate_attr_from_queries('new_attr',{'Control':q0,tmpString:q1})	
	new_attr = {'control': q0, 'shedder'+timeBin: q1}
	classificationAttr='control-vs-sheddder'+timeBin
	ccd.generate_attr_from_queries(classificationAttr, new_attr)
	idxs = ccd.find(classificationAttr, ['control', 'shedder'+timeBin])
	
	return ccd, idxs, classificationAttr

def generate_gse73072_dataset(dPath, tBin, study, out_dir):
    ccd, idx, classificationAttr = loadCalcomDataUsingStudy(dataPath=dPath,timeBin=tBin,study=study)
    
    data = ccd.generate_data_matrix(idx_list=idx)
    sample_ids = ccd.generate_labels('SampleID', idx_list=idx)

    probe_ids = list(ccd.variable_names)
    all_data = pd.DataFrame(data = data, columns = probe_ids, index = sample_ids)
    all_data.index.name = 'SampleID'

    subjects = ccd.generate_labels('SubjectID', idx_list=idx)
    subject_labels = pd.DataFrame(data = subjects, columns = ['subject'], index = sample_ids)
    subject_labels.index.name = 'SampleID'

    labels = ccd.generate_labels('shedding', idx_list=idx)
    all_labels = pd.DataFrame(data = labels, columns = ['label'], index = sample_ids)
    all_labels.index.name = 'SampleID'

    subject_labels.to_csv(f'{out_dir}gse73072_hrv_{tBin}_subjects.csv')
    all_data.to_csv(f'{out_dir}gse73072_hrv_{tBin}.csv')
    all_labels.to_csv(f'{out_dir}gse73072_hrv_{tBin}_labels.csv')

def generate_ebola_dataset(tissue, out_path):
    dPath = '/data3/darpa/all_CCD_processed_data/columbia-TCC-rnaseq.h5'
    ccd = calcom.io.CCDataSet(dPath)
    q0 = {'Tissue': tissue, 'Infection': 'Ebola'}
    idxs = ccd.find(q0)

    sample_ids = ccd.generate_labels('_id', idx_list = idxs)
    data = ccd.generate_data_matrix(idx_list=idxs)
    probe_ids = list(ccd.variable_names)
    all_data = pd.DataFrame(data = data, columns = probe_ids, index = sample_ids)

    labels = ccd.generate_labels('Phenotype', idx_list=idxs)
    all_labels = pd.DataFrame(data = labels, columns = ['label'], index = sample_ids)
    all_labels = all_labels.replace('Lethal/hemorrhage', 'Lethal')

    all_data.to_csv(f'{out_path}ebola_{tissue}.csv')
    all_labels.to_csv(f'{out_path}ebola_{tissue}_labels.csv')

def generate_salmonella_dataset(tissue, out_path):
    
    dPath = '/data3/darpa/all_CCD_processed_data/tamu-rnaseq-kranti.h5'
    ccd = calcom.io.CCDataSet(dPath)
    q0 = {'organ': tissue, 'phenotype':['tolerant', 'susceptible']}
    idxs = ccd.find(q0)

    sample_ids = ccd.generate_labels('_id', idx_list = idxs)
    data = ccd.generate_data_matrix(idx_list=idxs)
    probe_ids = list(ccd.variable_names)
    all_data = pd.DataFrame(data = data, columns = probe_ids, index = sample_ids)

    labels = ccd.generate_labels('phenotype', idx_list=idxs)
    all_labels = pd.DataFrame(data = labels, columns = ['label'], index = sample_ids)

    all_data.to_csv(f'{out_path}salmonella_{tissue}.csv')
    all_labels.to_csv(f'{out_path}salmonella_{tissue}_labels.csv')


if __name__ == '__main__':
    study = ['gse73072_uva', 'gse73072_duke']
    tBin = '48_64'
    dPath = '/data3/darpa/all_CCD_processed_data/ccd_gse73072_original_microarray.h5'
    out_dir = '/data4/mankovic/ModuleRefinement/data/'
    generate_gse73072_dataset(dPath, tBin, study, out_dir)

    generate_ebola_dataset('Liver', out_dir)
    generate_ebola_dataset('Spleen', out_dir)

    generate_salmonella_dataset('Liver', out_dir)
    generate_salmonella_dataset('Spleen', out_dir)
