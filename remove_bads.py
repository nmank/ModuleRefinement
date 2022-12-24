import shutil
import os

base_d = './refined_modules/'

for method in os.listdir(base_d):
    other_d = f'{base_d}{method}/'
    for fold_all in os.listdir(other_d):
        fold_all_d = f'{other_d}{fold_all}/'
        for f_name in os.listdir(fold_all_d):
            new_other_d = f'{fold_all_d}{f_name}'
            
            if 'gse' in f_name:
                if ('True' in f_name) or ('False' in f_name):
                    print(new_other_d)
                    os.remove(new_other_d)
	    


