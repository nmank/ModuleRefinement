import shutil
import os

base_d = './refined_modules/'

for method in os.listdir(base_d):
    other_d = f'{base_d}{method}/'
    for fold_all in other_d:
        fold_all_d = f'{other_d}{fold_all}/'
        for f_name in fold_all_d:
            other_d = f'{fold_all_d}{f_name}'
            if 'gse' in f_name:
                if ('True' in f_name) or ('False' in f_name):
                    shutil.remove(f'{other_d}/{f_name}')
	    


