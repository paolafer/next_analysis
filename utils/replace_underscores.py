import glob
import in_place
import re

import numpy as np


if __name__ == '__main__':

    folder = '/geometries'
    cc_files = np.array(glob.glob('source'+folder+'/*Demo*.cc'))
    h_files  = np.array(glob.glob('source'+folder+'/*Demo*.h'))
    all_files = np.concatenate((cc_files, h_files))
    #all_files = np.array(['prova.txt'])

    for the_file in all_files:

        with in_place.InPlace(the_file) as file:
            for line in file:

                # exclude comment or includes lines:  we don't want to touch _something in comments
                
                if line[0] == '#':
                    file.write(line)
                    continue

                words = line.split(' ')
                non_empty_words = [non_empty for non_empty in words if non_empty]

                # take care of '} // end...' lines
                if non_empty_words[0].strip() == '}':
                    file.write(line)
                    continue

                # take care of comment-only lines
                if non_empty_words[0].strip() == '//': # ex. // ACTIVE
                    file.write(line)
                    continue
                if non_empty_words[0].strip() == '///': # ex. /// ACTIVE ///
                    file.write(line)
                    continue
                if len(non_empty_words[0]) > 2:
                    if non_empty_words[0][0] == '/' and non_empty_words[0][1] == '/': # ex. //ACTIVE
                        file.write(line)
                        continue

                # move underscores
                words = re.split(r'\W+', line)
            
                for w in words:
                    if len(w) == 0:
                        continue
                    if w[0] == '_':
                        new_w = w[1:].strip()+'_'
                        line = line.replace(w, new_w)
                        
                file.write(line)
