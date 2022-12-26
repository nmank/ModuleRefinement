This folder contains platform files associated 
with microarray gene expression studies.

A "platform file" is essentially a table which 
indicates which gene a corresponding microarray 
probe (feature) is associated with. Typically 
we're looking to convert probe IDs into 
Entrez Gene ID, but these files often provide 
information about other gene formats as well.

NOTE:
The files which have no "_DATA" or "_META" 
suffix are the raw platform files. They have 
a header (indicated by leading hashtags "#") 
and the actual tabular data beginning with 
a leading row with column headers.

I've split these into "_DATA" and "_META" 
files when I initially did a quick-and-dirty 
hack putting together the platform_lookup module. 
These are nothing more than splitting the original 
file into its two pieces to make loading slightly easier.
Longer-term, this "_DATA" and "_META" convention 
should ***NOT*** be followed; rather, the lookup 
should just use the "skiprows" tool in pandas. 
If metadata is needed, a raw python file read can be used.

NOTE:

GPL571-**** is the platform file used in the 
microarray version of the data for the 7-study 
with accession number GSE73072. 

GPL-10558**** is the platform file used for 
the data corresponding to the study with 
accession number GSE61754.

============

Last edited 24 April 2020
Manuchehr Aminian
