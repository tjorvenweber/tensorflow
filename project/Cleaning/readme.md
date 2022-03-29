<img src="MrBean.jpg" align="right" />

# Cleaning process
> here you can find the material from our own cleaning process

To perform the cleaning yourself, you need to install deepface from PyPI beforehand

## supercleaning_of_imdb_clean.py

This Python file was used to perfom the cleaning of the IMDB_clean image folders.
The file needs to be put into the same folder hierarchy as the IMDB_clean folder "imdb-clean-1024".
To do the cleaning, enter the following into your cmd:

- python Supercleaning_of_imdb_clean.py

This is the outcome of the cleaning:
- several directories are made where the pictures will be sorted into
    - Supercleaning_of_imdb_clean is the folder containing subfolders
    - imdb_badpics contains 2 folders: noFace and multipleFaces
    - imdb_superclean contains the cleaned pictures

[imdb_clean](https://github.com/yiminglin-ai/imdb-clean) - This is where the original IMDB_clean can be downloaded from. Through this process we have obtained the mentioned "imdb-clean-1024" folder

Please note: the cleaning process can be very time consuming, when there are as many pictures as in the imdb_clean;)


## Csv-cleaning-test.py, Csv-cleaning-train.py and Csv-cleaning-valid.py

These python files were used to clean the corresponding csv-files containing metainformation about the images.
For every split we have a different file. To perform the cleaning, take the original csv-files (imdb_test_new_1024.csv, imdb_train_new_1024.csv, imdb_valid_new_1024.csv) from the imdb_clean and put them into a folder named "yiminglin_csvs".
Put the cleaning files in the same folder hierarchy as the yiminglin_csvs folder. Then enter the following command into your commandline:
- python Csv-cleaning-test.py
- python Csv-cleaning-train.py
- python Csv-cleaning-valid.py
