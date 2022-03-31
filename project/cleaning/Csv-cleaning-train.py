import csv
import os.path

with open("yiminglin_csvs/imdb_train_new_1024.csv") as csvdatei:
    csv_reader_object = csv.reader(csvdatei)

    zeilennummer = 0
    nr_of_supercleanPics = 0
    nr_of_deletedPics = 0
    for row in csv_reader_object:

        if zeilennummer == 0:
            #create the new csv file and write the column names in it
            with open('imdb_train_superclean.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
        else:
            #auf vermerkten Pfad des Bildes zugreifen: besteht aus Imdb-Ordnernummer/Bildname.jpg
            filename = row[0]
            #Pfad erweitern, so dass er zu meiner Ordnerstruktur passt
            filename = "./imdb_superclean/" + filename
            # test
            #print(filename)

            #überprüfen ob diese Datei im superclean ordner existiert
            if os.path.isfile(filename):
                nr_of_supercleanPics += 1
                print (f"Picture exists in superclean: {filename}")
                #row in cs file einfügen
                with open('imdb_train_superclean.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
            else:
                nr_of_deletedPics += 1
                print ("nope")

        zeilennummer += 1

    print(f'Anzahl Bilder in yiminglins train split war: {zeilennummer-1}') #note: es sind 56087
    print(f'Anzahl Bilder in der superclean Version: {nr_of_supercleanPics}')
    print(f'Anzahl Bilder die aussortiert wurden: {nr_of_deletedPics}')
