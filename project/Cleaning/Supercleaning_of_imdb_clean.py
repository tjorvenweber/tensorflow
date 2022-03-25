from deepface import DeepFace
import matplotlib.pyplot as plt
import os
from PIL import Image
import PIL

##make some directories to save the cleaned dataset in

#make directory, where we want to save the super cleaned imdb
os.mkdir("SuperCleaning_of_imdb_clean")
#here we will save the good images
os.mkdir("SuperCleaning_of_imdb_clean/imdb_superclean")
savingpath_good = "./SuperCleaning_of_imdb_clean/imdb_superclean/"
#here we will save the bad pictures
os.mkdir("SuperCleaning_of_imdb_clean/imdb_badpics")
savingpath_bad = "./SuperCleaning_of_imdb_clean/imdb_badpics/"
#pictures where no face was detected
os.mkdir("SuperCleaning_of_imdb_clean/imdb_badpics/noFace")
savingpath_bad_noFace = "./SuperCleaning_of_imdb_clean/imdb_badpics/noFace/"
#pictures with more than one face
os.mkdir("SuperCleaning_of_imdb_clean/imdb_badpics/multipleFaces")
savingpath_bad_multipleFaces = "./SuperCleaning_of_imdb_clean/imdb_badpics/multipleFaces/"
#folder where we get the images from
gettingpath = "./imdb-clean-1024/"
#temporary folder for faces that have been detected on one picture
os.mkdir("SuperCleaning_of_imdb_clean/temp")
temppath = "./SuperCleaning_of_imdb_clean/temp/"


# several detectors test whether they can detect a face in a picture
def detect_faces(picturename, foldername, gettingfrom):

    picturepath = gettingfrom + picturename
    print("Detectors try to detect in " + picturepath)

    #when the detection goes wrong, a ValueError will be thrown
    try:

        #opencv detector
        print("opencv is detecting")
        face = DeepFace.detectFace(img_path = picturepath, target_size = (224, 224), detector_backend = "opencv")
        plt.imsave(temppath + "opencv.jpg", face)


        #ssd detector
        print("ssd is detecting")
        face = DeepFace.detectFace(img_path = picturepath, target_size = (224, 224), detector_backend = "ssd")
        plt.imsave(temppath + "ssd.jpg", face)

        #dlib detector
        print("dlib is detecting")
        face = DeepFace.detectFace(img_path = picturepath, target_size = (224, 224), detector_backend = "dlib")
        plt.imsave(temppath + "dlib.jpg", face)

        #mtcnn detector
        print("mtcnn is detecting")
        face = DeepFace.detectFace(img_path = picturepath, target_size = (224, 224), detector_backend = "mtcnn")
        plt.imsave(temppath + "mtcnn.jpg", face)

        #retinaface detector
        print("retinaface is detecting")
        face = DeepFace.detectFace(img_path = picturepath, target_size = (224, 224), detector_backend = "retinaface")
        plt.imsave(temppath + "retinaface.jpg", face)

        #mediapipe detector
        print("mediapipe is detecting")
        face = DeepFace.detectFace(img_path = picturepath, target_size = (224, 224), detector_backend = "mediapipe")
        plt.imsave(temppath + "mediapipe.jpg", face)

        #if it gets to this point, all detectors have recognized a face in the picture. Now we can check whether they are all from the same person
        verify_faces(picturepath, picturename, foldername)
        #delete the temp dir and recreate an empty one...nope, it overwrites itself;)


    except ValueError as v:
        print("detection error!\n")
        badpic = Image.open(picturepath)
        badpic = badpic.save(savingpath_bad_noFace + foldername + "/" + picturename)


#check whether all detected faces in the temp folder are from the same person
def verify_faces(picturepath, picturename, foldername):
    allTheSameFace = True;
    #read in the face pictures from the temp folder
    with os.scandir(temppath) as entries:
        firstpic = next(entries)
        firstpic_name = firstpic.name
        for entry in entries:
            #verify face
            result = DeepFace.verify(img1_path = temppath + firstpic_name, img2_path = temppath + entry.name, enforce_detection=False)

            if (not(result["verified"])):
                allTheSameFace = False

    #sort the picture into the matching folder
    if allTheSameFace:
        print("This picture contains only 1 Face:)\n")
        goodpic = Image.open(picturepath)
        goodpic = goodpic.save(savingpath_good + foldername + "/" + picturename)
    else:
        print("MULTIPLE FACES DETECTED\n")
        badpic = Image.open(picturepath)
        badpic = badpic.save(savingpath_bad_multipleFaces + foldername + "/" + picturename)


#take all the pictures of the folders and let the face detection begin
with os.scandir(gettingpath) as folders:
    for folder in folders:
        bigger_gettingpath = gettingpath + folder.name + "/"
        os.mkdir("SuperCleaning_of_imdb_clean/imdb_superclean/" + folder.name)
        os.mkdir("SuperCleaning_of_imdb_clean/imdb_badpics/noFace/" + folder.name)
        os.mkdir("SuperCleaning_of_imdb_clean/imdb_badpics/multipleFaces/" + folder.name)

        with os.scandir(bigger_gettingpath) as entries:
            for entry in entries:
                detect_faces(entry.name, folder.name, bigger_gettingpath)

#temp Ordner aufräumen
with os.scandir("./SuperCleaning_of_imdb_clean/temp") as temps:
    for temp in temps:
        os.remove(temp)
os.rmdir("SuperCleaning_of_imdb_clean/temp")

print("Cleaning process 100% done.\n")
