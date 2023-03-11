import os
import logging
from PIL import Image


FILE_FORMAT = ".jpg"


def check_images(folder_name):
    logging.info("Checking folder " + folder_name + " for corrupted files...") 
    for filename in os.listdir(folder_name):
        if filename.endswith(FILE_FORMAT):
            try:
                img = Image.open(folder_name + '/' + filename)
                img.verify()
            except (IOError, SyntaxError) as e:
                logging.warning("Removing file:  " + filename) 
                os.remove(folder_name + '/' + filename)
