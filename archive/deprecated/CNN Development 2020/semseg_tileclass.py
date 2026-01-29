from PIL import Image
import argparse
import os

if __name__ == "__main__":

  

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="image directory to be processed")
	args = parser.parse_args()

    if args.images:
		image_directory = args.images
    
    color_red = (255,0,0)
    color_black = (0,0,0)
    color_white = (255,255,255)
    file_list = []
	for root, dirs, files in os.walk(os.path.abspath(image_directory)):
		for file in files:
			file_list.append(os.path.join(root, file))     
    result_file = open("result_file.txt","w")
    
    for file in file_list:
        im = Image.open(file)
        red_count = black_count = white_count = 0
        for pixel in im.getdata():
            if pixel == color_red:
                red_count += 1
            else if pixel == color_black:
                black_count += 1
            else if pixel ==  color_white:
                white_count += 1
        percentage_red = float(red_count) / float(65536)
        percentage_black = float(black_count) / float(65536)
        percentage_white = float(black_count) / float(65536)
        result_file.write(file + "\nred:" + percentage_red + "\nblack:" + percentage_black + "\nwhite:" + percentage_white)
        
        
        





