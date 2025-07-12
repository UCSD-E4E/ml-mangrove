import numpy as np
import cv2
import glob
 
def build_filters():
	filters = []
	ksize = 10
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.25, 0, ktype=cv2.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters
   
def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
 		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
  		np.maximum(accum, fimg, accum)
 	return accum
 
if __name__ == '__main__':
	import sys
 
 	#print __doc__
 	#try:
 	#	img_fn = sys.argv[1]
 	#except:
	#	img_fn = 'test.png'
        i = 0
 
        for img_fn in glob.glob("./white/*.jpg"):
 	        img = cv2.imread(img_fn)
 	        if img is None:
 		        print 'Failed to load image file:', img_fn
 		        sys.exit(1)
	        filters = build_filters()
                res1 = process(img, filters)
	        #cv2.imshow('result', res1)
                cv2.imwrite('white' + str(i) + '.jpg', res1);
                i+=1
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
        sys.exit(0)
        
