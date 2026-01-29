import cv2
import imutils
from skimage.measure import compare_ssim

#think about the data

a = cv2.imread("output1.tif")
b = cv2.imread("output2.tif") 

#mangrove_green = np.array([20,74,86])
#gotta identify green before that
#lowerGreen = np.array([0,100,100])
#upperGreen = np.array([30,255,255])

#hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)

#maskGreen = cv2.inRange(a, lowerGreen,upperGreen) 


# convert the images to grayscale
grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)


# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8") #DIFFERENCE IMAGE 
print("Change Percent: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
#thresh image --> overlay on after image 
thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE) #chain approx none saves all points, not just vertices 
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
#DO NOT DRAW BOUNDING BOXES.... DO INTENSITY
#more change = more darker
#cv2.imwrite("thresh_image.tif", thresh)
#c = cv2.imread("thresh_image.tif") 

#contours round mangrove.... if loss, put intensity 

# loop over the contours
for c in contours:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        area = cv2.contourArea(c)
        #if area > 1000:
         #       cv2.drawContours(a,c,-1,(255,0,0),3)
        if area > 300 and area < 5000:
                cv2.drawContours(a,c,-1, (0,255,0),3)
        if area > 200 and area < 300:
                cv2.drawContours(a,c,-1,(255,0,0),3)
         
        #arc_len = cv2.arcLength(c, True)
        #epsilon = 0.1*arc_len #epsilon = max dist from contour to approx contour
        #approx = cv2.approxPolyDP(c,epsilon,True) 
        #cv2.drawContours(a , [approx], -1,(0,255,0),2)
        #cv2.drawContours(b , [approx], -1,(0,255,0),2)

        
        #(x, y, w, h) = cv2.boundingRect(c)
        #cv2.rectangle(a, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.rectangle(b, (x, y), (x + w, y + h), (0, 0, 255), 2)
#dst = cv2.add(b,c)
#don't think you can add bc different num of channels

# show the output images
cv2.imshow("Before", a)
cv2.imshow("After", b)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
 
cv2.waitKey(0)
