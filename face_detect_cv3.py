import cv2
import sys
import numpy as np;
from sklearn.cluster import KMeans
#https://www.timpoulsen.com/2018/finding-the-dominant-colors-of-an-image.html
# USING THIS METHOD TO DETERMINE DOMINANT COLOR IN IMAGE SECTION
def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


def make_bar(height, width, color):
    """
    Create an image of a given color
    :param: height of the image
    :param: width of the image
    :param: BGR pixel values of the color
    :return: tuple of bar, rgb values, and hsv values
    """
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv_bar[0][0]
    return bar, (red, green, blue), (hue, sat, val)

# Get user supplied values
imagePath = "whiteshirt.jpeg"
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
detector = cv2.SimpleBlobDetector_create()

# Read the image
image = cv2.imread(imagePath)
gray = cv2.imread("whiteshirt.jpeg", cv2.IMREAD_GRAYSCALE)
# Detect blobs.
canny = cv2.Canny(gray,0.1,0.1,3,3)
cv2.imshow("Canny", canny)
#color list
colors = [0,0,0,255,255,255,255,0,0,0,255,0,0,0,255]
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print("x: " + str(w))
    print("y: " + str(h))
    cv2.rectangle(image, (x, y+h), (x+w, y+2*h), (0, 255, 0), 2)
    face = image[y+h:y+(h*2), x:x+w]
    cv2.imshow("lower", face)
    keypoints = detector.detect(face)
    im_with_keypoints = cv2.drawKeypoints(face, keypoints, np.zeros((1,1)), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    num_clusters = 5
    height, width, _ = np.shape(face)
    image = face.reshape((height * width, 3))
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(image)
    cv2.imshow("Keypoints", im_with_keypoints)
    histogram = make_histogram(clusters)
    combined = zip(histogram, clusters.cluster_centers_)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # finally, we'll output a graphic showing the colors in order
    bars = []
    hsv_values = []
    dist = 0
    mindist = 100000
    savedindex = 0
    index2 = 0
    for index, rows in enumerate(combined):
        bar, rgb, hsv = make_bar(100, 100, rows[1])
        if index == 0:
            while index2 < (len(colors)-3):
                #print(np.square((int(colors[index2])-rgb[0])))
                dist = np.sqrt(np.square((int(colors[index2])-rgb[0]))+np.square((int(colors[index2+1])-rgb[1]))+np.square((int(colors[index2+2])-rgb[2])))
                print(dist)
                if dist < mindist:
                    mindist = dist
                    savedindex = index2
                index2+=3
print(savedindex)
if savedindex == 0:
    print('the color is black')
if savedindex == 3:
    print('the color is white')
if savedindex == 6:
    print('the color is red')
if savedindex == 9:
    print('the color is green')
if savedindex == 12:
    print('the color is blue')
cv2.waitKey(0)
