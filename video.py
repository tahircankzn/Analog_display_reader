import cv2
import numpy as np
import time 

class bar():
    def __init__(self,width):
        def nothing(x):
            pass 
        self.width = width
        trackbar = np.zeros((300,512,3), np.uint8)
        cv2.namedWindow("Options")

        cv2.createTrackbar('level','Options',50,self.width,nothing)

video = cv2.VideoCapture("Adsız tasarım (1).mp4")

"""fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = frame_count / fps"""


width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

new_width = int(width / 2)
new_height = int(height / 2)

bar(new_width)

# Yeni boyutlar (örneğin, yarı boyutunda)


while True:
    
    current_time_sec = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
    minutes = current_time_sec
    if minutes >= 3.5:
        start_time_ms = 1 * 1000 
        video.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    level = cv2.getTrackbarPos('level','Options')
    
    if level <= 50 :
        level = 50

    # Resmi yükle
    
    
    ret, image = video.read()
    image = cv2.resize(image, (new_width, new_height))
    

    #image = cv2.imread('b.png')

    # Resmin boyutlarını al
    height, width = image.shape[:2]
    
    ###########################################################################################################################################
    #                                            RESİMİ DAİRE ŞEKLİNDE KIRPMA VE KIRPILAN KISMI DÖNDÜRME

    # Daireyi çizmek için bir maske oluştur
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(center[0], center[1]) * 0.8)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Maskeyi kullanarak resmi kes
    result = cv2.bitwise_and(image, image, mask=mask)

    # Resmi 90 derece saat yönünde döndür
    height, width = result.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -90, 1.0)
    rotated_image = cv2.warpAffine(result, rotation_matrix, (width, height))

    img = rotated_image.astype(np.float32)

    ###########################################################################################################################################
    #                                                         POLAR İMAGE

    #--- the following holds the square root of the sum of squares of the image dimensions ---
    #--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
    value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
    
    polar_image = cv2.linearPolar(img,center, value, cv2.WARP_FILL_OUTLIERS)
    
    polar_image = polar_image.astype(np.uint8)

    

    ###########################################################################################################################################

    hsv = cv2.cvtColor(polar_image[:,level-50:level], cv2.COLOR_BGR2HSV) # 100:200   # polar_image[:,level-50:level]
    # define range of a color in HSV
    lower_hue, upper_hue = np.array([0,0,0]) ,np.array([255,255,52]) # 40 # RENK AYARI
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_hue, upper_hue)
    # denem
    

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)


    rect = cv2.minAreaRect(largest_contour) # 
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(polar_image[:,level-50:level], [box], 0, (230, 0, 0), 2) #Ç polar_image[:,100:200]

    height, width = mask.shape

    


    # Dairenin merkezi (resmin ortası)
    centerX = width 
    centerY = height 
    
    seviye = 0
    for i in box:
        seviye += i[1]

    seviye = seviye / 4 

    val = (centerY - seviye) * (100/540)   # 540 => 10  centerY => 0  step => 0.2724795640326976

    cv2.putText(result, f"{str(val)[:5]}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    print(f"Value : {val}")

    height1, width1 = polar_image.shape[:2]
    cv2.rectangle(polar_image,(level-50,0),(level,width1),(0,0,255),thickness=5)

    #combined_img = cv2.hconcat([polar_image, result])
    #cv2.imshow("Output", combined_img)
    cv2.imshow("polar_image", polar_image)
    cv2.imshow("result", result)

    if cv2.waitKey(10) & 0xFF == ord("k"):
        
        cv2.destroyAllWindows()
        break
