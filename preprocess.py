import cv2 
import os


path = 'data/d1-1024x665.jpg'

def get_binary(path): 
    '''
    loads image from path and returns binary
    '''

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)
    return binary

if __name__ == '__main__': 

    for num, filename in enumerate(os.listdir('data')): 

        try: 

            path = os.path.join('data',filename)
            binary = get_binary(path)
            cv2.imwrite('processed_data/' + str(num) + '.jpeg', binary)
        except: 
            pass 
