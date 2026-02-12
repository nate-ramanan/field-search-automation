from ultralytics import YOLO
import math
import numpy as np
from PIL import Image
from configparser import ConfigParser

#get config data
file = './config.ini'
config = ConfigParser()
config.read(file)
infield_model = config['model_paths']['infield_model']
threshold = float(config['find_shift']['infield_threshold'])
'''
    This code is used to detect if there is a baseball infield. If there is a infield
    the image is zoomed into that location and the image is than saved. Returns 1 if infield is detected
    and the image is zoomed and 0 otherwise
'''

#method to calculate distance between two points
def euclidean_distance(point1,point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    
  
def find_shift(image, save_image, height, width):
    screen_center = [height//2,width//2]
    model = YOLO(infield_model) #loads infield detection model

    results = model.predict(source=image)
    results = results[0] #access the tensor(multidimensional array) that holds the information about detected objects
    if len(results) > 0: # if there is a infield detected
        
        weights = []
        distances = []
        
        for result in results:
            bounding_box = result.boxes.xyxy.tolist()[0] #gets box around each infield
            center = [((bounding_box[2] + bounding_box[0]) // 2), ((bounding_box[3] + bounding_box[1]) // 2)]
            distance = euclidean_distance(screen_center, center)
            distances.append(distance)
            
            '''weight is used so distance from center of image also plays a part in determining which infield 
               to select if there are multiple'''
            weight = 1 / distance
            weights.append(weight)
         
        confidence = [float(c) for c in results.boxes.conf.tolist()] #gets confidence which is a tensor and turns to a list
	
        #confidence, weights, and distances list all have the same amount of elements
        for i in range(len(confidence)-1, -1, -1): #loop backwards to prevent index out of range when removing
            if confidence[i] < threshold:
                del weights[i] 
                del confidence[i]
                del distances[i]
              
                
        #combines the elements with zip so [(c0,w0),(c1,w1)...] where (c1/w1=confidence/weight index 0) and multiplies them
        likelihood = [c * w for c, w in zip(confidence, weights)] #c
      
        
        if len(likelihood) != 0: #found a infield with large enough confidence
            maxConfidence = max(likelihood)
            index = likelihood.index(maxConfidence)
            
            bounding_box = results.boxes.xyxy.tolist()[index] #gets bounding box in xyxy and turns from a tensor to a list [[x,y,x,y]] and gets the inner list

            center = [((bounding_box[2] + bounding_box[0]) // 2), ((bounding_box[3] + bounding_box[1]) // 2)]

        
            zoom_factor = 3  
           
            # Calculate the new width and height of the zoomed region
            new_width = int(width / zoom_factor)
            new_height = int(height / zoom_factor)

            # Calculate the top-left and bottom-right coordinates of the zoomed region
            left = center[0] - (new_width // 2)
            top = center[1] - (new_height // 2)
            right = left + new_width
            bottom = top + new_height
            
            if left < 0:
                right -= left #in this case left with be a negative number
                left = 0
            if top < 0:
                bottom -= top
                top = 0
            if right > width:
                left -= (right - width)
                right = width
            if bottom > height:
                top -= (bottom - height)
                bottom = height

            im = Image.open(image)
            zoomed_image = im.crop((left, top, right, bottom)).resize((width, height))
            zoomed_image.save(save_image)
          
            return 1
        else:
            return 0
    else:
        return 0

