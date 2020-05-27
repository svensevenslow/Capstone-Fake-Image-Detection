import os 

from PIL import Image

def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    img = cropped_image.resize((256, 256), Image.BILINEAR) 
    img.save(saved_location)
    
for filename in os.listdir("/home/kainaat/Documents/PEC/Apple_Orange_Data_Train-20200526T143854Z-001/Apple_Orange_Data_Train/"):
    crop('/home/kainaat/Documents/PEC/Apple_Orange_Data_Train-20200526T143854Z-001/Apple_Orange_Data_Train/' + filename, (500, 285, 760, 580), '/home/kainaat/Documents/PEC/Apple_Orange_Data_Train-20200526T143854Z-001/Apple_Orange_Data_Train/' + filename)
    #print(filename)

#image = "/home/kainaat/Documents/PEC/CycleGan_Data_Test-20200525T152207Z-001/CycleGan_Data_Test/img34.jpg"
#crop(image, (500, 285, 760, 580), '/home/kainaat/Documents/PEC/CycleGan_Data_Test-20200525T152207Z-001/CycleGan_Data_Test/cropped.jpg')


