from flask import (Blueprint, request, make_response)
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS

bp = Blueprint('meta', __name__, url_prefix='/analysis')


@bp.route('/metadata', methods=['POST'])
def metadata_analysis():
    if 'image' not in request.files:
        return make_response({"message": "Image not sent"}, 500)
    '''
    file = request.files['image']
    file.save("/home/kainaat/Documents/PEC/sem 8/Major", 'tmp.jpg')

    
    img = Image.open('/home/kainaat/Documents/PEC/sem 8/Major/tmp.jpg');

    fakeness = 0;
    exif_data = img._getexif()
    tags = {}
    
    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        tags[decoded] = value

    tags_found = ""

    for key in tags:
        temp = str(tags[key])
        if "Photoshop" in temp:
            fakeness = fakeness + 5
            tags_found = tags_found + "Photoshop"
        if "Gimp" in temp:
            fakeness = fakeness + 5
            tags_found = tags_found + "Gimp"
        if "Corel" in temp:
            fakeness = fakeness + 5
            tags_found = tags_found + "Corel"
        if "Adobe" in temp:
            fakeness = fakeness + 3
            tags_found = tags_found + "Adobe"
    '''
    response = make_response({"message": "Image Upload Successful"}, 200)
    #response.headers['fakeness'] = fakeness
    #response.headers['tags_found'] = tags_found
    return response
