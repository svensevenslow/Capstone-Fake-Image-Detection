from flask import (Blueprint, request, make_response, render_template, flash)
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS

bp = Blueprint('meta', __name__, url_prefix='/analysis')


@bp.route('/metadata', methods=['POST'])
def metadata_analysis():
   
    meta_analysis = 1
    path = "uploads/tmp.jpg"
    file = request.files['image_file']
    file.save(path)
    
    img = Image.open(path);

    fakeness = 0;
    exif_data = img._getexif()
    tags = {}
    
    if exif_data is None:
        meta_analysis = 0
        return render_template('homepage.html', meta_analysis=meta_analysis)

    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        tags[decoded] = value

    tags_found = ""

    for key in tags:
        temp = str(tags[key])
        if "Photoshop" in temp:
            fakeness = fakeness + 5
            tags_found = tags_found + "Photoshop\n"
        if "Gimp" in temp:
            fakeness = fakeness + 5
            tags_found = tags_found + "Gimp\n"
        if "Corel" in temp:
            fakeness = fakeness + 5
            tags_found = tags_found + "Corel\n"
        if "Adobe" in temp:
            fakeness = fakeness + 3
            tags_found = tags_found + "Adobe\n"
    

    return render_template('homepage.html', meta_fakeness=fakeness, meta_tags_found=tags_found, meta_analysis=meta_analysis)
