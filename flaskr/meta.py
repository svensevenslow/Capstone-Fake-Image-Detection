from flask import Blueprint
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS

bp = Blueprint('meta', __name__, url_prefix='/analysis')


@bp.route('/metadata', methods=['GET'])
def metadata_analysis():
    return 'Welcome to the Metadata Module'
