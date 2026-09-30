import re

from liberopro.liberopro.envs.base_object import OBJECTS_DICT, VISUAL_CHANGE_OBJECTS_DICT

from .hope_objects import *
from .google_scanned_objects import *
from .articulated_objects import *
from .turbosquid_objects import *
from .site_object import SiteObject
from .target_zones import *
from .self_designed_object import RedSticker, LiberoMugYellow

# --- Fix for missing mug names ---
_bad_keys = {
    "white_white_porcelain_mug": "white_porcelain_mug", 
    "white_yellow_porcelain_mug": "yellow_porcelain_mug", 
    "white_red_porcelain_mug": "red_porcelain_mug"
}

for b_key, g_key in _bad_keys.items():
    if b_key not in OBJECTS_DICT:
        if g_key in OBJECTS_DICT:
            OBJECTS_DICT[b_key] = OBJECTS_DICT[g_key]
        elif "porcelain_mug" in OBJECTS_DICT:
            OBJECTS_DICT[b_key] = OBJECTS_DICT["porcelain_mug"]
# ---------------------------------

def get_object_fn(category_name):
    # print(OBJECTS_DICT.keys())
    try:
        s = OBJECTS_DICT[category_name.lower()]
    except:
        print(category_name)
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT
