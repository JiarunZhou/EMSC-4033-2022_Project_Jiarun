"""
Define test functions for each of functions:
    - get_images()
    - display_random_image()
    - subtract_meanRGB()
    - Image_Generator()
"""

import pytest
from functions import *

def test_my_documentation(type_my_doc = str): 
    # Test with type
    
    type_docu = type(my_documentation())
    len_docu = len(my_documentation())
    
    
    assert type_docu == type_my_doc, " *** Fail to return a documentation "

