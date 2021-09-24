from test import test_function
from compare import compare_function
from denoise import denoise_function
from eval import eval_function
from compare_damage import cd_function

def master_function():
    prefile='pre_disaster/'
    postfile='post_disaster/'
    test_function(prefile)
    test_function(postfile)
    compare_function()
    denoise_function()
    eval_function()
    cd_function()

