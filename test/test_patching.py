# FOR TESTING
# import sys
# sys.path.append("..")
# sys.path.append("../miniPSP")

from miniPSP.patch_generator import generate



################################################################################
# DEFAULTS
################################################################################
tdim = 256
input_fol = 'Data'
output_fol  = 'Data'
thresh = 8
percentage_ones  =0.25
strides = 128
train_test = False
save_details = True
################################################################################

# UPDATE IF REQUIRED
class _patch_args_helper_:
    def __init__(self, tdim, input_fol, output_fol, thresh, percentage_ones, strides, train_test, save_details):
        self.tdim = tdim
        self.input_fol  = input_fol
        self.output_fol  = output_fol
        self.thresh = thresh
        self.percentage_ones = percentage_ones
        self.strides = strides
        self.train_test = train_test
        self.save_details = save_details

# UPDATE IF REQUIRED
def test_patch():
    args = _patch_args_helper_(tdim=tdim, input_fol=input_fol, output_fol= output_fol,
                            thresh=thresh, percentage_ones=percentage_ones, strides = strides,
                            train_test = train_test,save_details=save_details)

    generate(args)
################################################################################
# 
# if __name__ == '__main__':
#
#     test_patch()
