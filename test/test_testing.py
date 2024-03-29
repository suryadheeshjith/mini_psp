## FOR TESTING
# import sys
# sys.path.append("..")
# sys.path.append("../mini_psp")


from mini_psp.test import evaluater
import os.path as osp

################################################################################
# DEFAULTS
################################################################################
class_names = ['1','2']
Data_fol  = 'test/Data'
Input_npy_file = 'sample_input.npy'
Output_npy_file = 'sample_output.npy'
json_file = 'sample_model.json'
weights_file = 'sample_model_final_weights.h5'
model_name = 'psp'
train_test = True
eval = True
save = True
plot = False
################################################################################

# UPDATE IF REQUIRED
class _test_args_helper_:
    def __init__(self, input_npy, output_npy, mjpath, mwpath, mname, train_test, eval, plot_conf, save_masks):
        self.input_npy = input_npy
        self.output_npy  = output_npy
        self.mjpath = mjpath
        self.mwpath = mwpath
        self.mname = mname
        self.train_test = train_test
        self.eval = eval
        self.plot_conf = plot_conf
        self.save_masks = save_masks

# UPDATE IF REQUIRED
def test_test():
    Input_npy_file_path = osp.join(Data_fol,Input_npy_file)
    Output_npy_file_path = osp.join(Data_fol,Output_npy_file)
    json_file_path = osp.join(Data_fol,json_file)
    weights_file_path = osp.join(Data_fol,weights_file)
    args = _test_args_helper_(input_npy=Input_npy_file_path, output_npy=Output_npy_file_path, mjpath=json_file_path,
                        mwpath=weights_file_path, mname=model_name, train_test=train_test, eval=eval, plot_conf=plot,
                        save_masks=save)
    evaluater(args, class_names)

################################################################################

## FOR TESTING
# if __name__=="__main__":
#
#     test_test()
