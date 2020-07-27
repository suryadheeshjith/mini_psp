# # FOR TESTING
import sys
sys.path.append("..")
sys.path.append("../miniPSP")

from miniPSP.train import train
import os.path as osp


################################################################################
# DEFAULTS
################################################################################
model_list = ['unet','fcn','psp']
Data_fol  = 'Data'
Input_npy_file = 'sample_input.npy'
Output_npy_file = 'sample_output.npy'
model_path = 'Data'
model_name = 'psp'
epochs = 2
batch_size = 1
train_test = True
plot = False
################################################################################

# UPDATE IF REQUIRED
class _train_args_helper_:
    def __init__(self, input_npy, output_npy, mpath, mname, epochs, batch_size, train_test, plot_hist):
        self.input_npy = input_npy
        self.output_npy  = output_npy
        self.mpath = mpath
        self.mname = mname
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_test = train_test
        self.plot_hist = plot_hist

# UPDATE IF REQUIRED
def test_train():
    Input_npy_file_path = osp.join(Data_fol,Input_npy_file)
    Output_npy_file_path = osp.join(Data_fol,Output_npy_file)
    args = _train_args_helper_(input_npy=Input_npy_file_path, output_npy=Output_npy_file_path, mpath=model_path,
                        mname=model_name, epochs = epochs, batch_size = batch_size, train_test=train_test,
                        plot_hist = plot)
    train(args)
################################################################################

# if __name__ == '__main__':
#
#     test_train()
