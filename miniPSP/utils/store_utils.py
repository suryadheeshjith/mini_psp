import os.path as osp


def save_details(args,input_shape,target_shape):

    save_path = args.output_fol+"/"+"data_details.txt"
    f= open(save_path,"w+")

    f.write("\nData details : \n\n")
    f.write("Input folder : {}\n".format(osp.abspath(args.input_fol)))
    f.write("Output folder : {}\n".format(osp.abspath(args.output_fol)))
    f.write("Strides taken : {}\n".format(args.strides))

    if(args.thresh>0):
        f.write("Selecting Tiles with Percentage ones and threshold : {},{}\n".format(args.percentage_ones,args.thresh))


    f.write('Input shape : {}\n'.format(input_shape))
    f.write('Target shape : {}'.format(target_shape))
    f.close()
