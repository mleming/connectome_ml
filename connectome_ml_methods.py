import os, sys, re, glob
import numpy as np
from scipy.stats.stats import pearsonr
# To import ann4brains
sys.path.insert(0,
				os.path.abspath(
				 os.path.join(
				  os.getcwd(),'/lustre/scratch/wbic-beta/ml784/ann4brains/all_connectomes')))
from read_in_connectomes import read_in_data_sets
from os import listdir
from os.path import isfile, join
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras.backend import batch_dot
from keras.optimizers import *

#import matplotlib.pyplot as plt

def is_float(n):
	try:
		float(n)
		return True
	except:
		return False


def get_connectome_name(arch,prefix="",covariate="gender"):
	arch_string = ""
	for layer,params in arch:
		if layer == 'e2e':
			arch_string = arch_string + 'e2e' +\
					str(params['n_filters'])
		if layer == 'e2n':
			arch_string = arch_string + 'e2n' +\
					str(params['n_filters'])
		if layer == 'fc':
			arch_string = arch_string + 'fc' +\
					str(params['n_filters'])
	#		fc_count+=1
	#		fc_num = params['n_filters']
	#		if n2g_num == None:
	#			n2g_num = params['n_filters']
	#if n2g_num == fc_num:
	#	n2g_num = None
	#elif n2g_num != None:
	#	fc_count -= 1
	#	
	#if fc_count > 0:
	#	arch_string = prefix + arch_string + str(fc_count) + \
	#	("" if n2g_num == None else "_n2g" + str(n2g_num) + "_") + "fc"\
	#		+ str(fc_num) + "_" + covariate
	arch_string = prefix + arch_string + "_" + covariate
	return arch_string

def get_latest_snapshot_num(cname):
	try:
		mypath='snapshot'
		onlyfiles = [f for f in listdir(mypath)\
			 	if isfile(join(mypath, f))]
		onlyfiles = filter(\
				lambda k: ("%s_iter_" % cname) in k, onlyfiles)
		file_nums = map(\
				lambda k: k.split("_")[-1].split(".")[0],\
				 onlyfiles)
		file_nums = map(lambda k: int(k), file_nums)
		return max(file_nums)
	except:
		return 0

def binary_predict(net, x,y,ret_preds=False):
	preds = net.predict(x) # Predict labels of test data
		
	sums = 0
	for i in range(len(preds)):
		sums += ((preds[i] > 0.5) == y[i])
	
	if ret_preds:
		return float(sums)/len(preds),preds
	else:
		return float(sums)/len(preds)

def correlation_predict(net,x,y,ret_preds=False):
	preds = net.predict(x)
	if ret_preds:
		return pearsonr(preds,y),preds
	else:
		return pearsonr(preds,y)

def get_list_of_models(covariate = "gender", correlation_type = "correlation"):
	model_list = []
	for f in os.listdir(os.path.join('.','models')):
		m = re.search("(" + correlation_type + "e2n.*" +\
			 covariate + ")\.pkl",f)
		if m:
			model_list.append(m.groups()[0])
	return model_list

def visualize_weights(net, layer_name, padding=4, filename=''):
	# The parameters are a list of [weights, biases]
	data = np.copy(net.params[layer_name][0].data)
	# N is the total number of convolutions
	N = data.shape[0]*data.shape[1]
	# Ensure the resulting image is square
	filters_per_row = int(np.ceil(np.sqrt(N)))
	# Assume the filters are square
	filter_size = data.shape[2]
	# Size of the result image including padding
	result_size = filters_per_row*(filter_size + padding) - padding
	# Initialize result image to all zeros
	result = np.zeros((result_size, result_size))
	
	# Tile the filters into the result image
	filter_x = 0
	filter_y = 0
	for n in range(data.shape[0]):
		for c in range(data.shape[1]):
			if filter_x == filters_per_row:
				filter_y += 1
				filter_x = 0
			for i in range(filter_size):
				for j in range(filter_size):
					result[filter_y*(filter_size + padding)\
					 + i, filter_x*(filter_size + padding)\
					 + j] = data[n, c, i, j]
			filter_x += 1
	
	# Normalize image to 0-1
	min = result.min()
	max = result.max()
	result = (result - min) / (max - min)
	
	# Plot figure
	plt.figure(figsize=(10, 10))
	plt.axis('off')
	plt.imshow(result, cmap='gray', interpolation='nearest')
	
	# Save plot if filename is set
	if filename != '':
		plt.savefig(filename, bbox_inches='tight', pad_inches=0)
	
	plt.show()

# Returns a specification of the architecture for the deep learning model.
def get_arch(h,w, type="",
	n_filters_e2e = 70,
	n_filters_e2n = 70,
	n_filters_fc  = 50,
	relu_slope=0.33):
	
	if isinstance(n_filters_e2e,int):
		n_filters_e2e = [n_filters_e2e]
	if isinstance(n_filters_fc,int):
		n_filters_fc = [n_filters_fc]
	if isinstance(relu_slope,float):
		relu_slope = [relu_slope]
	assert(len(relu_slope) in [1,len(n_filters_e2e) + len(n_filters_fc)+1])
	arch = []
	i = 0
	for f in n_filters_e2e:
		arch.append(['e2e',{'n_filters':f,'kernel_h':h,'kernel_w':w}])
		arch.append(['relu',{'negative_slope': relu_slope[i]}])
		i+=1
		i = i % len(relu_slope)
	arch.append(['e2n',{'n_filters':n_filters_e2n,'kernel_h':h,'kernel_w':w}])
	for f in n_filters_fc:
		arch.append(['dropout',{'dropout_ratio': 0.5}])
		arch.append(['relu',{'negative_slope': relu_slope[i]}])
		i+=1
		i = i % len(relu_slope)
		arch.append(['fc',{'n_filters':f}])
	arch.append(['relu',{'negative_slope': relu_slope[i]}])
	arch.append(['out',  {'n_filters': 2}])
	return arch

def get_arch_old3(h,w, type="",
	n_filters_e2e = 70,
	n_filters_e2n = 70,
	n_filters_fc  = 50,
	relu_slope=0.33):
	
	if isinstance(n_filters_e2e,int):
		n_filters_e2e = [n_filters_e2e]
	if isinstance(n_filters_fc,int):
		n_filters_fc = [n_filters_fc]
	if isinstance(relu_slope,float):
		relu_slope = [relu_slope]
	print "len(relu_slope): " + str(len(relu_slope))
	print "len(n_filters_fc): " + str(len(n_filters_fc))
	print "len(n_filters_e2e): " + str(len(n_filters_e2e))
	assert(len(relu_slope) in [1,len(n_filters_e2e) + len(n_filters_fc)+1])
	arch = []
	i = 0
	foomy=True
	for f in n_filters_e2e:
		if foomy:
			arch.append(['e2e',{'n_filters':f,'kernel_h':h,'kernel_w':w}])
		else:
			arch.append(['e2e',{'n_filters':f,'kernel_h': 58,'kernel_w': 58}])
		arch.append(['relu',{'negative_slope': relu_slope[i]}])
		if foomy:
			arch.append(['pool',{'kernel_size':2,'stride':2}])
			foomy=False
		i+=1
		i = i % len(relu_slope)
	arch.append(['e2n',{'n_filters':n_filters_e2n,'kernel_h': 58,'kernel_w': 58}])
	for f in n_filters_fc:
		arch.append(['dropout',{'dropout_ratio': 0.5}])
		arch.append(['relu',{'negative_slope': relu_slope[i]}])
		i+=1
		i = i % len(relu_slope)
		arch.append(['fc',{'n_filters':f}])
	arch.append(['relu',{'negative_slope': relu_slope[i]}])
	arch.append(['out',  {'n_filters': 1}])
	return arch

def get_arch_old2(h,w, type="",
	n_filters_e2e = 70,
	n_filters_e2n = 70,
	n_filters_fc  = 50,
	relu_slope=0.33):
	arch = [['e2e', {'kernel_w': 116, 'n_filters': 32, 'kernel_h': 116}],
		['relu', {'negative_slope': 0.33}],
		['pool', {'kernel_size':4,'stride':2}],
		['e2e', {'kernel_w': 57, 'n_filters': 32, 'kernel_h': 57}],
		['relu', {'negative_slope': 0.33}],
		['pool', {'kernel_size':4,'stride':2}],
		['e2e', {'kernel_w': 28, 'n_filters': 64, 'kernel_h': 28}],
		['relu', {'negative_slope': 0.33}],
		['pool', {'kernel_size':3,'stride':2}],
		['e2n', {'kernel_w': 13, 'n_filters': 128, 'kernel_h': 13}],
		['dropout', {'dropout_ratio': 0.5}],
		['relu', {'negative_slope': 0.05}],
		['fc', {'n_filters': 64}],
		['dropout', {'dropout_ratio': 0.5}],
		['relu', {'negative_slope': 0.05}],
		['fc', {'n_filters': 64}],
		['relu', {'negative_slope': 0.05}],
		['out', {'n_filters': 1}]]
	return arch

def get_arch_old1(h,w, type="",
	n_filters_e2e = 70,
	n_filters_e2n = 70,
	n_filters_fc  = 50,
	relu_slope=0.33):
	arch = [['e2e', {'kernel_w': 116, 'n_filters': 8, 'kernel_h': 116}],
		['e2n', {'kernel_w': 116, 'n_filters': 16, 'kernel_h': 116}],
		['dropout', {'dropout_ratio': 0.5}],
		['relu', {'negative_slope': 0.33}],
		['fc', {'n_filters': 32}],
		['dropout', {'dropout_ratio': 0.5}],
		['relu', {'negative_slope': 0.33}],
		['fc', {'n_filters': 32}],
		['relu', {'negative_slope': 0.0}],
		['out', {'n_filters': 2}]]
	return arch


# These two methods occlude certain parts of a network and checks what the
# classification percentage is with that.
def get_one_occlusion_set(X,x_ind, y_ind=None):
    #(65, 1, 116, 116) sample size
    x_len = X.shape[2]
    y_len = X.shape[3]
    ones = np.array([[1 for x in xrange(x_len)] for y in xrange(y_len)])
    if not isinstance(x_ind,list):
        x_ind = [x_ind]
    x_ind = np.array(x_ind)
    if y_ind != None and len(x_ind.shape) == 1:
        for i in x_ind:
            ones[i,y_ind] = 0
            ones[y_ind,i] = 0
    elif len(x_ind.shape) == 2:
        ones = x_ind
    elif len(x_ind.shape) == 1:
        for j in x_ind:
            ones[:,j] = 0
            ones[j,:] = 0
    return np.multiply(ones,X),ones

# Copied and pasted for later
def unscramble_mask(mean_salience_filepath,mask_folderpath):
	assert(os.path.isdir(mask_folderpath))
	assert(os.path.isfile(mean_salience_filepath))	
	with open(mean_salience_filepath,'r') as file:
		salience = np.genfromtxt(file,dtype=np.float32)
	mask = np.zeros((116,116))
	sorted_mask_dir = sorted(os.listdir(mask_folderpath))
	with open(os.path.join(mask_folderpath,'reformat_coords_x.txt')) as file:
		reformat_coords_x = np.genfromtxt(file)
		reformat_coords_x = reformat_coords_x.astype(int)
	with open(os.path.join(mask_folderpath,'reformat_coords_y.txt')) as file:
		reformat_coords_y = np.genfromtxt(file)
		reformat_coords_y = reformat_coords_y.astype(int)
	for x in range(reformat_coords_x.shape[0]):
		for y in range(reformat_coords_y.shape[1]):
			mask[reformat_coords_x[x,y],reformat_coords_y[x,y]] = salience[x,y]
			mask[reformat_coords_y[x,y],reformat_coords_x[x,y]] = salience[x,y]
	return mask

def apply_occlusion_set(net,X,y,conn_name="",columnwise = True):
    import random
    import time
    time.sleep(random.int(1,10000)/1000.0) # For parallelization
    x_len = X.shape[2]
    y_len = X.shape[3]
    dir_path = os.path.join('occlusion_preds')
    if conn_name != "":
        dir_path = os.path.join(dir_path,conn_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    or_preds_filename = os.path.join(dir_path,'original_preds.txt')
    or_y_filename     = os.path.join(dir_path,'original_y.txt')
    if not os.path.isfile(or_y_filename) or \
        not os.path.isfile(or_preds_filename):
        base_pred,base_preds = binary_predict(net,X,y,ret_preds=True)
        base_preds = np.array(base_preds)
        np.savetxt(or_preds_filename,base_preds)
        np.savetxt(or_y_filename,np.array(y))
    base_preds = np.loadtxt(or_preds_filename)
    y          = np.loadtxt(or_y_filename)
    if columnwise:
        for i in range(x_len):
            filename = os.path.join(dir_path,'oc_pred_%d.txt' % i)
            filename_arr = os.path.join(dir_path,'oc_arr_%d.txt' % i)
            if not os.path.isfile(filename) or not os.path.isfile(filename_arr):
                open(filename,'a').close() # For parallelization
                X_o,ones = get_one_occlusion_set(X,i)
                pred,preds = binary_predict(net,X_o,y,ret_preds=True)
                try:
                    np.savetxt(filename,np.array(preds))
                    np.savetxt(filename_arr,ones)
                except:
                    if os.path.isfile(filename):
                        os.remove(filename)
                    if os.path.isfile(filename_arr):
                        os.remove(filename_arr)
                    continue
    else: # Occludes one edge at a time
        for i in range(x_len):
            for j in range(i+1,x_len):
                filename = os.path.join(dir_path,'oc_pred_%d_%d.txt' % (i,j))
                filename_arr = os.path.join(dir_path,'oc_arr_%d_%d.txt' % (i,j))
                if not os.path.isfile(filename) or not \
                    os.path.isfile(filename_arr):
                    open(filename,'a').close() # For parallelization
                    X_o,ones = get_one_occlusion_set(X,i,j)
                    pred,preds = binary_predict(net,X_o,y,ret_preds=True)
                    try:
                        np.savetxt(filename,np.array(preds))
                        np.savetxt(filename_arr,ones)
                    except:
                        if os.path.isfile(filename):
                            os.remove(filename)
                        if os.path.isfile(filename_arr):
                            os.remove(filename_arr)
                        continue
                
        #preds = np.loadtxt(filename)

# Eliminates all of the nodes that didn't have a big effect on the prediction,
# then runs that through to see what happens.
def occlusion_elim_analysis(net,X,y,conn_name):
	apply_occlusion_set(net,X,y,conn_name)
	dir_path = os.path.join('occlusion_preds',conn_name)
	x_len = X.shape[2]
	i = 1
	while True:
		pred_path = os.path.join(dir_path,'node_elim_preds_%d.txt' % i)
		arr_path  = os.path.join(dir_path,'node_elim_arr_%d.txt' % i)
		i += 1
		if not(os.path.isfile(pred_path) and os.path.isfile(arr_path)):
			break
	rows_to_include = []
	for i in range(len(x_len)):
		filename = os.path.join(dir_path,'oc_pred_%d.txt' % i)
		preds = np.loadtxt(filename)
		ff = np.mean(np.abs(preds - y))
		print ff
		rows_to_include.append(ff > 0.01)
	ones = np.array([[1 for x in xrange(x_len)] for y in xrange(x_len)])
	for i in range(len(rows_to_include)):
		if rows_to_include[i]:
			ones[i,:] = 0
			ones[:,i] = 0
	np.savetxt(arr_path,ones)
	X_o = np.multiply(ones,X)
	pred,preds = binary_predict(net,X_o,y,ret_preds=True)
	np.savetxt(pred_path,preds)

# Should only be performed with one model. Outputs a lot of stuff
def get_activation_maximization_set(model,
                                    X_set,
                                    X_filenames,
                                    conn_name="",
                                    output_layer="e2n",
                                    save_directory="."):
    
    dir_path = os.path.join(save_directory,'activation_maximization_set',conn_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    layer = None
    if is_float(output_layer):
        layer = model.layers[int(output_layer)]
    else:
        for l in model.layers:
            if l.name == output_layer or l.name == "e2n" or l.name == "conv2d_8":
                layer = l
    if l == None:
        raise Exception('Layer name %s does not exist in model' % output_layer)
    get_layer_output = K.function([model.layers[0].input],[layer.output])
    activations = get_layer_output([X_set])[0]
    for i in range(len(X_set)):
        np.savetxt(os.path.join(dir_path,'%s_%s_activation.txt' % \
                (X_filenames[i],output_layer)),
                np.squeeze(activations[i,:,:,:]))

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def salience_visualization(model,save_directory,conn_name,output_x,output_y,
		output_f,verbose=True,clas=None,perc_output=1):
	from vis.visualization import visualize_saliency
	from vis.losses import ActivationMaximization
	from vis.optimizer import Optimizer
	from vis.utils import utils
	from vis.backprop_modifiers import get
	from keras import activations
	
	if clas != None:
		output_y = np.zeros(output_y.shape) + clas
	# Utility to search for layer index by name. 
	# Alternatively we can specify this as -1 since it corresponds to the last layer.
	#layer_idx = utils.find_layer_idx(model, 'preds')
	layer_idx = -1
	# Swap softmax with linear
	model.layers[layer_idx].activation = activations.linear
	model = utils.apply_modifications(model)
	
	###
	modifier = 'guided' # can be None (AKA vanilla) or 'relu'
	save_grads_path = os.path.join(save_directory,'salience',conn_name)
	if not os.path.isdir(os.path.join(save_directory,'salience')):
		os.mkdir(os.path.join(save_directory,'salience'))
	if not os.path.isdir(save_grads_path):
		os.mkdir(save_grads_path)
	print("Outputting saliency maps")
	if False:
		for the_file in os.listdir(save_grads_path):
			file_path = os.path.join(save_grads_path, the_file)
			try:
				if os.path.isfile(file_path):
					os.unlink(file_path)
			except Exception as e:
				print(e)
	modifier_fn = get(modifier)
	model = modifier_fn(model)
	for idx in range(output_x.shape[0]):
		if verbose:
			update_progress(float(idx) / output_x.shape[0])
		if float(idx) / output_x.shape[0] > perc_output:
			break
		#savename = os.path.join(save_grads_path,output_f[idx])
		#if os.path.isfile(savename):
		#	continue
		losses = [(ActivationMaximization(model.layers[layer_idx], int(output_y[idx][0])), -1)]
		opt = Optimizer(model.input, losses, wrt_tensor=None, norm_grads=False)
		grads = opt.minimize(seed_input=output_x[idx], max_iter=1, grad_modifier='absolute', verbose=False)[1]
		for i in range(grads.shape[3]):
			wc_subfolder = os.path.join(save_grads_path,"wc%d"%i)
			if not os.path.isdir(wc_subfolder):
				os.mkdir(wc_subfolder)
			np.savetxt(os.path.join(wc_subfolder,output_f[idx]),
				np.squeeze(grads[:,:,:,i]))
		#grads = visualize_saliency(model, layer_idx, filter_indices=int(output_y[idx][0]),
		#	seed_input=output_x[idx], backprop_modifier=modifier)
		#np.savetxt(savename,np.squeeze(grads))

def get_salience_visualization(model,save_directory,conn_name,output_x,output_y,
		output_f,verbose=True):
	salience_visualization(model,save_directory,conn_name,output_x,output_y,output_f,verbose=False)
	save_grads_path = os.path.join(save_directory,'salience',conn_name)
	smaps = np.zeros(output_x.shape)
	for i in range(output_x.shape[0]):
		for j in range(output_x.shape[3]):
			wc_subfolder = os.path.join(save_grads_path,"wc%d"%j)
			smaps[i,:,:,j] = np.genfromtxt(os.path.join(wc_subfolder,output_f[i]))
	return smaps

def get_optimizer(optimizer, lr, momentum):
	if lr == -1:
		"This option is here in case you want to use defaults"
	elif optimizer == 'adam':
		optimizer = Adam(clipnorm=1.,lr=lr)
	elif optimizer == 'nadam':
		optimizer = Nadam(clipnorm=1.) #Suggested to leave at default
	elif optimizer == 'adadelta':
		optimizer = Adadelta(clipnorm=1.,lr=lr)
	elif optimizer == 'adagrad':
		optimizer = Adagrad(lr=lr)
	elif optimizer == 'sgd':
		optimizer = SGD(lr=lr, momentum=momentum)
	return optimizer

def get_cross_model(h,w,no_channels,
	n_filters_e2e,n_filters_e2n,n_filters_fc,num_classes,relu_slope): 
	input_layer = Input((h,w,no_channels))
	concat_dx1_list = [Conv2D(n_filters_e2e,(h,1),\
		input_shape=(h,w,no_channels))\
		(input_layer) for x in range(w)]
	concat_1xd_list = [Conv2D(n_filters_e2e,(1,w),\
		input_shape=(h,w,no_channels))\
		(input_layer) for x in range(h)]
	concat_dx1_dxd = Concatenate(axis=1)(concat_dx1_list)
	concat_1xd_dxd = Concatenate(axis=2)(concat_1xd_list)
	sum_dxd = Add()([concat_dx1_dxd, concat_1xd_dxd])	
	LReLU1 = LeakyReLU(alpha=relu_slope)(sum_dxd)
	
	# E2N layer
	conv_1xd_e2n = Conv2D(n_filters_e2n,(h,1),\
		input_shape=(h,w),name="e2n")(LReLU1)
	dropout = Dropout(0.5)(conv_1xd_e2n)
	LReLU2 = LeakyReLU(alpha=relu_slope)(dropout)
	
	Flat = Flatten()(LReLU2)
	Dense1 = Dense(4096)(Flat)
	Activ1 = Activation('relu')(Dense1)
	Dropout1 = Dropout(0.5)(Activ1)
	BatchN1 = BatchNormalization()(Dropout1)
		
	Dense2 = Dense(4096)(BatchN1)
	Activ2 = Activation('relu')(Dense2)
	Dropout2 = Dropout(0.5)(Activ2)
	BatchN2 = BatchNormalization()(Dropout2)
		
	Dense3 = Dense(1000)(BatchN2)
	Activ3 = Activation('relu')(Dense3)
	Dropout3 = Dropout(0.5)(Activ3)
	BatchN3 = BatchNormalization()(Dropout3)
	

	
	# Fully-connected layers
	#Flat = Flatten()(LReLU2)
	#Dense1 = Dense(n_filters_fc)(Flat)
	#Dropout2 = Dropout(0.5)(Dense1)
	#LReLU3 = LeakyReLU(alpha=relu_slope)(Dropout2)
	#Dense2 = Dense(n_filters_fc)(LReLU3)
	#Dropout3 = Dropout(0.5)(Dense2)
	##LReLU4 = LeakyReLU(alpha=relu_slope)(Dropout3)
	##Dense3 = Dense(n_filters_fc)(LReLU4)
	##Dropout4 = Dropout(0.5)(Dense3)
	out = Dense(num_classes, activation='softmax')(BatchN3)
	
	return Model(inputs=input_layer,outputs=out)

def get_light_cross_model(h,w,no_channels,
	n_filters_e2e,n_filters_e2n,n_filters_fc,num_classes,relu_slope): 
	
	input_layer = Input((h,w,no_channels))
	
	# Cross-shaped filters
	concat_dx1 = Conv2D(n_filters_e2e,(h,1),\
		input_shape=(h,w,no_channels))(input_layer)
	concat_1xd = Conv2D(n_filters_e2e,(1,w),\
		input_shape=(h,w,no_channels))(input_layer)	
	concat_dx1_list = [concat_dx1 for x in range(w)]
	concat_1xd_list = [concat_1xd for x in range(h)]
	concat_dx1_dxd = Concatenate(axis=1)(concat_dx1_list)
	concat_1xd_dxd = Concatenate(axis=2)(concat_1xd_list)
	batch_dx1_dxd = BatchNormalization()(concat_dx1_dxd)
	batch_1xd_dxd = BatchNormalization()(concat_1xd_dxd)
	relu_dx1_dxd = ReLU()(batch_dx1_dxd)
	relu_1xd_dxd = ReLU()(batch_1xd_dxd)
	
	sum_dxd = Add()([relu_dx1_dxd, relu_1xd_dxd])	
	CBatch1 = BatchNormalization()(sum_dxd)
	CReLU1 = ReLU()(CBatch1)
	CDropout1 = Dropout(0.1)(CReLU1)
	
	# E2N layer
	conv_1xd_e2n = Conv2D(n_filters_e2n,(h,1),\
		input_shape=(h,w),name="e2n")(CDropout1)
	CBatch2 = BatchNormalization()(conv_1xd_e2n)
	CReLU2 = ReLU()(CBatch2)
	CDropout2 = Dropout(0.1)(CReLU2)
	
	# Fully-connected layers
	Flat = Flatten()(CDropout2)
	Dense1 = Dense(n_filters_fc)(Flat)
	Batch1 = BatchNormalization()(Dense1)
	ReLU1 = ReLU()(Batch1)
	Dropout1 = Dropout(0.5)(ReLU1)
	Dense2 = Dense(n_filters_fc)(Dropout1)
	Batch2 = BatchNormalization()(Dense2)
	ReLU2 = ReLU()(Batch2)
	Dropout2 = Dropout(0.5)(ReLU2)
	Dense3 = Dense(n_filters_fc)(Dropout2)
	Batch3 = BatchNormalization()(Dense3)
	ReLU3 = ReLU()(Batch3)
	Dropout3 = Dropout(0.5)(ReLU3)
	out = Dense(num_classes, activation='softmax')(Dropout3)
	
	return Model(inputs=input_layer,outputs=out)

def get_masks(folder_path):
	dir = sorted(os.listdir(folder_path))
	arr = np.genfromtxt(os.path.join(folder_path,dir[0]))
	all_masks = np.zeros((len(dir),arr.shape[0],arr.shape[1]),dtype=np.bool)
	for i in range(len(dir)):
		file = os.path.join(folder_path,dir[i])
		arr = np.genfromtxt(file,delimiter = ' ',dtype=np.float32) > 0.5
		all_masks[i,:,:] = arr
	return all_masks

def reformat_data_to_masks(X,masks):
	if isinstance(masks,str):
		if os.path.isdir(masks):
			masks = get_masks(masks)
	max_idx = -1
	for m in masks:
		max_idx = max(max_idx,np.sum(m))
	print (X.shape[0],max_idx,len(masks),X.shape[3])
	X_reformat = np.zeros((X.shape[0],max_idx,len(masks),X.shape[3]))
	for i in range(len(masks)):
		c = 0
		m = np.squeeze(masks[i,:,:])
		for x in range(m.shape[0]):
			for y in range(m.shape[1]):
				if m[x,y] == 1:
					X_reformat[:,c,i,:] = X[:,x,y,:]
					c += 1
	return X_reformat

def reformat_data_to_masks_1_2_1(X,reformat_coords_x,reformat_coords_y):
	if isinstance(reformat_coords_x,str):
		if os.path.isfile(reformat_coords_x):
			with open(reformat_coords_x,'r') as file:
				reformat_coords_x = np.genfromtxt(file,dtype=np.float32)
				reformat_coords_x = reformat_coords_x.astype(int)
	if isinstance(reformat_coords_y,str):
		if os.path.isfile(reformat_coords_y):
			with open(reformat_coords_y,'r') as file:
				reformat_coords_y = np.genfromtxt(file,dtype=np.float32)
				reformat_coords_y = reformat_coords_y.astype(int)
	assert(np.all(reformat_coords_y.shape == reformat_coords_x.shape))
	X_reformat = np.zeros((X.shape[0],
						reformat_coords_x.shape[0],
						reformat_coords_x.shape[1],
						X.shape[3]))
	for x in range(reformat_coords_x.shape[0]):
		for y in range(reformat_coords_x.shape[1]):
			X_reformat[:,x,y,:] = X[:,reformat_coords_x[x,y],reformat_coords_y[x,y],:]
	return X_reformat

def get_masked_model(matrix_size,no_channels,n_filters_e2e,n_filters_fc,
	num_classes,relu_slope,mask_dir):
	
	masks = get_masks(mask_dir)
	input_layer = Input((matrix_size,matrix_size,no_channels))	
	mask_layers = [Lambda(lambda x: x*np.reshape(m,(matrix_size,matrix_size,1)))(input_layer) for m in masks]
	concat = Concatenate(axis=3)(mask_layers)
	conv = Conv2D(n_filters_e2e*len(mask_layers),(matrix_size,matrix_size))(concat)
	LReLU2 = LeakyReLU(alpha=relu_slope)(conv)
	
	# Fully-connected layers
	Flat = Flatten()(LReLU2)
	Dense1 = Dense(n_filters_fc)(Flat)
	Dropout2 = Dropout(0.5)(Dense1)
	LReLU3 = LeakyReLU(alpha=relu_slope)(Dropout2)
	Dense2 = Dense(n_filters_fc)(LReLU3)
	Dropout3 = Dropout(0.5)(Dense2)
	out = Dense(num_classes, activation='softmax')(Dropout3)
	
	return Model(inputs=input_layer,outputs=out)

def get_rectangle_model(h,w,n_filters_conv,n_filters_fc,no_channels,num_classes,relu_slope):
	
	input_layer = Input((h,w,no_channels))
	Conv = Conv2D(n_filters_conv,(1,w),input_shape=(h,w,no_channels))(input_layer)
	BatchN = BatchNormalization(name="e2n")(Conv) 
	LReLU = LeakyReLU(alpha=relu_slope)(BatchN)
	
	Flat = Flatten()(LReLU)
	
	dense_1   = Dense(n_filters_fc)(Flat)
	batchn_1  = BatchNormalization()(dense_1)
	activat_1 = LeakyReLU(alpha=relu_slope)(batchn_1)
	dropout_1 = Dropout(0.5)(activat_1)
	
	dense_2   = Dense(n_filters_fc)(dropout_1)
	batchn_2  = BatchNormalization()(dense_2)
	activat_2 = LeakyReLU(alpha=relu_slope)(batchn_2)
	dropout_2 = Dropout(0.5)(activat_2)
	
	dense_3   = Dense(n_filters_fc)(dropout_2)
	batchn_3  = BatchNormalization()(dense_3)
	activat_3 = LeakyReLU(alpha=relu_slope)(batchn_3)
	dropout_3 = Dropout(0.5)(activat_3)
	
	dense_4   = Dense(num_classes)(dropout_3)
	out       = Activation('softmax')(dense_4)
	
	return Model(inputs=input_layer,outputs=out)


def get_rnn_model(matrix_size,row,col,pixel):
	col_hidden = matrix_size
	row_hidden = matrix_size
	#row, col, pixel = x_train.shape[1:]
	input_layer = keras.layers.Input(shape=(row, col, pixel))
	mask = keras.layers.Masking(mask_value=0., input_shape=(row, col, pixel))(input_layer)
	encoded_rows = keras.layers.TimeDistributed(LSTM(row_hidden))(mask)
	encoded_columns = keras.layers.LSTM(col_hidden)(encoded_rows)
	prediction = keras.layers.Dense(num_classes, activation='softmax')(encoded_columns)
	return Model(input_layer,prediction)

def get_alexnet_adaptation(matrix_size,no_channels):
	input_shape = (matrix_size,matrix_size,no_channels)
	
	input = Input(shape=input_shape)
	
	conv2d_1a = Conv2D(96, kernel_size=(29, 1))(input)
	activa_1a = Activation('relu')(conv2d_1a)
	pool2d_1a = MaxPooling2D(pool_size=(1,2),strides = (1,3))(activa_1a)
	batchn_1a = BatchNormalization()(pool2d_1a)
	
	conv2d_2a = Conv2D(256, kernel_size=(1,11))(batchn_1a)
	activa_2a = Activation('relu')(conv2d_2a)
	pool2d_2a = MaxPooling2D(pool_size=(2,1),strides = (3,1))(activa_2a)
	batchn_2a = BatchNormalization()(pool2d_2a)
	
	conv2d_3a = Conv2D(384, kernel_size=(5, 1), strides=(1,1))(batchn_2a)
	activa_3a = Activation('relu')(conv2d_3a)
	batchn_3a = BatchNormalization()(activa_3a)
		
	conv2d_4a = Conv2D(384, kernel_size=(1, 5), strides=(1,1))(batchn_3a)
	activa_4a = Activation('relu')(conv2d_4a)
	batchn_4a = BatchNormalization()(activa_4a)
	
	conv2d_5a = Conv2D(256, kernel_size=(25,1))(batchn_4a)
	activa_5a = Activation('relu')(conv2d_5a)
	pool2d_5a = MaxPooling2D(pool_size=(1, 2),strides = (1,2))(activa_5a)
	batchn_5a = BatchNormalization()(pool2d_5a)
	
	conv2d_6a = Conv2D(128, kernel_size=(1,12), strides=(1,1))(batchn_5a)
	activa_6a = Activation('relu')(conv2d_6a)
	batchn_6a = BatchNormalization()(activa_6a)
	
	flatten   = Flatten()(batchn_6a)
	
	dense1    = Dense(4096)(flatten)
	activat_1 = Activation('relu')(dense1)
	dropout1  = Dropout(0.4)(activat_1)
	batchn_7  = BatchNormalization()(dropout1)
	
	dense2    = Dense(4096)(batchn_7)
	activat_2 = Activation('relu')(dense2)
	dropout2  = Dropout(0.4)(activat_2)
	batchn_8  = BatchNormalization()(dropout2)
	
	dense3    = Dense(1000)(batchn_8)
	activat_3 = Activation('relu')(dense3)
	dropout3  = Dropout(0.4)(activat_3)
	batchn_9  = BatchNormalization()(dropout3)
	
	dense4    = Dense(2)(batchn_9)
	output    = Activation('softmax')(dense4)
	
	return Model(input,output)

def brainnetcnn_alt(matrix_size,no_channels):
	
	input_shape = (matrix_size,matrix_size,no_channels)
	input = Input(shape=input_shape)
	
	conv2d_1a = Conv2D(96, kernel_size=(matrix_size, 1))(input)
	activa_1a = Activation('relu')(conv2d_1a)
	batchn_1a = BatchNormalization()(activa_1a)
	
	conv2d_2a = Conv2D(96, kernel_size=(1,matrix_size))(input)
	activa_2a = Activation('relu')(conv2d_2a)
	batchn_2a = BatchNormalization()(activa_2a)
	
	batchn_1a_perm = Permute((3,2,1))(batchn_1a)
	batchn_2a_perm = Permute((3,2,1))(batchn_2a)
	
	matmul = Dot((3,2))([batchn_1a_perm,batchn_2a_perm])
	matmul_perm = Permute((3,2,1))(matmul)
	matmul_perm._keras_shape = (None, matrix_size, matrix_size, 96) # keras bug
	batchn = BatchNormalization()(matmul_perm)
	
	conv2d_3a = Conv2D(256, kernel_size=(1,matrix_size))(batchn)
	activa_3a = Activation('relu')(conv2d_3a)
	batchn_3a = BatchNormalization()(activa_3a)
	
	conv2d_4a = Conv2D(256, kernel_size=(matrix_size,1))(batchn_3a)
	activa_4a = Activation('relu')(conv2d_4a)
	batchn_4a = BatchNormalization()(activa_4a)
	
	flatten   = Flatten()(batchn_4a)
	
	dense1    = Dense(4096)(flatten)
	activat_1 = Activation('relu')(dense1)
	dropout1  = Dropout(0.4)(activat_1)
	batchn_7  = BatchNormalization()(dropout1)
	
	dense2    = Dense(4096)(batchn_7)
	activat_2 = Activation('relu')(dense2)
	dropout2  = Dropout(0.4)(activat_2)
	batchn_8  = BatchNormalization()(dropout2)
	
	dense3    = Dense(1000)(batchn_8)
	activat_3 = Activation('relu')(dense3)
	dropout3  = Dropout(0.4)(activat_3)
	batchn_9  = BatchNormalization()(dropout3)
	
	dense4    = Dense(2)(batchn_9)
	output    = Activation('softmax')(dense4)
	
	return Model(input,output)

def brainnetcnn_deeper(matrix_size,no_channels):
	input_shape = (matrix_size,matrix_size,no_channels)
	input = Input(shape=input_shape)
	
	conv2d_1a = Conv2D(256, kernel_size=(matrix_size/4, 1),strides=(matrix_size/4,1))(input)
	activa_1a = Activation('relu')(conv2d_1a)
	batchn_1a = BatchNormalization()(activa_1a)
	
	conv2d_1b = Conv2D(256, kernel_size=(4, 1))(batchn_1a)
	activa_1b = Activation('relu')(conv2d_1b)
	batchn_1b = BatchNormalization()(activa_1b)
	
	conv2d_2a = Conv2D(256, kernel_size=(1,matrix_size/4),strides=(matrix_size/4,1))(input)
	activa_2a = Activation('relu')(conv2d_2a)
	batchn_2a = BatchNormalization()(activa_2a)
	
	conv2d_2b = Conv2D(256, kernel_size=(1,4))(batchn_2a)
	activa_2b = Activation('relu')(conv2d_2b)
	batchn_2b = BatchNormalization()(activa_2b)
	
	batchn_1a_perm = Permute((3,2,1),input_shape=(None,1, 116, 256))(batchn_1b)
	batchn_2a_perm = Permute((3,2,1),input_shape=(None,116, 1, 256))(batchn_2b)
	
	matmul = Dot((3,2))([batchn_1a_perm,batchn_2a_perm])
	matmul_perm = Permute((3,2,1),input_shape=(None,256,116,116))(matmul)
	matmul_perm._keras_shape = (None, matrix_size, matrix_size, 256) # keras bug
	batchn = BatchNormalization()(matmul_perm)
	
	conv2d_3a = Conv2D(384, kernel_size=(1,matrix_size/4),strides=(1,matrix_size/4))(batchn)
	activa_3a = Activation('relu')(conv2d_3a)
	batchn_3a = BatchNormalization()(activa_3a)
	
	conv2d_3b = Conv2D(384, kernel_size=(1,4))(batchn_3a)
	activa_3b = Activation('relu')(conv2d_3b)
	batchn_3b = BatchNormalization()(activa_3b)
	
	conv2d_4a = Conv2D(384, kernel_size=(matrix_size/4,1),strides=(matrix_size/4,1))(batchn_3b)
	activa_4a = Activation('relu')(conv2d_4a)
	batchn_4a = BatchNormalization()(activa_4a)
	
	conv2d_4b = Conv2D(384, kernel_size=(4,1))(batchn_4a)
	activa_4b = Activation('relu')(conv2d_4b)
	batchn_4b = BatchNormalization()(activa_4b)
	
	flatten   = Flatten()(batchn_4a)
	
	dense1    = Dense(4096)(flatten)
	activat_1 = Activation('relu')(dense1)
	dropout1  = Dropout(0.4)(activat_1)
	batchn_7  = BatchNormalization()(dropout1)
	
	dense2    = Dense(4096)(batchn_7)
	activat_2 = Activation('relu')(dense2)
	dropout2  = Dropout(0.4)(activat_2)
	batchn_8  = BatchNormalization()(dropout2)
	
	dense3    = Dense(1000)(batchn_8)
	activat_3 = Activation('relu')(dense3)
	dropout3  = Dropout(0.4)(activat_3)
	batchn_9  = BatchNormalization()(dropout3)
	
	dense4    = Dense(2)(batchn_9)
	output    = Activation('softmax')(dense4)
	
	return Model(inputs,output)

def get_vertical_filter_model(matrix_size,n_filters_e2e,n_filters_e2n,n_filters_fc,no_channels,num_classes,relu_slope):
	
	input_shape = (matrix_size,matrix_size,no_channels)
	
	input = Input(shape=input_shape)
	
	conv2d_1a = Conv2D(n_filters_e2e,(1,matrix_size),input_shape=input_shape)(input)
	batchn_1a  = BatchNormalization(name="e2n")(conv2d_1a)
	activa_1a  = ReLU()(batchn_1a)
	dropout_1a = Dropout(0.3)(activa_1a)
	
	conv2d_1b  = Conv2D(n_filters_e2n,(matrix_size,1),\
				 input_shape=(matrix_size,1,n_filters_e2e))(dropout_1a)
	batchn_1b  = BatchNormalization()(conv2d_1b)
	activa_1b  = ReLU()(batchn_1b)
	dropout_1b = Dropout(0.3)(activa_1b)
	
	flatten    = Flatten()(dropout_1b)
	
	dense_1    = Dense(n_filters_fc)(flatten)
	batchn_1   = BatchNormalization()(dense_1)
	activat_1  = ReLU()(batchn_1)
	dropout_1  = Dropout(0.7)(activat_1)
	
	dense_2    = Dense(n_filters_fc)(dropout_1)
	batchn_2   = BatchNormalization()(dense_2)
	activat_2  = ReLU()(batchn_2)
	dropout_2  = Dropout(0.7)(activat_2)
	
	dense_3    = Dense(n_filters_fc)(dropout_2)
	batchn_3   = BatchNormalization()(dense_3)
	activat_3  = ReLU()(batchn_3)
	dropout_3  = Dropout(0.7)(activat_3)
	
	dense_4    = Dense(num_classes)(dropout_3)
	out        = Activation('softmax')(dense_4)
	
	return Model(input,out)


def get_vertical_filter_model2(matrix_size,n_filters_e2e,n_filters_e2n,n_filters_fc,no_channels,num_classes,relu_slope):
	
	input_shape = (matrix_size,matrix_size,no_channels)
	
	input = Input(shape=input_shape)
	
	conv2d_1a = Conv2D(n_filters_e2e,(1,matrix_size),input_shape=input_shape)(input)
	batchn_1a  = BatchNormalization()(conv2d_1a)
	activa_1a  = ReLU()(batchn_1a)
	dropout_1a = Dropout(0.3)(activa_1a)
	
	conv2d_1b  = Conv2D(n_filters_e2n,(matrix_size,1),\
				 input_shape=(matrix_size,1,n_filters_e2e))(dropout_1a)
	batchn_1b  = BatchNormalization(name="e2n")(conv2d_1b)
	activa_1b  = ReLU()(batchn_1b)
	dropout_1b = Dropout(0.3)(activa_1b)
	
	flatten    = Flatten()(dropout_1b)
	
	dense_1    = Dense(n_filters_fc)(flatten)
	batchn_1   = BatchNormalization()(dense_1)
	activat_1  = ReLU()(batchn_1)
	dropout_1  = Dropout(0.7)(activat_1)
	
	dense_2    = Dense(n_filters_fc)(dropout_1)
	batchn_2   = BatchNormalization()(dense_2)
	activat_2  = ReLU()(batchn_2)
	dropout_2  = Dropout(0.7)(activat_2)
	
	#dense_3    = Dense(n_filters_fc)(dropout_2)
	#batchn_3   = BatchNormalization()(dense_3)
	#activat_3  = ReLU()(batchn_3)
	#dropout_3  = Dropout(0.7)(activat_3)

	#dense_4    = Dense(n_filters_fc)(dropout_3)
	#batchn_4   = BatchNormalization()(dense_4)
	#activat_4  = ReLU()(batchn_4)
	#dropout_4  = Dropout(0.7)(activat_4)
	
	dense_5    = Dense(num_classes)(dropout_2)
	out        = Activation('softmax')(dense_5)
	
	return Model(input,out)
