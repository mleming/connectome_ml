#!/usr/bin/python2.7 -u

from __future__ import print_function
import argparse, os, sys, time, re

parser=argparse.ArgumentParser()

# Set a default for this to make it easier. To select the directory the script
# is in, uncomment the currently commented option. Note that this won't work
# if this script is submitted to a cluster.
parser.add_argument('--save_directory',\
	default='/lustre/scratch/wbic-beta/ml784/ann4brains/fmri_ml/data',\
	help='The working directory')
#parser.add_argument('--save_directory',\
#	default=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data'),\
#	help='The working directory')

parser.add_argument('--no_epochs', default='20', \
	help='Maximum number of iterations in training phase',type=int)
parser.add_argument('--covariate', default='asd_or_not', \
	metavar='<gender/sex,rest_or_task,age,asd_or_not,mdd_or_not,age_group>')
parser.add_argument('--dataset_size_cap', default='40000', \
	help='Max no. of connectomes to read in total',type=int)
parser.add_argument('--hardware', default='cpu', \
	metavar='<cpu,gpu>')
parser.add_argument('--train_batch_size', default='128', \
	help='No. of connectomes to train on each iteration',type=int)
parser.add_argument('--correlation_type', \
	default='correlation', \
	metavar='<correlation,partial_correlation,mutual_information,' + \
	'wavelet_correlation{1..5}>', \
	help='Type of connectome to be trained on')
parser.add_argument('--lr', default='-1', \
	help='Learning rate for the ML model. -1 to use default.',
	type=float,metavar='<0.00-1.00>')
parser.add_argument('--train_model', default='0', \
	help='1 to train the model, 0 to not (Default: 0)',type=int)
parser.add_argument('--matrix_size', default='116', \
	help='Size of connectome (one dimension of it)', \
	metavar='<116,230>',type=int)
parser.add_argument('--random_seed', default='302', \
	help='Integer to seed the randomizer, to make it reproducible', \
	metavar='<INT>',type=int)
parser.add_argument('--save_output', default='0', \
	help='Saves the predictions in a .txt file', \
	metavar='<0,1>',type=int)
parser.add_argument('--n_filters_fc', default='64', \
	help='Number of connections in the fully-connected layers', \
	metavar='<INT>',type=int)
parser.add_argument('--n_filters_e2e', default='256', \
	help='Number of connections in the edge-to-edge layer', \
	metavar='<INT>',type=int)
parser.add_argument('--n_filters_e2n', default='24', \
	help='Number of connections in the edge-to-node layer', \
	metavar='<INT>',type=int)
parser.add_argument('--occlusion_set', default='0',\
	help='If set to 1, applies occlusion analysis to the model.',\
	metavar='<0,1>',type=int)
parser.add_argument('--occlusion_threshold',default=0.5,type=str)
parser.add_argument('--apply_activation_maximization', default='0',\
	help='If set to 1, applies activation max. analysis to the model.',\
	metavar='<0,1>',type=int)
parser.add_argument('--include_datasets',default='',\
	help='Include a particular dataset(s) in the training. Separate ' + \
	'dataset names with commas', \
	metavar='<ndar,abcd,open_fmri,icbm,adni,biobank,abide,1000_fc>')
parser.add_argument('--exclude_datasets',default='',\
	help='Exclude a particular dataset(s) from the training. Separate ' + \
	'dataset names with commas', \
	metavar='<ndar,abcd,open_fmri,icbm,adni,biobank,abide,1000_fc>')
parser.add_argument('--rest_only',default='0',\
	help='1 to only include resting-state fMRI in analysis', \
	metavar='<0,1>',type=int)
parser.add_argument('--momentum',default='0',\
	help='Momentum of learning. Should be low at the beginning.', \
	metavar='<0.0-1.0>',type=float)
parser.add_argument('--weight_decay',default='0.0005',\
	help='Weight decay',\
	metavar='<0.0-1.0>',type=float)
parser.add_argument('--relu_slope',default='0.33',\
	help='The negative slope for the leaky ReLU units',\
	metavar='<0.0,0.1...>',type=float)
parser.add_argument('--input_conn_name',default='',\
	help='Name of the connectome, if you would like to force it to load.'+\
	' Supercedes all other architectural specifications. Warning: other'+\
	' input variables must be correct, like dataset loading specs.')
parser.add_argument('--output_conn_name',default='',\
	help='Output name of connectome, if different from the input')
parser.add_argument('--verbose',default='1',type=int,help='Print out stuff',
	metavar='<0,1>')
parser.add_argument('--optimizer',default='adam',
	metavar='<sgd,adagrad,adadelta,adam,adamax,rmsprop,nadam>',
	help='Optimizer selection')
parser.add_argument('--balance_by',default='datasets',
	metavar='<ages,collections,datasets,none>',
	help='Criteria by which training/test sets are balanced. For '+\
	'instance, if datasets, then for each test selected to be in the '+\
	'set, a corresponding control will be selected in the same dataset')
parser.add_argument('--get_salience_maps',default='',
	help='Output saliency networks',metavar='<0,1,class,preds>',type=str)
parser.add_argument('--cutoff_accuracy',default='0',metavar='<0-1>',
	help='Stop training the model after the validation accuracy reaches '+\
	'this point',type=float)
parser.add_argument('--loss_cutoff',default='-1',metavar='<0+>',
	help='Stops training when validation loss goes below this (unless -1).',
	type=int)
parser.add_argument('--save_every_iter',default='0',metavar='<0,1>',
	help='Saves an output model every iteration.',type=int)
parser.add_argument('--save_best_iter',default='0',metavar='<0,1>',
	help='Saves the output model with the highest validation accuracy',type=int)
parser.add_argument('--num_dynamic_fc_slices',default='4',metavar='2+',
	help='Number of slices if dynamic functional connectivity is selected',\
	type=int)
parser.add_argument('--no_timepoints',default='490',metavar='<1+>',
	help='Number of timepoints in the read-in timeseries array, if that is '+\
	'the correlation_type',type=int)
parser.add_argument('--class_ratio',default='0.5',metavar='<0-1>',
	help='Ratio of classes to one another in training',type=float)
parser.add_argument('--divvy_method',default='sort',metavar='<sort,hash>',
	help='Method used to divide data into test, training, and validation '+\
	'sets. Hash is always the same, but may enact slightly unequal numbers '+\
	'between sets. Sort balances better by collection, but only if the full '+\
	'dataset is used.')
parser.add_argument('--ml_model',default='brainnetcnn_lite',
	metavar='<brainnetcnn,masked_model,masked_model_lite,alexnet,brainnetcnn_lite,brainnetcnn_alt,brainnetcnn_deeper>')
parser.add_argument('--mask_subdir',default='',
	metavar='string')
parser.add_argument('--mask_dir',
	default='/lustre/scratch/wbic-beta/ml784/ann4brains/all_connectomes/masks/mask_folders',
	metavar='/path/to/mask/directory')
parser.add_argument('--shuffle',default='0',metavar='<0,1>',
	help='Shuffles the order of input data; input data may be different if ' +\
	'dataset_size_cap is under maximum limit',type=int)
parser.add_argument('--control_only',default='0',metavar='<0,1>',
	help='Only reads in controls',type=int)
parser.add_argument('--test_file_list',default='',metavar='<file.txt>',
	help='Optional input list of test-set files',type=str)
parser.add_argument('--rand_hash',default='0',metavar='<0,1>',
	help='Makes the hashing algorithm random')
parser.add_argument('--quartile_balance',default='',
	metavar='<asd_or_not,gender,rest_or_task>',help='Class balances between'+\
	' a subclass',type=str)

# Put parser arguments into local namespace
args = parser.parse_args()
globals().update(args.__dict__)
train_model = bool(train_model)
occlusion_set = bool(occlusion_set)
apply_activation_maximization = bool(apply_activation_maximization)
verbose = bool(verbose)
save_every_iter = bool(save_every_iter)
save_best_iter = bool(save_best_iter)
rest_only = bool(rest_only)
shuffle = bool(shuffle)
control_only=bool(control_only)
rand_hash = bool(rand_hash)

# Remaining imports
import keras
from keras.layers import *
from keras.models import load_model
sys.path.insert(0,
				os.path.abspath(
				 os.path.join(
				  os.getcwd(),
				  '/lustre/scratch/wbic-beta/ml784/ann4brains/all_connectomes')))
sys.path.insert(0,
				os.path.abspath(
				 os.path.join(
				  os.getcwd(),'/lustre/scratch/wbic-beta/ml784/ann4brains/fmri_ml')))
from read_in_connectomes import read_in_data_sets
from connectome_ml_methods import *

np.random.seed(seed=random_seed) # To reproduce results.
num_classes = 2
exclude_datasets = [] if exclude_datasets == '' \
	else (exclude_datasets).split(',')
include_datasets = [] if include_datasets == '' \
	else (include_datasets).split(',')
optimizer  = optimizer.lower()
balance_by = balance_by.lower()
input_conn_name  = os.path.splitext(os.path.basename(input_conn_name))[0]
output_conn_name = os.path.splitext(os.path.basename(output_conn_name))[0]
conn_name = input_conn_name	if input_conn_name != "" else "%s_%s_%d_%d" % \
	(covariate,correlation_type,matrix_size,int(time.time()))
model_path = os.path.abspath(os.path.join(save_directory,\
	'models','%s.h5' % conn_name))
output_conn_name = conn_name if output_conn_name == "" else output_conn_name
output_model_path = os.path.abspath(os.path.join(save_directory,'models',
	'%s.h5' % output_conn_name))

if verbose:
	for arg in sorted(vars(args)):
		print("%s: %s" % (arg,getattr(args, arg)))

(x_train, y_train, x_train_filenames),\
(x_test,  y_test,  x_test_filenames),\
(x_valid, y_valid, x_valid_filenames) = \
	read_in_data_sets(
	scale = True,
	limit = dataset_size_cap,
	covariate = covariate,
	matrix_size = matrix_size,
	correlation_type = correlation_type,
	exclude = exclude_datasets,
	include = include_datasets,
	rest_only = rest_only,
	balance_by = balance_by,
	quartile_balance = quartile_balance,
	num_dynamic_fc_slices = num_dynamic_fc_slices,
	class_ratio = class_ratio,
	no_timepoints = no_timepoints,
	divvy_method = divvy_method,
	test_file_list = test_file_list,
	rand_hash = rand_hash,
	shuffle=shuffle)

for ff in x_test_filenames:
	print(ff)

num_classes = int(np.max(np.concatenate((y_train,y_test,y_valid))) + 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test,  num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

no_channels = x_train.shape[3]

if verbose: print("Model path: %s" % model_path)

if os.path.isfile(model_path):
	if verbose: print("Loading model...")
	model = load_model(model_path)
else:
	model = None

if ml_model == 'masked_model_lite':
	if mask_subdir != "":
		mask_dir = os.path.join(mask_dir,mask_subdir)
	reformat_coords_x = os.path.join(mask_dir,'reformat_coords_x.txt')
	reformat_coords_y = os.path.join(mask_dir,'reformat_coords_y.txt')
	
	x_train = reformat_data_to_masks_1_2_1(x_train,
		reformat_coords_x,reformat_coords_y)
	x_test  = reformat_data_to_masks_1_2_1(x_test,
		reformat_coords_x,reformat_coords_y)
	x_valid = reformat_data_to_masks_1_2_1(x_valid,
		reformat_coords_x,reformat_coords_y)
	

if model == None:
	if ml_model == 'brainnetcnn':
		model = get_cross_model(matrix_size,matrix_size,no_channels,
			n_filters_e2e,n_filters_e2n,n_filters_fc,num_classes,relu_slope)
	elif ml_model == 'masked_model':
		model = get_masked_model(matrix_size,no_channels,n_filters_e2e,
			n_filters_fc,num_classes,relu_slope,mask_dir)
	elif ml_model == 'rnn_model':
		model = get_rnn_model(matrix_size,x_train.shape[1],x_train.shape[2],
			x_train.shape[3])
	elif ml_model == 'masked_model_lite' or ml_model == 'rectangle':
		#if mask_subdir != "":
		#	mask_dir = os.path.join(mask_dir,mask_subdir)
		#masks = get_masks(mask_dir)
		#x_train = reformat_data_to_masks(x_train,masks)
		#x_test  = reformat_data_to_masks(x_test, masks)
		#x_valid = reformat_data_to_masks(x_valid,masks)
		
		print(x_train.shape)
		model = get_rectangle_model(x_test.shape[1],x_test.shape[2],n_filters_e2e,
			n_filters_fc,no_channels,num_classes,relu_slope)
	elif ml_model == 'alexnet':
		model = get_alexnet_adaptation(matrix_size,no_channels)
	elif ml_model == 'brainnetcnn_lite':
		model = get_light_cross_model(matrix_size,matrix_size,no_channels,
			n_filters_e2e,n_filters_e2n,n_filters_fc,num_classes,relu_slope)
	elif ml_model == 'brainnetcnn_alt':
		model = brainnetcnn_alt(matrix_size,no_channels)
	elif ml_model == 'brainnetcnn_deeper':
		model = brainnetcnn_deeper(matrix_size,no_channels)
	elif ml_model == 'vertical_filter_model':
		model = get_vertical_filter_model(matrix_size,n_filters_e2e,n_filters_e2n,n_filters_fc,no_channels,num_classes,relu_slope)
	elif ml_model == 'vertical_filter_model2':
		model = get_vertical_filter_model2(matrix_size,n_filters_e2e,n_filters_e2n,n_filters_fc,no_channels,num_classes,relu_slope)
	else:
		raise Exception('Invalid ML model option: %s' % ml_model)
	optimizer = get_optimizer(optimizer,lr,momentum)
	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=optimizer,
				  metrics=['accuracy'])
	if verbose: model.summary()

if train_model:
	save = True
	score = model.evaluate(x_test,y_test,verbose=0)
	max_valid_acc = -np.inf
	max_assoc_test_acc = -np.inf
	max_test_iter = None
	match = re.search(r'.*_iter_([0-9]+)$',conn_name)
	i_start = 0 if match == None else int(match.group(1))
	for i in range(i_start,no_epochs + i_start):
		if cutoff_accuracy > 0 and score[1] > cutoff_accuracy:
			break
		if loss_cutoff > 0 and score[0] < loss_cutoff:
			break
		model.fit(x_train, y_train,
			batch_size=train_batch_size,
			epochs=1,
			verbose=verbose)
		score_v = model.evaluate(x_valid,y_valid,verbose=0)
		score_t = model.evaluate(x_test, y_test, verbose=0)
		if score_v[1] >  max_valid_acc:
			max_valid_acc = score_v[1]
			max_assoc_test_acc = score_t[1]
			max_test_iter = i
			if save_best_iter:
				model.save(output_model_path.replace('.h5','_best_iter.h5'))
		if verbose:
			print("Iter %d res:" % i)
			print('Valid loss:',score_v[0])
			print('Valid accuracy:',score_v[1])
			print('Test loss:', score_t[0])
			print('Test accuracy:', score_t[1])
			print('Best acc thus far: %.4f at iter %d (valid: %.4f)' % (max_assoc_test_acc,max_test_iter,max_valid_acc))
		if save_every_iter:
			model.save(output_model_path.replace('.h5','_iter_%d.h5' % i))
	if verbose:
		print("Results for %s classifying %s" % (covariate,correlation_type))
		print("Max test accuracy was %.4f at iteration %d (valid acc: %.4f)"\
			%(max_assoc_test_acc,max_test_iter,max_valid_acc))
	if save:
		model.save(output_model_path)

if output_conn_name != "":
	conn_name = output_conn_name

if save_best_iter and train_model:
	model = load_model(output_model_path.replace('.h5','_best_iter.h5'))

if save_output:
	if verbose: print("Saving output to files...")
	pred_dir = os.path.join(save_directory, 'pred_outputs')
#	with open(os.path.join(pred_dir,'%s_glob_eval.txt'%conn_name),'w+') as f:
#		f.write('Test acc: %.4f\n' % max_valid_acc)
#		f.write('Valid acc: %.4f\n' % max_assoc_test_acc)
	for x_,y_,filenames,string in [(x_test,y_test,x_test_filenames,'test'),
								   (x_valid,y_valid,x_valid_filenames,'valid'),
								   (x_train,y_train,x_train_filenames,'train')]:
		params = (conn_name,string)
		all_preds = model.predict(x_)
		out_preds_file = '%s_preds_%s.txt' % params
		out_preds_file = os.path.join(pred_dir,out_preds_file)
		np.savetxt(out_preds_file,np.array(all_preds))
		out_covar_file = '%s_covar_%s.txt' % params
		out_covar_file = os.path.join(pred_dir,out_covar_file)
		np.savetxt(out_covar_file,np.array(y_)) 
		out_names_file = '%s_filenames_%s.txt' % params
		out_names_file = os.path.join(pred_dir,out_names_file)
		with open(out_names_file,'w') as file:
			for filename in filenames:
				file.write("%s\n" % filename)

output_x = x_test
output_y = y_test
output_y_pred = model.predict(output_x)
output_f = x_test_filenames

if apply_activation_maximization:
	if verbose: print("Performing activation maximization...")
	get_activation_maximization_set(model,
									output_x,
									output_f,
									conn_name=conn_name,
									save_directory=save_directory)

if get_salience_maps == "class":
	if verbose: print("Outputting salience maps...")
	for i in np.unique(output_y):
		salience_visualization(model,save_directory,('%s_class_%d' % (conn_name,i)),output_x,output_y_pred,output_f,True,i,perc_output=0.15)
elif get_salience_maps == "preds":
	salience_visualization(model,save_directory,conn_name,output_x,output_y_pred,output_f,True,perc_output=0.15)

if occlusion_set:
	if verbose: print("Occluding parts of the input...")
	smaps = get_salience_visualization(model,save_directory,conn_name,output_x,output_y,output_f,verbose=False)	
	smaps_mean = np.mean(smaps,axis=0)
	for i in range(smaps.shape[0]):
		smaps[i,:,:,:] = smaps_mean
	#saldir = os.path.join(save_directory,'salience',conn_name)
	#X = np.zeros((len(saldir),output_x.shape[1],output_x.shape[2],output_x.shape[3]))
	#y = np.zeros(())
	#f = ["" for x in range(len(saldir))]
	#sal = np.zeros(shape(X))
	#for w in range(4):
	#	wc = "wc%d" % w
	#	listdir = os.listdir(os.path.join(saldir,wc))
	#	for i in range(len(listdir)):
	#		filename = listdir[i]
	#		filepath = os.path.join(saldir,wc,filename)
	#		with open(filepath,'r') as file:
	#			sal[i,:,:,w] = np.genfromtxt(file)
	#		X[i,:,:,w] = 
	for oc in sorted([float(xx) for xx in str(occlusion_threshold).split(",")],reverse=True):
		output_x[smaps > oc] = 0
		all_occlusion_preds = model.predict(output_x)
		if not os.path.isdir(os.path.join(save_directory,'occlusion')):
			os.mkdir(os.path.join(save_directory,'occlusion'))
		ocsaveline = os.path.join(save_directory,'occlusion','%s_%s_%.3f.txt')
		np.savetxt(ocsaveline%(conn_name,'occlusion',oc),all_occlusion_preds)
		np.savetxt(ocsaveline%(conn_name,'covar',oc),output_y)
		with open(ocsaveline%(conn_name,'filenames',oc),'w') as file:
			for filename in output_f:
				file.write("%s\n" % filename)
