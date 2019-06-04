import shutil as sh
import os
import numpy as np
import openslide
import ghalton
from PIL import Image
import itertools
import sys

import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import time
import argparse
from torch.optim import lr_scheduler
import copy
import torch.nn.parallel
import torch.optim as optim
#import data_aug as DA
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score
import sys
import torch.backends.cudnn as cudnn
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from scipy.interpolate import RegularGridInterpolator
import bottleneck as bn
from scipy import interpolate
from scipy.interpolate import griddata


def top_n_indices(arr, n):

    idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]

def gen_tile_idx():
   
    k = ghalton.Halton(1)
    points = k.get(10)
    points = list(itertools.chain.from_iterable(points))
    points = np.array(points)
 
    return points

def extract_tiles(slide_name, output_folder, idx_mode, idx_array=None):

    patch_size_40X = 101;

    try:
        oslide = openslide.OpenSlide(slide_name);
    #    mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
           mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        elif "XResolution" in oslide.properties:
           mag = 10.0 / float(oslide.properties["XResolution"]);
        else:
           mag = 10.0 / float(0.254);
        pw = int(patch_size_40X * mag / 40);
        width = oslide.dimensions[0];
        height = oslide.dimensions[1];
    except:
        print '{}: exception caught'.format(slide_name);
        exit(1);

    print (width, height)

    points = gen_tile_idx()

    if idx_mode == True:
	x = points * width
 	y = points * height
	
   	
	x = x.astype('int32')
	y = y.astype('int32')

	x_idx, y_idx= list(x), list(y)

	print slide_name, width, height;

	for x in x_idx:
	    for y in y_idx:
        	if x + pw > width:
	            pw_x = width - x;
       		else:
	            pw_x = pw;
	        if y + pw > height:
        	    pw_y = height - y;
	        else:
        	    pw_y = pw;
	        patch = oslide.read_region((x, y), 0, (pw_x, pw_y));
	        patch = patch.resize((patch_size_40X * pw_x / pw, patch_size_40X * pw_y / pw), Image.ANTIALIAS);
	        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_40X);
	        patch.save(fname);

#	open(fdone, 'w').close();
    
    elif idx_mode == False:
	x = idx_array[:,0] 
	y = idx_array[:, 1]

        x = x.astype('int32')
        y = y.astype('int32')

        x_idx, y_idx= list(x), list(y)
	for x in x_idx:
	    for y in y_idx:
        	if x + pw > width:
	            pw_x = width - x;
	        else:
        	    pw_x = pw;
	        if y + pw > height:
        	    pw_y = height - y;
	        else:
        	    pw_y = pw;
	        patch = oslide.read_region((x, y), 0, (pw_x, pw_y));
	        patch = patch.resize((patch_size_40X * pw_x / pw, patch_size_40X * pw_y / pw), Image.ANTIALIAS);
	        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_40X);
        	patch.save(fname);

#	open(fdone, 'w').close();
    return True
APS = 350;
PS = 101
#TileFolder = sys.argv[1] + '/';
#heat_map_out = "output_trial.txt"
BatchSize = 64;
#TileFolder = input_tile_directory
def mean_std(type = 'none'):
    if type == 'vahadane':
        mean = [0.8372, 0.6853, 0.8400]
        std = [0.1135, 0.1595, 0.0922]
    elif type == 'macenko':
        mean = [0.8196, 0.6938, 0.8131]
        std = [0.1417, 0.1707, 0.1129]
    elif type == 'reinhard':
        mean = [0.8364, 0.6738, 0.8475]
        std = [0.1315, 0.1559, 0.1084]
    elif type == 'macenkoMatlab':
        mean = [0.7805, 0.6230, 0.7068]
        std = [0.1241, 0.1590, 0.1202]
    else:
        mean = [0.7238, 0.5716, 0.6779]
        std = [0.1120, 0.1459, 0.1089]
    return mean, std

type = 'none'
mu, sigma = mean_std(type)

#mu = [0.5, 0.5, 0.5]
#sigma = [0.5, 0.5, 0.5]

mu = [0.6714873,  0, 0] # mean, std of YUV
sigma = [0.17164655, 1, 1]

device = torch.device("cuda")
data_aug = transforms.Compose([
    transforms.Scale(PS),
    transforms.ToTensor(),
    transforms.Normalize(mu, sigma)])

def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


def softmax_np(x):
    x = x - np.max(x, 1, keepdims=True)
    x = np.exp(x) / (np.sum(np.exp(x), 1, keepdims=True))
    return x

def load_data(img_list, rind, TileFolder):

    X = torch.zeros(size=(len(img_list), 3, PS, PS))
    inds = np.zeros(shape=(len(img_list,)), dtype=np.int32)
    coor = np.zeros(shape=(len(img_list), 2), dtype=np.int32)
    xind = 0
    cind = 0
    parts = 4
    lind = 0
    for img in img_list:
#       print (img_list)
        lind += 1
        full_img = TileFolder + '/' + img
        if not os.path.isfile(full_img):
            continue;
        if (len(img.split('_')) != parts) or ('.png' not in img):
            continue;
        try:
            x_off = float(img.split('_')[0]);
            y_off = float(img.split('_')[1]);
            svs_pw = float(img.split('_')[2]);
            png_pw = float(img.split('_')[3].split('.png')[0]);
        except:
            print('error reading image')

            continue;

        png = np.array(Image.open(full_img).convert('RGB'))
#        print(png.shape)

        if (whiteness(png) > 12):
            a = png
            a = cv2.cvtColor(a, cv2.COLOR_RGB2YUV)
            a = Image.fromarray(a.astype(np.uint8))
            a = data_aug(a)
            X[xind, :, :, :] = a
            inds[xind] = rind
            xind += 1
        coor[cind, 0] = np.int32(x_off)
        coor[cind, 1] = np.int32(y_off)
        cind += 1
        rind += 1
#    print (X.size())    
    return full_img[lind:], X, inds, coor, rind

def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model

def confusion_matrix(Or, Tr, thres):
    tpos = np.sum((Or>=thres) * (Tr==1));
    tneg = np.sum((Or< thres) * (Tr==0));
    fpos = np.sum((Or>=thres) * (Tr==0));
    fneg = np.sum((Or< thres) * (Tr==1));

    
    return tpos, tneg, fpos, fneg;

def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.to(device)
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=[5,6,7])
        cudnn.benchmark = True
        return model
def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;

model = 'models_cnn/HASHI_0mean_1std_YUV_none_1220_0027_0.8884745788940035_1.t7'

checkpoint = torch.load(model, map_location=lambda storage, loc: storage)
model = checkpoint['model']
model = unparallelize_model(model)
#model = parallelize_model(model)
model.to(device)
#torch.backends.cudnn.benchmark=True
model.eval()

def val_fn_epoch_on_disk(classn, model, TileFolder):

    img_list = os.listdir(TileFolder)
    rind = 0
    full_img, X, inds, coor, rind = load_data(img_list, rind, TileFolder)

    all_or = np.zeros(shape=(len(img_list), classn), dtype=np.float32)
    all_inds = np.zeros(shape=(len(img_list),), dtype=np.int32)
    all_coor = np.zeros(shape=(len(img_list),2),dtype=np.int32)


    with torch.no_grad():
        X = Variable(X.to(device))
        output =model(X)
    output = output.data.cpu().numpy()
    output = softmax_np(output)[:, 1]
 
    return output, inds, coor

def get_softmax_probs(heat_map_out, TileFolder):
    Or, inds, coor = val_fn_epoch_on_disk(1, model, TileFolder);
#Or_all = np.zeros(shape=(coor.shape[0],), dtype=np.float32)
#Or_all[inds] = Or[:, 0]
 #   print('len of all coor: ', coor.shape)
 #   print('shape of Or: ', Or.shape)
 #   print('shape of inds: ', inds.shape)

    fid = open( heat_map_out, 'w+');
    for idx in range(0, Or.shape[0]):
        fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or[idx]))

    fid.close();

#print('Elapsed Time: ', (time.time() - start)/60.0)
    print('DONE!');

#k = get_softmax_probs()

def interpolate2d(slide_name, output_tile_folder, output_np_array):

    patch_size_40X = 101;

    try:
        oslide = openslide.OpenSlide(slide_name);
    #    mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
           mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        elif "XResolution" in oslide.properties:
           mag = 10.0 / float(oslide.properties["XResolution"]);
        else:
           mag = 10.0 / float(0.254);
        pw = int(patch_size_40X * mag / 40);
        width = oslide.dimensions[0];
        height = oslide.dimensions[1];
    except:
        print '{}: exception caught'.format(slide_name);
        exit(1);


    hash_f = lambda x: int(np.ceil(((x-101.)/101.) + 1))

    hash_track = lambda x: int(np.ceil(((x-101.)/101.)))

    hash_inv = lambda x: ((x-1) * 101) + 101

    wsi_array_reduced = [[0.0]*hash_f(height)]*hash_f(width)


    lines = [line.rstrip('\n') for line in open(output_tile_folder)]

    k = []

    f = lambda l: [int(l[0]), int(l[1]), float(l[2])]
    for l in lines:
        ll = l.split(" ")
        ll = f(ll)
        k.append(ll)

    extracted_softmax_probs = np.array(k)
#    print extracted_softmax_probs    
    for i in range(extracted_softmax_probs.shape[0]):

 	wsi_array_reduced[hash_f(extracted_softmax_probs[i,0])][hash_f(extracted_softmax_probs[i,1])]= extracted_softmax_probs[i,2]

    print wsi_array_reduced[0][0]
    wsi_array_reduced = np.array(wsi_array_reduced, dtype=np.float32)
    
    wsi_idx_x, wsi_idx_y = np.arange(wsi_array_reduced.shape[0]), np.arange(wsi_array_reduced.shape[1])
    wsi_array_non_zero_idx = np.nonzero(wsi_array_reduced)
    wsi_array_non_zero_idx_T = np.transpose(wsi_array_non_zero_idx)

    x_idx, y_idx = wsi_array_non_zero_idx_T[:,0], wsi_array_non_zero_idx_T[:,1]

    z_idx = wsi_array_reduced[wsi_array_non_zero_idx]

    z = np.zeros((x_idx.shape[0], y_idx.shape[0]))


    zero_idx_x, zero_idx_y = np.setdiff1d(wsi_idx_x, x_idx), np.setdiff1d(wsi_idx_y, y_idx)


    wsi_array_reduced_masked = np.ma.masked_values(wsi_array_reduced, 0)

    xx,yy = np.meshgrid(wsi_idx_y, wsi_idx_x)
    print (xx.shape, yy.shape)
    x1, y1 = xx[~wsi_array_reduced_masked.mask], yy[~wsi_array_reduced_masked.mask]

    z1 = wsi_array_reduced_masked[~wsi_array_reduced_masked.mask]

    GD1 = interpolate.griddata((x1, y1), z1.ravel(), (xx, yy), method='cubic', fill_value=0)


    gradient_prob_map = np.gradient(GD1)


    mag_gradient_prob_map = np.sqrt(np.add(np.square(gradient_prob_map[0]), np.square(gradient_prob_map[1])))
    c = 0
    mag_gradient_prob_map_flat = np.ravel(mag_gradient_prob_map)

    res_idx = [list(elem) for elem in top_n_indices(mag_gradient_prob_map, 100)]

    res_idx = np.array(res_idx)

    res_idx_big = hash_inv(res_idx)
    np.save(ouput_np_array, GD1)
    return res_idx_big, mag_gradient_prob_map

def hashi_wsi_iter(wsi_dir_path, output_tile_path_init, heat_map_result, T):

    wsi_img_list = os.listdir(wsi_dir_path)
    fl = [f for f in os.listdir(wsi_dir_path) if os.path.isfile(os.path.join(, i_dir_path)) and f.endswith('.svs')]
    wsi_img_list = fl
    for wsi_img in wsi_img_list:
	slide_name = wsi_dir_path + '/' + wsi_img
	if not os.path.exists(output_tile_path_init):
	    os.mkdir(output_tile_path_init)
	stage1 = extract_tiles(slide_name, output_tile_path_init, True)
	print "First tiling of Image done"
	pred_iter = hashi_pred_iter(slide_name, output_tile_path_init, heat_map_result, T)
	print os.listdir(output_tile_path_init)
	sh.rmtree(output_tile_path_init)	
def hashi_pred_iter(slide_name, output_tile_path_iter, heat_map_result, T):
    c = 0
    
    for i in range(T):
        print "Pred Iter counter:", c
        if not os.path.exists(output_tile_path_iter):
            #os.mkdir(output_tile_path_iter)
            #print os.path.exists(output_tile_path_iter)
            print "No initial tiles"
             
        stage2 = get_softmax_probs(heat_map_result, output_tile_path_iter)
	print "Softmax Done"
	stage3 = interpolate2d(slide_name, heat_map_result, slide_name)
	print "Interpolation Done"
        sh.rmtree(output_tile_path_iter)
        print os.path.exists(output_tile_path_iter)

        os.mkdir(output_tile_path_iter)
        stage4 = extract_tiles(slide_name, output_tile_path_iter, False, idx_array = stage3[0])
	print "Second stage sampling done"
        print os.path.exists(output_tile_path_iter)
	c += 1
 

#k = interpolate2d("input_svs/TCGA-A1-A0SD-01Z-00-DX1.svs", "heat_map_res.txt", "TCGA-A1-A0SD-01Z-00-DX1.svs")


#kkk = hashi_wsi_iter("input_WSI_all/tcga_KE", "output_trial", "heat_map_res.txt", 100)
#print kkk

wsi_path = sys.argv[1]
output_tile_path = sys.argv[2]
softmax_text_file = sys.argv[3]

hashi_loop = hashi_wsi_iter("input_WSI_all/tcga_KE", "output_trial", "heat_map_res.txt", 100)


