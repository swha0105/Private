import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
#from scipy.misc import imresize
import re, os
import numpy as np
from read_roi import *
import zipfile
import pydicom
import nibabel as nib
from tqdm import *
import pylab 

#input: outline points 
#output: ground truth image
def get_area(pts,img_shape):
    img = Image.new('L',img_shape)
    ImageDraw.Draw(img).polygon(pts,outline=255,fill=255)
    return np.asarray(img)


def get_img_size(dicom_path,filename):
    
    #roi를 얻은 이미지가 ADC 인지 T2 인지 파악
    sequence = re.compile('adc|ADC|T2').findall(filename)[0]
    sequence = 'ADC' if sequence == 'adc' else sequence

    #roi파일에 대응되는 환자의 dicom 파일을 열어서 이미지 사이즈를 얻음     
    image_path = os.path.join(dicom_path,sequence)
    dc = dicom.read_file(os.path.join(image_path,os.listdir(image_path)[0]))
    img_shape = (dc.Rows,dc.Columns)
    
    return img_shape

def get_roi_arr(roi_folder_path,filename):
    #zip파일에서 roi 파일을 얻음
    stories_zip = zipfile.ZipFile(os.path.join(roi_folder_path,filename))
    roi_arr = []
    for file in stories_zip.namelist():
        #파일이 너무 클경우 roi가 아니거나 문제가 있는경우 이므로 예외
        if stories_zip.getinfo(file).file_size < 2**20:
            roi_path = (stories_zip.extract(file))
            roi_arr.append(read_roi_file(roi_path))
            os.remove(roi_path)
    stories_zip.close()
    
    return roi_arr

#데이터들이 포함된 path와 roi를 가지고 있는 zip 파일의 이름을 입력값으로 받음
def roizip2fig(dicom_path,roi_folder_path,roi_filename,isnii=False):
    fig = []
    position = []
    if isnii:
        img_shape = (512,512)
    else:
        img_shape = get_img_size(dicom_path,roi_filename)
    if os.path.splitext(roi_filename)[1] == '.zip':
        roi_arr = get_roi_arr(roi_folder_path,roi_filename)
    if os.path.splitext(roi_filename)[1] == '.roi':
        roi_arr = []
        roi_arr.append(read_roi_file(os.path.join(roi_folder_path,roi_filename)))        
    #roi를 이미지로 변환하고 사이즈를 512 512로 바꿈
    for roi in roi_arr:
        name = list((roi).keys())[0]
        x = roi[name]['x']
        y = roi[name]['y']
        area = get_area([tuple([p[0],p[1]]) for p in np.array([x,y]).T],img_shape)
        if img_shape != (512,512):
            im = Image.fromarray(np.uint8(area))
            area = (np.array(im.resize((512,512))))
        print(roi_filename[:-4],roi[name]['position'])
        position.append(roi[name]['position'])
        fig.append(area)
    #결과값은 roi 이미지와 slice number
    return fig,position

def get_int(data, base):
    b0 = data[base]
    b1 = data[base + 1]
    b2 = data[base + 2]
    b3 = data[base + 3]
    n = ((b0 << 24) + (b1 << 16) + (b2 << 8) + b3)
    return n

def roizip26to20(dicom_path, roi26_folder_path,roi20_folder_path, roi_filename):
    fig = []
    position = []
    img_shape = get_img_size(dicom_path,roi_filename)
    print(img_shape)
    stories_zip = zipfile.ZipFile(os.path.join(roi26_folder_path,roi_filename))
    roi_filename2 = roi_filename[:-4]+'_20slice.zip'
    roi_arr = []
    new_name = []
    for file in stories_zip.namelist():
        if stories_zip.getinfo(file).file_size < 1024*1024:
            roi_path = (stories_zip.extract(file))
            roi_dict = read_roi_file(roi_path)
            with open(roi_path, "r+b") as f:
                f.seek(56)
                f.write((int(roi_dict[file[:-4]]['position'])-3).to_bytes(4, byteorder='big'))
                f.close()
            parentdir = os.path.abspath(os.path.join(roi_path, os.pardir))
            file2 = '%04d'%(int(file[:4])-3) + file[4:]
            os.rename(roi_path,os.path.join(parentdir,file2))
            new_name.append(os.path.join(parentdir,file2))
    slice20_zip = zipfile.ZipFile(os.path.join(roi20_folder_path,roi_filename2), 'w')
    for roi20_name in new_name:
        slice20_zip.write(os.path.basename(roi20_name), compress_type = zipfile.ZIP_DEFLATED)
        os.remove(roi20_name)
    slice20_zip.close()

#nii 파일 데이터를 확인하는 함수 환자별로 받은 파일의 이미지 크기,갯수등이 올바른지 확인
#오류가없는 환자 리스트를 출력함
def nii_data_check(nii_path,nii_file_names):
    list_patient = []
    cnt = 0
    for name in os.listdir(nii_path):
        nii_folder = os.path.join(nii_path,name)
        nii_img_data = []
        file_flag = True
        size_flag = True
        #데이터를 불러올수 없는경우 -> file error
        for fname in nii_file_names:
            try:
                nii_img_data.append(nib.load(os.path.join(nii_folder,fname)).get_data())
            except:
                print('Error in %s (file:%s)'%(name,fname))
                file_flag = False
                break
        #이미지 크기,갯수가 안 맞는 경우 -> size error
        for i in range(len(nii_img_data)):
            if np.shape(nii_img_data[i]) != (512, 512, 20, 1):
                print(np.shape(nii_img_data[i]),nii_file_names[i])
                size_flag = False
        if not size_flag:
            print('Error in %s (size)'%name)
        if file_flag and size_flag:
            list_patient.append(name)
            cnt +=1
    print(cnt)
    return list_patient,cnt

#환자 리스트를 입력값으로 넣었을때 대응하는 ADC,T2의 dicom파일이 있는지, slice가 26장인지 확인함
def dicom_data_check(patient_list,prostate_dicom_path):
    # ADC T2 중 26장이 아닌게 있는 경우
    list_slice_error = []
    
    # dicom 파일중에 일부가 없거나 폴더명이 다른경우
    list_dicom_error = []
    
    # ADC와 T2 dicom 파일이 둘다 있고 slice 갯수가 각각 26개 있는 경우
    list_dicom_exist_26 = []
    
    #문제가 있는 경우 원인을 출력함
    for patient in patient_list:
        patient_path = os.path.join(prostate_dicom_path,patient)
        ADC_flag = True
        T2_flag = True
        file_flag = True
        try:
            if len(os.listdir(os.path.join(patient_path ,'ADC'))) != 26:
                print(patient,' len ADC : %d'%len(os.listdir(os.path.join(patient_path ,'ADC'))))
                ADC_flag = False
            if len(os.listdir(os.path.join(patient_path ,'T2'))) != 26:
                print(patient,' len T2 : %d'%len(os.listdir(os.path.join(patient_path ,'T2'))))
                T2_flag = False
        except:
            print('file error %s'%patient)
            file_flag = False
        
        if ADC_flag and T2_flag:
            pass
        else:
            list_slice_error.append(patient)

        if ADC_flag and T2_flag and file_flag:
            list_dicom_exist_26.append(patient)
        else:
            list_dicom_error.append(patient)

    return list_slice_error,list_dicom_error,list_dicom_exist_26

#dicom_roi_path : dicom 에서 만들어진 roi
#nii_list : nii file이 존재하는 환자 번호 목록
def get_dicom_roifig(dicom_roi_path,nii_list):
    roifig = []
    for dicom_file in os.listdir(prostate_dicom_path):
        dicom_num = re.compile('\d{1,4}').findall(dicom_file)
        dicom_num = dicom_num[0] if len(dicom_num) != 0 else -1 
        for roi_file in os.listdir(dicom_roi_path):
            roi_num = re.compile('\d{1,4}').findall(roi_file)
            roi_num = roi_num[0] if len(roi_num) != 0 else -2 
            if int(dicom_num) == int(roi_num):
                try:
                    #roi를 얻은 이미지가 ADC 인지 T2 인지 파악
                    sequence = re.compile('adc|ADC|T2').findall(roi_file)[0]
                    sequence = 'ADC' if sequence == 'adc' else sequence
                    dicom_list = os.listdir(os.path.join(*[prostate_dicom_path,dicom_file,sequence]))
                    print('len %s : %d'%(dicom_file,len(dicom_list)))
                    pos_diff = (len(dicom_list)-20)//2
                    fig,pos = roizip2fig(os.path.join(prostate_dicom_path,dicom_file),dicom_roi_path,roi_file,isnii=False)
                    roifig.append((dicom_num,fig,np.array(pos)-pos_diff))
                except NameError:
                    print('========error: ',dicom_file)
    return roifig

#nii를 보고 그려진 ROI를 그림으로 변환한다
def get_nii_roifig(nii_roi_path):
    nii_roifig = []
    for roi_file in os.listdir(nii_roi_path):
        if roi_file[-4:] == '.zip':
            roi_num = re.compile('\d{1,4}').findall(roi_file)
            roi_num = int(roi_num[0]) if len(roi_num) != 0 else -2
            fig,pos = roizip2fig('',nii_roi_path,roi_file,isnii=True)
            nii_roifig.append(('%04d'%roi_num,fig,pos))
        if roi_file[-4:] == '.roi':
            roi_num = re.compile('\d{1,4}').findall(roi_file)
            roi_num = int(roi_num[0]) if len(roi_num) != 0 else -2
            fig,pos = roizip2fig('',nii_roi_path,roi_file,isnii=True)
            nii_roifig.append(('%04d'%roi_num,fig,pos))
    return nii_roifig

#하나의 이미지에 그려진 여러개의 roi를 한 이미지로 다시 합침
def save_roi_sum_fig(nii_roifig,roi_npy_path):
    nii_roi_list = np.sort(np.array(list(set([roi[0] for roi in nii_roifig]))))
    nii_roi_result = []
    for roi_fig_idx in tqdm(nii_roi_list):
        temp_figs = [roi for roi in nii_roifig if roi_fig_idx == roi[0]]
        roi_dict = {}
        temp_fig_list = []
        temp_pos_list = [] 
        try:
            patient_pos_set = list(set().union(*[set(figs[2]) for figs in temp_figs]))
            patient_fig_list = np.concatenate([figs[1] for figs in temp_figs])
            patient_pos_list = np.concatenate([figs[2] for figs in temp_figs])
        except:
            print('error in roi_fig_idx: ',roi_fig_idx)
        for j in patient_pos_set:
            temp_roi_fig = np.zeros(np.shape(temp_figs[0][1][0]))
            for fig,k in zip(patient_fig_list,patient_pos_list):
                if j == k:
                    temp_roi_fig = np.vectorize(max)(temp_roi_fig,fig)
            temp_fig_list.append(temp_roi_fig.astype('uint8'))
            temp_pos_list.append(j)
        roi_dict['patient_number'] = int(roi_fig_idx)
        roi_dict['cancer_roi'] = temp_fig_list
        roi_dict['slice_index'] = temp_pos_list
        np.save(os.path.join(roi_npy_path,'%04d_ROI.npy'%roi_dict['patient_number']),roi_dict)
        
def mri_with_roi(mri,roi):
    w, h = mri.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    mri = mri/np.max(mri)*255
    ret[:, :, 0] = np.vectorize(max)(mri,roi).astype('uint8')
    ret[:, :, 1] = mri.astype('uint8')
    ret[:, :, 2] = mri.astype('uint8')
    return ret



#     fig = pyplot.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(x,y)
#     ax.legend(legendStrings, loc = 'best')
#     fig.savefig('himom.png')
#     #new bit here
#     pylab.close(fig) #where f is the figure

def save_mri_roi_plot(nii_path,nii_list,roi_npy_path,output_folder,is_save=True):
    no_roi_list = []
    for patient_filename in tqdm(nii_list):
        patient_number = re.findall('\d{1,4}',patient_filename)
        patient_number = int(patient_number[0]) if len(patient_number)>0 else -1    
        t2_nii_data = nib.load(os.path.join(*[nii_path,patient_filename,'T2.nii'])).get_data()
        #adc_nii_data = nib.load(os.path.join(*[nii_path,patient_filename,'ADC.nii'])).get_data()
        try:
            roi_data = np.load(os.path.join(roi_npy_path,'%04d_ROI.npy'%patient_number)).item()
        except FileNotFoundError:
            print('FileNotFoundError')
            no_roi_list.append(patient_filename)
            continue
        for pos,fig in zip(roi_data['slice_index'],roi_data['cancer_roi']):
            figure = plt.figure(figsize=[10,5])
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace = 0.01,hspace = 0.01)
            figure.add_subplot(121).imshow(t2_nii_data[:,:,pos-1,0].T,cmap='gray')
            plt.title('%s (slice:%02d)  T2 '%(patient_filename,pos))
            plt.axis('off')
            figure.add_subplot(122).imshow(mri_with_roi(t2_nii_data[:,:,pos-1,0].T,fig))
            plt.title('%s (slice:%02d)  T2 & ROI '%(patient_filename,pos))
            plt.axis('off')
            '''
            plt.subplot(223).imshow(adc_nii_data[:,:,pos-1,0].T,cmap='gray')
            plt.title('%s (slice:%02d)  ADC '%(patient_filename,pos))
            plt.axis('off')
            plt.subplot(224).imshow(mri_with_roi(adc_nii_data[:,:,pos-1,0].T,fig))
            plt.title('%s (slice:%02d)  ADC & roi ROI '%(patient_filename,pos))
            plt.axis('off')
            '''
            #plt.subplots_adjust(wspace=0, hspace=0.1)
            if is_save:
                plt.savefig(os.path.join(output_folder,'%04d_%02d'%(patient_number,pos)))
            else:
                plt.show()
            pylab.close(figure)
            
            
def nii_ecdf_plot(nii_image):
    #nii data 확인 및 확률값으로 변환해서 시각화
    from statsmodels.distributions.empirical_distribution import ECDF
    for i in range(8):
        ecdf = ECDF(np.reshape(nii_image[i][:,:,10,0],-1))  
        plt.figure(figsize=[20,10])
        plt.subplot(121).imshow(np.flip((nii_image[i][:,:,10,0]).T,axis=1),cmap='gray')
        plt.axis('off')
        plt.subplot(122).imshow(np.flip(ecdf(nii_image[i][:,:,10,0]).T,axis=1),cmap='gray')
        plt.axis('off')
        plt.show()