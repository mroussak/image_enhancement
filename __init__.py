from flask import Flask, render_template, request, redirect, flash, send_file, send_from_directory
from zipfile import ZipFile
import shutil
import math
import os
import cv2
import traceback
import numpy as np

__author__= 'Maxim R.'

app = Flask(__name__)
app.debug = True

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    masked = np.ma.array(matrix, mask=low_mask, fill_value=low_value)
    high_mask = matrix > high_value
    masked = np.ma.array(matrix, mask=high_mask, fill_value=high_value)
    return masked.filled()


def apply_white_balance(channel, half_percent):
    # find the low and high precentile values (based on the input percentile)
    height, width = channel.shape
    vec_size = width * height
    flat = channel.reshape(vec_size)
    flat = np.sort(flat)
    n_cols = flat.shape[0]
    low_val = flat[int(math.floor(n_cols * half_percent))]
    high_val = flat[int(math.ceil(n_cols * (1.0 - half_percent)))]
    # saturate below the low percentile and above the high percentile
    thresholded = apply_threshold(channel, low_val, high_val)
    # scale the channel
    normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
    return normalized

def exposure_fusion(exposures):
    ##Mertens##
    # Align input images
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(exposures, exposures)
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(exposures)
    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    return res_mertens_8bit

def rename_inputs(input_dir):
    exposures = []
    for idx, img in enumerate(os.listdir(input_dir)):
        exposures.append(os.path.join(input_dir,img))
    exposures.sort()
    print('exposures original filename')
    print(exposures)
    pic_counter = 1
    ev_counter = 1
    for exp in exposures:
        print('processing :',exp)
        os.rename(exp,os.path.join(input_dir,'{}-{}.JPG'.format(str(pic_counter).zfill(3),ev_counter)))
        if ev_counter%3==0:
            ev_counter=1
            pic_counter+=1
        else:
            ev_counter+=1

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/process_images", methods=['GET', 'POST'])
def process_images():
    if request.method == 'POST':
        try:
            zip_file = request.files['file']
            print(zip_file.filename)
            if '.zip' not in zip_file.filename:
                flash('Please upload a zip file')
                return redirect(request.url)
            # Open zip file uploaded by user
            input_dir = '/output/input_images/'
            os.makedirs(input_dir, exist_ok=True)
            with ZipFile(zip_file, 'r') as zipObj:
                for zip_info in zipObj.infolist():
                    if (zip_info.filename[-1] == '/') or 'MAC' in (zip_info.filename) or 'DS_Store' in (zip_info.filename):
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    if ('.JPG' in zip_info.filename) or ('.jpg' in zip_info.filename):
                        zipObj.extract(zip_info, input_dir)
            if len(os.listdir(input_dir))==0:
                flash('no jpegs have been detected in the zip file')
                return redirect(request.url)
            rename_inputs(input_dir)
            output_dir = '/output/enhanced_images'
            os.makedirs(output_dir,exist_ok=True)
            input_exposures = [input for input in os.listdir(input_dir) if '-' in input]
            merged_exposures = np.unique([input_img.split('-')[0] for input_img in input_exposures])
            image_index = 0
            paths_to_outputs = []
            for idx, merged_exposure in enumerate(merged_exposures):
                #Load exposures for a single image
                exposures = [cv2.imread(os.path.join(input_dir, img)) for img in input_exposures if merged_exposure in img]
                exposure_paths = [os.path.join(input_dir, img) for img in input_exposures if merged_exposure in img]
                print('exposure_paths')
                print(exposure_paths)
                print('image_shapes')
                print([i.shape for i in exposures])
                #fuse exposures
                fused_exposures = exposure_fusion(exposures)
                half_percent = 7.5 / 200.0
                #split channels into r,g and b
                channels = cv2.split(fused_exposures)
                out_channels = []
                #Apply color balancing to each channel
                for channel in channels:
                    normalized = apply_white_balance(channel, half_percent)
                    out_channels.append(normalized)
                #Merge Result
                white_balanced_img = cv2.merge(out_channels)
                # Converting image to LAB Color model
                lab = cv2.cvtColor(white_balanced_img, cv2.COLOR_BGR2LAB)
                # Splitting the LAB image to different channels
                l, a, b = cv2.split(lab)
                # Applying CLAHE to L-channel (For contrast enhancement)
                clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
                lcl = clahe.apply(l)
                # Merge the CLAHE enhanced L-channel with the a and b channel
                limg = cv2.merge((lcl, a, b))
                # Converting image from LAB Color model to RGB model
                enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                output_path = '/output/enhanced_images/{}-output.jpg'.format(idx)
                cv2.imwrite(output_path, enhanced_image)
                paths_to_outputs.append(output_path)
                image_index+=1
                if image_index==len(merged_exposures):
                    break
            # writing files to a zipfile
            with ZipFile('/output/enhanced_images.zip', 'w') as zip:
                for path_to_output in paths_to_outputs:
                    zip.write(path_to_output)
            # os.chmod('/output/enhanced_images.zip', 0o777)
            shutil.rmtree(input_dir)
            shutil.rmtree(output_dir)
            return send_from_directory('/output', filename='enhanced_images.zip')
        except Exception as e:
            print('error', e)
            print(traceback.print_exc())
            render_template('index.html')
    else:
        return render_template('index.html')

#
# if __name__ == "__main__":
#     app.secret_key = 'image_editing_flask-app-secret-key'
#     app.run()
