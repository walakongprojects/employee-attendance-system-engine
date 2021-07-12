from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
from datetime import datetime, timedelta
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pymongo

# pre trained model
modeldir = './model/20170511-185253.pb'
# trained classifier
classifier_filename = './class/classifier.pkl'
# npy files
npy='./npy'
# get all the names of trained images
train_img="./train_img"

# Setup database
# dev -------
# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# mydb = myclient["emp-attendance-dev"]
# prod -------
myclient = pymongo.MongoClient("mongodb+srv://niconiconi:niconiconi@cluster0.qskbw.mongodb.net/attendace-prod?retryWrites=true&w=majority")
mydb = myclient["attendace-prod"]
col_employees = mydb["employees"]
col_attendance = mydb["attendances"]

with tf.Graph().as_default():
    
    # setup for gpu
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

    # setup for cpu
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        
        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        print(embeddings)
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')

        video_capture = cv2.VideoCapture(0)
        # video_capture = cv2.VideoCapture(-1)
        c = 0

        ## custom
        num_detected = 0
        name_detected = "Sample name"

        print('Start Recognition')
        prevTime = 0
        checker = True
        img_counter = 0
        while checker:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)    #resize frame (optional)

            curTime = time.time()+1    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)
                    ## mode 3
                    
                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue
                        if(i>len(cropped)):
                            print('Running')
                            break
                        else:
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            # print(predictions)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            # print("predictions")
                            # print(best_class_indices,' with accuracy ',best_class_probabilities)

                            
                            img_counter += 1
                            print(best_class_probabilities)
                            if best_class_probabilities>0.70:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                #plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                
                                # print('Result Indices: ', best_class_indices[0])
                                # print(HumanNames)
                                print(HumanNames[best_class_indices[0]])

                                # custom
                                if num_detected <= 4:
                                    if name_detected == HumanNames[best_class_indices[0]]:
                                        num_detected+=1
                                    else:
                                        name_detected = HumanNames[best_class_indices[0]]
                                        num_detected = 0
                                else:
                                    start_date = datetime.today()
                                    end_date = datetime.today() + timedelta(days=1)
                                    start_date = start_date.replace(hour=0, minute=0, second=0)
                                    print(start_date)
                                    print(end_date)
                                    total_documents = col_attendance.count_documents({"name": HumanNames[best_class_indices[0]], 'date': {'$lt': end_date, '$gte': start_date}})
                                    print(total_documents)
                                    if total_documents == 0:
                                        curr_date = datetime.now()
                                        attendance = {}

                                        for x in col_employees.find({"name": HumanNames[best_class_indices[0]]}):
                                            attendance = {
                                                "employeeName": HumanNames[best_class_indices[0]],
                                                "timeIn": curr_date,
                                                # "age": x["age"],
                                                # "phone_number": x["phone_number"],
                                                # "employee_number": x["employee_number"],
                                                # "date_string": curr_date.strftime("%m/%d/%Y, %H:%M:%S")
                                            }

                                        # myquery = { "name": HumanNames[best_class_indices[0]] }
                                        # newvalues = { "$set": { "is_present": 1 }}
                                        # x = mycol2.update_one(myquery, newvalues)

                                        x = col_attendance.insert_one(attendance)
                                        print(HumanNames[best_class_indices[0]], " is sent to database.")
                                        print("inserted on database")
                                        attendance = {}
                                        num_detected = 0
                                        name_detected = "sample name"
                                    else:
                                        print("Employee already login today")
                                    
                                    # db.employees.insert({"name": "John Nico Austria", "age": 18, "employee_number": "001", "phone_number": "09999999999"})
                                    # db.employees.insert({"name": "sssss Cueto", "age": 20, "employee_number": "006", "phone_number": "09321536313"})
                                    # db.employees.insert({"name": "Ernest Nicole Penales", "age": 21, "employee_number": "003", "phone_number": "099072918293"})
                                    # db.employees.insert({"name": "Sammuel Tumanguil", "age": 22, "employee_number": "004", "phone_number": "09339992199"})

                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        # result_names = HumanNames[best_class_indices[0]]
                                        result_names = "{}: {}".format(HumanNames[best_class_indices[0]],best_class_probabilities)
                                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    .7, (0, 0, 255), thickness=1, lineType=2)
                            else:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                                #plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    .7, (255, 0, 0), thickness=1, lineType=2)
                                
                else:
                    print('Alignment Failure')
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
