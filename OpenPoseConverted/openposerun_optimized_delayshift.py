import cv2
import time
import numpy as np
import argparse
from os.path import exists
from os import system
from model_processor_optimized import ModelProcessor
from acl_resource import AclResource
from multiprocessing import Pool, Queue


def prerunner(queue_in, queue_mid1, modelFile, inWidth, inHeight):
    output_frame = 0
    max_frame_time = 0
    while True:
        time1 = time.time()
        frame = queue_in.get()
        if frame is not None:
            output_frame += 1
            preprocessed_img = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
            queue_mid1.put((preprocessed_img, output_frame, frame))
        else:
            queue_mid1.put((frame, output_frame, frame))
        time2 = time.time()
        max_frame_time = max(max_frame_time, time2-time1)
        print("Max prerunner time: ", max_frame_time)


def modelrunner(queue_mid1, queue_mid2, modelFile, inWidth, inHeight):
    output_frame = 0
    acl_resource = AclResource()
    acl_resource.init()
    model_parameters = {
        'model_dir': modelFile,
        'width': inWidth,  # model input width
        'height': inHeight,  # model input height
    }
    # perpare model instance: init (loading model from file to memory)
    # model_processor: preprocessing + model inference + postprocessing
    model_processor = ModelProcessor(acl_resource, model_parameters)
    max_frame_time = 0

    while True:
        time1 = time.time()
        frame, input_frame, orig_frame = queue_mid1.get()
        if frame is not None:
            output = model_processor.runmodel(frame)[0]
            output_frame += 1
            queue_mid2.put((output, output_frame, orig_frame))
        else:
            queue_mid2.put((frame, output_frame, orig_frame))
        # print("Model time: ", time2-time1)
        time2 = time.time()
        max_frame_time = max(max_frame_time, time2-time1)
        print("Max model time: ", max_frame_time)



def postrunner(queue_mid, queue_out, nPoints, frameWidth, frameHeight, threshold):
    output_frame = 0
    input_frame = 0
    max_frame_time = 0
    while True:
        time1 = time.time()
        output, input_frame, orig_frame = queue_mid.get()
        if output is not None:
            output_frame += 1
            time1 = time.time()
            points = []
            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold:
                    points.append((int(point[0]), int(point[1])))
                else:
                    points.append(None)
            queue_out.put((points, output_frame, orig_frame))
        else:
            queue_out.put((output, output_frame, orig_frame))
        time2 = time.time()
        max_frame_time = max(max_frame_time, time2-time1)
        # print("Postrunner: ", time2-time1)
        print("Max postrunner time: ", max_frame_time)


def main(): 
    parser = argparse.ArgumentParser(description='Specify Input File')
    parser.add_argument('--input', type=str, default='inputs/asl.mp4')

    inputArgs = parser.parse_args()

    input_frame = 0
    nPoints = 22
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
                  [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    threshold = 0.2

    input_source = inputArgs.input
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    aspect_ratio = frameWidth/frameHeight

    vid_name = input_source.split("/")[1].split(".")[0]
    # inHeight = 368
    inHeight = 250
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    modelFile = "model/openposehand" + str(inHeight) + str(inWidth) + ".om"
    # modelFile = "model/openposehands.om"
    if not exists(modelFile):
        print("Model doesn't exist for your video size, please wait while we create the offline model")
        system('./model/atc_create.sh model/pose_deploy.prototxt model/pose_iter_102000.caffemodel ' + str(inHeight) + ' ' + str(inWidth))
    vid_writer = cv2.VideoWriter('outputs/delayshift_' + vid_name + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                                 (frame.shape[1], frame.shape[0]))


    queue_in = Queue(maxsize=1)
    queue_mid1 = Queue(maxsize=1)
    queue_mid2 = Queue(maxsize=1)
    queue_out = Queue(maxsize=1)
    pool_pre = Pool(1, prerunner, (queue_in, queue_mid1, modelFile, inWidth, inHeight))
    pool_model = Pool(1, modelrunner, (queue_mid1, queue_mid2, modelFile, inWidth, inHeight))
    pool_post = Pool(1, postrunner, (queue_mid2, queue_out, nPoints, frameWidth, frameHeight, threshold))
    output_frame = 0
    start_time = time.time()
    max_frame_time = 0


    while True:
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame and queue_in.empty() and queue_mid1.empty() and queue_mid2.empty() and queue_out.empty() and input_frame == output_frame:
            cv2.waitKey()
            break
        elif hasFrame:
            time1 = time.time()
            input_frame += 1
            queue_in.put(frame)

        # net.setInput(inpBlob)
        # output2 = net.forward()
        time1 = time.time()
        try:
            points, output_frame, frame = queue_out.get(False)
        except:
            points = None

        # Empty list to store the detected keypoints
        if points is not None:
            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            vid_writer.write(frame)
            time2 = time.time()
            max_frame_time = max(max_frame_time, time2-time1)
            print(1 / (time2 - t))
    total_time = time.time() - start_time
    print("Number of frames: ", output_frame)
    print("Total time: ", total_time)
    print("Average framerate ", output_frame / total_time)
    vid_writer.release()


if __name__ == "__main__":
    main()

