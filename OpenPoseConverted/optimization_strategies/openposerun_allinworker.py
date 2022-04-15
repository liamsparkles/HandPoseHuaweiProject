import cv2
import time
import numpy as np
from model_processor import ModelProcessor
from acl_resource import AclResource
from multiprocessing import Pool, Queue


input_frame = 0
def worker(queue_in, queue_out):
    output_frame = 0
    acl_resource = AclResource()
    acl_resource.init()
    model_parameters = {
        'model_dir': modelFile,
        'width': inWidth,  # model input width
        'height': inHeight,  # model input height
        'frame_w': frameWidth,
        'frame_h': frameHeight,
    }
    # perpare model instance: init (loading model from file to memory)
    # model_processor: preprocessing + model inference + postprocessing
    model_processor = ModelProcessor(acl_resource, model_parameters)

    while True:
        time1 = time.time()
        frame = queue_in.get()
        if frame is not None:
            output = model_processor.predict(frame)
            output_frame += 1
            queue_out.put((output, output_frame))
            # print("Output frame: ", output_frame)
        else:
            queue_out.put((frame, output_frame))
        time2=time.time()
        print("Frame time: ", time2-time1)

protoFile = "model/pose_deploy.prototxt"
weightsFile = "model/pose_iter_102000.caffemodel"
modelFile = "model/openposehand2.om"
nPoints = 22
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
              [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

threshold = 0.2

input_source = "asl.mp4"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth/frameHeight

inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)

vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                             (frame.shape[1], frame.shape[0]))

# acl_resource = AclResource()
# acl_resource.init()
# model_parameters = {
#     'model_dir': modelFile,
#     'width': inWidth,  # model input width
#     'height': inHeight,  # model input height
# }
# perpare model instance: init (loading model from file to memory)
# model_processor: preprocessing + model inference + postprocessing
# model_processor = ModelProcessor(acl_resource, model_parameters)


queue_in = Queue(maxsize=1)
queue_out = Queue(maxsize=1)
pool = Pool(1, worker, (queue_in, queue_out))
output_frame = 0
start_time = time.time()

while True:
    t = time.time()
    hasFrame, frame = cap.read()
    # frameCopy = np.copy(frame)
    if not hasFrame and queue_in.empty() and queue_out.empty() and input_frame == output_frame:
        cv2.waitKey()
        break
    elif hasFrame:
        time1 = time.time()
        input_frame += 1
        queue_in.put(frame)
        # print("Frame add time ", time.time()-time1)

    # net.setInput(inpBlob)
    # output2 = net.forward()
    time1 = time.time()
    try:
        points, output_frame = queue_out.get(False)
    except:
        points = None
    # print("Wait output time: ", time.time()-time1)
    # output = model_processor.predict(frame)
    # print("time per frame: ", time2-t)
    # This output needs to be 4 dimensional, dim 0 is of size 1, dim 1 is of size nPoints, then 46 and 57
    # model_end = time.time()

    # print("model = {}".format(model_end - t))
    # Empty list to store the detected keypoints
    if points is not None:
        time1 = time.time()

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                # cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                # cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # print("Post processing = {}".format(time.time() - model_end))
        vid_writer.write(frame)
        time2 = time.time()
    # print("total = {}".format(time.time() - t))
        print("PP time: ", time2-time1)
        # print("Input frame: ", input_frame)
        # print("output frame: ", output_frame)
        # print("Frame rate: {}".format(1 / (time.time() - t)))

# print("total time: ", time.time() - start_time)
print("Average framerate ", output_frame / (time.time() - start_time))
vid_writer.release()
