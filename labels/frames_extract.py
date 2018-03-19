import cv2, numpy as np
from time import sleep
from os import listdir
from os.path import isfile, join
import csv
import hashlib
import os, errno

def flick(x):
    pass


def frames_extract(cap, result_dict):
    cv2.namedWindow('image')
    cv2.moveWindow('image',250,150)
    cv2.namedWindow('controls')
    cv2.moveWindow('controls',250,50)
    
    controls = np.zeros((50,950),np.uint8)
    cv2.putText(controls, "W/w: Play, S/s: Stay, A/a: Prev, D/d: Next, E/e: Fast, Q/q: Slow, Esc: Exit, [: Start selection, ]: End selection", (40,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    
    tots = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    i = 0
    cv2.createTrackbar('S','image', 0,int(tots)-1, flick)
    cv2.setTrackbarPos('S','image',0)
    
    cv2.createTrackbar('F','image', 1, 100, flick)
    frame_rate = 30
    cv2.setTrackbarPos('F','image',frame_rate)
    
    def process(im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    status = 'stay'
    frames_list = []
    annotation_num = 0
    
    while True:
        cv2.imshow("controls",controls)
        try:
            if i==tots-1:
                i=0
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, im = cap.read()
            if not ret:
                return result_dict
            r = 750.0 / im.shape[1]
            dim = (750, int(im.shape[0] * r))
            im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
            if im.shape[0]>600:
                im = cv2.resize(im, (500,500))
                controls = cv2.resize(controls, (im.shape[1],25))
    #        cv2.putText(im, status, )
            cv2.imshow('image', im)
            status = { ord('s'):'stay', ord('S'):'stay',
                        ord('w'):'play', ord('W'):'play',
                        ord('a'):'prev_frame', ord('A'):'prev_frame',
                        ord('d'):'next_frame', ord('D'):'next_frame',
                        ord('q'):'slow', ord('Q'):'slow',
                        ord('e'):'fast', ord('E'):'fast',
                        ord('c'):'snap', ord('C'):'snap',
                        ord('['):'frames_start', 
                        ord(']'):'frames_end', 
                        -1: status, 
                        27: 'exit'}[cv2.waitKey(10)]
        
            if status == 'play':
                frame_rate = cv2.getTrackbarPos('F','image')
#                sleep(1.0/(frame_rate+1))
                i+=1
                frames_list.append(i)
                cv2.setTrackbarPos('S','image',i)
                continue
            if status == 'stay':
                i = cv2.getTrackbarPos('S','image')
            if status == 'exit':
                break
            if status=='prev_frame':
                i-=1
                cv2.setTrackbarPos('S','image',i)
                status='stay'
            if status=='next_frame':
                i+=1
                cv2.setTrackbarPos('S','image',i)
                status='stay'
            if status=='slow':
                frame_rate = max(frame_rate - 5, 0)
                cv2.setTrackbarPos('F', 'image', frame_rate)
                status='play'
            if status=='fast':
                frame_rate = min(100,frame_rate+5)
                cv2.setTrackbarPos('F', 'image', frame_rate)
                status='play'
            if status=='snap':
                cv2.imwrite("./"+"Snap_"+str(i)+".jpg",im)
                print "Snap of Frame",i,"Taken!"
                status='stay'
                
            if status == 'frames_start':
                frames_list = []                
                status = 'play'
            if status == 'frames_end':
                result_dict[annotation_num] = frames_list
                annotation_num += 1
                print 'annotation: ', annotation_num, '\t', frames_list
                status = 'play'
                
                
    
        except KeyError:
            print "Invalid Key was pressed"
    cv2.destroyWindow('image')
    print result_dict
    return result_dict





def frames_export(cap, result_dict):
    frames_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),result_dict['label_type'])
    try:
        os.makedirs(frames_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for key, frames_list in result_dict.items():
        if str(key).isdigit(): 
            frames_group = hashlib.md5('-'.join(map(str, [result_dict['video_path'], result_dict['label_type']] + frames_list))).hexdigest()
            print frames_group            
            if frames_list != []:
                for index in range(len(frames_list)):
                    frame_num = frames_list[index]
                    print frame_num
                    cap.set(1, frame_num)
                    ret, frame = cap.read()
                    print ret
                    frame_path = os.path.join(frames_directory
                                                ,result_dict['label_type'] + '_' + frames_group + '_' + str(index) + '.jpg')
                    print frame_path
                    cv2.imwrite(frame_path, frame)
                    

    

if __name__ == '__main__':
    video_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'videos')
    csv_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'csv')

    video_files_list = [f for f in listdir(video_folder) if isfile(join(video_folder, f))]
    anno_type = raw_input("input forhand/backhand/serve/volley/move etc. : ")

    for index, video in enumerate(video_files_list):
        print video
        video_path = video_folder + '/' + video
        vidcap = cv2.VideoCapture(video_path)
        
        anno_file = '_'.join([anno_type,video]) + '.csv'
        anno_path = csv_folder + '/' + anno_file
        
        print '\n', 'selected frames will be stored in file: ', anno_path, '\n'
         
    #video = sys.argv[1] 
#        video = './Tennis - Women\'s Singles - 002 (World team tennis - Irvine NTRP 4.0) [720p].mp4'
#        cap = cv2.VideoCapture(video)
         
        result_dict = dict()
        result_dict['video_path'] = video_path
        result_dict['sllected_frames_file'] = anno_path
        result_dict['label_type'] = anno_type
        result = frames_extract(vidcap, result_dict)
        
        frames_export(vidcap, result)

        with open(anno_path, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in result_dict.items():
                writer.writerow([key, value])
        vidcap.release()
