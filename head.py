import time
import cv2
import dlib
import fdetect

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '../dlibcascades/shape_predictor_68_face_landmarks.dat')


def face_pose(video_capture, facecascade):
    video = fdetect.video_read(480, 640)
    prev_time = 0
    new_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # --- NEW CODE START ---
        if not ret: 
            break
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5) 
        # --- NEW CODE END ---

        if ret:
            dets = detector(frame, 1)
            for k, d in enumerate(dets):
                # Get the landmarks/parts for the face in box d.
                shape = predictor(frame, d)
                mid_x = [(shape.part(1).x+shape.part(15).x)/2,
                         (shape.part(1).y+shape.part(15).y)/2]
                mid_y = [(shape.part(27).x+shape.part(66).x)/2,
                         (shape.part(27).y+shape.part(66).y)/2]
                nose = [shape.part(30).x, shape.part(30).y]
                final_x = 5*nose[0]-4*mid_x[0]
                final_y = 5*nose[1]-4*mid_y[1]
                print(f"Nose X: {nose[0]} | Gaze X: {int(final_x)}") # print nose,final_x
                cv2.circle(frame, (int(final_x), int(final_y)), 2, (0, 0, 255))
                cv2.circle(frame, (int(nose[0]), int(nose[1])), 2, (0, 0, 255))
                cv2.line(frame, (int(nose[0]), int(nose[1])), (int(final_x), int(final_y)), (255, 0, 0), 3)
        # --- FPS COUNTER ---
                new_time = time.time()
                fps = 1 / (new_time - prev_time)
                prev_time = new_time
                fps_text = "FPS: " + str(int(fps))
                cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # -------------------
            # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release video capture
    video.release()
    cv2.destroyAllWindows()

