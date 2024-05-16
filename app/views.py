import os
import cv2
import pafy
import math
import random
import string
import base64
import logging
import tempfile
import numpy as np
from ultralytics import YOLO
from django.views import View
from django.urls import reverse
from django.conf import settings
from django.core.files import File
from django.contrib import messages
from app.models import ProcessedVideo
from django.contrib.messages import get_messages
from django.core.files.temp import NamedTemporaryFile
from django.shortcuts import render,redirect,HttpResponseRedirect
from django.http import JsonResponse, FileResponse, StreamingHttpResponse, HttpResponseBadRequest, HttpResponseServerError



logger = logging.getLogger(__name__)
# Create your views here.
class YoloView(View):
  def get(self,request):
      
      return render(
        request,
        'app/home.html',
        {
            
        }
      )
  

class UploadVideoView(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the YOLOv8 model when the view is instantiated
        model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'best.pt')
        try:
            self.model = YOLO(model_path)
 
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # messages.error(request, f'Failed to load YOLOv8 model')
            
    def get(self, request):
        return render(request, 'app/upload_video.html')

    def post(self, request):
        if self.model is None:
            messages.error(request, 'Failed to load YOLOv8 model')
            return render(request, 'app/upload_video.html')

        video_file = request.FILES.get('video_file')
        if not video_file:
            messages.error(request, 'No video file provided')
            return render(request, 'app/upload_video.html')

        try:
            # Create a temporary file and write the uploaded file contents to it
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for chunk in video_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name

            # Open the video file using the temporary file path
            cap = cv2.VideoCapture(temp_file_path, cv2.CAP_FFMPEG)

        except Exception as e:
            logger.error(f"Failed to open video: {e}")
            messages.error(request, 'An error occurred during video processing.')
            return render(request, 'app/upload_video.html')
        # try:
        #     cap = cv2.VideoCapture(video_file.file.name, cv2.CAP_FFMPEG)
        #     # cap = cv2.VideoCapture(video_file.file)
        #     # cap = cv2.VideoCapture(video_file.temporary_file_path())
        # except Exception as e:
        #     logger.error(f"Failed to open video: {e}")
        #     messages.error(request, 'An error occurred during video processing.')
        #     return render(request, 'app/upload_video.html')
        # file_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        # file_name=f'{video.videoid}{random.randint(100000, 999999)}'

        # Generate a file name with the original name and a random extension
        original_filename = video_file.name
        file_name = f"{os.path.splitext(original_filename)[0]}_{random.randint(100000, 999999)}"
        print(file_name)
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', f'{file_name}.webm')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Use VP8 codec for WebM
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        

        def generate_frames():
            frame_count = 0
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for chunk in video_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name

            cap = cv2.VideoCapture(temp_file_path, cv2.CAP_FFMPEG)
            while True:
                ret, frame = cap.read()
                if not ret:
                    # logger.info("End of video stream")
                    messages.info(request, f'End of video stream')
                    break
                try:
                    results = self.model(frame)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                except Exception as e:
                    # logger.error(f"Failed to process frame {frame_count}: {e}")
                    messages.error(request, 'Some Error occurred while processing')
                    break
                # Encode the annotated frame as base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                yield (b'--frame\r\n'
                    b'Content-Type: text/plain\r\n\r\n' + frame_bytes.encode() + b'\r\n')
                frame_count += 1
            cap.release()
            out.release()
          
        # Generate and save thumbnail
        # cap = cv2.VideoCapture(video_file.temporary_file_path())
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                for chunk in video_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name

        # Open the video file using the temporary file path
        cap = cv2.VideoCapture(temp_file_path, cv2.CAP_FFMPEG)
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, encoded_img = cv2.imencode('.png', img)
            thumbnail = encoded_img.tobytes()

            thumbnail_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', 'thumbnails')
            os.makedirs(thumbnail_path, exist_ok=True)
            thumbnail_filename = f'{file_name}.png'
            thumbnail_file_path = os.path.join(thumbnail_path, thumbnail_filename)

            with open(thumbnail_file_path, 'wb') as thumbnail_file:
                thumbnail_file.write(thumbnail)
        cap.release()
        processed_video = ProcessedVideo.objects.create(
            video_id=file_name,
            title=os.path.splitext(original_filename)[0],
            processed_video=os.path.join('processed_videos', f'{file_name}.webm'),
            thumbnail_file=os.path.join('processed_videos', 'thumbnails', thumbnail_filename),
        )

        # Open the processed video file and save it to the ProcessedVideo object
        # with open(output_path, 'rb') as f:
        #     processed_video.processed_video.save(os.path.basename(output_path), File(f))

        response = StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
        response['Cache-Control'] = 'no-cache'  # Disable caching
        messages.success(request, 'Video processing completed successfully')
        # return render(request, 'app/upload_video.html', context)
        return response

    
class RealtimeVideoView(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the YOLOv8 model when the view is instantiated
        model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'best.pt')
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # messages.error(request, f'{e}')

    def get(self, request):
        return render(request, 'app/realtime_video.html')

    def post(self, request):
        # Start webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        def generate_frames():
            classNames =['bus', 'car', 'motorcycle', 'person', 'traffic', 'truck']
            while True:
                success, img = cap.read()
                if not success:
                    # logger.error("Failed to read frame from webcam")
                    messages.error(request, f'Failed to read frame from webcam')
                    break
                try:
                    results = self.model(img, stream=True)

                    for r in results:
                        boxes = r.boxes

                        for box in boxes:
                            # Bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                            # Confidence
                            confidence = math.ceil((box.conf[0] * 100)) / 100

                            # Class name
                            cls = int(box.cls[0])
                            class_name = classNames[cls]

                            # Display class name
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2
                            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', img)
                    frame_bytes = base64.b64encode(buffer).decode('utf-8')
                    yield (b'--frame\r\n'
                           b'Content-Type: text/plain\r\n\r\n' + frame_bytes.encode() + b'\r\n')

                except Exception as e:
                    # logger.error(f"Failed to process frame: {e}")
                    messages.error(request, f'Failed to process frame')
                    break

        response = StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
        response['Cache-Control'] = 'no-cache'  # Disable caching
        messages.success(request, 'Realtime Object Detection Intrupted! Closing Webcam.')
        return response

class VideoLibraryView(View):
  def get(self,request):
    processed_videos = ProcessedVideo.objects.all()

    context = {
        'processed_videos': processed_videos
    }
    return render(request, 'app/video_library.html', context)



class ProcessVideoView(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the YOLOv8 model when the view is instantiated
        model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'best.pt')
        try:
            self.model = YOLO(model_path)
 
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            # messages.error(request, f'Failed to load YOLOv8 model')
            
    def get(self, request):
        return render(request, 'app/process_video.html')

    def post(self, request):
        video_link = request.POST.get('video_link')
        if not video_link:
            # logger.error("No video link provided")
            messages.error(request, 'No video link provided')
            return HttpResponseRedirect(reverse('process_video')) 
        
        # Load the YOLOv8 model
        # model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'best.pt')
        # try:
        #     model = YOLO(model_path)
        # except Exception as e:
        #     logger.error(f"Failed to load YOLOv8 model: {e}")
        #     return HttpResponseServerError("Failed to load YOLOv8 model")

        # Process the video using the YOLOv8 model
        try:
            video = pafy.new(video_link)
            best = video.getbest()
            cap = cv2.VideoCapture(best.url)
        except Exception as e:
            # logger.error(f"Failed to open video: {e}")
            messages.error(request, 'An error occurred during video processing.')
            return HttpResponseRedirect(reverse('process_video'))  
            
        file_name=f'{video.videoid}{random.randint(100000, 999999)}'
        print(file_name)
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', f'{file_name}.webm')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Use VP8 codec for WebM
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        

        def generate_frames():
            frame_count = 0
            cap = cv2.VideoCapture(best.url)
            while True:
                ret, frame = cap.read()
                if not ret:
                    # logger.info("End of video stream")
                    messages.info(request, f'End of video stream')
                    break
                try:
                    results = self.model(frame)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                except Exception as e:
                    # logger.error(f"Failed to process frame {frame_count}: {e}")
                    messages.error(request, 'Some Error occured while processing')
                    break
                # Encode the annotated frame webmmkvas base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = base64.b64encode(buffer).decode('utf-8')
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n' + frame_bytes.encode() + b'\r\n')
                frame_count += 1
            cap.release()
            out.release()
          
        # Generate and save thumbnail
        cap = cv2.VideoCapture(best.url)
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, encoded_img = cv2.imencode('.png', img)
            thumbnail = encoded_img.tobytes()

            thumbnail_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', 'thumbnails')
            os.makedirs(thumbnail_path, exist_ok=True)
            thumbnail_filename = f'{file_name}.png'
            thumbnail_file_path = os.path.join(thumbnail_path, thumbnail_filename)

            with open(thumbnail_file_path, 'wb') as thumbnail_file:
                thumbnail_file.write(thumbnail)
        cap.release()
        processed_video = ProcessedVideo.objects.create(
            video_id=file_name,
            title=video.title,
            processed_video=os.path.join('processed_videos', f'{file_name}.webm'),
            thumbnail_file=os.path.join('processed_videos', 'thumbnails', thumbnail_filename),
        )

        # Open the processed video file and save it to the ProcessedVideo object
        # with open(output_path, 'rb') as f:
        #     processed_video.processed_video.save(os.path.basename(output_path), File(f))

        response = StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
        response['Cache-Control'] = 'no-cache'  # Disable caching
        messages.success(request, 'Video processing completed successfully')
        # return render(request, 'app/process_video.html', context)
        return response
    
    def delete(self, request):
        try:
            # Retrieve the video object
            video_id = request.headers.get('X-Video-ID')
            video = ProcessedVideo.objects.get(id=video_id)
            
            # Delete video file
            if os.path.exists(video.processed_video.path):
                os.remove(video.processed_video.path)
            
            # Delete thumbnail file
            if os.path.exists(video.thumbnail_file.path):
                os.remove(video.thumbnail_file.path)
            
            # Delete video object from database
            video.delete()
            messages.success(request, f'Video Deleted Successfully')
            
            return JsonResponse({'message': 'Video deleted successfully'})
        except ProcessedVideo.DoesNotExist:
            messages.error(request, f'Video not found')
            return HttpResponseBadRequest("Video not found")
        except Exception as e:
            messages.error(request, f'Failed to delete video')
            return HttpResponseServerError(f"Failed to delete video: {e}")


# class ProcessVideoView(View):
#     def get(self, request):
#         return render(request, 'app/process_video.html')

#     def post(self, request):
#         video_link = request.POST.get('video_link')

#         if not video_link:
#             logger.error("No video link provided")
#             return HttpResponseBadRequest("No video link provided")

#         # Load the YOLOv8 model
#         model_path = os.path.join(settings.BASE_DIR, 'app', 'models', 'best.pt')
#         try:
#             model = YOLO(model_path)
#         except Exception as e:
#             logger.error(f"Failed to load YOLOv8 model: {e}")
#             return HttpResponseServerError("Failed to load YOLOv8 model")

#         # Process the video using the YOLOv8 model
#         try:
#             video = pafy.new(video_link)
#             best = video.getbest()
#             cap = cv2.VideoCapture(best.url)
#         except Exception as e:
#             logger.error(f"Failed to open video: {e}")
#             return HttpResponseServerError("Failed to open video")

#         def generate_frames():
#             frame_count = 0
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     logger.info("End of video stream")
#                     break

#                 try:
#                     results = model(frame)
#                     annotated_frame = results[0].plot()
#                 except Exception as e:
#                     logger.error(f"Failed to process frame {frame_count}: {e}")
#                     break

#                 # Encode the annotated frame as base64
#                 _, buffer = cv2.imencode('.jpg', annotated_frame)
#                 frame_bytes = base64.b64encode(buffer).decode('utf-8')

#                 yield (b'--frame\r\n'
#                        b'Content-Type: text/plain\r\n\r\n' + frame_bytes.encode() + b'\r\n')

#                 frame_count += 1

#         response = StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
#         response['Cache-Control'] = 'no-cache'  # Disable caching
#         return response
  