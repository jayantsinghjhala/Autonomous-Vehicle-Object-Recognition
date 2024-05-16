from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('',views.YoloView.as_view(),name='home'),
    path('process_video/', views.ProcessVideoView.as_view(), name='process_video'),
    path('video_library/', views.VideoLibraryView.as_view(), name='video_library'),
    path('realtime_video/', views.RealtimeVideoView.as_view(), name='realtime_video'),
    path('upload_video/', views.UploadVideoView.as_view(), name='upload_video'),
] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)