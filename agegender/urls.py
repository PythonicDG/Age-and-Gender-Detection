from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
    path('uploaded_results/<int:image_id>/', views.uploaded_results, name='uploaded_results'),
    path('live_detection/', views.live_detection, name='live_detection'),
    path('live_feed/', views.live_video_feed, name='live_video_feed'),
]
