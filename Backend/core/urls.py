from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FilesViewSet, TrainViewSet, ArffViewSet, get_metadata, predict, arffa
from . import views

router = DefaultRouter()
router.register('files', FilesViewSet, basename='files')
router.register('train', TrainViewSet, basename='train')
router.register('arff', ArffViewSet, basename='arff')

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/<str:filename>/metadata', get_metadata, name='get_metadata'),
    path('api/<str:filename>/predict', predict, name='predict'),
    path('api/<str:filename>/arffa', arffa, name='arffa'),
    
]