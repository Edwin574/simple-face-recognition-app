from django.urls import path
from . import views

from . import views

# myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train, name='train'),
    path('recognize/', views.recognize, name='recognize'),
    path('upload/', views.upload, name='upload'),
]
