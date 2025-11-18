from django.urls import path
from .views import PrecipitationView

urlpatterns = [
    path('precipitation/', PrecipitationView.as_view(), name='precipitation'),
]