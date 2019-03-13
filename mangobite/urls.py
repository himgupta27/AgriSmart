
from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.homepage),
	path('home/', views.homepage, name='home'),
    path('dpage/', views.diseasepage, name='dpage'),
    path('fpage/', views.feedback, name='fpage'),
    path('sapage/', views.soilanalysispage, name='sapage'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
