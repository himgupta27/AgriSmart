from django.db import models

# Create your models here.
class Imageclass(models.Model):
	image=models.ImageField(upload_to='media/')
	name=models.CharField(max_length=20, default='noname')

	def __str__(self):
		return self.name
