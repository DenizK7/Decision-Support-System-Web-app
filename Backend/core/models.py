from django.db import models

# Create your models here.


class Files(models.Model):
    file_s = models.FileField(upload_to='store/file/')##pdf is changed to file !! and change the direction of saved foılders

    def __str__(self):
        return self.file_s
class MyModel(models.Model):
    arff = models.CharField(max_length=100, default='')
    accuracy = models.CharField(max_length=100, default='')
    filename = models.FileField(upload_to='store/file/',default='')
    def __str__(self):
        return self.name