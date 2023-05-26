from rest_framework import serializers
from .models import Files, MyModel


class FilesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Files
        fields = ['id','file_s']
        
class MyModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MyModel
        fields = ['id', 'arff','accuracy','filename']