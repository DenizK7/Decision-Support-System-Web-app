# Generated by Django 4.2.1 on 2023-05-14 16:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_mymodel_accuracy'),
    ]

    operations = [
        migrations.AddField(
            model_name='mymodel',
            name='filename',
            field=models.FileField(default='', upload_to='store/file/'),
        ),
    ]
