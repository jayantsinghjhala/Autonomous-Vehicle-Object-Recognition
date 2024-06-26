# Generated by Django 5.0 on 2024-03-21 15:57

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ProcessedVideo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video_id', models.CharField(max_length=255)),
                ('title', models.CharField(max_length=255)),
                ('processed_video', models.FileField(upload_to='processed_videos/')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
