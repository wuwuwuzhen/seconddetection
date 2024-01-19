# docker
```shell
docker exec -it 69 /bin/zsh
ps -ef | grep python
cd data/code/hf_bus
pkill -f python
kill -9 $(pgrep gunicorn)
nohup gunicorn --workers 24 --timeout 3600 --bind 0.0.0.0:5000 main:app --log-level debug --access-logfile - --error-logfile - &
```

# curl 
```sehll
curl --location 'http://127.0.0.1:5000/seconddetection' \
--header 'Content-Type: application/json' \
--data '[
    {
        "id": 60,
        "plate": "???A05217D",
        "day": "2023-12-25",
        "alarm_begin_time": "2023-12-25 08:10:45",
        "alarm_end_time": "2023-12-25 08:10:45",
        "exception_name": "????????????",
        "exception_type": 15,
        "video_url": "[\"http://10.2.137.174:8950/openUrl/6xhUKU8/playback.mp4?beginTime=2023-12-25T08:10:46.000+08:00&endTime=2023-12-25T08:10:59.000+08:00&playBackMode=1\"]",
        "picture_url": "[\"https://10.2.137.173:443/ngx/proxy?i=aHR0cDovLzEwLjIuMTM3LjEyOTo2MTIwL3BpYz85ZGQ5MDZpY2QtZSo1N2YxMDAzNmVlNmEtLTQ4MTljMjI2YjdlYTBpY2I1Kj01ZDBpM3MxKj1pZHAzKj0qZDBpMnQ9cGUwbTcxMTQ0Yy1hMzBzNTA2MHo2N2RpPTI9\"]"
    }
]'

```