# docker
```shell
docker exec -it 69 /bin/zsh
pkill -f python ## 杀死所有的进程
nohup gunicorn --workers 24 --timeout 3600 --bind 0.0.0.0:5000 main:app --log-level debug --access-logfile - --error-logfile - &
```