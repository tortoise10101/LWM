# LWM

Language World Model

### load vae model
```
./loader.sh
```


### docker setup
dockder-composeを使ってる．
GUI環境を想定．

コンテナ名: LWM

コンテナのビルド
```
docker-compose build
```

コンテナの起動
```
docker-compose up -d
```

実行例 (train.pyの実行)
```
docker exec -it lwm python train.py
```