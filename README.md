# llama2-webui
llama2-webui

# 前提
自行下载[Llama2](https://github.com/facebookresearch/llama)项目代码和模型，将webapp.py文件放置于llama目录中，与download.sh同级位置。

# 运行
类似于官方example，使用torchrun命令运行即可
```bash
torchrun --nproc_per_node 1 webapp.py
```

# 效果

![image](https://github.com/learningpro/llama2-webui/assets/1081377/beea3cef-dfdc-44d7-bdba-bf831c0b1a5f)
