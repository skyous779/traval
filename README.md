# 第四届MindCon-西安旅游主题图像分类

## 数据集目录如下
```text
    ├── data
        ├─image
        │   ├─0
        │   │ ├ img_1.jpg
        │   │ ├ ...
        │   ├─2
        │   │ ├ img_n.jpg
        │   │ ├ ...
        │   ├─ ...
        │   └─53      
        ├─test
        │   ├─1.jpg
        │   ├─2.jpg
        │   ├─ ...
        │   └─269.jpg  
```

## 推理脚本
```bash
python eval.py  --data_url ../data --batch_size 1   --pretrained True --eval 1 --ckpt_url ./ckpt_0/best.ckpt  --result_url ./ --num_classes 54
```
- data_url:数据集路径
- batch_size：推理batch_size
- pretrained: 推理模型使能
- eval：推理使能
- ckpt_url：推理模型路径
- num_classes：推理类别
- result_url：结果保存路径，最后得到的结果保存命名为result_sort.txt
