"""eval"""
import os

import mindspore.dataset as ds
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
import numpy as np
from src.args import args
from src.tools.cell import cast_amp, WithEvalCell
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer
# from mindvision.engine.loss import CrossEntropySmooth
set_seed(args.seed)


def main():
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
    set_device(args)

    # get model
    net = get_model(args)
    cast_amp(net)
    criterion = get_criterion(args)

    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained(args, net)

    obs_data_url = args.data_url
    if not os.path.exists(args.data_url):
        os.mkdir(args.data_url)
        
    # try:
    #     import moxing as mox
    #     mox.file.copy_parallel(obs_data_url, args.data_url)
    #     print("Successfully Download {} to {}".format(obs_data_url, args.data_url))
    # except Exception as e:
    #     print('moxing download {} to {} failed: '.format(obs_data_url, args.data_url) + str(e))

    
    
    
    
    
    data = get_dataset(args, training=False)
    testset = data.test_dataset
    # testset = testset.batch(16)
    dataset_test = ds.GeneratorDataset(testset, ["data", "label"], shuffle=False)
    # model = ms.Model(net_with_loss, metrics=eval_metrics)
    

    
    batch_num = data.test_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)
    eval_network = WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    eval_metrics = {'Loss': nn.Loss(),
                    'Top1-Acc': nn.Top1CategoricalAccuracy(),
                    'Top5-Acc': nn.Top5CategoricalAccuracy()}
    # network_loss = CrossEntropySmooth(sparse=True,
    #                               reduction="mean",
    #                               smooth_factor=0.1,
    #                               classes_num=num_classes)
    model = Model(net, criterion, metrics=eval_metrics)
    


    path = os.path.join(args.data_url, 'test')
    fir = os.listdir(path)
    fir.sort()
    list = []
    print("###################start predicting!!!############################")
    for i,data in enumerate(dataset_test.create_dict_iterator()):
        # print(data['data'].shape,data['label'], data['label'].shape)
        image = data['data']
        # print(i)
        # print(image.shape)
        prob = model.predict(image)
        label0 = np.argmax(prob[0].asnumpy(), axis=1)
        label1 = np.argmax(prob[1].asnumpy(), axis=1)
        # print(label0, label1)
        # if label0 !=label1:
        #     print(i, label0, label1)
        # else:
            # print("label0 == label1")
        # print('img_'+str(i)+'.jpg, '+str(label0[0]))
        log = fir[i]+', '+str(label0[0])
        list.append(log)

        # # 保存result.txt文件
        # filename = 'result.txt'
        # file_path = os.path.join('/home/ma-user/work', filename)
        # with open(file_path, 'a+') as file:
        #     file.write(log+"\n")
        #     file.close()
    
    # print(list)
    
    num_result = []
    for numb in list:
        p1 = numb.find('.')
        order_n = int(numb[:p1])
        num_result.append(order_n)
        # print(order_n)  
        
    ### 把对应的结果放入output中
    output = []
    for numb in list:
        p2 = numb.find(',')
        order_n = int(numb[p2+1:]) ## order_n = int(numb[4:p1])
        output.append(order_n)
        # print(order_n)
 
    # 保存没有排序的output
    # with open("./result_output.txt", "w") as f:
    #     for item in range(len(output)):
    #         print(output[item])
    #         f.writelines(str(output[item]))
    #         f.writelines('\n')
    #     f.close()
        
    import json
    # json_data = json.load('label_id_name.json')
    with open('./label_id_name.json','r',encoding='utf8')as fp:
        json_data = json.load(fp) #读取json文件
        
    order_list = []
    for i in range(len(num_result)):
        order_list.append((num_result[i], output[i], json_data.get(str(output[i]))))
    # print(order_list) # (205, 0, '工艺品/仿唐三彩'),
    
    sorted_by_second = sorted(order_list, key=lambda tup: tup[0])
    with open(os.path.join(args.result_url, "result_sort.txt"), "w") as f:
        for item in sorted_by_second:
            # print(item[1])
            f.writelines(str(item[1]))
            f.writelines('\n')
        f.close()

if __name__ == '__main__':
    main()
## python eval.py --run_openi True --data_url /home/ma-user/work/data --batch_size 1  --num_classes 54 --pretrained True --eval 1 --ckpt_url ./ckpt_0/best.ckpt
## python eval.py  --data_url ../data --batch_size 1   --pretrained True --eval 1 --ckpt_url ./ckpt_0/best.ckpt  --result_url ./