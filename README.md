# Implementation-of-Question-Answering-System

##this project for q-a system
####first,you should download the BERT chinese mmodel chinese_L-12_H-768_A-12
secend，you should start the bert-server frmo hanxiao
 ---
  >if you are win10 cd to this dir /bert-as-service 
  bert-serving-start -model_dir G:/pythonworkspace/chinese_L-12_H-768_A-12 -num_worker=4
  这里要注意-num_worker=4表示工作个数，一般就是你机器的cpu核数，如果超过，效率会降低。
 ---
 > if you are in Linux ubuntu 17.0
  pip install bert-serving-service
  pip install bert-serving-client
  cd to /bert-as-service,there is not cls: ert-serving-start 
  you can use find ./bert-serving-start 
  and find in /home/xxx/.local/bin
  now you can in python>3.5 and tensorflow>1.10 if your tf is 1.8,you can pip3 install tensorflow=1.10
  then  python3 bert-serving-start -model_dir /home/chinese_L-12_H-768_A-12 -num_worker=4
  
  ---
  
  usage: bert-serving-start -model_dir /home/liujiefei/pythonworkspace/06bertserver/02bertapp/chinese_L-12_H-768_A-12 -num_worker=4
                 ARG   VALUE
__________________________________________________
           ckpt_name = bert_model.ckpt
         config_name = bert_config.json
                cors = *
                 cpu = False
          device_map = []
  fixed_embed_length = False
                fp16 = False
 gpu_memory_fraction = 0.5
       graph_tmp_dir = None
    http_max_connect = 10
           http_port = None
        mask_cls_sep = False
      max_batch_size = 256
         max_seq_len = 25
           model_dir = /home/liujiefei/pythonworkspace/06bertserver/02bertapp/chinese_L-12_H-768_A-12
          num_worker = 4
       pooling_layer = [-2]
    pooling_strategy = REDUCE_MEAN
                port = 5555
            port_out = 5556
       prefetch_size = 10
 priority_batch_size = 16
show_tokens_to_client = False
     tuned_model_dir = None
             verbose = False
                 xla = False

I:VENTILATOR:[__i:__i: 66]:freeze, optimize and export graph, could take a while...
I:GRAPHOPT:[gra:opt: 52]:model config: /home/liujiefei/pythonworkspace/06bertserver/02bertapp/chinese_L-12_H-768_A-12/bert_config.json
I:GRAPHOPT:[gra:opt: 55]:checkpoint: /home/liujiefei/pythonworkspace/06bertserver/02bertapp/chinese_L-12_H-768_A-12/bert_model.ckpt
I:GRAPHOPT:[gra:opt: 59]:build graph...
I:GRAPHOPT:[gra:opt:128]:load parameters from checkpoint...
I:GRAPHOPT:[gra:opt:132]:optimize...
I:GRAPHOPT:[gra:opt:140]:freeze...
I:GRAPHOPT:[gra:opt:145]:write graph to a tmp file: /tmp/tmp6k2v1ha_
I:VENTILATOR:[__i:__i: 74]:optimized graph is stored at: /tmp/tmp6k2v1ha_
I:VENTILATOR:[__i:_ru:106]:bind all sockets
I:VENTILATOR:[__i:_ru:110]:open 8 ventilator-worker sockets
I:VENTILATOR:[__i:_ru:113]:start the sink
I:SINK:[__i:_ru:270]:ready
I:VENTILATOR:[__i:_ge:188]:get devices
W:VENTILATOR:[__i:_ge:203]:only 1 out of 1 GPU(s) is available/free, but "-num_worker=4"
W:VENTILATOR:[__i:_ge:205]:multiple workers will be allocated to one GPU, may not scale well and may raise out-of-memory
I:VENTILATOR:[__i:_ge:221]:device map: 
		worker  0 -> gpu  0
		worker  1 -> gpu  0
		worker  2 -> gpu  0
		worker  3 -> gpu  0
I:WORKER-0:[__i:_ru:478]:use device gpu: 0, load graph from /tmp/tmp6k2v1ha_
I:WORKER-1:[__i:_ru:478]:use device gpu: 0, load graph from /tmp/tmp6k2v1ha_
I:WORKER-2:[__i:_ru:478]:use device gpu: 0, load graph from /tmp/tmp6k2v1ha_
I:WORKER-3:[__i:_ru:478]:use device gpu: 0, load graph from /tmp/tmp6k2v1ha_
I:WORKER-0:[__i:gen:506]:ready and listening!
I:WORKER-1:[__i:gen:506]:ready and listening!
I:WORKER-2:[__i:gen:506]:ready and listening!
I:WORKER-3:[__i:gen:506]:ready and listening!
---

now the bert-server is start
---
you can cd G:/Implementation-of-Question-Answering-System/qa
or :~/pythonworkspace/qa-bertapp$ python3 bertClient.py

#now the qa system is start
#open your broswer:
#http://youip:8080/?question=你好
get your answer

