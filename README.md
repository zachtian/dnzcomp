# dnzcmp
modelv1 1, 16, 256\
modelv2 0, 16, 256\
modelv3 stack noise with only 1\*1 conv layer and add 2 more 3\*3 conv\
modelv4 use rep - delta\
modelv5 stack noise with only 1\*1 conv layer
* modelv5l1 stack noise with only 1\*1 conv layer with loss weight 1
(loss = imgage loss + loss weight \* sparsity loss)
* modelv5l01 stack noise with only 1\*1 conv layer with loss weight 0.1
* modelv5l001 stack noise with only 1\*1 conv layer with loss weight 0.01

modelv6 stack noise for all layer