import os

from matplotlib.pyplot import axis
import onnxruntime as rt
import numpy as np

import torch



#for conv 
def weights_rebuild_for_conv3x3_pytorch_nchw(weights,shape,mul,weight_zp=0):
   #(out_channel,inchannel_,h,w)
    out_channel,in_c,_,_= shape
    new_shape = (shape[0]*mul,shape[1]*mul,shape[2],shape[3])
    weights_1 = np.ones(new_shape)*weight_zp  #must multiply weight_zp!!!!!!!
    for i in range(0,mul):
        for out_c in range(out_channel):
            o_index= i*out_channel+out_c  
            if i==0:
                weights_1[o_index,(mul-1)*in_c:mul*in_c,:,0] = weights[out_c,:,:,0]
                weights_1[o_index,0:in_c,:,1] = weights[out_c,:,:,1]
                weights_1[o_index,in_c:in_c*2,:,1] = weights[out_c,:,:,2]
            elif i!=mul-1:
                weights_1[o_index,(i-1)*in_c:i*in_c,:,1] = weights[out_c,:,:,0]
                weights_1[o_index,(i)*in_c:(i+1)*in_c,:,1] = weights[out_c,:,:,1]
                weights_1[o_index,(i+1)*in_c:(i+2)*in_c,:,1] = weights[out_c,:,:,2]
            else :
                weights_1[o_index,(mul-2)*in_c:(mul-1)*in_c,:,1] = weights[out_c,:,:,0]
                weights_1[o_index,(mul-1)*in_c:(mul)*in_c,:,1] = weights[out_c,:,:,1]
                weights_1[o_index,0:in_c,:,2] = weights[out_c,:,:,2]
    return weights_1

def repeat_mul_times(data,shape,mul):
    data_rebuild = np.tile(data,mul)
    return data_rebuild




#ringt for deconv 2multiply
def deconv_w2c_validate_2multiply_sr_fake_data_nhwc_v4(deconv_weight):
    #假设deconv的weight为[5,5,2,2]  weight[k_h,k_w,in_channel,out_channle]   stride=3,padding=1
    
    # deconv_weight = np.arange(5*5*2*2).reshape((5,5,2,2))
    deconv_weight = deconv_weight.reshape((5,5,2,2))
    #(10,10,2)
    input = np.arange(10*10*2).reshape(10,10,2)
    #
    deconv_weight_1 = np.zeros((5,5,4,4))  #ik_h,k_w,in_channel,out_channel,
    
    # 第一列：
    #第一个**output0的位置，参考【d】的计算，需要的是【2 3】和【4 5】 以d为中心【2 3】[4 5]在deconv_weight的位置是0和3 ，所以这边deconv_weight的w是0 和 3
    #以d为中心【2 3】deconv_weight_1的位置是也是0和3，所以deconv_weight_1中的w也是0 和 3
    #这个[2:4]是【2 3】在新的channel中的位置，【0:2】是【4 5】在mul后的input中的位置，所以这边deconv_weight_1的inchannel是【2:4】和 【0:2】
    #input的first channel
    deconv_weight_1[:,0,2:4,0] = deconv_weight[:,0,:,0]  #【2 3】
    deconv_weight_1[:,3,0:2,0] = deconv_weight[:,3,:,0]  #【4 5】
    #input的second channel
    deconv_weight_1[:,0,2:4,1] = deconv_weight[:,0,:,1]
    deconv_weight_1[:,3,0:2,1] = deconv_weight[:,3,:,1]   

   
    #第二个[0 1]的值 计算【0 1】点的时候以其为中心需要的就是他自己，所以deconv_weight是w=2 【0 1 2 3 4】的中心是2  然后【0 1】在mul后的input中的inchannel位置是【0:2】
    #因为是以第2列【0 0 0 0 】为中心计算的[0 1]，所以deconv_weight_1 w=3
    #第一个channnel
    deconv_weight_1[:,3,0:2,2] = deconv_weight[:,2,:,0]
    #第二个channel
    deconv_weight_1[:,3,0:2,3] = deconv_weight[:,2,:,1]
   
    # 第二列：
    #第三个aa的值
    
    deconv_weight_1[:,2,0:2,0] = deconv_weight[:,1,:,0]
    deconv_weight_1[:,2,2:4,0] = deconv_weight[:,4,:,0]
    #second channel 
    deconv_weight_1[:,2,0:2,1] = deconv_weight[:,1,:,1]
    deconv_weight_1[:,2,2:4,1] = deconv_weight[:,4,:,1]

    #第四个[b b]
    
    deconv_weight_1[:,2,0:2,2] = deconv_weight[:,0,:,0]
    deconv_weight_1[:,2,2:4,2] = deconv_weight[:,3,:,0]

    #second channel 
    deconv_weight_1[:,2,0:2,3] = deconv_weight[:,0,:,1]
    deconv_weight_1[:,2,2:4,3] = deconv_weight[:,3,:,1]
    
    # 第三列：
    #第5个[2 3]
    deconv_weight_1[:,1,2:4,0] = deconv_weight[:,2,:,0]
    deconv_weight_1[:,1,2:4,1] = deconv_weight[:,2,:,1]
    
    #第6个[c c]  需要【2 3】 【4 5】deconv_weight是1 和4 
    #因为是以第4列【0 0 0 0 】为中心计算的[c c]，所以deconv_weight_1 w=1和4
    deconv_weight_1[:,1,2:4,2] = deconv_weight[:,1,:,0]
    deconv_weight_1[:,4,0:2,2] = deconv_weight[:,4,:,0]

    deconv_weight_1[:,1,2:4,3] = deconv_weight[:,1,:,1]
    deconv_weight_1[:,4,0:2,3] = deconv_weight[:,4,:,1]
    return deconv_weight,deconv_weight_1



    #未multiply的矩阵  x轴是w，y轴是channel
    #0 a b 2 c d  4 e f 6 g h 8 i j 10 k l 12
    #1 a b 3 c d  5 e f 7 g h 9 i j 11 k l 13 以b为中心时，需要 【0 1】和【2 3】 所以这边是0 和 3
    #4倍multiply的矩阵.开头的时候是padding的3列0
    # 0 0     0       0       0       0      8
    # 0 0     0       1       0       0      9
    # 0 0     0       2       0       0      10
    # 0 0     0       3       0       0      11
    # 0 0     0       4       0       0      12
    # 0 0     0       5       0       0      13
    # 0 0     0       6       0       0      14
    # 0 0     0       7       0       0      15
            
    #        [*]    [2 3]    [e e]   [h h]
    #        [0 1]  [c c]    [f f]   [8 9]
    #        [a a]  [d d]    [6 7]
    #        [b b]  [4 5]    [g g]

    #


    #第3列[0 0 0 0 0 0 0 0]为中心要计算[0* 0 1 a b】
    #第4列[0 1 2 3 4 5 6 7]为中心要计算[2 c d 4]
    #第5列[0 0 0 0 0 0 0 0]为中心要计算[e f 6 g]
#如果是4倍的话那么就要计算12个数字
#2倍的话是6个数字，因为这里的deconv是扩大3倍
#right for deconv 4 multiply
def deconv_w2c_validate_4multiply_sr_fake_data_nhwc_v4(deconv_weight):
    #假设deconv的weight为[5,5,2,2]  weight[k_h,k_w,in_channel,out_channle]   stride=3,padding=1
    deconv_weight = deconv_weight.reshape((5,5,2,2))
    
    #
    deconv_weight_1 = np.zeros((5,5,8,8))  #ik_h,k_w,inchannel,out_channel,
    #第一排

    #第一个output0的位置，第一个**output0的位置，参考【h】的计算，需要的是【6 7】和【8 9】 以h为中心【6 7】[8 9]在deconv_weight的位置是0和3 ，所以这边deconv_weight的w是0 和 3
    #以第5列[0 0 0 0 0 0 0 0]为中心【6 7】 【8 9】 deconv_weight_1的位置是也是0和3
    #这个[6:8]是【6 7】在新的channel中的位置，【0:2】是【8 9】在mul后的input中的位置，所以这边deconv_weight_1的inchannel是【2:4】和 【0:2】
    deconv_weight_1[:,0,6:8,0] = deconv_weight[:,0,:,0]

    deconv_weight_1[:,3,0:2,0] = deconv_weight[:,3,:,0]

    deconv_weight_1[:,0,6:8,1] = deconv_weight[:,0,:,1]
    deconv_weight_1[:,3,0:2,1] = deconv_weight[:,3,:,1]

   
    #第二个[0 1]的值 计算【0 1】点的时候以其为中心需要的就是他自己，所以deconv_weight是w=2 
    #以第2列[0 0 0 0 0 0 0 0]为中心计算的【0 1】所以在deconv_weight_1的w=3
    # 然后【0 1】在mul后的input中的inchannel位置是【0:2】
    #第一个channnel
    deconv_weight_1[:,3,0:2,2] = deconv_weight[:,2,:,0]
    
    #第二个channel
    deconv_weight_1[:,3,0:2,3] = deconv_weight[:,2,:,1]

    
    
    #第三个aa的值  需要【0 1】和【2 3】 ,以【a a】为中心【0 1】【2 3】在deconv_weight的w为1 ,4
    #以第2列[0 0 0 0 0 0 0 0]为中心计算的【a a】所以在deconv_weight_1的w=3
    # 然后【0 1】在mul后的input中的inchannel位置是【0:2】【2 3】为[2:4]
    deconv_weight_1[:,3,0:2,4] = deconv_weight[:,1,:,0]
    deconv_weight_1[:,3,2:4,4] = deconv_weight[:,4,:,0]
    #second channel 
    deconv_weight_1[:,3,0:2,5] = deconv_weight[:,1,:,1]
    deconv_weight_1[:,3,2:4,5] = deconv_weight[:,4,:,1]

    #第四个[b b]  需要【0 1】和【2 3】，以【b b】为中心【0 1】【2 3】在deconv_weight的w为0 ，3
    #以第2列[0 0 0 0 0 0 0 0]为中心计算的【b b】所以在deconv_weight_1的w=3
    deconv_weight_1[:,3,0:2,6] = deconv_weight[:,0,:,0]
    deconv_weight_1[:,3,2:4,6] = deconv_weight[:,3,:,0]

    #second channel 
    deconv_weight_1[:,3,0:2,7] = deconv_weight[:,0,:,1]
    deconv_weight_1[:,3,2:4,7] = deconv_weight[:,3,:,1]
    
    #第二排
    #第5个[2 3] 计算【2 3】点的时候以其为中心需要的就是他自己，deconv_weight中对应的w是2
    #以第3列【0 1 2 3 4 5 6 7】的为中心计算的【2 3】所以此时deconv_weight_1的w=2
    deconv_weight_1[:,2,2:4,0] = deconv_weight[:,2,:,0]
    deconv_weight_1[:,2,2:4,1] = deconv_weight[:,2,:,1]
    
    #第6个[c c] 需要【2 3】【4 5】 deconv_weight中对应的w是1 和 4
    #以第3列【0 1 2 3 4 5 6 7】的为中心计算的【c c】所以此时deconv_weight_1的w=2

    deconv_weight_1[:,2,2:4,2] = deconv_weight[:,1,:,0]
    deconv_weight_1[:,2,4:6,2] = deconv_weight[:,4,:,0]

    deconv_weight_1[:,2,2:4,3] = deconv_weight[:,1,:,1]
    deconv_weight_1[:,2,4:6,3] = deconv_weight[:,4,:,1]

    #第7个[d d]
    deconv_weight_1[:,2,2:4,4] = deconv_weight[:,0,:,0]  #[2 3]
    deconv_weight_1[:,2,4:6,4] = deconv_weight[:,3,:,0]
    deconv_weight_1[:,2,2:4,5] = deconv_weight[:,0,:,1]  #[2 3]
    deconv_weight_1[:,2,4:6,5] = deconv_weight[:,3,:,1]

    #第8个[4 5]
    deconv_weight_1[:,2,4:6,6] = deconv_weight[:,2,:,0]
    deconv_weight_1[:,2,4:6,7] = deconv_weight[:,2,:,1]

    #第三排
    #第9个[e e] 需要的是【4 5】和【6 7】deconv_weight中对应的w是 1和4
    #以第4列【0 0 0 0 0 0 0 0】为中心计算【e e】需要的值在deconv_weight_1中的w=1
    deconv_weight_1[:,1,4:6,0] = deconv_weight[:,1,:,0]
    deconv_weight_1[:,1,6:8,0] = deconv_weight[:,4,:,0]

    deconv_weight_1[:,1,4:6,1] = deconv_weight[:,1,:,1]
    deconv_weight_1[:,1,6:8,1] = deconv_weight[:,4,:,1]

    #第10个[f f]需要的是【4 5】和【6 7】deconv_weight中对应的w是 0和3
    #以第4列【0 0 0 0 0 0 0 0】为中心计算【e e】需要的值在deconv_weight_1中的w=1
    deconv_weight_1[:,1,4:6,2] = deconv_weight[:,0,:,0]
    deconv_weight_1[:,1,6:8,2] = deconv_weight[:,3,:,0]
    deconv_weight_1[:,1,4:6,3] = deconv_weight[:,0,:,1]
    deconv_weight_1[:,1,6:8,3] = deconv_weight[:,3,:,1]

    #第11个[6 7]
    deconv_weight_1[:,1,6:8,4] = deconv_weight[:,2,:,0]
    deconv_weight_1[:,1,6:8,5] = deconv_weight[:,2,:,1]

    #第12个[g g]
    deconv_weight_1[:,1,6:8,6] = deconv_weight[:,1,:,0]
    deconv_weight_1[:,4,0:2,6] = deconv_weight[:,4,:,0]
    deconv_weight_1[:,1,6:8,7] = deconv_weight[:,1,:,1]
    deconv_weight_1[:,4,0:2,7] = deconv_weight[:,4,:,1]
  
    return deconv_weight,deconv_weight_1



#  # deconv_weight_pytorch = weights.reshape((shape))#这种是浅copy，deconv_weight_pytorch改变，weights也会改变
#step1:pytorch nchw [in,out,kh,kw] reverse[kh,kw] and transpose (2,3,0,1) to TF [kh,kw.in,out]
#step2: TF weight to rebuild,so rebuild is [kh,kw,in,out]
#step3: TF rebuild wegiht [kh,kw,in*mul,out*mul] first transpose(2,3,0,1)[in*mul,out*mul,kh,kw] and reverse
 
def deconv_w2c_2mul_pytorch_nchw_whole(weights,shape,mul,weight_zp=0):
    #  weight[in_channel,out_channle,k_h,k_w]   stride=3,padding=1
    in_channel,out_channel,k_h,k_w = shape
    import copy
    deconv_weight_pytorch = copy.deepcopy(weights)
    #step1: pytorch nchw [in,out,kh,kw] reverse[kh,kw] and transpose (2,3,0,1) to TF [kh,kw.in,out]
    for in_c in range(in_channel):
        for out_c in range(out_channel):
            deconv_weight_pytorch[in_c,out_c,:,:] = deconv_weight_pytorch[in_c,out_c,:,:].flatten()[::-1].reshape((1,1,k_h,k_w))
    deconv_weight_tf = deconv_weight_pytorch.transpose((2,3,0,1))
    #step2:TF weight to rebuild,so rebuild is [kh,kw,in*mul,out*mul]
    new_shape = (shape[2],shape[3],shape[0]*mul,shape[1]*mul)
    deconv_weight_tf_rebuild = np.ones(new_shape)*weight_zp
    for i in range(mul):
        for out_c in range(out_channel):
            #第一个值
            o_index= i*out_channel+out_c
            if(i==0):
                #第一列：第一个值：**[kh,kw,in,out]
                deconv_weight_tf_rebuild[:,0,(mul-1)*in_channel:mul*in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]  #【2 3】
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]  #【4 5】
                #第二列：第一个值：aa的值
                deconv_weight_tf_rebuild[:,2,0:in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                #对于2倍这边是2:4
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
                # 第三列：第一个值：[2 3]
                deconv_weight_tf_rebuild[:,1,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
            if(i==1):
                #第一列：第二个值[0 1]
                #第二个[0 1]的值 计算【0 1】点的时候以其为中心需要的就是他自己，所以deconv_weight是w=2 【0 1 2 3 4】的中心是2  然后【0 1】在mul后的input中的inchannel位置是【0:2】
                #因为是以第2列【0 0 0 0 】为中心计算的[0 1]，所以deconv_weight_1 w=3
                #第一个channnel  第二个channel ：for out_c in range(out_channel):
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
                #第二列：第二个值：[b b]
                deconv_weight_tf_rebuild[:,2,0:in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]
                # 第三列：第二个值：[c c]
                deconv_weight_tf_rebuild[:,1,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,4,0:in_channel,o_index,] = deconv_weight_tf[:,4,:,out_c]
    #step3:TF rebuild wegiht [kh,kw,in*mul,out*mul] first transpose(2,3,0,1)[in*mul,out*mul,kh,kw] and reverse
    deconv_weight_tf_rebuild_transpose = copy.deepcopy(deconv_weight_tf_rebuild.transpose((2,3,0,1)))
     # deconv_weight_tf_rebuild_transpose = deconv_weight_tf_rebuild.transpose((2,3,0,1))#(8,2,5,5)//这种是浅copy
    #reverse
    for in_c in range(in_channel*mul):
        for out_c in range(out_channel*mul):
            deconv_weight_tf_rebuild_transpose[in_c,out_c,:,:] = deconv_weight_tf_rebuild_transpose[in_c,out_c,:,:].flatten()[::-1].reshape((1,1,k_h,k_w))
    deconv_weight_pytorch_rebuild = deconv_weight_tf_rebuild_transpose
    return deconv_weight_tf,deconv_weight_tf_rebuild,weights,deconv_weight_pytorch_rebuild


 

def deconv_w2c_4multiply_pytorch_nchw_whole(weights,shape,in_channel,out_channel,mul=4,weight_zp=0):
    #假设deconv的weight为[2,2,5,5]  weight[in_channel,out_channle,k_h,k_w]   stride=3,padding=1
    in_channel,out_channel,k_h,k_w = shape
    # deconv_weight = np.arange(5*5*2*2).reshape((2,2,5,5))
    import copy
    deconv_weight_pytorch = copy.deepcopy(weights)
    # deconv_weight_pytorch = weights.reshape((shape))#这种是浅copy，deconv_weight_pytorch改变，weights也会改变
    #step1: reverse and transpose

    for in_c in range(in_channel):
        for out_c in range(out_channel):
            deconv_weight_pytorch[in_c,out_c,:,:] = deconv_weight_pytorch[in_c,out_c,:,:].flatten()[::-1].reshape((1,1,k_h,k_w))

    deconv_weight_tf = deconv_weight_pytorch.transpose((2,3,0,1))
    
    #step2:TF weight to rebuild
    new_shape = (shape[2],shape[3],shape[0]*mul,shape[1]*mul)#(5,5)
    print("new_shape is",new_shape)
    deconv_weight_tf_rebuild = np.ones(new_shape)*weight_zp
  
    for i in range(mul):
        for out_c in range(out_channel):
            o_index= i*out_channel+out_c
            if(i==0):
                #第一列：第一个值：**(mul-1)*in_channel:mul*in_channel=[6:8]
                deconv_weight_tf_rebuild[:,0,(mul-1)*in_channel:mul*in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]  #【2 3】
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]  #【4 5】
                #第二列：第一个值[2 3]
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
                #第三列：第一个值[e e]
                deconv_weight_tf_rebuild[:,1,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]

            if(i==1):
                #第一列：第二个值[0 1]  i=1，o_index=2，3
                #第二个[0 1]的值 计算【0 1】点的时候以其为中心需要的就是他自己，所以deconv_weight是w=2 【0 1 2 3 4】的中心是2  然后【0 1】在mul后的input中的inchannel位置是【0:2】
                #因为是以第2列【0 0 0 0 】为中心计算的[0 1]，所以deconv_weight_1 w=3
                #第一个channnel  第二个channel ：for out_c in range(out_channel):
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
                #第二列：第二个值[c c]
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,2,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
                #第三列：第二个值[f f]
                deconv_weight_tf_rebuild[:,1,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]



            if(i==2):
                #第一列：第三个值[a a]  i=2，o_index=4，5
                #第一个channnel  第二个channel ：for out_c in range(out_channel):
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,3,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
                #第二列：第三个值[d d]
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]  #[2 3]
                deconv_weight_tf_rebuild[:,2,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]
                #第三列：第三个值[6 7]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]

            if(i==3):
                #第一列：第四个值[b b]  i=3，o_index=6，7
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]
                deconv_weight_tf_rebuild[:,3,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]
                #第二列：第四个值[4 5]
                deconv_weight_tf_rebuild[:,2,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
                #第三列：第四个值[g g]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,4,0:in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
    #step3:first transpose(2,3,0,1) and reverse
    deconv_weight_tf_rebuild_transpose = copy.deepcopy(deconv_weight_tf_rebuild.transpose((2,3,0,1)))
    # deconv_weight_tf_rebuild_transpose = deconv_weight_tf_rebuild.transpose((2,3,0,1))#(8,2,5,5)//这种是浅copy
    #reverse
    for in_c in range(in_channel*mul):
        for out_c in range(out_channel*mul):
            deconv_weight_tf_rebuild_transpose[in_c,out_c,:,:] = deconv_weight_tf_rebuild_transpose[in_c,out_c,:,:].flatten()[::-1].reshape((1,1,k_h,k_w))
    deconv_weight_pytorch_rebuild = deconv_weight_tf_rebuild_transpose
    return deconv_weight_tf,deconv_weight_tf_rebuild,weights,deconv_weight_pytorch_rebuild





import copy
def deconv_w2c_4multiply_pytorch_nchw_whole(weights,shape,in_channel,out_channel,mul=4,weight_zp=0):
    in_channel,out_channel,k_h,k_w = shape
    deconv_weight_pytorch = copy.deepcopy(weights)
    #step1: reverse and transpose
    for in_c in range(in_channel):
        for out_c in range(out_channel):
            deconv_weight_pytorch[in_c,out_c,:,:] = deconv_weight_pytorch[in_c,out_c,:,:].flatten()[::-1].reshape((1,1,k_h,k_w))
    deconv_weight_tf = deconv_weight_pytorch.transpose((2,3,0,1))
    #step2:TF weight to rebuild
    new_shape = (shape[2],shape[3],shape[0]*mul,shape[1]*mul)#(5,5)
    deconv_weight_tf_rebuild = np.ones(new_shape)*weight_zp
    for i in range(mul):
        for out_c in range(out_channel):
            o_index= i*out_channel+out_c
            if(i==0):
                #第一列：第一个值： #第二列：第一个值[2 3] #第三列：第一个值[e e]
                deconv_weight_tf_rebuild[:,0,(mul-1)*in_channel:mul*in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]  #【2 3】
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]  #【4 5】
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
                deconv_weight_tf_rebuild[:,1,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
            if(i==1):
                #第一列：第二个值[0 1]  i=1，o_index=2，3 #第二列：第二个值[c c] #第三列：第二个值[f f]
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,2,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
                deconv_weight_tf_rebuild[:,1,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]
            if(i==2):
                #第一列：第三个值[a a]  i=2，o_index=4，5  #第二列：第三个值[d d] #第三列：第三个值[6 7]
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,3,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
                deconv_weight_tf_rebuild[:,2,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]  #[2 3]
                deconv_weight_tf_rebuild[:,2,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
            if(i==3):
                #第一列：第四个值[b b]  i=3，o_index=6，7  #第二列：第四个值[4 5] #第三列：第四个值[g g]
                deconv_weight_tf_rebuild[:,3,0:in_channel,o_index] = deconv_weight_tf[:,0,:,out_c]
                deconv_weight_tf_rebuild[:,3,in_channel:2*in_channel,o_index] = deconv_weight_tf[:,3,:,out_c]
                deconv_weight_tf_rebuild[:,2,2*in_channel:3*in_channel,o_index] = deconv_weight_tf[:,2,:,out_c]
                deconv_weight_tf_rebuild[:,1,3*in_channel:4*in_channel,o_index] = deconv_weight_tf[:,1,:,out_c]
                deconv_weight_tf_rebuild[:,4,0:in_channel,o_index] = deconv_weight_tf[:,4,:,out_c]
    #step3:first transpose(2,3,0,1) and reverse
    deconv_weight_tf_rebuild_transpose = copy.deepcopy(deconv_weight_tf_rebuild.transpose((2,3,0,1)))
    #reverse
    for in_c in range(in_channel*mul):
        for out_c in range(out_channel*mul):
            deconv_weight_tf_rebuild_transpose[in_c,out_c,:,:] = deconv_weight_tf_rebuild_transpose[in_c,out_c,:,:].flatten()[::-1].reshape((1,1,k_h,k_w))
    deconv_weight_pytorch_rebuild = deconv_weight_tf_rebuild_transpose
    return deconv_weight_tf,deconv_weight_tf_rebuild,weights,deconv_weight_pytorch_rebuild



