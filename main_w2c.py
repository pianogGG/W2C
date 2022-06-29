import os
import onnx
import cv2
from collections import OrderedDict
from onnx import numpy_helper
import onnxruntime as rt
import numpy as np
from conv_c2w_core import weights_rebuild_for_conv3x3_pytorch_nchw,repeat_mul_times
from onnx import onnx_pb as onnx_proto

def weight_w2c(model,mul_dict):
   
    w2c_weights=[]
    w2c_bias=[]
    init_maps={}
    input_maps = {}
    keys = []
    weight_mul={}
    for inp in model.graph.input:
        input_maps[inp.name] = inp
        keys.append(inp.name)

    for node in model.graph.node:
        print(node)
        if(node.op_type=="Conv"):
            if node.name in mul_dict.keys():
                if(node.attribute[2].name=="kernel_shape" and node.attribute[2].ints==[3,3] and node.attribute[3].ints==[1,1,1,1] and node.attribute[4].ints==[1,1]):
                    w2c_weights.append(node.input[1])
                    weight_mul[node.input[1]]=mul_dict[node.name]
                    if(len(node.input)>2):
                        w2c_bias.append(node.input[2])
                        weight_mul[node.input[2]]=mul_dict[node.name]

               

    for init in model.graph.initializer:
        init_maps[init.name] = init
    for weight_name in w2c_weights:
        weight = init_maps[weight_name]
        mul = weight_mul[weight_name]
        new_dims=[weight.dims[0]*mul,weight.dims[1]*mul,weight.dims[2],weight.dims[3]]
        weight_data = numpy_helper.to_array(weight)
        weight_rebuild = weights_rebuild_for_conv3x3_pytorch_nchw(weight_data,weight_data.shape,mul)
        weight_rebuild = np.reshape(weight_rebuild,new_dims)
        weight_rebuild = weight_rebuild.astype(np.float32).flatten()
        new_weight = onnx.helper.make_tensor(weight.name, onnx.TensorProto.FLOAT, new_dims, weight_rebuild)
        model.graph.initializer.remove(weight)
        model.graph.initializer.extend([new_weight])

        # weight_input = input_maps[weight_name]
        # model.graph.input.remove(weight_input)
        new_weight_input = onnx.helper.make_tensor_value_info(weight_name, onnx.TensorProto.FLOAT, new_dims)
        model.graph.input.extend([new_weight_input])

        print("finished")
    for bias_name in w2c_bias:
        bias = init_maps[bias_name]
        mul = weight_mul[bias_name]
        new_dims = [bias.dims[0]*mul]
        bias_data = numpy_helper.to_array(bias)
        bias_rebuild = repeat_mul_times(bias_data,bias_data.shape,mul)
        bias_rebuild = np.reshape(bias_rebuild,new_dims)
        bias_rebuild = bias_rebuild.astype(np.float32).flatten()
        new_bias = onnx.helper.make_tensor(bias.name, onnx.TensorProto.FLOAT, new_dims, bias_rebuild)
        model.graph.initializer.remove(bias)
        model.graph.initializer.extend([new_bias])
        # bias_input = input_maps[bias_name]
        # model.graph.input.remove(bias_input)
        new_bias_input = onnx.helper.make_tensor_value_info(bias_name, onnx.TensorProto.FLOAT, new_dims)
        model.graph.input.extend([new_bias_input])




        print("123")
    #insert Reshape

    base_index=0
    for key in mul_dict.keys():
        if 'Reshape' in key:
            #insert reshape
            reshape_info = mul_dict[key]
            reshape_shape = reshape_info[:-1]
            reshape_shape = np.array(reshape_shape).astype(np.int64)
            reshape_index = reshape_info[-1]+base_index
            node = model.graph.node[reshape_index]

            reshape_input_shape = node.name + "_reshape_shape"
            init_shape = onnx.helper.make_tensor(reshape_input_shape, onnx_proto.TensorProto.INT64, [4],
                                             reshape_shape)
            model.graph.initializer.extend([init_shape])

            new_node = onnx.helper.make_node(
                'Reshape',
                name='Reshape'+node.name,
                inputs=[node.output[0],reshape_input_shape],
                outputs=[node.output[0]+'_reshape'],
                
            )
            
            model.graph.node.insert(reshape_index+1,new_node)
            next_node = model.graph.node[reshape_index+2]
            next_node.input[0]=new_node.output[0]
            base_index+=1
            print("reshape insert")


def compare_float(a,b,pre=0.0001):
    for i ,j in zip(a,b):
        if abs(i-j)>pre:
            return False
    return True



def w2c_fucntion(input_float_model_path,out_model_path,mul=2):
  
    
    width=1280
    height=720
    case = 0
    if case == 0:
        img = cv2.imread('onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/test_images/fd_input/img_18.jpg')
        img = cv2.resize(img, (1280, 720))  #hw:720x1280
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        
        input_data = y/255.0
        # input_data = y
        input_data = input_data.astype(np.float32)
        print(y.shape)
        print(input_data.min(),input_data.max())
        nhwc_data = np.expand_dims(input_data, axis=2)
        nhwc_data = np.expand_dims(nhwc_data, axis=0)
        nchw_data = nhwc_data.transpose(0,3,1,2)  # ONNX Runtime standard  (1,1,450,348)
    model_name = os.path.basename(input_float_model_path)
    model_path = os.path.dirname(input_float_model_path)
    
    model_float = onnx.load(input_float_model_path)
    onnx.checker.check_model(model_float)
    model_float = onnx.shape_inference.infer_shapes(model_float)


    sess = rt.InferenceSession(model_float.SerializeToString())
    # sess = rt.InferenceSession(model_float.SerializeToString())
    
    features = [x.name for x in sess.get_outputs()]
    input_name = sess.get_inputs()[0].name
    res = OrderedDict(zip(features, sess.run(features, {input_name: nchw_data})))
    mul_dict = {'Conv_0':8,'Conv_2':2,'Conv_4':8,'Reshape_0':(1,8,720,640,1),'Reshape_1':(1,32,720,160,3)}
    # mul_dict = {'Conv_0':8,'Conv_2':8,'Conv_4':8}
    mul=8
    weight_w2c(model_float,mul_dict)
    

    
    model_float.graph.input[0].type.tensor_type.shape.dim[1].dim_value=mul
    model_float.graph.output[0].type.tensor_type.shape.dim[1].dim_value=mul
    onnx.checker.check_model(model_float)
    onnx.save(model_float,out_model_path)
    
    # onnx.save(model_float,out_model_path)
    model_w2c = onnx.load(out_model_path)
  
    sess_2 = rt.InferenceSession(model_w2c.SerializeToString())
   
    

    #save data to binary for trtexec for w2c  这边的数据不要转到fp16
    
    nchw_data_fp16 = nchw_data.astype(np.float16)
    # nchw_data_fp16 = nchw_data
    nchw_data_fp16.tofile("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd_input_fp16_oriNCHW.bin")
    data_nhwc = nchw_data.transpose((0,2,3,1))
    data_nhwc = data_nhwc.astype(np.float16)
    data_nhwc_mul8 = data_nhwc.reshape((1,720,160,8))
    data_nhwc_mul8.tofile("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd_input_mul8_fp16_NHWC.bin")

    print(np.abs(data_nhwc.flatten()-data_nhwc_mul8.flatten()).max())








    features_2 = [x.name for x in sess_2.get_outputs()]
    input_name = sess_2.get_inputs()[0].name
    nchw_data =nchw_data.transpose(0,2,3,1)
    nchw_data = nchw_data.reshape(1,720,int(1280/mul),mul)
    nchw_data =nchw_data.transpose((0,3,1,2))
    
    
    
    
    res_2 = OrderedDict(zip(features_2, sess_2.run(features, {input_name: nchw_data})))
    res_nhwc_1= res['output'].transpose((0,2,3,1))#nchw to nhwc
    res_nhwc_2 =res_2['output'].transpose((0,2,3,1))
    res_nhwc_2 =res_nhwc_2.reshape((1,720,1280,1))
    bool_res = compare_float(res_nhwc_2.flatten(),res_nhwc_1.flatten())

    print((res_nhwc_2.flatten()==res_nhwc_1.flatten()).all())
    print(np.abs(res_nhwc_2-res_nhwc_1).max())
    return res['output'],res_2['output']


    
def add_features_to_output(model):
    # del m.graph.output[:]
    # m.graph.output.extend(m.graph.value_info)
    del model.graph.output[:]
    for node in model.graph.node:
        for output in node.output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])    
  
    

import json
def trt_res(input_float_model_path,out_model_path,mul):
    f = open("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd_output_orifp16_nchw.json")
    jnd_output_ori = json.load(f)
    jnd_output_ori_data = np.array(jnd_output_ori[0]['values']).reshape((1,1,720,1280))
    # jnd_output_ori_data = jnd_output_ori_data.transpose((0,2,3,1))
    f_1 =open("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd_output_fp16_mul8.json")
    jnd_output_mul8 =  json.load(f_1)
    jnd_output_mul8_data = np.array(jnd_output_mul8[0]['values']).reshape((1,8,720,160))
    jnd_output_mul8_data = jnd_output_mul8_data.transpose((0,2,3,1)).reshape((1,720,1280,1)).transpose((0,3,1,2))
    bool_res = compare_float(jnd_output_mul8_data.flatten(),jnd_output_ori_data.flatten())
    print(np.abs(jnd_output_mul8_data.flatten()-jnd_output_ori_data.flatten()).max())

    # jnd_output_mul4_data = jnd_output_mul4_data.transpose((0,2,3,1))
    # jnd_output_ori_data =jnd_output_ori_data.transpose((0,2,3,1))
    nchw_data_fp32 = np.fromfile("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd_input_fp32_oriNCHW.bin",dtype=np.float32)
    nchw_data_fp32 = nchw_data_fp32.reshape(1,1,720,1280)
    # model_float = onnx.load(input_float_model_path)
    model_float = onnx.load("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/models/jnd_qp27_fp16.onnx")
    add_features_to_output(model_float)
    case=0
    if case == 0:
        img = cv2.imread('onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/test_images/fd_input/img_18.jpg')
        img = cv2.resize(img, (1280, 720))  #hw:720x1280
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        
        input_data = y/255.0
        # input_data = y
        input_data = input_data.astype(np.float32)
        print(y.shape)
        print(input_data.min(),input_data.max())
        nhwc_data = np.expand_dims(input_data, axis=2)
        nhwc_data = np.expand_dims(nhwc_data, axis=0)
        nchw_data = nhwc_data.transpose(0,3,1,2)  # ONNX Runtime standard  (1,1,450,348)

    sess = rt.InferenceSession(model_float.SerializeToString())
    save_data=False
    if save_data:
        nhwc_data.tofile("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd_input_fp32_oriNCHW.bin")
        data_nhwc = nchw_data.transpose((0,2,3,1))  #nchw to nhwc
        data_nhwc_mul8 = data_nhwc.reshape((1,720,160,8))
        data_nhwc_mul8.tofile("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd_input_mul8_fp32_NHWC.bin")
    
    
    
    features = [x.name for x in sess.get_outputs()]
    input_name = sess.get_inputs()[0].name
    res = OrderedDict(zip(features, sess.run(features, {input_name: nchw_data_fp32})))
    save_img=False
    if save_img:
        res_out = res['output']*255
        res_out = res_out.clip(0,255).astype(np.uint8)
        res_out_img = res_out.transpose((0,2,3,1))
        res_out_img = np.squeeze(res_out_img,axis=0)
        cv2.imwrite("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/trt_jnd_data/jnd.png",res_out_img)
    res_final = res['graph_output_cast_0']
    res_bool =compare_float(res_final.flatten(),jnd_output_ori_data.flatten())
    print((res_final-jnd_output_ori_data).max())

    python_jnd_output_ori ,python_jnd_output_mul4= JND_w2c(input_float_model_path,out_model_path,mul)

    
    
    
    
    
    bool_res = compare_float(jnd_output_ori_data.flatten(),python_jnd_output_ori.flatten())
    print(np.abs(jnd_output_ori_data-python_jnd_output_ori).max())

  









    print("finished")
    
import onnx
from onnxsim import simplify



def modify_opset(model_path):
    import onnx
    from onnx import version_converter, helper

   
    original_model = onnx.load(model_path)

    print('The model before conversion:\n{}'.format(original_model))

    # A full list of supported adapters can be found here:
    # https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
    # Apply the version conversion on the original model
    converted_model = version_converter.convert_version(original_model, 9)

    print('The model after conversion:\n{}'.format(converted_model))

    onnx.save(converted_model,"./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/models/S1.0_sim_merged_opset9.onnx")

def simplify_test():
    model = onnx.load("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/models/S1.0_sim_plugin_allconv.onnx")
    
    index_conv=0
    index_mul=0
    index_sigmoid=0
    index_add = 0
    
    for node in model.graph.node:
        if node.op_type=='Conv':
            node.name = 'Conv_'+str(index_conv)
            index_conv+=1
        if node.op_type == 'Mul':
            node.name = 'Mul_'+str(index_mul)
            index_mul+=1
        if node.op_type == 'Sigmoid':
            node.name = 'Sigmoid_'+str(index_sigmoid)
            index_sigmoid+=1
        if node.op_type == 'Add':
            node.name = 'Add_'+str(index_add)
            index_add+=1
    onnx.save(model,"./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/models/S1.0_sim_plugin_allconv_name.onnx")
    print("finished")
        
        
        

    # model_simp, check = simplify(model)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp,"./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/models/sim_nsfw_S1_0.onnx")
  

   
 
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",required=True,help="input model")
    parser.add_argument("--output_model",required=True,help="input model")
    parser.add_argument("--mul",required=True,help="w2c mul",type=int,default=2)
    parser.add_argument("--input_path",default="./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/s1.jpg",help="input data path")
    
    parser.add_argument("--model_format",default='float')
    args = parser.parse_args()
    return args




    

def main():
    args = get_args()
    input_path = args.input_path
    input_float_model_path = args.input_model
    out_model_path = args.output_model
    # modify_opset("./onnxruntime/python/tools/quantization/E2E_example_model/image_classification/cpu/models/S1.0_sim_merged.onnx")
    simplify_test()
    mul=args.mul
   
    
    w2c_fucntion(input_float_model_path,out_model_path,mul)
 
   
if __name__ == "__main__":
    main()