This is a TensorFlow implementation of A Deep Learning Model of Coalbed Methane Production Prediction Considering Time, Space, and Geological Features.

# ð THANK YOU
        The paper was finally accepted, which was a very painful, bitter and dark process.
        Thank the world, thank Wenquxing, thank alprazolam tablets, and thank Oncomelanone hydrochloride tablets.
        Thank professors and all co-authors for their support.

        è®ºæç»äºè¢«æ¥æ¶äºï¼è¿æ¯ä¸ä¸ªæ æ¯çç¬ãè¦æ¶©ãæ¦æçè¿ç¨ã
        æè°¢å¨ä¸çï¼æè°¢ææ²æï¼æè°¢é¿æ®åä»çï¼æè°¢çé¸éèºç¯é®çã
        æè°¢ææä»¬ä¸ææåèèçæ¯æã
        æè°¢åæ®èµhxdmã


# The manuscript
#### A Deep Learning Model of Coalbed Methane Production Prediction Considering Time, Space, and Geological Features
#### https://doi.org/10.1016/j.cageo.2023.105312


#  The Code
## ð§¾ Requirements:
* Tensorflow(The best is tensorflow GPU = = 1.14.0)
* scipy
* numpy
* matplotlib
* pandas
* math

## ð Geological features
*  DTW_matrix.py

        You just need to run it
        Generate an adjacency matrix with spatial and geological features.
        The generated file is saved in "data \ testdata \ test_ohe. CSV"

* Spatial_matrix.py

        Generate an adjacency matrix with spatial features.
        The generated file is saved in "data \ testdata \ test_oh1. CSV"
## ð» T-DGCN model
* main.py

        You just need to run it.
        And provide the following parameter adjustment interface:
        * learning_rate
        * training_epoch
        * gru_units
        * seq_len
        * pre_len
        * train_rate
        * batch_size
        * dataset
        * model_name
* Out

        * See console for real-time parameters
        * See "/out" for output file 
        ð Due to the small amount of test data, the experimental results will be slightly different from the accuracy in this paper.
        


## ð Implement

    In the CBM dataset, we set the parametersï¼
    seq_len: 7 days 
    pre_len: 1, 3, 5, 7 days. 

## ð¾ Data Description
#### As the CBM data is classified and controlled by the state, the original data used in the paper is not included in this project.


    However, we prepared some test data to verify the usability of the code. 
    Due to the small amount of data, the performance of the model did not meet the expectations. 
    Please refer to the manuscript for the theoretical accuracy of the model. 
    In addition, for the sake of confidentiality, we have moderately transformed the data, so these data seem to be slightly different from common sense.
    ðSpecial note: the encryption method of test data does not affect the accuracy.


# ð Special Thanks
Special thanks to Zhao et al. For their research papers:<br>
T-GCN: A temporal graph convolutional network for traffic prediction
The manuscript can be visited at https://ieeexplore.ieee.org/document/8809901   or  https://arxiv.org/abs/1811.05320 

# ð§ e-mail
    1185702573@qq.com
    ts20010005a31tm@cumt.edu.cn

â¨
