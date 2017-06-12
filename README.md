## Inertial-Gesture-Recognition

________________
### Introduction
Mobile phone plays an important role in our daily life. This paper develops a gesture recognition benchmark based on sensors of mobile phone. The built-in micro gyroscope and accelerometer of mobile phone can efficiently measure the accelerations and angular velocities along x-,y- and z-axis, which are used as the input data. We calculate the energy of the input data to reduce the effect of the phone’s posture variations. A large database is collected, which contains more than 1,000 samples of 8 gestures. The Hidden Markov Model (HMM), K-NearestNeighbor (KNN) and Support Vector Machine (SVM) are tested on the benchmark. The experimental results indicated that the employed methods can effectively recognize the gestures. To promote research on this topic, the source code and database are made available to the public.

________________
### Instructions

1. git clone https://github.com/JasonZhao001/Inertial-Gesture-Recognition

2. modify the datafile to your specific path that point to the "struc.mat" data

3. run the following commond in the terminal at the project root path:

   (1) train: 
   
    $ python LSTM.py -t False
    
   (2) test only:
   
    $ python LSTM.py -t True
    
   ** one thing to note is that the test result will automatically printed after training
   
   ** the argparse is added for your performing the test process straightly.
   
##### Notes:

1. The data is divided into 120*8 training set and 20*8 testing set.

2. The model size is only  863.6 kB and run in real time, so it can be easily built in the cell phone. 

3. Result: test accuracy: 100.00% 


________________
### Notes

The only file that I created is LSTM.py, others are the files from the original repo of Professor Baochang Zhang.

To learn more please visit https://github.com/bczhangbczhang/Inertial-Gesture-Recognition

If you use the database, please cite this paper

Chunyu Xie, Shangzhen Luan, Hainan Wang, Baochang Zhang: Gesture Recognition Benchmark Based on Mobile Phone. CCBR 2016: 432-440


________________
### Contact

Baochang Zhang

bczhang@buaa.edu.cn

