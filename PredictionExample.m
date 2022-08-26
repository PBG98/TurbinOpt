[XTrain,YTrain] = digitTrain4DArrayData;

layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm');

rng('default')
net = trainNetwork(XTrain,YTrain,layers,options);

[XTest,YTest] = digitTest4DArrayData;
YPred = predict(net,XTest);

YTest(1:10,:)

YPred(1:10,:)