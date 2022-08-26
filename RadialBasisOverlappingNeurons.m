P = -1:.1:1;
T = [-.9602 -.5770 -.0729  .3771  .6405  .6600  .4609 ...
      .1336 -.2013 -.4344 -.5000 -.3930 -.1647  .0988 ...
      .3072  .3960  .3449  .1816 -.0312 -.2189 -.3201];
plot(P,T,'+');
title('Training Vectors');
xlabel('Input Vector P');
ylabel('Target Vector T');

eg = 0.02; % sum-squared error goal
sc = 100;  % spread constant
net = newrb(P,T,eg,sc);

Y = net(P);
hold on;
plot(P,Y);
hold off;