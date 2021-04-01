
x = linspace(0,5,100);

f1 = sin(x);
f2 = 2*sin(2*x-1);
f3 = sin(x/4+4);
f4 = 0.5*sin(6*x+2);

f = f1+f2+f3+f4;
subplot(4,1,1)
plot(x,f3)
subplot(4,1,2)
plot(x,f1)
subplot(4,1,3)
plot(x,f2)
subplot(4,1,4)
plot(x,f4)
figure
plot(x,f)