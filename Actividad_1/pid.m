X = -[0; 0; 0];
index = 0;
h = 1e-7;
ref = 1;
simTime = 1e-3;
Kp=1
Ki=1.1
Kd=1.9
%Kp = 0.1;
%Ki = 0.01;
%Kd = 5;
samplingPeriod = h;
A = ((2*Kp*samplingPeriod)+(Ki*(samplingPeriod^2))+(2*Kd))/(2*samplingPeriod);
B = (-2*Kp*samplingPeriod+Ki*(samplingPeriod^2)-4*Kd)/(2*samplingPeriod);
C = Kd/samplingPeriod;
e = zeros(simTime/h,1);
u = 0;
theta = zeros(simTime/h,1); % Agrega esta línea para inicializar theta
for t = 0:h:simTime
    index = index+1;
    k = index+2;
    X = motorModel(h,X,u);
    e(k) = ref-X(3);
    u = u+A*e(k)+B*e(k-1)+C*e(k-2);
    theta(index) = X(3); % Almacena el valor de theta en cada paso de tiempo
end
t = 0:h:simTime;
plot(t,theta)
title('Salida Controlada por PID: K_p=0.1, K_i=0.01, K_D=5') 
ylabel('\theta [rad]') 
grid
