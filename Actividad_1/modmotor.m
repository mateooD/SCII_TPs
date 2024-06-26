function [X] = motorModel(t, prevX, u)
  L = 4.7857;
  J = 4.9285e-13;
  R = 2;
  B = 9.8544e-8;
  K = 0.01896;
  Va = u;
  h = 1e-7;
  omega = prevX(1);
  wp = prevX(2);
  theta = prevX(3);
  for ii = 1:t/h
    wpp =(-wp*(R*J+L*B)-omega*(R*B+K*K)+Va*K)/(J*L);
    wp = wp+h*wpp;
    omega = omega + h*wp;
    thetap = omega;
    theta = theta + h*thetap;
  end
  X = [omega,wp,theta];
end
