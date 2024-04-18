pkg load control

% Función para simular el PID con los valores de Kp, Ki y Kd dados
function [theta, t] = simulate_PID(Kp, Ki, Kd)
    X = -[0; 0; 0];
    index = 0;
    h = 1e-7;
    ref = 1;
    simTime = 1e-3;
    samplingPeriod = h;
    A = ((2*Kp*samplingPeriod)+(Ki*(samplingPeriod^2))+(2*Kd))/(2*samplingPeriod);
    B = (-2*Kp*samplingPeriod+Ki*(samplingPeriod^2)-4*Kd)/(2*samplingPeriod);
    C = Kd/samplingPeriod;
    e = zeros(simTime/h,1);
    u = 0;
    theta = zeros(simTime/h,1);
    for t = 0:h:simTime
        index = index+1;
        k = index+2;
        X = modmotor(h,X,u);
        e(k) = ref-X(3);
        u = u+A*e(k)+B*e(k-1)+C*e(k-2);
        theta(index) = X(3);
    end
    t = 0:h:simTime;
    % Calcular el error en estado estacionario
    error_ss = (ref - theta(end))*100;
    disp(['Error en estado estacionario: ', num2str(error_ss)]);
end

% Define la función para la actualización del gráfico con los nuevos valores de Kp, Ki y Kd
function updatePlot(h, event, Kp_slider, Ki_slider, Kd_slider, Kp_label, Ki_label, Kd_label)
    Kp = get(Kp_slider, 'value');
    Ki = get(Ki_slider, 'value');
    Kd = get(Kd_slider, 'value');

    % Actualiza las etiquetas de los sliders
    set(Kp_label, 'string', ['Kp: ', num2str(Kp)]);
    set(Ki_label, 'string', ['Ki: ', num2str(Ki)]);
    set(Kd_label, 'string', ['Kd: ', num2str(Kd)]);

    % Código de simulación con los valores de Kp, Ki y Kd dados
    [theta, t] = simulate_PID(Kp, Ki, Kd);

    % Actualiza el gráfico con los nuevos valores de Kp, Ki y Kd
    plot(t, theta)
    title(['Salida Controlada por PID: K_p=', num2str(Kp), ', K_i=', num2str(Ki), ', K_D=', num2str(Kd)])
    xlabel('Tiempo [s]')
    ylabel('\theta [rad]')
    grid on
end

% Crea una nueva figura
figure;

% Crea sliders para ajustar los valores de Kp, Ki y Kd
Kp_slider = uicontrol('style', 'slider', 'min', 0.01, 'max', 1, 'value', 0.1, 'position', [100, 20, 120, 20]);
Ki_slider = uicontrol('style', 'slider', 'min', 0.001, 'max', 3, 'value', 0.1, 'position', [100, 50, 120, 20]);
Kd_slider = uicontrol('style', 'slider', 'min', 1, 'max', 10, 'value', 5, 'position', [100, 80, 120, 20]);

% Crea etiquetas para mostrar los valores actuales de Kp, Ki y Kd
Kp_label = uicontrol('style', 'text', 'position', [250, 20, 50, 20], 'string', ['Kp: ', num2str(get(Kp_slider, 'value'))]);
Ki_label = uicontrol('style', 'text', 'position', [250, 50, 50, 20], 'string', ['Ki: ', num2str(get(Ki_slider, 'value'))]);
Kd_label = uicontrol('style', 'text', 'position', [250, 80, 50, 20], 'string', ['Kd: ', num2str(get(Kd_slider, 'value'))]);

% Establece el callback de los sliders
set(Kp_slider, 'callback', {@updatePlot, Kp_slider, Ki_slider, Kd_slider, Kp_label, Ki_label, Kd_label});
set(Ki_slider, 'callback', {@updatePlot, Kp_slider, Ki_slider, Kd_slider, Kp_label, Ki_label, Kd_label});
set(Kd_slider, 'callback', {@updatePlot, Kp_slider, Ki_slider, Kd_slider, Kp_label, Ki_label, Kd_label});

% Ejecuta la función para simular el PID con los valores iniciales de Kp, Ki y Kd
[theta, t] = simulate_PID(0.1, 0.01, 5);



