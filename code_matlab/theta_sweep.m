clc; clear; 
rng('default');

%% simulation theta sweeps
N = 512;
J0 = 1;%1/N*512;
a = 0.4;
tau = 3;
tau_v = 144;
dt = tau/10;
k = 5;%5/N*512;% global inhibition factor

%%%%%%%%%%%%%%%%%%Matrix Construction
J = zeros(N, 1);
Iext = zeros(N, 1);

% Map all neurons to [-pi, pi)
x=linspace(-pi, pi, N+1);
pos = x(1: N);
for i = 1: N
    dx = min(pos(i)-pos(1), pi-pos(i));
    % choose the weight matrix to be a 2D Gaussian Distribution
    J(i) = J0/(sqrt(2*pi)*a) * exp(-(dx^2)/(2*a^2));
end
Jfft = fft(J);
%%%%%%%%%%%%%%
alpha = 0.19;
% h = (151*tau/tau_v - m)/alpha;
h = 16.4;
pos = pos';
vbar = 0.5;
v=a/tau_v*vbar;
T = 4*pi/v;
tic
timestamp = 1;
j = 1;
noise_m = 0;
n_m = 11;
mbar = linspace(0,1,n_m);
amplitude = zeros(n_m,1);
omega= zeros(n_m,1);
for mi = 1:n_m
    m = mbar(mi)* tau/tau_v;
    U = zeros(N, 1);
    V = zeros(N, 1);
    r = zeros(N, 1);
    loc=-pi*5/8;
    t = 0;
    centerx_U = zeros(1,length(T/dt));
    centerx_I = zeros(1,length(T/dt));
    while t < T
        loc = loc + v * dt;
        if loc >= pi
            loc = loc - 2 * pi;
        end
        dis = min(abs(pos - loc), 2 * pi - abs(pos - loc));
        Iext = alpha  * exp(-(dis.^2/(4*a^2)));
        Irec=ifft(Jfft.*fft(r));
        dU = dt * (-U - V + Iext+Irec)/tau;
        U = U + dU;
        U = max(U, 0);
        m_rec = m+alpha*h;
        dV = dt * (-V + m_rec.*U) / tau_v;
        V = V + dV;
        r = U.^2./(1+k.*sum(U(:).^2));%.*(ratio)
        
        if t>T/2
            centerx_I(1,j)=loc;
            maxp = angle(exp(-1i.*pos)'*r);
            if loc > pi - 4*a && maxp < -pi+4*a
                maxp = maxp + 2 * pi;
            end
            
            if maxp > pi - 4*a && loc < -pi+4*a
                maxp = maxp - 2 * pi;
            end
            centerx_U(1,j)=maxp;
            
            j=j+1;
        end
        t = t + dt;
    end
    L_diff = centerx_U - centerx_I;
    [pks_pos,index] = findpeaks(L_diff);
    [pks_neg,~] = findpeaks(-L_diff);
    period = mean(diff(index))*dt*1e-3;
    omega(mi) = 1/period;
    amplitude(mi) = mean(pks_pos) - mean(pks_neg);
    fprintf('Simulated theta sweeps with adaptation strength is %f\n', mbar(mi));

end
toc

%% simulation replays
N = 512;
J0 = 1;%1/N*512;
a = 0.4;
tau = 3;
tau_v = 144;
dt = tau/10;
k = 5;%5/N*512;% global inhibition factor

%%%%%%%%%%%%%%%%%%Matrix Construction
J = zeros(N, 1);
Iext = zeros(N, 1);
% Map all neurons to [-pi, pi)
x=linspace(-pi, pi, N+1);
pos = x(1: N);
for i = 1: N
		dx = min(pos(i)-pos(1), pi-pos(i));
		% choose the weight matrix to be a 2D Gaussian Distribution
		J(i) = J0/(sqrt(2*pi)*a) * exp(-(dx^2)/(2*a^2));
end
Jfft = fft(J);
%%%%%%
alpha = 0.19;
h = 16.4;
pos = pos';
vbar = 0.5;
v=a/tau_v*vbar;
T = 20*pi/v;
loc=-pi*5/8;
n_m = 11;
mbar = linspace(0,1,n_m);
tic
for mi = 1:n_m
    U = zeros(N, 1);
    V = zeros(N, 1);
    r = zeros(N, 1);
    centerx_U = zeros(1,length(T/dt));
    centerx_I = zeros(1,length(T/dt));
    t = 0;
    r_t = zeros(N,length(T/dt));
    timestamp = 1;
    j = 1;
    m = mbar(mi)* tau/tau_v;
    while t < T
            loc = loc + v * dt;
        if loc >= pi
            loc = loc - 2 * pi;
        end
        dis = min(abs(pos - loc), 2 * pi - abs(pos - loc));
        if t < 500
            noise = 0.04*randn(N,1);
            alpha = 0.01;
        else
            noise = 0.04*randn(N,1);
            alpha = 0;
        end
        m_rec = m+0.02*randn(1);
        Iext = alpha  * exp(-(dis.^2/(4*a^2)));
        Irec=ifft(Jfft.*fft(r));%;
        dU = dt * (-U - V + Iext+Irec+noise)/tau;
        U = U + dU;
        U = max(U, 0);
        dV = dt * (-V + m_rec.*U) / tau_v;
        V = V + dV;
        r = U.^2./(1+k.*sum(U(:).^2));%.*(ratio)

        if t>T/2
            centerx_I(1,j)=loc;
            maxp = angle(exp(-1i.*pos)'*r);
            centerx_U(1,j)=maxp;
            j=j+1;
        end
        t = t + dt;
    end
    speed = abs(diff(centerx_U));
    speed(speed>pi) = 2*pi-speed(speed>pi);
    mean_speed(mi) = mean(speed);
    var_speed(mi) = var(speed);
    fprintf('Simulated replay with adaptation strength is %f\n', mbar(mi));
end
toc

%% 
% plot sweep amplitude and frequency against adaptation strength

% A4 paper size in inches
a4WidthInches = 8.27;
a4HeightInches = 11.69;

% Calculate the desired figure width
figureWidthInches = (2/3) * a4WidthInches;

% Create the figure with the specified width
figure('Units', 'inches', 'Position', [0 0 figureWidthInches figureWidthInches]);

subplot(3,1,1)
plot(mbar, amplitude,'-o', 'Color','#009FB9','linewidth',1);
hold on
scatter(mbar, amplitude, 50,'MarkerFaceColor', '#F18D00', 'MarkerEdgeColor', '#F18D00');
ylabel('Sweeps amp.', 'FontName', 'Arial', 'FontSize', 10)
set(gca, 'LineWidth', 1.0);
xticks([]);
yticks(linspace(min(amplitude), max(amplitude), 3));
ytickformat('%.2f');
box off;

subplot(3,1,2)
plot(mbar, omega,'-o', 'Color','#009FB9','linewidth',1);
hold on
scatter(mbar, omega, 50,'MarkerFaceColor', '#F18D00', 'MarkerEdgeColor', '#F18D00');

xticks([]);
yticks(linspace(min(omega), max(omega), 3));
ytickformat('%.2f');
ylabel('Sweeps freq. (Hz)', 'FontName', 'Arial', 'FontSize', 10)
set(gca, 'LineWidth', 1.0);
box off;

% plot replay step size
subplot(3,1,3)
plot(mbar,mean_speed,'Color','#009FB9','linewidth',1);
hold on
scatter(mbar, mean_speed, 50,'MarkerFaceColor', '#F18D00', 'MarkerEdgeColor', '#F18D00');
%shadedErrorBar(mbar,mean_speed,var_speed,'lineprops','-b','patchSaturation',0.33)

xticks(mbar(1:2:end));
yticks(linspace(min(mean_speed), max(mean_speed), 3));
ytickformat('%.2f');
xlabel('Adaptation strength (\times \tau_u/\tau_v)','FontName', 'Arial', 'FontSize', 10)
ylabel('Step size', 'FontName', 'Arial', 'FontSize', 10)

set(gca, 'LineWidth', 1.0);

box off;


set(gcf, 'defaultAxesLooseInset', [0 0 0 0.0])