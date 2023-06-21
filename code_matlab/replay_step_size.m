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
    fprintf('Simulated adaptation strength is %f\n', mbar(mi));
end
toc
%%
figure
plot(mbar,mean_speed,'Color','#009FB9','linewidth',1);
hold on
scatter(mbar, mean_speed, 50,'MarkerFaceColor', '#F18D00', 'MarkerEdgeColor', '#F18D00');
%shadedErrorBar(mbar,mean_speed,var_speed,'lineprops','-b','patchSaturation',0.33)

xticks(mbar(1:2:end));
yticks(linspace(min(mean_speed), max(mean_speed), 4));
ytickformat('%.2f');
xlabel('Adaptation strength (\times \tau_u/\tau_v)','FontName', 'Arial', 'FontSize', 10)
ylabel('Step size', 'FontName', 'Arial', 'FontSize', 10)

set(gca, 'LineWidth', 1.0);