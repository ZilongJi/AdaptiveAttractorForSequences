ccc
N = 512;
vbar = 0;
mbar = 0.5;
k0 = 1;
J0 = 1;
%%%%%%%%%%%%%%%%Parameters Settings
J0=J0/N*512;
a =0.4;
tau = 1;
tau_v =48;
dt = tau/10;
m = mbar* tau/tau_v;
k=k0/N*512;% global inhibition factor

%%%%%%%%%%%%%%%%%%Matrix Construction
J = zeros(N, 1);
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
alpha=0.1;
pos = pos';
v=a/tau_v*vbar;
T=210;
loc=0;
sigma = 0:0.015:0.15;
% sigma = 0.2;
n = length(sigma);
tic
for sigma_i = 1:n
    Iext = zeros(N, 1);
    U = zeros(N, 1);
    V = zeros(N, 1);
    r = zeros(N, 1);
    t = 0;
    j = 1;
    while t < T
        dis = min(abs(pos - loc), 2 * pi - abs(pos - loc));
        Irec=ifft(Jfft.*fft(r));%;
        noise = sigma(sigma_i)*randn(N,1);
        if t < 200
            Iext = alpha  * exp(-(dis.^2/(4*a^2)));
        else
            Iext = 0;
        end

        dU = dt * (-U - V + Iext+Irec+noise)/tau;
        U = U + dU;
        dV = dt * (-V + m.*U) / tau_v;
        V = V + dV;
        U = max(U, 0);
        r = U.^2./(1+k.*sum(U(:).^2));%.*(ratio)
%         plot(pos,r),hold on
%         plot(pos,Iext*0.1),hold off
%         drawnow
        j=j+1;
        t = t + dt;
    %     disp(t/T)
    end
    [A,resnorm, y_fit] = select_cann(pos,r,1);
    amplitude = A(1);
    mu = A(2);
    sigma_fit(sigma_i) = A(3);
    loss_fit(sigma_i) = resnorm;
end
toc
plot(pos,r,'b','linewidth',2),hold on
plot(pos,y_fit,'r','linewidth',2)
legend('Simulated neural data','fit results')
xlabel('pos')
ylabel('neural activities')
% ylim([0 1e-6])
box off
set(gca,'linewidth',2,'fontsize',25,'fontname','Arial');
set(gcf,'unit','centimeters','position',[25,17,20,15])
A(3)
figure
plot(sigma,sigma_fit,'linewidth',2),hold on
plot(sigma,sigma_fit,'r.','markersize',20)
xlabel('noise level')
ylabel('bump width')
ylim([0.3 0.5])
box off
set(gca,'linewidth',2,'fontsize',25,'fontname','Arial');
set(gcf,'unit','centimeters','position',[25,17,20,15])
figure
plot(sigma,loss_fit,'linewidth',2),hold on
plot(sigma,loss_fit,'r.','markersize',20)
xlabel('noise level')
ylabel('Resnorm')
ylim([0 1e-6])
box off
set(gca,'linewidth',2,'fontsize',25,'fontname','Arial');
set(gcf,'unit','centimeters','position',[25,17,20,15])


