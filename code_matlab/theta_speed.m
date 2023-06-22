ccc
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
h = 17;
pos = pos';
loc=-pi*5/8;

draw = 0;
save_video = 0;
save_video = draw*save_video;
if save_video == 1
    %%%%%%%%%%%%%%%Video initialization
    filename = ['videos\Adaptive_SFA_',num2str(alpha),'.mp4'];
    myVideo = VideoWriter(filename, 'MPEG-4'); 
    myVideo.FrameRate = 40; 
    open(myVideo); 
end
noise_m = 0;
m_record = load('m_record.mat').m_record;
tic
n_simu = 10;
for vi = 1:n_simu
    U = zeros(N, 1);
    V = zeros(N, 1);
    r = zeros(N, 1);
    vbar = 0.5 + randn(1)*0.1;
    v=a/tau_v*vbar;
    v_record(vi) = v;
    T = 8800;
    centerx_U = zeros(1,length(T/dt));
    centerx_I = zeros(1,length(T/dt));
    t = 0;
    r_t = zeros(N,length(T/dt));
    timestamp = 1;
    j = 1;
    m = m_record(vi);

    while t < T
            loc = loc + v * dt;
        if loc >= pi
            loc = loc - 2 * pi;
        end
        dis = min(abs(pos - loc), 2 * pi - abs(pos - loc));
        Iext = alpha  * exp(-(dis.^2/(4*a^2)));
        Irec=ifft(Jfft.*fft(r));%;
        noise = 0.0*randn(N,1);
        dU = dt * (-U - V + Iext+Irec+noise)/tau;
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

            if rem(floor(t/dt),30) == 0 && t<T*0.75 && draw == 1
                plot(pos,Iext,'r','linewidth',2),hold on
                plot(pos,U,'b','linewidth',2),hold off
                axis([-pi pi 0 0.2])
                xlabel('x')
                ylabel('neural activities')
                set(gcf,'unit','centimeters','position',[20,10,20,13])
                set(gca,'linewidth',3,'fontsize',15,'fontname','Cambria Math');
                drawnow
                if save_video == 1
                    frame = getframe(gcf);
                    im = frame2im(frame); 
                    writeVideo(myVideo,im); 
                end
            end
            j=j+1;
        end
        t = t + dt;
    end
    if save_video == 1
        close(myVideo)
    end
    % m_rec*tau_v/tau
    toc
    speed = abs(diff(centerx_U));
    speed(speed>pi) = 2*pi-speed(speed>pi);
    Speed(vi,:) = speed;
    disp(vi/n_simu)
end
save('speed_theta.mat','Speed')
save('v_record.mat','v_record')

