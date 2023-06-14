% this code is to select cann cells from CA3 cells using weight fit and max
% firing rate

function [A,resnorm, y_fit] = select_cann(x,y,method)
if method ==1    
    % Define g(x) = a*exp(-(x-mu)^2/(2*sigma^2)):
    g = @(A,X) A(1)*exp(-(X-A(2)).^2/(2*A(3)^2));   
    % Cut Gaussian bell data
    ymax=max(y); xnew=[]; ynew=[];
    for n=1:length(x)
        if y(n)>0.05*ymax
            xnew=[xnew,x(n)]; 
            ynew=[ynew,y(n)]; 
        end
    end
    
    % Fitting
    ylog=log(ynew); xlog=xnew; B=polyfit(xlog,ylog,2);
    
    % Compute Parameters
    sigma=sqrt(-1/(2*B(1))); mu=B(2)*sigma^2; a=exp(B(3)+mu^2/(2*sigma^2)); 
    A=[a,mu,sigma]; 
    y_fit = g(A,x);
    resnorm = sum((y-y_fit).^2)/length(x);
elseif method ==2
    % Define g(x) = a*exp(-(x-mu)^2/(2*sigma^2)):
    g = @(A,X) A(1)*exp(-(X-A(2)).^2/(2*A(3)^2));
    A0 = [max(y),0,0.1];
    % Fit using Matlab's Least Squares function
    [A,resnorm] = lsqcurvefit(g,A0,x,y);
    y_fit = g(A,x);

elseif method ==3
%     circGauss = @(v,x) v(1).*exp(v(4)*(cos((x-v(3)))-1)) + v(2);   
%     circGauss = @(v,x) v(1).*exp(v(4)*(cos(x-v(3)))) + v(2);  
%     x = x(:);
%     y = y(:);
%     [~,init_phase] = max(y);
%     vinit(1) = max(y)-min(y); % Amplitude
%     vinit(2) = min(y);        % offset
%     vinit(3) = x(init_phase); % x-phase
%     vinit(4) = 0.5*pi;        % sigma of gaussian
%     
%     options = optimoptions('lsqnonlin', 'MaxIter', 500,...
% 	    'Algorithm', 'trust-region-reflective', 'Tolx', 10^-15, 'TolFun', 10^-15,...
% 	    'ScaleProblem', 'none', 'MaxFunEvals', length(x)*200, 'Display', 'off');
%     
%     ub = [abs(vinit(1)*2), max(y)*2, 2*pi, 2*pi];
%     lb = [0, 0, 0, -2*pi];
% 	    
%     initphz = (-180:30:180)*pi/180;
%     efunc = @(v) abs((circGauss(v,x) - y));
% 
%     for i = 1:length(initphz)
% 	    vinit_tmp = vinit;
% 	    vinit_tmp(3) = initphz(i);
% 	    [params{i}, f(i)] = ...
%           lsqnonlin(efunc, vinit_tmp, lb, ub,options);		
%     end
%     [~, o] = min(f);
%     A = params{o};
%     resnorm = f(o);
%     y_fit =  circGauss(A, x);

    
    circGauss = @(v,x) exp(v(1)*(cos(x-v(2))-1));
    x = x(:);
    y = y(:);
    [~,init_phase] = max(y);
    vinit(1) = x(init_phase); % x-phase
    vinit(2) = 0.5*pi;        % sigma of gaussian
    options = optimoptions('lsqnonlin', 'MaxIter', 500,...
	    'Algorithm', 'trust-region-reflective', 'Tolx', 10^-15, 'TolFun', 10^-15,...
	    'ScaleProblem', 'none', 'MaxFunEvals', length(x)*200, 'Display', 'off');
    
    ub = [100, pi];
    lb = [-100, -pi];
	    
    initphz = (-180:30:180)*pi/180;
    efunc = @(v) abs((circGauss(v,x) - y));

    for i = 1:length(initphz)
	    vinit_tmp = vinit;
	    vinit_tmp(2) = initphz(i);
	    [params{i}, f(i)] = ...
          lsqnonlin(efunc, vinit_tmp, lb, ub,options);		
    end
    [~, o] = min(f);
    A = params{o};
    resnorm = f(o);
    y_fit =  circGauss(A, x);

elseif method ==4
    f = fit(x.',y.','gauss1');
    y_fit = f.a1 .* exp(-((x-f.b1)/f.c1).^2);
    resnorm = sum((y- y_fit).^2)/length(y);
    A = [f.a1 f.b1 f.c1/2];
end
%     if plot_flag >0 % Plot fitting curve
%     % figure
%         scatter(x,y_fit,'r');hold on;
%         scatter(x,y,'b');hold off;
%         drawnow;
%     end

end

