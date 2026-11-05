%------------------------------------------------------------------
% This script illustrates the convergence analysis of Bregman proximal gradient
% methods (BPGM) presented in [1], on the problem
%
%      min_{x >= 0}  KL(y; Ax) + lamb * ||x||_1
%
% where KL denotes the generalized Kullback-Leibler divergence and
% lamb > 0 is the regularization parameter.
%
% Four algorithms are compared:
%   - BPGM with the Burg entropy as kernel (generating function),
%   - BPGM with a smoothed Burg entropy as kernel,
%   - PGM, i.e., BPGM with the squared Euclidean norm as kernel,
%   - Richardson-Lucy.
%
% The solution of the problem can be chosen to lie either on the boundary
% or in the interior of the nonnegative orthant. The two cases lead to
% different convergence rates, as discussed in [1].
%
% For illustration purposes, small simulated problems are considered,
% with a controlled conditioning of A.
% 
% [1] On The Linear Convergence of Bregman Proximal Gradient Methods with Applications to Kullback-Leibler Regression.
% Jonathan Chirinos-Rodriguez, Christian Daniele, Cédric Févotte, and Emmanuel Soubies.
%
% -- 2026
%------------------------------------------------------------------

clear; close all; warning off; rng(1)

%% Parameters
N = 100;           % Problem size
M = 50;
Q = 100;           % Average number of counts per entry in the data   
b = 1e-3;          % Background level 
K = 10^5;          % Number of iterations
lamb=5e-2;         % Regularization parameter 
solint = false;    % To force the solution being in the interor of the domain (only possible for lamb sufficiently small)

%% Problem Generation
% -- Generation of forward matrix A
A = rand(M,N); 
[U,S,V] = svd(A);
ss = diag(S);
s_max = max(ss(:));
s_min = min(ss(:));
cond_target = 50;
ss =  s_max - (s_max - s_max / cond_target) * (s_max - ss) / (s_max - s_min);
DD = zeros(size(A)); DD(:,1:M)=diag(ss);
A = (U*DD*V'); A = A - min(A(:)); 
disp(['Conditioning of generated A : ',num2str(cond(A))]);
A=A/norm(A);

% -- Generation of ground truth vector xt and measurements y
if solint
    % To have bar x in int(C) (for lamb sufficiently small)
    xt = Q*ones(N,1);
    yoiseless = A*xt + b;
    y = yoiseless/Q;
else
    % To have bar x in dC
    xt = zeros(N,1);
    xt(randperm(N,N)) = rand(1,N)*Q;
    y = random('Poisson',A*xt + b)/Q;
end

% -- Objective function and gradient
F = @(x) sum(A*x + b - y.*log(A*x+b));
gF = @(x) A'*(1 - y./(A*x+b));
J = @(x) F(x) + lamb*sum(x);

% -- Init point 
x0 = rand(N,1)*5*10^4;

%% Optimization  
disp('Run BPGM with SBurg ...')
Fevol_sB4 = zeros(1,K);
xi=0.6;
x =  x0;
Fevol_sB4(1) = J(x);
tau = 1;
beta =0.9;
Dphi = @(x,y) sum(log1p((y-x)./(x+xi))) + sum((x-y)./(y+xi));
tauth = 1 / sum(max((A*ones(N,1)*xi)./b,1).^2.*y);
tic;
for k = 2:K   
    gFx = gF(x);
    x_new = max((x+xi)./(1 + (gFx+lamb)*tau.*(x+xi))-xi,0);
    Axnew = A*x_new;
    Ax = A*x;
    while sum(y.*((Axnew-Ax)./(Ax+b)-log1p((Axnew -Ax)./(Ax + b)))) >  Dphi(x_new,x)/ tau
        tau = tau * beta;
        x_new = max((x+xi)./(1 + (gFx+lamb)*tau.*(x+xi))-xi,0);
        Axnew = A*x_new;
        if tau < tauth
            disp('Step-size below global theoretical one')
        end
    end
    x = x_new;
    Fevol_sB4(k) = J(x);
end
timesB = toc;
disp(['   NNZ sBurg sol:',num2str(nnz(x))])

% L2 geometry
disp('Run PGM (BPGM with L2) ...')
Fevol_L2 = zeros(1,K);
x = x0;
tau = 1;
Fevol_L2(1) = J(x);
L_l2 = (max(y) / b^2);
tic;
for k = 2:K
    tmp= x - tau*gF(x);
    x_new = max(sign(tmp).*max(abs(tmp) - tau*lamb,0),0);
    Axnew = A*x_new;
    Ax = A*x;
    while sum(y.*((Axnew-Ax)./(Ax+b)-log1p((Axnew -Ax)./(Ax + b)))) >  norm(x_new-x)/ tau
        tau = tau * beta;
        tmp= x - tau*gF(x);
        x_new = max(sign(tmp).*max(abs(tmp) - tau*lamb,0),0);
        Axnew = A*x_new;
    end
    x = x_new;
    Fevol_L2(k) = J(x);
end
timeL2 = toc;

% Burg geometry
disp('Run BPGM with Burg ...')
Fevol_B = zeros(1,K);
x =  x0;
Fevol_B(1) = J(x);
tau = 1;
beta =0.99;
Dphi = @(x,y) sum(log1p((y./x)-1) + x./y - 1);
tic;
for k = 2:K   
    gFx = gF(x);
    x_new = max(x./(1 + (gFx+lamb)*tau.*x),0);
    Axnew = A*x_new;
    Ax = A*x;
    while sum(y.*((Axnew-Ax)./(Ax+b)-log1p((Axnew -Ax)./(Ax + b)))) / Dphi(x_new,x)* tau >1 || any(x_new<eps)
        tau = tau * beta;
        x_new = max(x./(1 + (gFx+lamb)*tau.*x),0);
        Axnew = A*x_new;
    end
    x = x_new;
    Fevol_B(k) = J(x);
end
timeB=toc;

% -- RL
disp('Run RL ...')
Fevol_RL = zeros(1,K);
x =  x0;
Fevol_RL(1) = J(x);
dem = A'*ones(M,1)+lamb;
tic;
for k = 2:K   
    x =A'*(y./(A*x + b)).*x./dem;
    Fevol_RL(k) = J(x);
end
timeRL = toc;

%% Plots
% Plot objective decrease in terms of iterations
figure; hold all;
minval=min([Fevol_L2(end),Fevol_sB4(end),Fevol_B(end),Fevol_RL(end)]);
semilogy(Fevol_L2 - minval,'linewidth',2);
semilogy(Fevol_B - minval,'linewidth',2);
semilogy(Fevol_sB4 - minval,'linewidth',2);
semilogy(Fevol_RL - minval,'linewidth',2);
legend('L2','Burg','s-Burg','RL');
xlabel('Iterations');
grid; set(gca,'fontsize',16,'YScale','log');

% Plot objective decrease in terms of elapsed time
figure; hold all;
minval=min([Fevol_L2(end),Fevol_sB4(end),Fevol_B(end),Fevol_RL(end)]);
tt = linspace(0,1,K);
semilogy(tt*timeL2,Fevol_L2 - minval,'linewidth',2);
semilogy(tt*timeB,Fevol_B - minval,'linewidth',2);
semilogy(tt*timesB,Fevol_sB4 - minval,'linewidth',2);
semilogy(tt*timeRL,Fevol_RL - minval,'linewidth',2);
legend('L2','Burg','s-Burg','RL');
xlabel('Time (s)');
grid; set(gca,'fontsize',16,'YScale','log');


