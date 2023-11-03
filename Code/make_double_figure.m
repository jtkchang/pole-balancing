% pendulum-cart model with feedback delay and modep predcitvie (MP) control
% stab chart for linear model (without deadzone and saturation)
% stab chart for nonlinear model (with deadzone and saturation)
% obtained using the zeroth-order semi discretization
% P1, D1: control gains for the angular position of the stick
% P2, D2: control gains for the location of the pivot point (cart)
% DV(1): deadzone for angular position of the pendulum
% DV(2): deadzone for position of the pivot point
% DV(3): deadzone for angular velocity of the pendulum
% DV(4): deadzone for velocity of the pivot point

clear
load FS_Milton0100c.dat
X=FS_Milton0100c;

Xtot=[];
 
    x1=X(:,1);
    x2=X(:,2);
    x3=X(:,3);
    x4=X(:,4);
    x5=X(:,5);
    x6=X(:,6);
    
   
for i=55:length(x1)
   tt(i)=i+1;
   ell_1(i)=sqrt((x1(i)-x4(i))^2+(x2(i)-x5(i))^2+(x3(i)-x6(i))^2);
   ang_sin_1(i)=sqrt((x1(i)-x4(i))^2+(x2(i)-x5(i))^2)/ell_1(i);
   ang_sin_x_1(i)=(x1(i)-x4(i))/ell_1(i);
   ang_sin_y_1(i)=(x2(i)-x5(i))/ell_1(i);
   ang_1(i)=asind(ang_sin_1(i));
   ang_x_1(i)=asind(ang_sin_x_1(i));
   dang_x(i)=asind(ang_sin_x_1(i-54));
   ang_y_1(i)=asind(ang_sin_y_1(i));   
   dev(i)=sqrt((x1(i)-x4(i))^2+(x2(i)-x5(i))^2); 
   ext(i)=sqrt((x4(i)-x4(1))^2+(x5(i)-x5(1))^2+(x6(i)-x6(1))^2);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parameters to give
L = 0.56;       % [m] stick length
m = 0.0247*L;     % mass of stick [kg]
m0 = 1.2;        % mass of the cart (hand) [kg]
c = L/2;    % location of the cente of gravity [m]
dt = 0.01;   % discretization time step [s]
r = 0.23/dt;       % discrete delay, tau = r*dt
g = 9.81;    % grav. acc. [m/s2]
tmax = 300;   % [s]
acc_max = 50;  % max acceleration of the fingertip m/s^2
jerk_max = 600; % max jerk of the fingertip m/s^2
x_check_lim = 0.31;   % [m]
%x_check_lim=0.43;
phi_check_lim = 10;  % deg John: 10deg or 14deg
%phi_check_lim=20;
angle=0.464;

% control gains
P1 = 120;
%P1=240;
D1 = 28;
%D1=30;
P2 = 14;
%P2=13;
D2 = 19;
%D2=30;

% deadzones DV=[angle, position, ang.velocity, velocity]
% in [deg, m, deg/s, m/s]
% no deadzone for the efferent copies
DVdeg = [1.75; 0; 0; 0];
% DV in rad:
DV = [DVdeg(1)*pi/180; DVdeg(2); DVdeg(3)*pi/180; DVdeg(4)];

% initial conditons
phi0deg = angle;  % [deg]
dphi0deg = 0;   % [deg/s]
x0 = 0;      % [m]
dx0 = 0;     % [m/s]
phi0 = phi0deg/180*pi;  % [rad]
dphi0 = dphi0deg/180*pi;  % [rad/s]

% end of parameters to give
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters, system matrices for semi-discreization
Ic = m*L^2/12;
M = [[m*c, m+m0]; [Ic+m*c^2, m*c]];
K = [[0, 0]; [-m*g*c, 0]];
invM = inv(M);
B = [0; 0; invM*[1; 0]];
A = [[zeros(2), eye(2)]; [-invM*K, zeros(2)]];
P = expm(A*dt);
A_int = @(s)expm(A*(dt-s));
R = quadv(A_int,0,dt);
Phi = zeros(4+r) + diag(ones(r+3,1),-1);
Phi(1:4,1:4) = P;
Phi(1:4,4+r) = R*B;
D = [P1, P2, D1, D2];
S = D*P^r;
Phi(5,1:4) = S;
for j = 1:r
    V(j) = D*P^(j-1)*R*B;
    Phi(5,4+j) = V(j);
end

% max control force
m0corr = 1/([0,1]*inv(M)*[1;0]);
% u_max = acc_max*m0corr;   % from M*x"=u
% u_max = acc_max*m0;   % simplified calculation from m0*a=F

% simulation
% initial (r+4)-dimensional vector for simulation

X = zeros(r+4,1);
X(1:4) = [phi0; x0; dphi0; dx0];
x_check = 0;
phi_check = 0;
u_prev = 0;
ix = 0;
iphi = 0;
for i = 1:(tmax/dt)
    t = (i-1)*dt;
    phi = X(1)*180/pi;
    dphi = X(3)*180/pi;
    x = X(2);
    dx = X(4);
    % with matrix multiplication
    %     X0 = X;
    %     X = Phi*X0;
    % without matrix multiplication
    X0 = X;
    DX0 = X0(1:4).*(abs(X0(1:4))>DV); % set to zero if within deadzone
    X(1:4) = P*X0(1:4) + R*B*X0(r+4);
    % control force
    u = S*DX0(1:4) + V*X0(5:(4+r));
    if abs(u)/m0corr < acc_max   % limitation by max acceleration
        ua = u;
    else
        ua = sign(u)*acc_max*m0corr;
    end
    if abs(ua-u_prev)/m0corr/dt < jerk_max   % limitation by max jerk
        uaj = ua;
    else
        uaj = u_prev + sign(ua-u_prev)*jerk_max*m0corr*dt;
    end
    %         % if there is no limitation in acceleration and jerk
    %         ua = u;
    %         uaj = ua;
    X(5) = uaj;
    u_prev = uaj;
    X(6:(r+4)) = X0(5:(r+3));

% Note that X is the actual augmented state vector and X0 is the state
% vector in the previous step.

% x_check_lim limitation is effective is x=X(2) reaches x_check_lim AND the
% fingertip velocity v=x'=X(4) has the same sign as x (the sticks falls
% further). This is expressed by:

    if (abs(X(2)) >= x_check_lim)*(X(4)*X(2)>0)
        ix = ix+1;
        tx_out(ix)=t;
        x_out(ix)=x;
    % If x and the force has the same sign (either the fingertip 
    % is closest to the person and the force is towards the person, 
    % or the fingertip furthest from the person and the force is away 
    % from the person), then we have free (open loop) dynamics since 
    % no control force can be exerted any more:
        if X0(2)*X0(r+4) > 0
            X(1:4) = P*X0(1:4);
        end
    % At the time instant when x reaches x_check_lim, the stick's bottom
    % cannot move further, it is a sudden fixation (almost like an impact). 
    % The stick's angular velocity changes suddenly. The angular velocity
    % after the fixation can be calculated by the concept that the angularm
    % momentum about the fixation point (stick's bottom) is preserved
    % during fixation:
        if abs(X0(2)) < x_check_lim
            X(3) = (c*X0(3)+X0(4)*cos(X0(1)))/c;  % new angular velocity
        end
        X(2) = sign(X0(2))*x_check_lim; % x is fixed to +/-x_check_lim
        X(4) = 0;   % fingertip velocity is set to zero
    % This will plot the actual control force including the case when it
    % becomes zero due to armlength limitation:
        if X0(2)*X0(r+4) > 0    % 
            % uv(i) = heaviside(-X0(2)*X0(r+4),0)*X0(r+4);
            uv(i) = 0;
        end
    else
        uv(i) = X0(r+4);
    end
    if abs(X(1)*180/pi) >= phi_check_lim   % check if theta>20deg
        iphi = iphi+1;
        tphi_out(iphi)=t;
        phi_out(iphi)=phi;
        X(6:(r+4)) = zeros(r-1,1);
    end

    x_check = abs(X(2));
    phi_check = abs(X(1)*180/pi);
    tv(i) = t;
    phiv(i) = phi;
    dphiv(i) = dphi;
    xv(i) = x;
    dxv(i) = dx;    % acceleration by discrete derivation
%     uv(i) = uaj;
    if phi_check > phi_check_lim  % simulation is terminated if theta>45deg
        break
    end
end
figure

subplot(2,1,1)
plot(tt/250,ang_x_1,'k.')
%plot(t,x4,'ko-')
subtitle('HUMAN POLE BALANCING')
%axis([27 37 -12 12])
axis([27 37 -10 25])
% hline(10,'k--')
line([36.12 36.12], [10 15],'color','k')
%hold on
%axis([1 38 -12 12])
%hline(10,'k--')
%hline(-10,'k--')
%ylabel('\theta [deg]')
%xlabel('TIME [s]')
%xline(35.12,'k--','falling','LineWidth',2)
%xline(33.84,'k--','forecasting','LineWidth',2)
%xline(33.16,'k--','prediction','LineWidth',2)
%hline(1.74,'k--')
%hline(-1.74,'k--')
ylabel('\theta [deg]')
%xlabel('TIME [s]')
text(35.9,18,'LCT')

subplot(2,1,2)
plot(tv,phiv,'k.')
%hold on
%plot(tt/250,ang_x_1,'r.')
subtitle('SYNTHETIC POLE BALANCING')
hold on
ylabel('\theta [deg]')
xlabel('TIME [s]')
%axis([27 37 -12 12])
%hline(1.74,'k--')
%hline(-1.74,'k--')
axis([27 37 -10 20])
%hline(-10,'k--')
%hline(10,'k--')

