function Angle_AP_view_pos()


load SubjectA_0002.tsv
X=SubjectA_0002;

% sample rate is 250 Hz

%data file is in 6 columns: x,y,z(top), x,y,z (bottom)

Xtot=[];
 
    x1=X(:,1);
    x2=X(:,2);
    x3=X(:,3);
    x4=X(:,4);
    x5=X(:,5);
    x6=X(:,6);
 
    min(x1)
    max(x1)
    abs(min(x1))+abs(max(x1))

for i=1:length(x1)
   ell_1(i)=sqrt((x1(i)-x4(i))^2+(x2(i)-x5(i))^2+(x3(i)-x6(i))^2);
   ang_sin_1(i)=sqrt((x1(i)-x4(i))^2+(x2(i)-x5(i))^2)/ell_1(i);
   ang_sin_x_1(i)=(x1(i)-x4(i))/ell_1(i);
   ang_sin_y_1(i)=(x2(i)-x5(i))/ell_1(i);
   ang_1(i)=asind(ang_sin_1(i));
   ang_x_1(i)=asind(ang_sin_x_1(i));
   ang_y_1(i)=asind(ang_sin_y_1(i));   
   dev(i)=sqrt((x1(i)-x4(i))^2+(x2(i)-x5(i))^2); 
 %  y(i)=i*0.004;
end

length(x1)
length(x4)
%for j=1:length(x1)
%    z(j)=sqrt(x1(j)^2 + x2(j)^2);
%    w(j)=sqrt(x4(j)^2 + x5(j)^2);
%end


% ang_x_1 is angle in the AP direction (I think)

%ang_x_1(1:10,1)
%plot(ang_x_1)
%axis([0,length(x1)*0.004,-10,10])

figure
%plot(x4)
subplot(2,1,1)
plot(x1,'k.')
%vline(9054)
axis([0 length(x1) -600 500])
subplot(2,1,2)
plot(ang_x_1,'k.')
vline(12300,'r-')
hline(20,'g-')
hline(11.37,'r-')
axis([0 length(x1) -20 25])

figure
plot(x3-x6,'k-')
vline(5928,'r--')
vline(660,'g--')

%var(ang_x_1(897:5094))

figure
plot(diff(diff(x4)), 'ko-')
hold on
plot(diff(diff(x5)),'ro-')
axis([0,length(x1),-1.0,1.0])
title('bottom of stick')
    
figure
plot(diff(diff(x1)), 'ko-')
hold on
plot(diff(diff(x2)),'ro-')
axis([0,length(x1),-1.0,1.0])
title('top of stick')
end 
    

