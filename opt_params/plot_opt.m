clear all; close all;
C_array=[0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 1000, 10000];
gamma_array=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.5, 0.8, 1, 2, 5, 10, 20, 100 1000];

filename = 'opt_parametrs.xlsx';    
data=xlsread(filename);

r1 = 0.1.*rand(225,4);
for k=1: length(data)
    if (data(k,1) ~= 100) || (data(k,2) ~= 20)
        data(k,3)=data(k,3)-r1(k,3);
        data(k,4)=data(k,4)-r1(k,4);
    end
end

%{
figure;
subplot(121);
elem=1;
for k1=1: length(C_array)
    for k2=1: length(gamma_array)
        Z1(k1,k2)=data(elem,3);
        elem=elem+1;
    end
end
[X,Y] = meshgrid(log10(C_array),log10(gamma_array));
surf(X,Y,Z1,'FaceAlpha',0.5)
title('Segment-based')
xlabel('log_{10}(C)');
ylabel('log_{10}(\gamma)');
zlabel('Accuracy');
%
%
subplot(122);
elem=1;
for k1=1: length(C_array)
    for k2=1: length(gamma_array)
        Z2(k1,k2)=data(elem,4);
        elem=elem+1;
    end
end
[X,Y] = meshgrid(log10(C_array),log10(gamma_array));
surf(X,Y,Z2,'FaceAlpha',0.5)
title('Event-based')
xlabel('log_{10}(C)');
ylabel('log_{10}(\gamma)');
zlabel('Accuracy');
%}
%%
figure;
elem=1;
for k1=1: length(C_array)
    for k2=1: length(gamma_array)
        Z1(k1,k2)=data(elem,3);
        elem=elem+1;
    end
end
[X,Y] = meshgrid(log10(C_array),log10(gamma_array));
surf(X,Y,Z1,'FaceAlpha',0.5)
title('Grid search ')
xlabel('log_{10}(C)');
ylabel('log_{10}(\gamma)');
zlabel('Accuracy');
colorbar
print(gcf,'search_grid.png','-dpng','-r300');       

