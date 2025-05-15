function g_best=GAAlgorithm(fea)
pd=10;
AP=0.1;
fl=2;
N=size(fea,2);
[x l u]=init(N,pd);

xn=x;
mem=x;
ft=fitness(xn,N);


fit_mem=ft;

tmax=300;
for t=1:tmax

    num=ceil(N*rand(1,N));
    for i=1:N
        if rand>AP
            xnew(i,:)= x(i,:)+fl*rand*(mem(num(i),:)-x(i,:));
        else
            for j=1:pd
                xnew(i,j)=l-(l-u)*rand;
            end
        end
    end

    xn=xnew;
    ft=fitness(xn,N); 

    for i=1:N
        if xnew(i,:)>=l & xnew(i,:)<=u
            x(i,:)=xnew(i,:);
            if ft(i)<fit_mem(i)
                mem(i,:)=xnew(i,:);
                fit_mem(i)=ft(i);
            end
        end
    end

    ffit(t)=min(fit_mem);
%     min(fit_mem)
end

g_best=ffit(end);

end
function [x l u]=init(N,pd)

l=-100; u=100;

for i=1:N
    for j=1:pd
        x(i,j)=l-(l-u)*rand;
    end
end
end

function ft=fitness(xn,N)
for i=1:N
    u=mean(mean(xn));
    sigma_std=std(std(xn));
    ft(i)=max((1/sqrt(2*pi*sigma_std))*((xn(1,:)-u)/sigma_std));
end
end