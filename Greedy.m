function [Elite_antlion_fitness,Elite_antlion_position,Convergence_curve]=Greedy(N,Max_iter,lb,ub,dim,Featur)
global antlion_position ant_position sorted_indexes
antlion_position=initialization(Featur,dim,ub,lb);
ant_position=initialization(Featur,dim,ub,lb);
Sorted_antlions=zeros(N,dim);
Elite_antlion_position=zeros(1,dim);
Elite_antlion_fitness=inf;
Convergence_curve=zeros(1,Max_iter);
antlions_fitness=zeros(1,N);
ants_fitness=zeros(1,N);

for i=1:size(antlion_position,1)
    antlions_fitness(1,i)=objectiveALO(antlion_position(i,:));
end

[sorted_antlion_fitness,sorted_indexes]=sort(antlions_fitness);

for newindex=1:N
    Sorted_antlions(newindex,:)=antlion_position(sorted_indexes(newindex),:);
end

Elite_antlion_position=Sorted_antlions(1,:);
Elite_antlion_fitness=sorted_antlion_fitness(1);

Current_iter=2;
while Current_iter<Max_iter+1
    
    for i=1:size(ant_position,1)
        Rolette_index=RouletteWheelSelection(1./sorted_antlion_fitness);
        if Rolette_index==-1
            Rolette_index=1;
        end
        RA=Random_walk_around_antlion(dim,Max_iter,lb,ub, Sorted_antlions(Rolette_index,:),Current_iter);
        [RE]=Random_walk_around_antlion(dim,Max_iter,lb,ub, Elite_antlion_position(1,:),Current_iter);
        
        ant_position(i,:)= (RA(Current_iter,:)+RE(Current_iter,:))/2; % Equation (2.13) in the paper
    end
    
    for i=1:size(ant_position,1)
        Flag4ub=ant_position(i,:)>ub;
        Flag4lb=ant_position(i,:)<lb;
        ant_position(i,:)=(ant_position(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        ants_fitness(1,i)=objectiveALO(ant_position(i,:));
        
    end
    double_population=[Sorted_antlions;ant_position];
    double_fitness=[sorted_antlion_fitness ants_fitness];
    
    [double_fitness_sorted I]=sort(double_fitness);
    double_sorted_population=double_population(I,:);
    
    antlions_fitness=double_fitness_sorted(1:N);
    Sorted_antlions=double_sorted_population(1:N,:);
    if antlions_fitness(1)<Elite_antlion_fitness
        Elite_antlion_position=Sorted_antlions(1,:);
        Elite_antlion_fitness=antlions_fitness(1);
    end
    Sorted_antlions(1,:)=Elite_antlion_position;
    antlions_fitness(1)=Elite_antlion_fitness;
    Convergence_curve(Current_iter)=Elite_antlion_fitness;
    if mod(Current_iter,50)==0
        display(['At iteration ', num2str(Current_iter), ' the elite fitness is ', num2str(Elite_antlion_fitness)]);
    end
    
    Current_iter=Current_iter+1;
end
end
function o=objectiveALO(x)
o=max(x.^2);
end
function X=initialization(Featur,dim,ub,lb)

Boundary_no= size(ub,2);
if Boundary_no==1
    X=rand(size(Featur,1),dim).*(ub-lb)+lb;
end
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        X(:,i)=rand(size(Featur,1),1).*(ub_i-lb_i)+lb_i;
    end
end
end

function choice = RouletteWheelSelection(weights)
accumulation = cumsum(weights);
p = rand() * accumulation(end);
chosen_index = -1;
for index = 1 : length(accumulation)
    if (accumulation(index) > p)
        chosen_index = index;
        break;
    end
end
choice = chosen_index;
end

function [RWs]=Random_walk_around_antlion(Dim,max_iter,lb, ub,antlion,current_iter)
if size(lb,1) ==1 && size(lb,2)==1
    lb=ones(1,Dim)*lb;
    ub=ones(1,Dim)*ub;
end

if size(lb,1) > size(lb,2)
    lb=lb';
    ub=ub';
end

I=1;
if current_iter>max_iter/10
    I=1+100*(current_iter/max_iter);
end

if current_iter>max_iter/2
    I=1+1000*(current_iter/max_iter);
end

if current_iter>max_iter*(3/4)
    I=1+10000*(current_iter/max_iter);
end

if current_iter>max_iter*(0.9)
    I=1+100000*(current_iter/max_iter);
end

if current_iter>max_iter*(0.95)
    I=1+1000000*(current_iter/max_iter);
end


lb=lb/(I);
ub=ub/(I);

if rand<0.5
    lb=lb+antlion;
else
    lb=-lb+antlion;
end

if rand>=0.5
    ub=ub+antlion;
else
    ub=-ub+antlion;
end

for i=1:Dim
    X = [0 cumsum(2*(rand(max_iter,1)>0.5)-1)'];
    a=min(X);
    b=max(X);
    c=lb(i);
    d=ub(i);
    X_norm=((X-a).*(d-c))./(b-a)+c;
    RWs(:,i)=X_norm;
end
end




