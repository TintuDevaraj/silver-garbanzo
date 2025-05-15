
function [BestChrom_Fitness1,BestChrom_Gene]  = GeneticAlgorithm (M , N, MaxGen , Pc, Pm , Er , visuailzation,FeaturesFile)
global MinIdx
cgcurve = zeros(1 , MaxGen);

[ population ] = initialization(M,FeaturesFile(:,1:N));
for i = 1 : M
    population.Chromosomes(i).fitness = objectiveGA( population.Chromosomes(i).Gene(:) );
end

g = 1;
disp(['Generation #' , num2str(g)]);
[max_val , indx] = sort([ population.Chromosomes(:).fitness ] , 'descend');
cgcurve(g) = population.Chromosomes(indx(1)).fitness;

for g = 2 : MaxGen
    disp(['Generation #' , num2str(g)]);
    for i = 1 : M
        population.Chromosomes(i).fitness = objectiveGA( population.Chromosomes(i).Gene(:) );
    end
    
    for k = 1: 2: M
        [ parent1, parent2] = selection(population);
        
        [child1 , child2] = crossover(parent1 , parent2, Pc, 'single');
        
        [child1] = mutation(child1, Pm);
        [child2] = mutation(child2, Pm);
        
        newPopulation.Chromosomes(k).Gene = child1.Gene;
        newPopulation.Chromosomes(k+1).Gene = child2.Gene;
    end
    
    for i = 1 : M
        newPopulation.Chromosomes(i).fitness = objectiveGA( newPopulation.Chromosomes(i).Gene(:) );
    end
    [ newPopulation ] = elitism(population, newPopulation, Er);
    
    cgcurve(g) = newPopulation.Chromosomes(1).fitness;
    
    population = newPopulation;
end
for ii=1:length(population.Chromosomes)
    BestChrom_Gene(ii,:)    = population.Chromosomes(ii).Gene;
    BestChrom_Fitness(ii) = population.Chromosomes(ii).fitness;
end
[minVal,MinIdx]=min(BestChrom_Fitness);
BestChrom_Fitness1=BestChrom_Fitness(MinIdx);
BestChrom_Gene1=BestChrom_Gene(MinIdx(1),:);
end

function  [fitness_value] = objectiveGA( X )
fitness_value = min(min(X-mean(X))/(X-std(X)));
end

function [parent1, parent2] = selection(population)

M = length(population.Chromosomes(:));

if any([population.Chromosomes(:).fitness] < 0 )
    a = 1;
    b = abs( min(  [population.Chromosomes(:).fitness] )  );
    Scaled_fitness = a *  [population.Chromosomes(:).fitness] + b;
    
    normalized_fitness = [Scaled_fitness] ./ sum([Scaled_fitness]);
else
    normalized_fitness = [population.Chromosomes(:).fitness] ./ sum([population.Chromosomes(:).fitness]);
end
[sorted_fintness_values , sorted_idx] = sort(normalized_fitness , 'descend');

for i = 1 : length(population.Chromosomes)
    temp_population.Chromosomes(i).Gene = population.Chromosomes(sorted_idx(i)).Gene;
    temp_population.Chromosomes(i).fitness = population.Chromosomes(sorted_idx(i)).fitness;
    temp_population.Chromosomes(i).normalized_fitness = normalized_fitness(sorted_idx(i));
end


cumsum = zeros(1 , M);

for i = 1 : M
    for j = i : M
        cumsum(i) = cumsum(i) + temp_population.Chromosomes(j).normalized_fitness;
    end
end


R = rand();
parent1_idx = M;
for i = 1: length(cumsum)
    if R > cumsum(i)
        parent1_idx = i - 1;
        break;
    end
end

parent2_idx = parent1_idx;
while_loop_stop = 0;
while parent2_idx == parent1_idx
    while_loop_stop = while_loop_stop + 1;
    R = rand();
    if while_loop_stop > 20
        break;
    end
    for i = 1: length(cumsum)
        if R > cumsum(i)
            parent2_idx = i - 1;
            break;
        end
    end
end

parent1 = temp_population.Chromosomes(parent1_idx);
parent2 = temp_population.Chromosomes(parent2_idx);

end

function [child1 , child2] = crossover(parent1 , parent2, Pc, crossoverName)
switch crossoverName
    case 'single'
        Gene_no = length(parent1.Gene);
        ub = Gene_no - 1;
        lb = 1;
        Cross_P = round (  (ub - lb) *rand() + lb  );
        
        Part1 = parent1.Gene(1:Cross_P);
        Part2 = parent2.Gene(Cross_P + 1 : Gene_no);
        child1.Gene = [Part1, Part2];
        
        Part1 = parent2.Gene(1:Cross_P);
        Part2 = parent1.Gene(Cross_P + 1 : Gene_no);
        child2.Gene = [Part1, Part2];
    case 'double'
        Gene_no = length(parent1);
        
        ub = length(parent1.Gene) - 1;
        lb = 1;
        Cross_P1 = round (  (ub - lb) *rand() + lb  );
        
        Cross_P2 = Cross_P1;
        
        while Cross_P2 == Cross_P1
            Cross_P2 = round (  (ub - lb) *rand() + lb  );
        end
        
        if Cross_P1 > Cross_P2
            temp =  Cross_P1;
            Cross_P1 =  Cross_P2;
            Cross_P2 = temp;
        end
        
        Part1 = parent1.Gene(1:Cross_P1);
        Part2 = parent2.Gene(Cross_P1 + 1 :Cross_P2);
        Part3 = parent1.Gene(Cross_P2+1:end);
        
        child1.Gene = [Part1 , Part2 , Part3];
        
        
        Part1 = parent2.Gene(1:Cross_P1);
        Part2 = parent1.Gene(Cross_P1 + 1 :Cross_P2);
        Part3 = parent2.Gene(Cross_P2+1:end);
        
        child2.Gene = [Part1 , Part2 , Part3];
end
R1 = rand();
if R1 <= Pc
    child1 = child1;
else
    child1 = parent1;
end
R2 = rand();

if R2 <= Pc
    child2 = child2;
else
    child2 = parent2;
end
end

function [ newPopulation2 ] = elitism(population , newPopulation, Er)
M = length(population.Chromosomes);
Elite_no = round(M * Er);
[max_val , indx] = sort([ population.Chromosomes(:).fitness ] , 'descend');
for k = 1 : Elite_no
    newPopulation2.Chromosomes(k).Gene  = population.Chromosomes(indx(k)).Gene;
    newPopulation2.Chromosomes(k).fitness  = population.Chromosomes(indx(k)).fitness;
end

for k = Elite_no + 1 :  M
    newPopulation2.Chromosomes(k).Gene  = newPopulation.Chromosomes(k).Gene;
    newPopulation2.Chromosomes(k).fitness  = newPopulation.Chromosomes(k).fitness;
end
end


function [ population ] = initialization(M,dataFile)

for i = 1 : M
    for j = 1 : size(dataFile,2)
        population.Chromosomes(i).Gene(j) =round(rand());
    end
end

end

function [child] = mutation(child, Pm)

Gene_no = length(child.Gene);

for k = 1: Gene_no
    R = rand();
    if R < Pm
        child.Gene(k) = ~ child.Gene(k);
    end
end

end
