%% Performance Evaluation for the Proposed Approach
clear,clc

%Parameters initalization
M=10; %numero domini
n_sim=1;

%Step 1
I=zeros(5,8); %8 id 
N=size(I,1);
I(:,2) = [0.200 2 0.100 0.200 1]; %deadline in s 
I(:,1) = [0.1*10^6 1000*10^6 100*10^6 0.1*10^6 10*10^6]; %thr
I(:,4) = [100 90 30 60 10]; %budg
I(:,3) = [3 1.5 1 2 1.8]; %impaxt
I(:,6) = [1 2.5 3 2.5 2.8]; %risk appetite
I(:,7) = [5 8 8 10 6]; %pi
I(:,8) = [1 2 3 4 5]; % intent category id

for int=1:N
    I(int,5)=randi([300*10^3,600*10^3]); %Kbit
end
    

while n_sim<=250
    num = 10;

    if (0 <= n_sim) && (n_sim <= 50)
        lim_inf = 750;
        lim_sup = 1000;
        n_utenti = 750;
        I_input = zeros(n_utenti, 8);
    
    elseif (51 <= n_sim) && (n_sim <= 100)
        lim_inf = 1000;
        lim_sup = 1500;
        n_utenti = 1000;
        I_input = zeros(n_utenti, 8);
    
    elseif (101 <= n_sim) && (n_sim <= 150)
        lim_inf = 1500;
        lim_sup = 2000;
        n_utenti = 1500;
        I_input = zeros(n_utenti, 8);
    
    
    elseif (151 <= n_sim) && (n_sim <= 200)
        lim_inf = 2000;
        lim_sup = 2500;
        n_utenti = 2000;
        I_input = zeros(n_utenti, 8);
     elseif (201 <= n_sim) && (n_sim <= 250)
        lim_inf = 2500;
        lim_sup = 2800;
        n_utenti = 2500;
        I_input = zeros(n_utenti, 8);
    
    elseif n_sim > 250
        % if n_sim > 250, return
        return;
    else
        error('n_sim is out of the expected range');
    end


    rng(13)
    for shift=0:8:72
        for time=1:48
            D_tot(time,1+shift)=randi([40*10^6,2000*10^6]); 
            D_tot(time,2+shift)=rand; 
            D_tot(time,3+shift)=randi([1,2])*10^(-3);
            D_tot(time,4+shift)=randi([1,2])*10^(-3);
            D_tot(time,5+shift)=randi([1,2]);
            D_tot(time,6+shift)=randi([3,5])*10^(-3);
            D_tot(time,7+shift)=randi([lim_inf,lim_sup]);
            D_tot(time,8+shift)=randi([100*10^6,1000*10^6]);
        end
    end
    
    categories = [1 2 3 4 5]; % Intent categories

    %Reqs distribution for scenario. ex: 40% category 1 reqs, 10%
    %category 2 reqs...
    
    scenario_1 = [1 1 1 2 3 3 4 4 4 5];
    scenario_2 = [1 2 2 2 3 3 3 4 5 5];
    scenario_3 = [1 2 2 3 3 4 5 5 5 5];
    scenario_4 = [1 1 1 1 2 3 4 4 5 5];
    
    rng(n_sim);

    %The scenario must be changed accordingly.
    for num=1:n_utenti
        indx = randi(length(scenario_4),1); % Index selection 
        I_input(num,:) = [I(scenario_4(indx),:)];% Number of requests belonging to a given intent category is queued 
    end


    D=zeros(M,8);
    N=size(I_input,1); 
    
    signs_d = 1; 
    signs_i = [1 -1 -1 -1];
    
    allocated_c = zeros(N,M+1); 
    intents_in = zeros(5,11);
    intents_in(:,1)=1:5;
    intents = zeros(N,M+1);
    intents(:,1)=1:N;
    domains = zeros(M,N+1);
    domains(:,1)=1:M;
    matchings = zeros(M,N); %values in the (i,j) position are the allocated tasks
    q_domains = zeros(1,M); %q_j
    q_intents=zeros(1,N);
    
    pref_d=cell(48,M); 
    pref_i=cell(48,N);
    
    results_ca=cell(48,20); % to store the results

    results_ca{48,19}=I_input;
    
    tc_tot=0;
    tc_tot_task=zeros(1,N);
    deadline_outages=0;
    risk_tot_task=zeros(1,N);
    risk_outages=0;
    thr_tot_task=zeros(1,N);
    throughput_outages=0;
    budg_tot_task=zeros(1,N);
    rev_dom=zeros(1,M);
    budget_outages=0;
    
    time=1;


    while time<=48
        dom=1;
        for shift=0:8:72
            D(dom,:)=D_tot(time,1+shift:8+shift); 
            dom=dom+1;
        end
        
    
        for d=1:M
            q_domains(1,d) = D(d,7);%q_j
        end

        og_availability = q_domains;
        
        for i=1:N
            q_intents(1,i) = I_input(i,7);
        end
        
    
        for k=1:M
            [decision_matrix_d] = dm_d(D,I_input,k);
            decision_matrix_d = decision_matrix_d./max(decision_matrix_d);
            w_obj_d=1;
            [preferences] = topsis(decision_matrix_d,w_obj_d,signs_d); 
            domains(k,2:end) = real(preferences)'; 
            pref_d{time,k}=preferences;
            clear w w_obj_d norm_mat_d decision_matrix_d preferences;
            beta_parameter=0.5;
        end
    
        for l=1:size(I,1)
            decision_matrix_i = dm_i(D,I_input,l);
            decision_matrix_i = decision_matrix_i./max(decision_matrix_i);
            [w_obj_i,norm_mat_i] = ewm_i(decision_matrix_i,signs_i);
            [preferences] = topsis(norm_mat_i,w_obj_i,signs_i); 
            intents_in(l,2:end) = real(preferences);
            pref_i{time,l}=preferences;
            clear w w_obj_i norm_mat_i decision_matrix_i preferences;
            beta_parameter=0.5;
        end 
        
        k=0; l=0;
        [intents,pref_i]=populate_preferences(I_input,intents_in,pref_i,time);      
        condizione=~isempty(intents(:,2:11));
    
        while (condizione)
            for k=1:size(intents,1) %intents send out proposals to their most preferred domain
                [rc,id_dom] = max(intents(k,2:end)); 
                
                if (q_intents(k)~=0)
                    if (q_domains(id_dom) > 0) % if room
                        if (q_intents(k) <= q_domains(id_dom)) % if the domain can completely execute the request 
                            matchings(id_dom,k) = q_intents(k); % added as a matched domain with the right # of tasks
                            q_domains(id_dom)=q_domains(id_dom)-q_intents(k); % updated tasks, preference and availability
                            q_intents(k)=0; 
                            intents(k,id_dom+1) = 0; 
                            allocated_c(k,:) = intents(k,:); %added to this list in case it has to be replaced
                            intents(k,2:end) = 0;
                            
                        else %partially executed by a domain
                            matchings(id_dom,k)=q_domains(id_dom); 
                            q_intents(k)=q_intents(k)-q_domains(id_dom); 
                            q_domains(id_dom)=0; 
                            intents(k,id_dom+1) = 0;
                            allocated_c(k,:) = intents(k,:);             
                        end
                        
                    else % no space available, looking for a replacement
                        [rc_m,id_int_m] = min(domains(id_dom,2:end));
                        if (rc_m < rc && matchings(id_dom,id_int_m) ~= 0)
                            matchings(id_dom,k) = matchings(id_dom,id_int_m); 
                            q_intents(k) = q_intents(k)-matchings(id_dom,k); 
                            q_intents(id_int_m)=q_intents(id_int_m)+matchings(id_dom,id_int_m); 
                            allocated_c(k,:) = intents(k,:); 
                            intents(id_int_m,:) = allocated_c(id_int_m,:);
                            
                            if (matchings(id_dom,id_int_m)==q_intents(id_int_m)) % if the intent was completely matched with a domain
                                allocated_c(id_int_m,:) = 0;
                                intents(k,id_dom+1) = 0;  
                            else
                                intents(k,id_dom+1) = 0;
                            end 
                            matchings(id_dom,id_int_m) = 0;
            
                        else 
                            intents(k,id_dom+1) = 0;
                        end
                    end
                end
                r=(intents(:,2:11)==0);
                exit=all(q_intents == 0);
                ex=all(q_domains == 0);
                if (r == 1) | exit
                    condizione = false;
                    break;
                elseif exit && ex
                    condizione = false;
                    break;
                end
            end
        end
     
        [dim_deadline_p] = t_c_p(D,I_input,matchings);
        [riskapp] = risks_p(D,I_input,matchings);
        [below_budg,revenues] = costs_p(D,I_input,matchings);
        [guarant_thr] = thr_p(D,I_input,matchings);
        ind_risk=0;
        ind_costs=0;
        all_met = 0;
        tot_risk_appetite=sum(I_input(:,6));

    % computation of metrics for the results
        
        for c=1:size(matchings,2) %for each intent
            num = sum(matchings(:,c)>0);
            tc_tot_task(c)=max(dim_deadline_p(c,2:end)); 
            risk_tot_task(c)=sum(riskapp(c,2:end))/num;
            thr_tot_task(c)=sum(guarant_thr(c,2:end));
            budg_tot_task(c)=sum(below_budg(c,2:end));
    
            if tc_tot_task(c)>dim_deadline_p(c,1) 
                deadline_outages=deadline_outages+1;
            end
            if risk_tot_task(c)>riskapp(c,1)
               risk_outages=risk_outages+1;
            end
            if thr_tot_task(c)<guarant_thr(c,1) && thr_tot_task(c) ~= inf
                throughput_outages=throughput_outages+1;
            end
            if budg_tot_task(c)>below_budg(c,1)
                budget_outages=budget_outages+1;
            end
            
            if thr_tot_task(c) == inf
                thr_tot_task(c) = 0;
            end

            if tc_tot_task(c)<dim_deadline_p(c,1) && risk_tot_task(c)<riskapp(c,1) && thr_tot_task(c)>guarant_thr(c,1) && budg_tot_task(c)<below_budg(c,1) 
                all_met = all_met + 1;
            end
            
            ind_costs=ind_costs+(below_budg(c,1)-budg_tot_task(c));
            risk_tot_task(c)=(risk_tot_task(c)*I_input(c,6))/tot_risk_appetite; %risk appetite
            
        end
    
        for c=1:size(matchings,1)
            rev_dom(c)=sum(revenues(c,:));
        end
    
    
        tc_tot=max(tc_tot_task);  
        thr_tot=sum(thr_tot_task);
        risk_tot=sum(risk_tot_task);
        results_ca{time,1}=tc_tot;
        tc_tot=0;
        results_ca{time,2}=deadline_outages;
        results_ca{time,3}=risk_outages;
        results_ca{time,4}=throughput_outages;
        results_ca{time,5}=budget_outages;
    
        count_not_allocated=0;
    
        for r=1:size(matchings,2) %for each intent
            if all(matchings(:,r) == 0) 
                count_not_allocated=count_not_allocated+1;
            end
        end
    
        count_all_allocated=0;
    
        for r=1:size(matchings,2)
            if sum(matchings(:,r)) == I_input(r,7) 
                count_all_allocated=count_all_allocated+1;
            end
        end
    
        count_partially_allocated=0;
    
        for r=1:size(matchings,2)
            if (sum(matchings(:,r)) ~= I_input(r,7) && sum(matchings(:,r) ~= 0)) 
                count_partially_allocated=count_partially_allocated+1;
            end
        end
    
        results_ca{time,6}=count_not_allocated;
        results_ca{time,7}=count_all_allocated;
        results_ca{time,8}=all_met;
        risk_index = sum(ind_risk)/n_utenti; 
        cost_index = sum(ind_costs)/n_utenti; 
        results_ca{time,15}=sum(risk_tot_task);
        results_ca{time,16}=cost_index;
        results_ca{time,17}=sum(rev_dom)/M;
        thr_tot=0;
        
        risk_index=0;
        cost_index=0;
        ind_risk=0;
        ind_costs=0;
        revenues=0;
        rev_dom=0;
    
    
        avg_tc = max(tc_tot_task)/n_utenti; 
        avg_risk = sum(risk_tot_task)/n_utenti;
        avg_th = sum(thr_tot_task)/n_utenti;
        avg_costs = sum(budg_tot_task)/n_utenti;
    
        results_ca{time,9}=avg_tc;
        results_ca{time,10}=avg_risk;
        results_ca{time,11}=avg_th;
        results_ca{time,12}=avg_costs;
    
        satisfaction_d=zeros(1,N);
        satisfaction_i=zeros(1,N);
    
        %if matchings(i,j) !=0, selectpref_d{j} = preferences and its (i,j)-th element
    
    
        for c=1:size(matchings,2) %for each intent
            for r=1:size(matchings,1) %for each domain
                if matchings(r,c) ~= 0
                    temp_pref_d=pref_d{time,r}';
                    max_pref_d=max(temp_pref_d);
                    satisfaction_d(c)=satisfaction_d(c)+temp_pref_d(c)/max_pref_d;
    
                    temp_pref_i=pref_i{time,c}';
                    max_pref_i=max(temp_pref_i);
                    satisfaction_i(c)=satisfaction_i(c)+temp_pref_i(r)/max_pref_i;           
                end
            end
        end
        
        
        for c=1:length(q_domains)
            og_availability(c) = ((og_availability(c)-q_domains(c))/og_availability(c))*100;
        end
    
        results_ca{time,13}=sum(satisfaction_d)/n_utenti;
        results_ca{time,14}=sum(satisfaction_i)/n_utenti;
        results_ca{time,18}=sum(og_availability)/10;
        results_ca{time,20}=matchings;
        matchings = zeros(M,N);
        allocated_c = zeros(N,M+1);
        intents = zeros(N,M+1);
        intents(:,1)=1:N;
        domains = zeros(M,N+1);
        domains(:,1)=1:M;
        matchings = zeros(M,N); 
        q_domains = zeros(1,M); %q_j
        q_intents=zeros(1,N);
        tc_tot_task=zeros(1,N);
        deadline_outages=0;
        risk_tot_task=zeros(1,N);
        risk_outages=0;
        thr_tot_task=zeros(1,N);
        throughput_outages=0;
        budg_tot_task=zeros(1,N);
        budget_outages=0;
        

        if time == 48
            if (0 <= n_sim) && (n_sim <= 50)
                n_sim_print = n_sim;
            elseif (51 <= n_sim) && (n_sim <= 100)
                n_sim_print = n_sim - 50;
            elseif (101 <= n_sim) && (n_sim <= 150)
                n_sim_print = n_sim - 100;
            elseif (151 <= n_sim) && (n_sim <= 200)
                n_sim_print = n_sim - 150; 
            elseif (201 <= n_sim) && (n_sim <= 250)
                n_sim_print = n_sim - 200;
            else
                error('n_sim is out of the expected range');
            end
     
            fname = sprintf('sca_s1_%d_%d.mat',n_utenti,n_sim_print);
            save(fname, 'results_ca');
        end

        clear avg_costs avg_th avg_risk avg_tc;
        time=time+1;
    end

    n_sim=n_sim+1;
end


%% FUNZIONI

function [decision_matrix_d] = dm_d(matriceD,matriceI,k)
%A decision matrix is created for each domain, specifically the k-th domain among the M considered.

decision_matrix_d = zeros(size(matriceI,1),1); % revenues

for r=1:size(matriceI,1)
    t_c = (1000*matriceI(r,5))/matriceD(k,1);
    decision_matrix_d(r,1) = matriceD(k,5)*t_c-(matriceD(k,3)+matriceD(k,4)*matriceD(k,7)+matriceD(k,6)*matriceD(k,2));
end
end

function [decision_matrix_i] = dm_i(matriceD,matriceI,k)
%Analogous function as ^. It is generated a decision matrix for the k-th intent. 

decision_matrix_i = zeros(size(matriceD,1),4); %thr, t_c,c_tot,risk

for r=1:size(matriceD,1)
    decision_matrix_i(r,1) = matriceD(r,8); 
    decision_matrix_i(r,2) =(1000*matriceI(k,5))/matriceD(r,1)+matriceI(k,5)/matriceD(r,8);
    decision_matrix_i(r,3) = matriceD(r,5)*decision_matrix_i(r,2);
    decision_matrix_i(r,4) = matriceD(r,2)*matriceI(k,3);
end
end


function [w_obj_i,mat_norm_I] = ewm_i(matrice,signs_i)

[m, n] = size(matrice);
R = zeros(m, n);
for j = 1:n
    if signs_i(j) == 1
        R(:, j) = matrice(:, j) / sqrt(sum(matrice(:, j).^2));
    elseif signs_i(j) == -1
        R(:, j) = min(matrice(:, j)) ./ matrice(:, j);
    end
end
P = R ./ sum(R);
mat_norm_I=R;

E_i = zeros(1,4);
w_obj_i = zeros(1,4);
M =size(mat_norm_I,1);

epsilon = 1e-12; 
for j = 1:4
    Ej = -sum(matrice(:, j) .* log(matrice(:, j) + epsilon)) / log(M);
    E_i(j) = Ej;
end

d = 1 - E_i;

w_obj_i = d / sum(d);


end


function [preferences] = topsis(decision_matrix,weights,signs)

preferences = zeros(1,size(decision_matrix,1));

%WEIGHTED MATRIX
V = weights .* decision_matrix; 

%PIS, NIS
V_IS = signs .* V;
maximum = abs(max(V_IS)); 
minimum = abs(min(V_IS));

max_matrix=repmat(maximum,size(V,1),1); % vertically stack max vector for number-of-rows times 
min_matrix=repmat(minimum,size(V,1),1); % vertically stack min vector for number-of-rows times
     
%DISTANCE
s_plus=zeros(size(V_IS,1),1);
s_minus=zeros(size(V_IS,1),1);

s_p=(V_IS-max_matrix).^2;
s_m=(V_IS-min_matrix).^2;

for k=1:size(V_IS,1)
    s_plus(k,1)=sum(s_p(k,:));
    s_minus(k,1)=sum(s_m(k,:));
end

dpos=sqrt(s_plus);
dneg=sqrt(s_minus);

sumT=dneg+dpos;
preferences=dneg./sumT;

    for col=1:size(decision_matrix,1)
        if isnan(preferences(col,:)) 
            preferences(col,:) = 0;
        end
    end
end


function [dim_deadline_p] = t_c_p(D,I,matchings)

dim_deadline_p = zeros(size(I,1),size(D,1)+1); %intents for each row, e2e columns
dim_deadline_p(:,1) = I(:,2); % deadline

for r=1:size(I,1)
    for c=1:size(D,1)
        if matchings(c,r)==0
            dim_deadline_p(r,c+1)=0;
        else
            dim_deadline_p(r,c+1)=(1000*I(r,5)/I(r,7))/(D(c,1))+((I(r,5)/I(r,7))/D(c,8));
        end
    end
end

end

function [guarant_thr] = thr_p(D,I,matchings)

guarant_thr = zeros(size(I,1),size(D,1)+1); %intents for each row, thr columns
guarant_thr(:,1) = I(:,1); % req thr

for r=1:size(I,1)
    for c=1:size(D,1)
        if matchings(c,r)~=0
            guarant_thr(r,c+1)=D(c,8); 
        end
    end
end

end

function [below_budg,revenues] = costs_p(D,I,matchings)

below_budg = zeros(size(I,1),size(D,1)+1); %intents rows, costs columns
below_budg(:,1) = I(:,4); % budg
revenues=zeros(size(I,1),size(D,1));

for r=1:size(I,1)
    for c=1:size(D,1)
        if matchings(c,r)==0
            below_budg(r,c+1)=0;
        else
            below_budg(r,c+1)= D(c,5)*(1000*I(r,5)/I(r,7))/(D(c,1));
        end
    end
end

for r=1:size(D,1)
    for c=1:size(I,1)
        if matchings(r,c)==0
            revenues(r,c)=0;
        elseif sum(matchings(:,r)) == I(r,7)
            revenues(r,c)= D(r,5)*(1000*I(c,5)/I(c,7))/(D(r,1))-(D(r,3)+D(r,4)+D(r,6)*D(r,2));
        else
            revenues(r,c)=0;
        end
    end
end

end

function [riskapp] = risks_p(D,I,matchings)

riskapp = zeros(size(I,1),size(D,1)+1); %intents rows, risk columns
riskapp(:,1) = I(:,6); % risk appetite

for r=1:size(I,1)
    for c=1:size(D,1)
        if matchings(c,r)~=0
            riskapp(r,c+1)= D(c,2)*(I(r,3));
        end
    end
end
end


function [intents,pref_i] = populate_preferences(I_input,intents_in,pref_i,time)

intents = zeros(size(I_input,1), size(intents_in,2));

% id extraction
IDs_I_input = I_input(:, 8);
IDs_intents_in = intents_in(:, 1);

% iteration
for i = 1:length(IDs_I_input)
    % find index
    idx = find(IDs_intents_in == IDs_I_input(i));
    
    if ~isempty(idx)
        % copy row
        intents(i, :) = intents_in(idx, :);
    end
    
    pref_i{time,i}=intents(i, 2:end);
end


end

