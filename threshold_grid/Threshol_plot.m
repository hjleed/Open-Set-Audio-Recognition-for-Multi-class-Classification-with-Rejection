clear all; close all;
% Initialize
classList = {'knock', 'printer', 'keys', 'drawer', 'speech', 'keyboard','unknown'};
thresholds=[1.95, 1.975, 2.0, 2.025, 2.05, 2.075, 2.10, 2.125, 2.15, 2.175, 2.20, 2.225, 2.25, 2.275, 2.3];
N=length(classList);

BasePath  = 'xlxs';
S = dir(fullfile(BasePath,'*.xlsx'));
frame_aeer=[];
event_aeer=[];
frame_acc=[];
event_acc=[];
frame_j=[];
event_j=[];
frame_f1=[];
event_f1=[];

for ii = 1:length(S)
    fprintf('---------- Threshod: %2.3f -----------\n', thresholds(ii))
    Fname = fullfile(S(ii).folder,S(ii).name);
    t1 = readtable(Fname);
    Gtrue = t1.file;
    Gpred = t1.predicted;
    %%  Compute confusion matrix for Frame
    fprintf('(( Frame-based )) \n')
    tp=0; fp=0;
    tn=0; fn=0;
    for j=1:length(Gpred)
        if ismember(regexprep(Gtrue{j},'[\d"]',''),classList)
            Ntrue=regexprep(Gtrue{j},'[\d"]','');
            if strcmp(Ntrue,Gpred{j})
                tp=tp+1;
            else
                fp=fp+1;
            end    
        else 
            Ntrue='unknown';
            if strcmp(Ntrue,Gpred{j})
                tn=tn+1;
            else
                fn=fn+1; 
            end 
        end
        
       aeer=(fp+fn)/(tp+tn+fn+fp);
       acc= (tp+tn)/(tp+tn+fp+fn);
       j_v=(tp/tp+fn)+(fp/fp+tn);
       PPV = tp/(tp+fp); % precision
       TNR = tn/(tn+fp); %Specificity
       NPV = tn/(tn+fn);
       F1Score = 2 * (PPV * TNR) / (PPV + TNR);
    end

     %---------------------------------
    frame_aeer=[frame_aeer,aeer];
    frame_acc =[frame_acc, acc];
    frame_j =[frame_j, (j_v)];
    frame_f1 =[frame_f1, F1Score];
    
    %% Compute confusion matrix for event
    fprintf('\n  (( Event-based ))  \n')
    
    holdPar=Gtrue{1};
    psr_sum=0; k=1;
    temp = {}';
    NPtrue = {}';
    NPpred = {}';
    for j=1:length(Gtrue)
        
        if find(strmatch(holdPar, Gtrue{j}, 'exact'))
            temp{end+1} = Gpred{j};
        else
            NPtrue{end+1} = holdPar;
            [sval,~,jval]=unique(temp);
            NPpred{end+1} = sval{mode(jval)};
            k=k+1;
            holdPar=Gtrue{j}; 
            temp = {};
            psr_sum=0;
        end    
    end
    tp=0; fp=0;
    tn=0; fn=0;
    for j=1:length(NPtrue)
        if ismember(regexprep(NPtrue{j},'[\d"]',''),classList)
            Ntrue=regexprep(NPtrue{j},'[\d"]','');
            if strcmp(Ntrue,NPpred{j})
                tp=tp+1;
            else
                fp=fp+1;
            end  
        else 
            Ntrue='unknown';

            if strcmp(Ntrue,NPpred{j})
                tn=tn+1;
            else
                fn=fn+1; 
            end 
        end
       aeer=(fp+fn)/(tp+tn+fn+fp);
       acc= (tp+tn)/(tp+tn+fp+fn);
       j_v=(tp/tp+fn)+(fp/fp+tn);
       PPV = tp/(tp+fp); % precision
       TNR = tn/(tn+fp); %Specificity
       NPV = tn/(tn+fn);
       F1Score = 2 * (PPV * TNR) / (PPV + TNR);
    end
   
     %---------------------------------
    event_aeer=[event_aeer,aeer];
    event_acc =[event_acc, acc];
    event_j =[event_j, (j_v)];
    event_f1 =[event_f1, F1Score];

end

figure,subplot(121)
plot(thresholds,frame_acc,thresholds,frame_aeer);grid
legend('accuracy','Error Rate','location','best')
xlabel('threshold')
title('frame-based evaluation')

subplot(122),plot(thresholds,event_acc,thresholds,event_aeer);
grid
legend('accuracy','Error Rate','location','best')
xlabel('threshold')
title('event-based evaluation')


figure, plot(event_j)
figure,plot(event_f1)
