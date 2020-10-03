function runCaiCVPR2011MVSC()
% 
% function runCaiCVPR2011MVSC()
% 

dbstop if error

maxiter = 1; % run testing for this many times
outpath = 'output';

% ***** Full version
% dbnames = {'handwritten', 'Caltech101-7', 'Caltech101-20', 'NUSWIDEOBJ','Reuters'};
% rvalues = [10.^0.5, 10.^0.5, 10.^0.5, 10.^1.1, 10.^1.3];
% ***** Small demo
dbnames = {'handwritten'};
rvalues = [10.^0.5];

for dbID = 1:length(dbnames);
    dbname = dbnames{dbID};
    [X, truth, nc] = loaddata(dbname); 

    if ~exist(outpath, 'dir'),
        mkdir(outpath);
    end

    %[r, bestAC] = serchBestR(X, truth, nc);
    % r = 10.^1.3; % UCI Handwritten 5 view
    % r = 10.^0.5; % UCI Handwritten 6 view
    % r = 10.^1.3; % Reuters
    % r = 10.^0.5; % Caltech-7
    % r = 10.^1.1; % nus_wide
    % r = 10.^0.5; % MNIST
    r = rvalues(dbID);

    %===================================================================
    % Experiment
    [mFBase mP mR mNMI mRI mPrt mAC mTime] = deal([]);
    j = 1;
    if min(truth) ==0, truth = truth + 1; end
    for i = 1:maxiter,
        i
        tidID = tic; 
        [res, ~] = MVSC(X, nc, r);
        if min(res) ==0, res = res + 1; end
        elapseTime = toc(tidID);
        
        res = bestMap(truth,res);
        AC = length(find(truth == res))/length(truth)
        MIhat = MutualInfo(truth,res)
        [purityprt] = purity(res, truth , nc)
        %[mynmi] = nmi(gnd, res)
        [~, mynmi] = compute_nmi(truth, res)
        [Fmeasure,Precision, Recall] = compute_f(res, truth)
        [ARI]=RandIndex(truth,res)

        [mFBase(j, i) mP(j, i) mR(j, i) mNMI(j, i)...
            mRI(j, i) mPrt(j, i) mAC(j, i) mTime(j, i)] = ...
        deal(Fmeasure,Precision, Recall, mynmi, ARI, purityprt, AC, elapseTime);
    end

    fn = fullfile(outpath, sprintf('cai11_%s.mat', dbname));
    save(fn, 'mFBase', 'mP', 'mR', 'mNMI', 'mRI', 'mPrt', 'mAC', 'mTime');

end

% *************************************************************************
function [bestr, bestAC] = serchBestR(X, truth, nc)
% 
% function [bestr, bestAC] = serchBestR(X, truth, nc)
% 
bestAC = -inf;
bestr = 0;
Ln = [];
for r = 10.^0.5,%10.^(0.1:0.2:2), %r = 10.^0.3;%
    r
    %***** Bipartite clustering
    if isempty(Ln),
        tic;[res, Ln] = MVSC(X, nc, r); toc
    else
        tic;[res] = MVSC(X, nc, r, 'kmeans', Ln); toc
    end
    res = bestMap(truth,res);
    AC = length(find(truth == res))/length(truth)
    MIhat = MutualInfo(truth,res)
    [purityprt] = purity(res, truth , nc)
    %[mynmi] = nmi(gnd, res)
    [~, mynmi] = compute_nmi(truth, res)
    [Fmeasure,Precision, Recall] = compute_f(res, truth)
    if bestAC < AC,
        bestAC = AC;
        bestr = r;
    end
end

bestAC
bestr