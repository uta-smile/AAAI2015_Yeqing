function runProposedMVSC()
% 
% runProposedMVSC()
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

    % ***** Parameter setting
    % opts.p = 200;
    opts.p = 400;
    opts.r = 5;
    opts.kmMaxIter = 30;
    opts.maxWghtIter = 50;
    % opts.wr = 0;
    opts.thresh = 1e-6;
    opts.kertype = 'Gaussian';
    if strcmp(dbnames, 'Reuters'),
        opts.kertype = 'Linear';
    end
    % rand('twister',5489) 

    %[r, bestAC] = serchBestR(X, truth, nc);
    % r = 10.^0.7; % UCI Handwritten 6 view
    % r = 10.^1.7; % Reuters
    % r = 10.^1.1; % Caltech-7
    % r = 10.^1.7; % nus_wide
    % r = 10.^0.1; % MNIST
    r = rvalues(dbID);

    %===================================================================
    % Experiment
    %***** Bipartite clustering compute all results
    X = X(1);
    [mFBase mP mR mNMI mRI mPrt mAC mTime] = deal([]);
    j = 1;
    if min(truth) ==0, truth = truth + 1; end
    for i = 1:maxiter,
        i
        opts.wr = r;
        tidID = tic;
        [res, markslbl, marks, obj, Zv, alpha] = multiviewBiSC(X, nc, opts);
        if min(res) ==0, res = res + 1; end
        elapseTime = toc(tidID);
        
        %figure(2); plot(obj); title('Objective function values (BiSC)')
        res = bestMap(truth,res);
        AC = length(find(truth == res))/length(truth)
        MIhat = MutualInfo(truth,res)
        [purityprt] = purity(res, truth , nc)
        %[mynmi] = nmi(gnd, res)
        [~, mynmi] = compute_nmi(truth, res)
        [ARI]=RandIndex(truth,res)
        [Fmeasure,Precision, Recall] = compute_f(res, truth)

        [mFBase(j, i) mP(j, i) mR(j, i) mNMI(j, i)...
            mRI(j, i) mPrt(j, i) mAC(j, i) mTime(j, i)] = ...
        deal(Fmeasure,Precision, Recall, mynmi, ARI, purityprt, AC, elapseTime);
    
        %[res] = outofsample(Zv, markslbl, nc);
    end

    fn = fullfile(outpath, sprintf('MMVCC_%s.mat', dbname));
    save(fn, 'mFBase', 'mP', 'mR', 'mNMI', 'mRI', 'mPrt', 'mAC', 'mTime');

end

% *************************************************************************
function [bestr, bestAC] = serchBestR(X, truth, nc)
% 
% function [bestr, bestAC] = serchBestR(X, truth, nc)
% 
bestAC = -inf;
bestr = 0;
X = 0;
if min(truth) ==0, truth = truth + 1; end
%***** Bipartite clustering search best parameters
for r = 10.^(0.1:0.2:2),
    opts.wr = r;
    tic;[res, markslbl, marks, obj, Zv, alpha] = multiviewBiSC(X, nc, opts);toc
    if min(res) ==0, res = res + 1; end
    figure(2); plot(obj); title('Objective function values (BiSC)')
    res = bestMap(truth,res);
    AC = length(find(truth == res))/length(truth)
    MIhat = MutualInfo(truth,res)
    [purityprt] = purity(res, truth , nc)
    %[mynmi] = nmi(gnd, res)
    [~, mynmi] = compute_nmi(truth, res)
    [Fmeasure,Precision, Recall] = compute_f(res, truth)
    
    if bestAC < purityprt,
        bestAC = purityprt;
        bestr = r;
    end
end

% bestAC
% bestr
% *************************************************************************
function [res] = outofsample(Zv, markslbl, nc)
% 
% function outofsample()
% 
disp('Out-of-sample problem')
[nSmp, p, numView] = size(Zv);
truth = 0;

Z = sum(bsxfun(@times, Zv, alpha.^opts.wr), 3);
% Z = sum(bsxfun(@times, Zv, alpha), 3);
% Z = sparse(Z);
Z = sparse(double(Z > 0));

A = bsxfun(@eq, markslbl, 1:nc);
f = Z*A;
[~, res] = max(f, [], 2);
res = bestMap(truth,res);
AC = length(find(truth == res))/length(truth)
MIhat = MutualInfo(truth,res)
[purityprt] = purity(res, truth , nc)
[mynmi] = nmi(truth, res)
[Fmeasure,Precision, Recall] = compute_f(res, truth)
