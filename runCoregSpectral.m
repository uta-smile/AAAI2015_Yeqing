function runCoregSpectral()
% 
% function runCoregSpectralExp()
% 

dbstop if error

maxiter = 1; % run testing for this many times

% ***** Full version
% dbnames = {'handwritten', 'Caltech101-7', 'Caltech101-20', 'NUSWIDEOBJ','Reuters'};
% ***** Small demo
dbnames = {'handwritten'};

for dbID = 1:length(dbnames);
    dbname = dbnames{dbID};
    [X, truth, numClust] = loaddata(dbname); 
    num_views = length(X);

    lambda = 0.5; 
    numiter = 5;

    outpath = 'output';
    if ~exist(outpath, 'dir'),
        mkdir(outpath);
    end

    [mFBase mP mR mNMI mRI mPrt mAC mTime] = deal([]);

    %===================================================================
    % 
    % ***** Single view feature spectral clustering
    % 
    for i = 1:maxiter,

        fprintf('Running iteration %d with the single feature\n', i);
        for j = 1:num_views,
            ticID = tic;
            [V E F P R nmi avgent AR prt AC] = baseline_spectral(X{i},numClust,optSigma(X{i}),truth);
            [mFBase(j, i) mP(j, i) mR(j, i) mNMI(j, i)...
                mRI(j, i) mPrt(j, i) mAC(j, i) mTime(j, i)] = ...
                deal(F, P, R, nmi, AR, prt, AC, toc(ticID));
        end
    end

    %===================================================================
    % 
    % ***** Concat feature spectral clustering
    % 
    Xall = cell2mat(X); % [X1 X2 X3 X4 X5]
    for i = 1:maxiter,

        j = num_views + 1;
        fprintf('Running iteration %d with the feature concatenation of all views\n',i);
        ticID = tic;
        [V E F P R nmi avgent AR prt AC] = baseline_spectral(Xall,numClust,optSigma(Xall),truth);
        [mFBase(j, i) mP(j, i) mR(j, i) mNMI(j, i)...
            mRI(j, i) mPrt(j, i) mAC(j, i) mTime(j, i)] = ...
            deal(F, P, R, nmi, AR, prt, AC, toc(ticID));
    end    
    clear Xall

    %===================================================================
    % 
    % ***** Pairwise/Centroid consens multi-view spectral clustering
    % 
    sigma = zeros(1, num_views);
    for v = 1:num_views,
        sigma(v) = optSigma(X{v});
    end
    for i = 1:maxiter,
        % % *****  multiview spectral (pairwise): more than 2 views
        % fprintf('Multiview spectral with 3 views\n');
        % [F P R nmi avgent AR] = spectral_pairwise_multview(X,num_views,numClust,sigma,lambda,truth,numiter);

        j = num_views + 2;
        % *****  multiview spectral (centroid): more than 2 views
        fprintf('Running  iteration %d Multiview spectral \n', i);
        % lambda = [0.5 0.5 0.5];
        lambda = ones(1, num_views)*0.5;
        ticID = tic;
        [F P R nmi avgent AR prt AC] = spectral_centroid_multiview(X,num_views,numClust,sigma,lambda,truth,numiter);
        [mFBase(j, i) mP(j, i) mR(j, i) mNMI(j, i)...
            mRI(j, i) mPrt(j, i) mAC(j, i) mTime(j, i)] = ...
            deal(F(end), P(end), R(end), nmi(end), AR(end), prt(end), AC(end), toc(ticID));
    end

    fn = fullfile(outpath, sprintf('coreg_%s.mat', dbname));
    save(fn, 'mFBase', 'mP', 'mR', 'mNMI', 'mRI', 'mPrt', 'mAC', 'mTime');
end 
