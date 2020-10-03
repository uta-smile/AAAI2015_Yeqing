function runPrintResult()
% 
% function runPrintResult()
% 
% Print the clustering accuracy measures stored in output/ diriectory.
% Used by AAAI15 MVSC paper
% 

algo = {'coreg', 'MMVCC'};
% algo = {'MMVCC'};
% algo = {'cai11','MMVCC'};
dbs = {'handwritten','Caltech101-7','Caltech101-20','Reuters','NUSWIDEOBJ'};
% dbs = {'reuters', 'nus_wide', 'MNIST'};

% flds = {'mAC','mFBase','mNMI','mP','mPrt','mR','mRI','mTime'};
flds = {'mNMI','mPrt','mRI','mFBase','mTime'};

outpath = 'output';

for i = 1:length(dbs),
    disp([dbs{i} '================================']);
    for j = 1:length(algo),
        disp([algo{j} '-------------------------']);
        fn = fullfile(outpath, sprintf('%s_%s.mat', algo{j}, dbs{i}));
        if ~exist(fn, 'file'), continue; end
        t = load(fn);
        ret = [];
        disp(flds)
        for k = 1:length(flds)-1,
            ret = [ret mean(t.(flds{k}),2)];
        end
        disp(ret)
        fprintf('%.2f\n', mean(t.mTime,2))
    end
end