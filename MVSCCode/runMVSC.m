function runMVSC

addpath func

for dbID = 6

load('handwritten6')
T = data;
gnd = labels;
r = 10.^0.7;
%r = 10.^1.3;

nc = length(unique(gnd));

opts.p = 200;
opts.r = 5;
opts.kmMaxIter = 30;
opts.maxWghtIter = 50;
opts.thresh = 1e-6;
opts.kertype = 'Gaussian';
if dbID == 2,
    opts.kertype = 'Linear';
end

maxiter = 5; 
totalAcc = 0;
totalTime = 0;
for i = 1:maxiter,
    i
    opts.wr = r;
    tidID = tic;
    [res, markslbl, marks, obj, Zv, alpha] = MVSC(T, nc, opts);
    if min(res) ==0, res = res + 1; end
    elapseTime = toc(tidID);
    res = bestMap(gnd,res);
    AC = length(find(gnd == res))/length(gnd)
    totalAcc = totalAcc + AC;
    totalTime = totalTime + elapseTime;
end

disp('Average Accuracy')
totalAcc/maxiter
disp('Average Time')
totalTime/maxiter

end

