clear
t = [];
ti = [];
for nsample = [500,1000,2000,4000,8000,12000]
    load(sprintf('RJ-MLE%d',nsample))
    t = [t; mean(e2), std(e2)];
    ti = [ti;  mean(time2), std(time2)];
end

figure;
subplot(2,1,1)
hold on; h= errorbar([500,1000,2000,4000,8000,12000], t(:,1),t(:,2), 'linewidth',2);
h = gca;
h.XScale = 'log';
grid on;

subplot(2,1,2)
hold on; h = errorbar([500,1000,2000,4000,8000,12000], ti(:,1), ti(:,2), 'linewidth', 2);
h = gca;
h.XScale = 'log';
grid on;

load SM1.mat
subplot(2,1,1)
h = rectangle('position', [500,mean(e1) - std(e1), 12000-500, std(e1)*2]);
h.FaceColor = [1,0,0,0.2];
h.EdgeColor = 'none';
plot([500,12000],[mean(e1), mean(e1)],'r','linewidth',2); 
h = ylabel('$\|\theta^* - \hat{\theta}\|$'); h.Interpreter = 'latex';

subplot(2,1,2)
h = rectangle('position', [500,mean(time1) - std(time1), 12000-500, std(time1)*2]);
h.FaceColor = [1,0,0,0.2];
h.EdgeColor = 'none';
plot([500,12000],[mean(time1),mean(time1)],'r','linewidth',2);
ylabel('Single run CPU time (s)')
xlabel('# Samples for Rej. Sampling')

