figure;
for i = 0:9
subplot(2,5,i+1)
load(sprintf('res%d',i))
errorbar(.1:.1:.5, mean(sTruncSM,1),std(sTruncSM),'r','linewidth',2)
hold on;
errorbar((.1:.1:.5)-.0025, mean(sMLE,1),std(sMLE),'k','linewidth',2)
errorbar((.1:.1:.5)+.0025, mean(sMLEt,1),std(sMLEt),'b','linewidth',2)
axis([.05, .55, -19,-13])
title(sprintf('class %d', digit))
grid on
end