filename = 'logFile150.txt';

fid = fopen(filename);

train_size = [];
score = [];
reliability = [];
while ~feof(fid)
     line = fgets(fid);
     if str2num(line) == 0.0001
         line = fgets(fid);
         train_size = [train_size; str2num(line)];
     elseif startsWith(line,'score')
         score = [score; str2num(line(9:end))];
         line = fgets(fid);
         if line ~= -1, reliability = [reliability; str2num(line)]; end
     end
end

fclose(fid);

plot(score);


save('self150.mat','score','train_size','reliability');