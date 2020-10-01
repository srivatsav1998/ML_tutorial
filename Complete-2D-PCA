fno=1;
images = zeros(112,93);
for i=1:40 
    foldername = strcat(strcat('/home/pvsuryachaitanya/Desktop/Matlab/Project/orl_faces/s',num2str(fno)),'/');
    files = fullfile(foldername,'*.pgm');
    faceimages = dir(files);
    for j = 1:length(faceimages)
        baseFileName = faceimages(j).name;
        fullFilename = fullfile(foldername,baseFileName);
        temp=imread(fullFilename);
        cls = zeros(112,1);
        cls(1,1)=i; 
        temp = horzcat(temp,cls);
        images = cat(3,images,temp);
    end
    fno = fno + 1 ;
end
images = double(images);
images(:,:,1) = [ ];
rng(20);
p = randperm(400);
tr = p(1:300);
tst = p(301:400);
traindata = images(:,:,tr);
testdata = images(:,:,tst);
meanimage = mean(traindata(:,1:92,:),3);
G1 = zeros(112,112);
G1 = double(G1);
G2 = zeros(92,92);
G2 = double(G2);
for i=1:length(traindata)
    G1 = G1 + (traindata(:,1:92,i)-meanimage)*transpose(traindata(:,1:92,i)-meanimage);
end
for i=1:length(traindata)
    G2 = G2 + transpose(traindata(:,1:92,i)-meanimage)*(traindata(:,1:92,i)-meanimage);
end
[V1,D1] = eig(G1);
[V2,D2] = eig(G2);
[c , ind] = sort(diag(D1),'descend');
V1 = V1(:,ind);
[c , ind] = sort(diag(D2),'descend');
V2 = V2(:,ind);
V1 = V1(1:6,:);
V2 = V2(:,1:6);
projectedimages = zeros(6,6);
for i=1:length(traindata)
    projectedimages(:,:,i)=V1*traindata(:,1:92,i)*V2;
end
count=0.0;
for i=1:100
    featuretest = V1*testdata(:,1:92,i)*V2;
    distance = zeros(300);
    for j = 1:length(traindata)
        distance(j) = norm(projectedimages(:,:,j)-featuretest); 
    end
    [dist, ind] = min(distance);
    subplot(2,2,1);
    imshow(testdata(:,:,i),[]);
    title(['Tested Face = ' num2str(i)]);
    subplot 222;
    imshow(traindata(:,:,ind(1)),[]);
    title(['Recognized Face = ' num2str(ind(1))]);
    if(traindata(1,93,ind(1))==testdata(1,93,i))
        count = count + 1;
    end
end