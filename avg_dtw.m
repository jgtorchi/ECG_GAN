function avgDTW = avg_dtw(a,b)
    totalDTW = 0;
    for i = 1:size(a,2)
        for j = 1:size(b,2)
            totalDTW = totalDTW + dtw(a(:,i),b(:,j),'euclidean');
        end
    end
    avgDTW = totalDTW/(size(a,2)*size(b,2));
end