function avgED = avg_ed(a,b)
    totalED = 0;
    for i = 1:size(a,2)
        for j = 1:size(b,2)
            totalED = totalED + sqrt(sum((a(:,i)-b(:,j)).^2));
        end
    end
    avgED = totalED/(size(a,2)*size(b,2));
end