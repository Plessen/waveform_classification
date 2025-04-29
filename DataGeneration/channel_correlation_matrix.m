function cov = channel_correlation_matrix(Number_antennas, fc, d)
    lambda = 3e8 / fc;

    vdis = 0:d:(d * Number_antennas);
    Mdis = zeros(Number_antennas, Number_antennas);

    for i=1:Number_antennas
        for j=1:Number_antennas
            Mdis(i, j) = abs(vdis(i) - vdis(j));
        end
    end
    
    cov = besselj(0,2*pi*Mdis/lambda);
end