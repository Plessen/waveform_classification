function noisy_signal = fading_strategy(signal, SNR, strategy, Nr, numPaths_range, pathDelay_range, pathGain_range, Kfactor_range, fs, fc)
    N = length(signal); % Length of the input signal
    if strategy == "cooperative"
        ricianChan = cell(1, Nr);

        for i = 1:Nr
            PathNum = randi(numPaths_range,1,1);
            pathDelayConstant = randi(pathDelay_range,1,1);
            avgPathGainsConstant = randi(pathGain_range,1,1);
            
            pathDelays = [0:PathNum-1].*pathDelayConstant*1e-9;
            avgPathGains = -1*[0:PathNum-1].*avgPathGainsConstant;
            K = randi(Kfactor_range,1,1);

            ricianChan{i} = comm.RicianChannel(...
                'SampleRate', fs,...
                'PathDelays', pathDelays,...
                'AveragePathGains', avgPathGains,...
                'KFactor', K,...
                'MaximumDopplerShift', 0.001, ... 
                'DirectPathInitialPhase', rand() * 2 * pi,...
                'ChannelFiltering', true);
        end

        received_signal = zeros(N, Nr);
        for i = 1:Nr
            fadingOutput = ricianChan{i}(signal.');
            received_signal(:, i) = awgn(fadingOutput, SNR, 'measured');
        end
        
        aligned_signals = zeros(N, Nr);
        aligned_signals(:, 1) = received_signal(:, 1);
        for i = 2:Nr
            [r, ~] = xcorr(received_signal(:, 1), received_signal(:, i));
            phi_est = angle(r(N));
            aligned_signal = received_signal(:, i) * exp(1i*phi_est);
            aligned_signals(:, i) = aligned_signal;
        end
        noisy_signal = sum(aligned_signals, 2);
    elseif strategy == "independent" || strategy == "correlated"
        PathNum = randi(numPaths_range,1,1);
        pathDelayConstant = randi(pathDelay_range,1,1);
        avgPathGainsConstant = randi(pathGain_range,1,1);
        
        pathDelays = [0:PathNum-1].*pathDelayConstant*1e-9;
        avgPathGains = -1*[0:PathNum-1].*avgPathGainsConstant;
        K = randi(Kfactor_range,1,1);
        
        if strategy == "independent"
            cov = eye(Nr);
        else
            cov = channel_correlation_matrix(Nr, fc, 0.00001); % All antennas are fully correlated and equal to each other
        end
        mimoChan = comm.MIMOChannel("SampleRate",fs,"FadingDistribution","Rician", "PathDelays",pathDelays, "AveragePathGains",avgPathGains,...
        "KFactor",K, "MaximumDopplerShift", 0.001, "ChannelFiltering",true, "ReceiveCorrelationMatrix",cov, "TransmitCorrelationMatrix",1, "SpatialCorrelationSpecification","Separate Tx Rx",...
        "NormalizeChannelOutputs",false);

        fadingOutput = mimoChan(signal.');
        received_signal = awgn(fadingOutput, SNR, "measured");

        aligned_signals = zeros(N, Nr);
        aligned_signals(:, 1) = received_signal(:, 1);
        for i = 2:Nr
            [r, ~] = xcorr(received_signal(:, 1), received_signal(:, i));
            phi_est = angle(r(N));
            aligned_signal = received_signal(:, i) * exp(1i*phi_est);
            aligned_signals(:, i) = aligned_signal;
        end

        noisy_signal = sum(aligned_signals, 2);
    elseif strategy  == "selection"
        PathNum = randi(numPaths_range,1,1);
        pathDelayConstant = randi(pathDelay_range,1,1);
        avgPathGainsConstant = randi(pathGain_range,1,1);
        
        pathDelays = [0:PathNum-1].*pathDelayConstant*1e-9;
        avgPathGains = -1*[0:PathNum-1].*avgPathGainsConstant;
        K = randi(Kfactor_range,1,1);

        mimoChan = comm.MIMOChannel("SampleRate",fs,"FadingDistribution","Rician", "PathDelays",pathDelays, "AveragePathGains",avgPathGains,...
        "KFactor",K, "MaximumDopplerShift", 0.001, "ChannelFiltering",true, "ReceiveCorrelationMatrix",eye(Nr), "TransmitCorrelationMatrix",1, "SpatialCorrelationSpecification","Separate Tx Rx",...
        "NormalizeChannelOutputs",false);

        fadingOutput = mimoChan(signal.');
        received_signal = awgn(fadingOutput, SNR, "measured");
        
        [~, index] = max(sum(abs(received_signal).^2, 1));
        noisy_signal = received_signal(:, index);
    end
end