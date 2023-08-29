classdef sincNetLayer < nnet.layer.Layer

properties (Access = private)
        % Number of sincNet filters in the sincNet layer
        NumFilters;
        % Sampling frequency 
        SampleRate;
        % Length of the sincNet filter
        FilterLength;
        % Number of channels in the speech signal
        NumChannels;
        % Internal parameters computed within the layer
        Window;
        % Time axis for the filters
        TimeStamps;
        % Minumum starting frequency for the bandpass filters
        MinimumFrequency;
        % Minimum bandwidth for the bandpass filters
        MinimumBandwidth;       
    end

    properties (Learnable)
        % Starting frequency of the bandpass filter
        FilterFrequencies;
        % Bandwidth of the band pass filter
        FilterBandwidths;
    end
    
    methods
        function layer = sincNetLayer(NumFilters, FilterLength, SampleRate, NumChannels, Name)
            
            layer.NumFilters = NumFilters;
            layer.FilterLength = FilterLength;
            layer.SampleRate = SampleRate;
            layer.MinimumFrequency = 50.0;
            layer.MinimumBandwidth = 50.0;
            
            % Mel Initialization of the filterbanks
            % Following lines of code are adapted from the author's Python
            % code for sincNet
            low_freq_mel = 80;
            high_freq_mel = hz2mel(SampleRate/2 - layer.MinimumFrequency - layer.MinimumBandwidth);  % Convert Hz to mel
            mel_points = linspace(low_freq_mel,high_freq_mel,NumFilters);  % Equally spaced in mel scale
            f_cos = mel2hz(mel_points); % Convert mel to Hz
            b1 = circshift(f_cos,1);
            b2 = circshift(f_cos,-1); 
            b1(1) = 30; % Min b1 of filter = 30 Hz
            b2(end) = (SampleRate/2)-100;
            
            N = layer.FilterLength;
            layer.TimeStamps = linspace(-(N-1)/2,(N-1)/2,N)/layer.SampleRate;
            
            if mod(N,2) == 1
                layer.TimeStamps(round(N/2)) = eps;
            end
            
            % Hamming Window
            n = linspace(0,N,N);
            layer.Window = 0.54 - 0.46*cos(2*pi*n/N);
            
            % sincNet layer learnable parameters
            layer.FilterFrequencies = b1/SampleRate;
            layer.FilterBandwidths = (b2 - b1)/SampleRate;
            
            % Set layer name.
            layer.Name = Name;
            
            % Set layer description.
            layer.Description = "Sinc Layer with " + NumChannels + " channels";
            
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            F = generateSincFilter(layer);
            Filters = reshape(F,1,layer.FilterLength,1,layer.NumFilters,1);
            % Convolve input with sinc filters
            Z = dlconv(X,Filters,0,'DataFormat','SSCB');
        end
        
        function plotNFilters(layer,n)
            % This layer plots n filters with equally spaced filter indices
            [H,Freq] = freqDomainFilters(layer);
            idx = linspace(1,layer.NumFilters,n);
            idx = round(idx);
            for jj = 1:n
                subplot(ceil(sqrt(n)),ceil(n/ceil(sqrt(n))),jj);
                plot(Freq(:,idx(jj)),H(:,idx(jj)));
                sgtitle("Filter Frequency Response")
                xlabel("Frequency (Hz)")
            end
        end

    end
    
    methods (Access = private)
        
        function F = generateSincFilter(layer)
            % Returns the time-domain response of the Sinc parametrized
            % bandpass filters
            filt_beg_freq = abs(layer.FilterFrequencies) + layer.MinimumFrequency/layer.SampleRate;
            filt_end_freq = filt_beg_freq + (abs(layer.FilterBandwidths) + layer.MinimumBandwidth/layer.SampleRate);
            
            % Define Filter values
            low_pass1 = 2*filt_beg_freq.*sin(2*pi*layer.SampleRate*layer.TimeStamps'*filt_beg_freq) ...
                ./(2*pi*layer.SampleRate*layer.TimeStamps'*filt_beg_freq);
            low_pass2 = 2*filt_end_freq.*sin(2*pi*layer.SampleRate*layer.TimeStamps'*filt_end_freq) ...
                ./(2*pi*layer.SampleRate*layer.TimeStamps'*filt_end_freq);
            band_pass = low_pass2 - low_pass1;
            band_pass = band_pass./max(band_pass);
            F = reshape(layer.Window'.*band_pass,layer.FilterLength,layer.NumFilters);
        end
        
         function [H,Freq] = freqDomainFilters(layer)
            % Returns the magnitude the frequency domain representation of
            % the Sinc parametrized bandpass filters
            F = generateSincFilter(layer);
            H = zeros(size(F));
            Freq = zeros(size(F));
            for ii = 1:size(F,2)
               [h,f] = freqz(F(:,ii),1,layer.FilterLength,layer.SampleRate);
               H(:,ii) = abs(h);
               Freq(:,ii) = f;
            end
        end
    end
end