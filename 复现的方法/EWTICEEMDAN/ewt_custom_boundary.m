function [signal_below_4Hz, signal_above_4Hz, wfb] = ewt_custom_boundary(eeg_signal, fs, cutoff_freq)
    % Implements the core principle of EWT with a user-defined frequency boundary
    % as described in the provided literature.
    %
    % INPUTS:
    %   eeg_signal  - The input single-channel EEG signal (a vector).
    %   fs          - The sampling frequency in Hz.
    %   cutoff_freq - The desired cutoff frequency in Hz (e.g., 4).
    %
    % OUTPUTS:
    %   signal_below_4Hz - The decomposed signal component below the cutoff.
    %   signal_above_4Hz - The decomposed signal component above the cutoff.
    %   wfb              - The filter bank used for decomposition.

    % Ensure the input is a column vector
    eeg_signal = eeg_signal(:);
    N = length(eeg_signal);

    % 1. Go to the Frequency Domain
    f_hat = fft(eeg_signal);

    % Create the angular frequency vector from 0 to 2*pi
    w = (0:N-1)' * 2 * pi / N;

    % 2. Define the Frequency Boundary
    % Convert the cutoff frequency from Hz to radians per sample
    omega_n = cutoff_freq * 2 * pi / fs;

    % 3. Build the Filter Bank (Empirical Scaling and Wavelet functions)
    % This part builds the filters around the boundary omega_n. We need a small
    % transition width (gamma) for the Meyer wavelet construction.
    
    % A simple transition width choice (can be adjusted)
    gamma = 0.1 * omega_n; 

    % Initialize filter bank (2 filters for 1 boundary)
    wfb = zeros(N, 2);

    % --- Build the Low-Pass Filter (Scaling Function phi) ---
    for k = 1:N
        if w(k) <= omega_n - gamma
            wfb(k, 1) = 1;
        elseif w(k) >= omega_n + gamma
            wfb(k, 1) = 0;
        else % Transition region
            wfb(k, 1) = cos( (pi/2) * beta( (w(k) - (omega_n - gamma)) / (2*gamma) ) );
        end
    end
    
    % --- Build the High-Pass Filter (Wavelet Function psi) ---
    for k = 1:N
        if w(k) <= omega_n - gamma
            wfb(k, 2) = 0;
        elseif w(k) >= omega_n + gamma && w(k) <= pi % up to Nyquist for real signals
            wfb(k, 2) = 1;
        elseif w(k) > pi
             wfb(k, 2) = 0; % Symmetrical part for real fft
        else % Transition region
            wfb(k, 2) = sin( (pi/2) * beta( (w(k) - (omega_n - gamma)) / (2*gamma) ) );
        end
    end

    % For real signals, the filter bank must be symmetric
    % We only defined it for the first half, now we mirror it.
    wfb(floor(N/2)+2:end, :) = wfb(ceil(N/2):-1:2, :);

    % 4. Apply the Filters and Reconstruct
    % Low-frequency component
    f_hat_low = f_hat .* wfb(:, 1);
    signal_below_4Hz = real(ifft(f_hat_low));

    % High-frequency component
    f_hat_high = f_hat .* wfb(:, 2);
    signal_above_4Hz = real(ifft(f_hat_high));
end

% Helper function beta(x) for Meyer wavelet construction, as seen in EWT theory
function y = beta(x)
    if x < 0
        y = 0;
    elseif x > 1
        y = 1;
    else
        y = x^4 * (35 - 84*x + 70*x^2 - 20*x^3);
    end
end