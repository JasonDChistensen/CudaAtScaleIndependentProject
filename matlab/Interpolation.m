%% Digital Interpolation Filter
% The purpose of this script is to write the code that produces that same output 
% as the built in Matlab function |filter|.  Which then can be used as a prototype 
% for other languages such as C++ or CUDA.
%% 
% Design a digital interpolation filter to upsample a signal by |upfactor|, 
% using the bandlimited method. Specify a "bandlimitedness" factor of 0.5 and 
% use $2\times2$ samples in the interpolation.

upfactor = 7;
alpha = 0.5;
h1 = intfilt(upfactor,2,alpha);
%% 
% Plot the filter coefficients.

stem(h1);
title(strcat("Interpolation Filter h1.  Upfactor: ", num2str(upfactor)))
xlabel('filter index')
ylabel('filter value')
saveas(gcf,strcat("InterpolationFilter", num2str(upfactor), ".jpg"))
%% 
% The filter works best when the original signal is bandlimited to |alpha| times 
% the Nyquist frequency. Create a bandlimited noise signal by generating N Gaussian 
% random numbers and filtering the sequence with a 40th-order FIR lowpass filter. 
% Reset the random number generator for reproducible results.

lowp = fir1(40,alpha);
N = 500;
rng('default')
x = filter(lowp,1,randn(N,1));
%% 
% Increase the sample rate of the signal by inserting zeros between each pair 
% of samples of |x|.

xr = upsample(x,upfactor);
%% 
% Use the Matlab |filter| function to produce an interpolated signal.

tic
matlabInterpolated = filter(h1,1,xr);
MatlabElapsedTime = toc
%  It takes about 1 second to upsample by 7 and number of sample for input signal is 50e6.
% Computer specs:
%   1)  11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz   2.50 GHz
%   2)  64.0 GB (63.6 GB usable)
%% 
% Use the user defined function |myInterpolation| to produce the interpolated 
% signal.

tic
userInterpolated = myInterpolation(x, upfactor, h1);
UserElapsedTime = toc
%  It takes about 23 seconds to upsample by 7 and the number of sample for input signal is 50e6.
% Computer specs:
%   1)  11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz   2.50 GHz
%   2)  64.0 GB (63.6 GB usable)
%% 
% 
%% 
% Compare the Matlab interpolated signal vs. the user defined interpolated signal

error = sum(abs(userInterpolated - matlabInterpolated))
% Error is 9.1057e-09 when upsampling by 7 and the number of sample for input signal is 50e6.
%% 
% 
%% 
% Store the input signal, interpolation filter, upsampled input signal,  and 
% the Matlabe interpolated signal.

writeToFile(strcat('inputSignal', num2str(upfactor), '.bin'), x)
writeToFile(strcat('interpFilter', num2str(upfactor), '.bin'), h1)
writeToFile(strcat('inputSignalUpsampled', num2str(upfactor), '.bin'), xr)
writeToFile(strcat('matlabInterpolatedOutput', num2str(upfactor), '.bin'), matlabInterpolated)
%% 
% 
%% 
% |Create a file that can be used to validate reading input files.|

writeToFileValidation
%% 
% Plot the results

x_to_plot = x(1:500);
matlabInterpolated_to_plot = matlabInterpolated(1:500);

% Compensate for the delay introduced by the filter. Plot the original and interpolated signals.
delay = mean(grpdelay(h1));
matlabInterpolated_to_plot(1:delay) = [];

stem(1:upfactor:upfactor*length(x_to_plot),x_to_plot)
hold on
plot(matlabInterpolated_to_plot)
xlim([50 50+upfactor*20])
title(strcat("Input signal (x) interpolated by ", num2str(upfactor)))
xlabel('sample index')
ylabel('sample value')
legend('Input Signal','Interpolated Signal')
saveas(gcf,strcat("InterpolationOutput", num2str(upfactor), ".jpg"))

%% 
% 
% 
% 
% 
% 
% 
% 
% 
%